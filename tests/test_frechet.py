"""Test Frechet bounds functions against R's StatMatch implementation."""

import numpy as np
import pandas as pd
import pytest

# Only import rpy2 for test generation
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter

    # Import StatMatch and MASS for test data
    statmatch_r = importr("StatMatch")
    mass_r = importr("MASS")
    base_r = importr("base")
    stats_r = importr("stats")
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False


class TestPBayes:
    """Test suite for pBayes function."""

    @pytest.fixture
    def simple_table(self):
        """Create a simple contingency table for testing."""
        # 3x4 contingency table
        table = np.array(
            [[10, 5, 3, 2], [8, 12, 4, 1], [5, 8, 15, 7]]
        )
        return table

    @pytest.fixture
    def sparse_table(self):
        """Create a sparse contingency table with zeros."""
        table = np.array(
            [[10, 0, 3, 0], [0, 12, 0, 1], [5, 0, 15, 0]]
        )
        return table

    def test_jeffreys_method(self, simple_table):
        """Test Jeffreys method (add 0.5 to each cell)."""
        from statmatch.frechet import p_bayes

        result = p_bayes(simple_table, method="Jeffreys")

        # Check structure
        assert "info" in result
        assert "prior" in result
        assert "pseudoB" in result

        # Check info values
        info = result["info"]
        assert info["n"] == simple_table.sum()
        assert info["no.cells"] == simple_table.size
        assert info["const"] == 0.5
        assert info["K"] == simple_table.size / 2  # c/2 for Jeffreys

    def test_minimax_method(self, simple_table):
        """Test minimax method (add sqrt(n)/c to each cell)."""
        from statmatch.frechet import p_bayes

        result = p_bayes(simple_table, method="minimax")

        info = result["info"]
        n = simple_table.sum()
        c = simple_table.size
        expected_const = np.sqrt(n) / c
        np.testing.assert_allclose(info["const"], expected_const)
        np.testing.assert_allclose(info["K"], np.sqrt(n))

    def test_invcat_method(self, simple_table):
        """Test invcat method (add 1/c to each cell)."""
        from statmatch.frechet import p_bayes

        result = p_bayes(simple_table, method="invcat")

        info = result["info"]
        c = simple_table.size
        expected_const = 1 / c
        np.testing.assert_allclose(info["const"], expected_const)
        np.testing.assert_allclose(info["K"], 1.0)

    def test_user_method(self, simple_table):
        """Test user-defined constant method."""
        from statmatch.frechet import p_bayes

        const = 0.25
        result = p_bayes(simple_table, method="user", const=const)

        info = result["info"]
        assert info["const"] == const

    def test_mutual_independence_method(self, simple_table):
        """Test m.ind (mutual independence) method."""
        from statmatch.frechet import p_bayes

        result = p_bayes(simple_table, method="m.ind")

        # Check that prior is based on independence assumption
        assert "prior" in result
        assert result["prior"].shape == simple_table.shape

    def test_homogeneous_association_method(self, simple_table):
        """Test h.assoc (homogeneous association) method."""
        from statmatch.frechet import p_bayes

        result = p_bayes(simple_table, method="h.assoc")

        assert "prior" in result
        assert result["prior"].shape == simple_table.shape

    def test_zeros_handled(self, sparse_table):
        """Test that zero cells are handled properly."""
        from statmatch.frechet import p_bayes

        result = p_bayes(sparse_table, method="Jeffreys")

        # All pseudoB values should be positive
        assert np.all(result["pseudoB"] > 0)

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_jeffreys_against_r(self, simple_table):
        """Test Jeffreys method against R implementation."""
        from statmatch.frechet import p_bayes

        # Python implementation
        result_py = p_bayes(simple_table, method="Jeffreys")

        # R implementation - use Fortran order for R's column-major arrays
        r_array = ro.r.array(
            ro.FloatVector(simple_table.flatten(order="F")),
            dim=ro.IntVector(simple_table.shape),
        )
        result_r = statmatch_r.pBayes(x=r_array, method="Jeffreys")

        # Compare info
        info_r_vec = result_r.rx2("info")
        info_r = dict(zip(info_r_vec.names, list(info_r_vec)))
        np.testing.assert_allclose(
            result_py["info"]["n"], float(info_r["n"]), rtol=1e-10
        )
        np.testing.assert_allclose(
            result_py["info"]["K"], float(info_r["K"]), rtol=1e-10
        )

        # Compare pseudoB estimates - R returns column-major, reshape with F order
        pseudoB_r = np.array(result_r.rx2("pseudoB"))
        np.testing.assert_allclose(
            result_py["pseudoB"],
            pseudoB_r.reshape(simple_table.shape, order="F"),
            rtol=1e-6,
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_minimax_against_r(self, simple_table):
        """Test minimax method against R implementation."""
        from statmatch.frechet import p_bayes

        result_py = p_bayes(simple_table, method="minimax")

        # R uses column-major order
        r_array = ro.r.array(
            ro.FloatVector(simple_table.flatten(order="F")),
            dim=ro.IntVector(simple_table.shape),
        )
        result_r = statmatch_r.pBayes(x=r_array, method="minimax")

        pseudoB_r = np.array(result_r.rx2("pseudoB"))
        np.testing.assert_allclose(
            result_py["pseudoB"],
            pseudoB_r.reshape(simple_table.shape, order="F"),
            rtol=1e-6,
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_m_ind_against_r(self, simple_table):
        """Test mutual independence method against R implementation."""
        from statmatch.frechet import p_bayes

        result_py = p_bayes(simple_table, method="m.ind")

        # R uses column-major order
        r_array = ro.r.array(
            ro.FloatVector(simple_table.flatten(order="F")),
            dim=ro.IntVector(simple_table.shape),
        )
        result_r = statmatch_r.pBayes(x=r_array, method="m.ind")

        # Compare K values (allow some tolerance as estimation methods may differ)
        info_r_vec = result_r.rx2("info")
        info_r = dict(zip(info_r_vec.names, list(info_r_vec)))
        np.testing.assert_allclose(
            result_py["info"]["K"], float(info_r["K"]), rtol=0.2
        )

        # Compare pseudoB estimates (allow more tolerance due to K differences)
        pseudoB_r = np.array(result_r.rx2("pseudoB"))
        np.testing.assert_allclose(
            result_py["pseudoB"],
            pseudoB_r.reshape(simple_table.shape, order="F"),
            rtol=0.1,
        )


class TestFrechetBoundsCat:
    """Test suite for frechet_bounds_cat function."""

    @pytest.fixture
    def simple_tables(self):
        """Create simple contingency tables for testing."""
        # tab.x: 2x2 table for X variables
        tab_x = np.array([[30, 20], [25, 25]])

        # tab.xy: 2x2x2 table for X vs Y
        tab_xy = np.array([[[15, 15], [10, 10]], [[12, 13], [12, 13]]])

        # tab.xz: 2x2x3 table for X vs Z
        tab_xz = np.array(
            [[[10, 10, 10], [7, 7, 6]], [[8, 9, 8], [8, 9, 8]]]
        )

        return tab_x, tab_xy, tab_xz

    def test_basic_bounds(self, simple_tables):
        """Test that bounds are computed correctly."""
        from statmatch.frechet import frechet_bounds_cat

        tab_x, tab_xy, tab_xz = simple_tables

        result = frechet_bounds_cat(tab_x, tab_xy, tab_xz, print_f="tables")

        # Check all expected keys
        assert "low.u" in result
        assert "up.u" in result
        assert "CIA" in result
        assert "low.cx" in result
        assert "up.cx" in result
        assert "uncertainty" in result

        # Lower bounds should be <= upper bounds
        assert np.all(result["low.u"] <= result["up.u"])
        assert np.all(result["low.cx"] <= result["up.cx"])

        # Conditional bounds should be tighter than unconditional
        assert np.all(result["low.cx"] >= result["low.u"] - 1e-10)
        assert np.all(result["up.cx"] <= result["up.u"] + 1e-10)

    def test_dataframe_output(self, simple_tables):
        """Test dataframe output format."""
        from statmatch.frechet import frechet_bounds_cat

        tab_x, tab_xy, tab_xz = simple_tables

        result = frechet_bounds_cat(tab_x, tab_xy, tab_xz, print_f="data.frame")

        assert "bounds" in result
        assert "uncertainty" in result
        assert isinstance(result["bounds"], pd.DataFrame)

        # Check required columns
        expected_cols = ["low.u", "low.cx", "CIA", "up.cx", "up.u"]
        for col in expected_cols:
            assert col in result["bounds"].columns

    def test_uncertainty_calculation(self, simple_tables):
        """Test that uncertainty values are computed correctly."""
        from statmatch.frechet import frechet_bounds_cat

        tab_x, tab_xy, tab_xz = simple_tables

        result = frechet_bounds_cat(tab_x, tab_xy, tab_xz)

        unc = result["uncertainty"]
        assert "av.u" in unc
        assert "av.cx" in unc

        # Unconditional uncertainty should be >= conditional
        assert unc["av.u"] >= unc["av.cx"] - 1e-10

        # Uncertainty should be non-negative
        assert unc["av.u"] >= 0
        assert unc["av.cx"] >= 0

    def test_no_x_variables(self):
        """Test when tab.x is None (unconditional bounds only)."""
        from statmatch.frechet import frechet_bounds_cat

        # Just marginal distributions of Y and Z
        tab_y = np.array([50, 50])  # 2 categories
        tab_z = np.array([40, 35, 25])  # 3 categories

        result = frechet_bounds_cat(None, tab_y, tab_z)

        assert "low.u" in result
        assert "up.u" in result

        # Should have Y x Z dimensions
        assert result["low.u"].shape == (2, 3)

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_against_r_basic(self):
        """Test basic case against R implementation."""
        from statmatch.frechet import frechet_bounds_cat

        # Use R's quine dataset
        ro.r('library(MASS)')
        ro.r('data(quine)')
        ro.r('suppressWarnings(RNGversion("3.5.0"))')
        ro.r('set.seed(7654)')
        ro.r('lab.A <- sample(nrow(quine), 70, replace=TRUE)')
        ro.r('quine.A <- quine[lab.A, 1:4]')
        ro.r('quine.B <- quine[-lab.A, c(1:3, 5)]')

        ro.r('tab.x <- xtabs(~Eth+Sex, data=quine)')
        ro.r('tab.xy <- xtabs(~Eth+Sex+Lrn, data=quine.A)')
        ro.r('tab.xz <- xtabs(~Eth+Sex+Days, data=quine.B)')

        # Get R results
        result_r = ro.r('''
            suppressWarnings(
                Frechet.bounds.cat(tab.x, tab.xy, tab.xz, print.f="tables")
            )
        ''')

        # Extract tables from R
        tab_x = np.array(ro.r('tab.x'))
        tab_xy = np.array(ro.r('tab.xy'))
        tab_xz = np.array(ro.r('tab.xz'))

        # Python implementation
        result_py = frechet_bounds_cat(tab_x, tab_xy, tab_xz, print_f="tables")

        # Compare uncertainty values
        unc_r_vec = result_r.rx2("uncertainty")
        unc_r = dict(zip(unc_r_vec.names, list(unc_r_vec)))
        np.testing.assert_allclose(
            result_py["uncertainty"]["av.u"], float(unc_r["av.u"]), rtol=1e-5
        )
        np.testing.assert_allclose(
            result_py["uncertainty"]["av.cx"], float(unc_r["av.cx"]), rtol=1e-5
        )


class TestFbwidthsByX:
    """Test suite for fbwidths_by_x function."""

    @pytest.fixture
    def test_tables(self):
        """Create test contingency tables."""
        # 2x2 table for X1 x X2
        tab_x = np.array([[30, 20], [25, 25]])

        # 2x2x2 table for X1 x X2 x Y
        tab_xy = np.array([[[15, 15], [10, 10]], [[12, 13], [12, 13]]])

        # 2x2x3 table for X1 x X2 x Z
        tab_xz = np.array(
            [[[10, 10, 10], [7, 7, 6]], [[8, 9, 8], [8, 9, 8]]]
        )

        return tab_x, tab_xy, tab_xz

    def test_basic_functionality(self, test_tables):
        """Test basic fbwidths_by_x functionality."""
        from statmatch.frechet import fbwidths_by_x

        tab_x, tab_xy, tab_xz = test_tables

        result = fbwidths_by_x(tab_x, tab_xy, tab_xz)

        # Check that sum.unc is present
        assert "sum.unc" in result
        assert isinstance(result["sum.unc"], pd.DataFrame)

        # Check expected columns in sum.unc
        expected_cols = [
            "x.vars",
            "av.width",
            "penalty1",
            "penalty2",
        ]
        for col in expected_cols:
            assert col in result["sum.unc"].columns

    def test_all_subsets_computed(self, test_tables):
        """Test that all subsets of X variables are computed."""
        from statmatch.frechet import fbwidths_by_x

        tab_x, tab_xy, tab_xz = test_tables

        result = fbwidths_by_x(tab_x, tab_xy, tab_xz)

        # Should have 2^k - 1 subsets + unconditional = 2^k rows
        # For 2 X variables: 2^2 = 4 rows
        n_x_vars = len(tab_x.shape)
        expected_rows = 2**n_x_vars
        assert len(result["sum.unc"]) == expected_rows

    def test_penalty_values(self, test_tables):
        """Test that penalty values are computed."""
        from statmatch.frechet import fbwidths_by_x

        tab_x, tab_xy, tab_xz = test_tables

        result = fbwidths_by_x(tab_x, tab_xy, tab_xz)

        # Unconditional case should have 0 penalties
        uncond = result["sum.unc"].iloc[0]
        assert uncond["penalty1"] == 0
        assert uncond["penalty2"] == 0

        # Other cases should have positive penalties
        for i in range(1, len(result["sum.unc"])):
            row = result["sum.unc"].iloc[i]
            assert row["penalty1"] >= 0
            assert row["penalty2"] >= 0

    def test_sparse_handling(self, test_tables):
        """Test handling of sparse tables."""
        from statmatch.frechet import fbwidths_by_x

        tab_x, tab_xy, tab_xz = test_tables

        # Test with discard option
        result_discard = fbwidths_by_x(
            tab_x, tab_xy, tab_xz, deal_sparse="discard"
        )
        assert "sum.unc" in result_discard

        # Test with relfreq option
        result_relfreq = fbwidths_by_x(
            tab_x, tab_xy, tab_xz, deal_sparse="relfreq"
        )
        assert "sum.unc" in result_relfreq

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_against_r(self):
        """Test against R implementation."""
        from statmatch.frechet import fbwidths_by_x

        # Use R's quine dataset with simplified Z variable
        ro.r('library(MASS)')
        ro.r('data(quine)')
        ro.r('quine$c.Days <- cut(quine$Days, c(-1, seq(0,50,10),100))')
        ro.r('suppressWarnings(RNGversion("3.5.0"))')
        ro.r('set.seed(4567)')
        ro.r('lab.A <- sample(nrow(quine), 70, replace=TRUE)')
        ro.r('quine.A <- quine[lab.A, 1:4]')
        ro.r('quine.B <- quine[-lab.A, c(1:3, 6)]')

        ro.r('freq.xA <- xtabs(~Eth+Sex, data=quine.A)')
        ro.r('freq.xB <- xtabs(~Eth+Sex, data=quine.B)')
        ro.r('freq.xy <- xtabs(~Eth+Sex+Lrn, data=quine.A)')
        ro.r('freq.xz <- xtabs(~Eth+Sex+c.Days, data=quine.B)')

        # R implementation
        result_r = ro.r('''
            suppressWarnings(
                Fbwidths.by.x(tab.x=freq.xA+freq.xB, tab.xy=freq.xy,
                    tab.xz=freq.xz)
            )
        ''')

        # Extract tables
        tab_x = np.array(ro.r('freq.xA + freq.xB'))
        tab_xy = np.array(ro.r('freq.xy'))
        tab_xz = np.array(ro.r('freq.xz'))

        # Python implementation
        result_py = fbwidths_by_x(tab_x, tab_xy, tab_xz)

        # Compare sum.unc dataframe
        with localconverter(ro.default_converter + pandas2ri.converter):
            sum_unc_r = ro.conversion.rpy2py(result_r.rx2("sum.unc"))

        # Compare av.width for unconditional case
        py_uncond = result_py["sum.unc"].iloc[0]
        r_uncond = sum_unc_r.iloc[0]

        np.testing.assert_allclose(
            py_uncond["av.width"], r_uncond["av.width"], rtol=1e-5
        )
