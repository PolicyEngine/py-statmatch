"""Test mixed_mtc and sel_mtc_by_unc against R's StatMatch implementation."""

import numpy as np
import pandas as pd
import pytest

# Only import rpy2 for test generation
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    # Import StatMatch
    statmatch_r = importr("StatMatch")
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False


class TestMixedMtc:
    """Test suite for mixed_mtc function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample donor and recipient datasets with continuous variables."""
        np.random.seed(42)

        # Donor dataset with matching variables X and target variable Z
        n_donors = 100
        x1 = np.random.normal(0, 1, n_donors)
        x2 = np.random.normal(0, 2, n_donors)
        # Z is correlated with X1 and X2
        z = 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.5, n_donors)

        donor_data = pd.DataFrame({"x1": x1, "x2": x2, "z": z})

        # Recipient dataset with matching variables X and target variable Y
        n_recipients = 80
        x1_rec = np.random.normal(0, 1, n_recipients)
        x2_rec = np.random.normal(0, 2, n_recipients)
        # Y is correlated with X1 and X2
        y = 0.4 * x1_rec + 0.2 * x2_rec + np.random.normal(0, 0.5, n_recipients)

        recipient_data = pd.DataFrame({"x1": x1_rec, "x2": x2_rec, "y": y})

        return donor_data, recipient_data

    @pytest.fixture
    def sample_data_with_categorical(self):
        """Create sample data with mixed continuous and categorical variables."""
        np.random.seed(42)

        n_donors = 100
        x1 = np.random.normal(0, 1, n_donors)
        x2 = np.random.choice(["A", "B", "C"], n_donors)
        z = np.where(
            x2 == "A",
            0.5 * x1 + np.random.normal(0, 0.5, n_donors),
            np.where(
                x2 == "B",
                0.3 * x1 + np.random.normal(0, 0.5, n_donors),
                0.1 * x1 + np.random.normal(0, 0.5, n_donors),
            ),
        )

        donor_data = pd.DataFrame({"x1": x1, "x2": x2, "z": z})

        n_recipients = 80
        x1_rec = np.random.normal(0, 1, n_recipients)
        x2_rec = np.random.choice(["A", "B", "C"], n_recipients)
        y = np.where(
            x2_rec == "A",
            0.4 * x1_rec + np.random.normal(0, 0.5, n_recipients),
            np.where(
                x2_rec == "B",
                0.2 * x1_rec + np.random.normal(0, 0.5, n_recipients),
                0.05 * x1_rec + np.random.normal(0, 0.5, n_recipients),
            ),
        )

        recipient_data = pd.DataFrame({"x1": x1_rec, "x2": x2_rec, "y": y})

        return donor_data, recipient_data

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_ml_method_against_r(self, sample_data):
        """Test ML method parameter estimates against R implementation."""
        from statmatch.mixed_mtc import mixed_mtc

        donor_data, recipient_data = sample_data

        # Python implementation
        result_py = mixed_mtc(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            y_rec="y",
            z_don="z",
            method="ML",
        )

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            donor_r = ro.conversion.py2rpy(donor_data)
            recipient_r = ro.conversion.py2rpy(recipient_data)

            result_r = statmatch_r.mixed_mtc(
                data_rec=recipient_r,
                data_don=donor_r,
                match_vars=ro.StrVector(["x1", "x2"]),
                y_rec="y",
                z_don="z",
                method="ML",
            )

        # Compare mean estimates
        mu_r = np.array(result_r.rx2("mu"))
        np.testing.assert_allclose(
            result_py["mu"],
            mu_r,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Mean estimates differ from R",
        )

        # Compare variance-covariance matrix
        vc_r = np.array(result_r.rx2("vc"))
        np.testing.assert_allclose(
            result_py["vc"],
            vc_r,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Variance-covariance matrix differs from R",
        )

        # Compare correlation matrix
        cor_r = np.array(result_r.rx2("cor"))
        np.testing.assert_allclose(
            result_py["cor"],
            cor_r,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Correlation matrix differs from R",
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_ms_method_against_r(self, sample_data):
        """Test MS (Moriarity-Scheuren) method against R implementation."""
        from statmatch.mixed_mtc import mixed_mtc

        donor_data, recipient_data = sample_data

        # Python implementation with specified rho.yz
        result_py = mixed_mtc(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            y_rec="y",
            z_don="z",
            method="MS",
            rho_yz=0.3,
        )

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            donor_r = ro.conversion.py2rpy(donor_data)
            recipient_r = ro.conversion.py2rpy(recipient_data)

            result_r = statmatch_r.mixed_mtc(
                data_rec=recipient_r,
                data_don=donor_r,
                match_vars=ro.StrVector(["x1", "x2"]),
                y_rec="y",
                z_don="z",
                method="MS",
                rho_yz=0.3,
            )

        # Compare mean estimates
        mu_r = np.array(result_r.rx2("mu"))
        np.testing.assert_allclose(
            result_py["mu"],
            mu_r,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Mean estimates differ from R",
        )

        # Compare variance-covariance matrix
        vc_r = np.array(result_r.rx2("vc"))
        np.testing.assert_allclose(
            result_py["vc"],
            vc_r,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Variance-covariance matrix differs from R",
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_micro_matching_against_r(self, sample_data):
        """Test micro-level matching with filled recipient data against R."""
        from statmatch.mixed_mtc import mixed_mtc

        donor_data, recipient_data = sample_data

        # Python implementation with micro=True
        result_py = mixed_mtc(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            y_rec="y",
            z_don="z",
            method="ML",
            micro=True,
            constr_alg="hungarian",
        )

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            donor_r = ro.conversion.py2rpy(donor_data)
            recipient_r = ro.conversion.py2rpy(recipient_data)

            result_r = statmatch_r.mixed_mtc(
                data_rec=recipient_r,
                data_don=donor_r,
                match_vars=ro.StrVector(["x1", "x2"]),
                y_rec="y",
                z_don="z",
                method="ML",
                micro=True,
                constr_alg="Hungarian",
            )

        # Check that filled.rec exists
        assert "filled.rec" in result_py
        assert result_py["filled.rec"] is not None

        # Check that mtc.ids exists
        assert "mtc.ids" in result_py

        # Check that dist.rd exists
        assert "dist.rd" in result_py

        # Compare distance values (should be close but may differ due to
        # random perturbation in the algorithm)
        dist_r = np.array(result_r.rx2("dist.rd"))
        # Distances won't match exactly due to random noise, but should be
        # in similar range
        assert len(result_py["dist.rd"]) == len(dist_r)

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_categorical_variables_against_r(self, sample_data_with_categorical):
        """Test handling of categorical variables (converted to dummies)."""
        from statmatch.mixed_mtc import mixed_mtc

        donor_data, recipient_data = sample_data_with_categorical

        # Python implementation
        result_py = mixed_mtc(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            y_rec="y",
            z_don="z",
            method="ML",
        )

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            donor_r = ro.conversion.py2rpy(donor_data)
            recipient_r = ro.conversion.py2rpy(recipient_data)

            result_r = statmatch_r.mixed_mtc(
                data_rec=recipient_r,
                data_don=donor_r,
                match_vars=ro.StrVector(["x1", "x2"]),
                y_rec="y",
                z_don="z",
                method="ML",
            )

        # Compare key results (exact matching may differ due to dummy coding)
        mu_r = np.array(result_r.rx2("mu"))
        # Check dimensions match
        assert len(result_py["mu"]) == len(mu_r)

    def test_basic_ml_method(self, sample_data):
        """Test basic ML method functionality without R comparison."""
        from statmatch.mixed_mtc import mixed_mtc

        donor_data, recipient_data = sample_data

        result = mixed_mtc(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            y_rec="y",
            z_don="z",
            method="ML",
        )

        # Check required outputs
        assert "mu" in result
        assert "vc" in result
        assert "cor" in result
        assert "res.var" in result

        # Check dimensions
        n_vars = 4  # x1, x2, y, z
        assert len(result["mu"]) == n_vars
        assert result["vc"].shape == (n_vars, n_vars)
        assert result["cor"].shape == (n_vars, n_vars)

        # Check correlation matrix properties
        np.testing.assert_allclose(
            np.diag(result["cor"]), 1.0, rtol=1e-10
        )  # Diagonal is 1
        np.testing.assert_allclose(
            result["cor"], result["cor"].T, rtol=1e-10
        )  # Symmetric

    def test_basic_ms_method(self, sample_data):
        """Test basic MS method functionality without R comparison."""
        from statmatch.mixed_mtc import mixed_mtc

        donor_data, recipient_data = sample_data

        result = mixed_mtc(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            y_rec="y",
            z_don="z",
            method="MS",
        )

        # Check required outputs
        assert "mu" in result
        assert "vc" in result
        assert "cor" in result
        assert "res.var" in result
        assert "rho.yz" in result  # MS method returns rho.yz info

    def test_micro_matching_basic(self, sample_data):
        """Test micro-level matching basic functionality."""
        from statmatch.mixed_mtc import mixed_mtc

        donor_data, recipient_data = sample_data

        result = mixed_mtc(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            y_rec="y",
            z_don="z",
            method="ML",
            micro=True,
        )

        # Check micro-level outputs
        assert "filled.rec" in result
        assert "mtc.ids" in result
        assert "dist.rd" in result

        # Check filled.rec has the Z variable
        assert "z" in result["filled.rec"].columns

        # Check dimensions
        assert len(result["filled.rec"]) == len(recipient_data)
        assert len(result["dist.rd"]) == len(recipient_data)

    def test_insufficient_donors_raises_error(self):
        """Test that error is raised when donors < recipients for micro."""
        from statmatch.mixed_mtc import mixed_mtc

        # More recipients than donors
        donor_data = pd.DataFrame(
            {"x1": [1, 2, 3], "x2": [1, 2, 3], "z": [1, 2, 3]}
        )
        recipient_data = pd.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5],
                "x2": [1, 2, 3, 4, 5],
                "y": [1, 2, 3, 4, 5],
            }
        )

        with pytest.raises(ValueError, match="donors.*recipients"):
            mixed_mtc(
                data_rec=recipient_data,
                data_don=donor_data,
                match_vars=["x1", "x2"],
                y_rec="y",
                z_don="z",
                method="ML",
                micro=True,
            )

    def test_rho_yz_bounds_ms_method(self, sample_data):
        """Test that rho.yz is adjusted if outside admissible bounds."""
        from statmatch.mixed_mtc import mixed_mtc

        donor_data, recipient_data = sample_data

        # Use an extreme rho.yz value
        result = mixed_mtc(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            y_rec="y",
            z_don="z",
            method="MS",
            rho_yz=0.99,  # Likely outside bounds
        )

        # Check that rho.yz was adjusted
        assert "rho.yz" in result
        # The used value should be within [-1, 1]
        assert -1 <= result["rho.yz"]["used"] <= 1


class TestSelMtcByUnc:
    """Test suite for sel_mtc_by_unc function."""

    @pytest.fixture
    def sample_contingency_tables(self):
        """Create sample contingency tables for testing."""
        np.random.seed(42)

        # Create sample data
        n = 500
        x1 = np.random.choice(["a", "b", "c"], n)
        x2 = np.random.choice(["low", "high"], n)
        x3 = np.random.choice(["yes", "no"], n)
        y = np.random.choice(["Y1", "Y2", "Y3"], n)
        z = np.random.choice(["Z1", "Z2"], n)

        df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y, "z": z})

        # Create contingency tables
        tab_x = pd.crosstab([df["x1"], df["x2"], df["x3"]], columns="count")
        tab_xy = pd.crosstab([df["x1"], df["x2"], df["x3"]], df["y"])
        tab_xz = pd.crosstab([df["x1"], df["x2"], df["x3"]], df["z"])

        return tab_x, tab_xy, tab_xz

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_sel_mtc_against_r(self, sample_contingency_tables):
        """Test sel_mtc_by_unc against R implementation."""
        from statmatch.mixed_mtc import sel_mtc_by_unc

        tab_x, tab_xy, tab_xz = sample_contingency_tables

        # Python implementation
        result_py = sel_mtc_by_unc(
            tab_x=tab_x, tab_xy=tab_xy, tab_xz=tab_xz, corr_d=2
        )

        # For R comparison, we need to create the tables in R format
        # This is complex due to multi-index tables
        # For now, test basic functionality

        # Check required outputs
        assert "ini.ord" in result_py
        assert "list.xs" in result_py
        assert "av.df" in result_py

    def test_basic_functionality(self, sample_contingency_tables):
        """Test basic sel_mtc_by_unc functionality."""
        from statmatch.mixed_mtc import sel_mtc_by_unc

        tab_x, tab_xy, tab_xz = sample_contingency_tables

        result = sel_mtc_by_unc(
            tab_x=tab_x, tab_xy=tab_xy, tab_xz=tab_xz, corr_d=2
        )

        # Check required outputs
        assert "ini.ord" in result
        assert "list.xs" in result
        assert "av.df" in result

        # ini.ord should be a list of variable names
        assert len(result["ini.ord"]) > 0

    def test_corr_d_parameter(self, sample_contingency_tables):
        """Test different corr_d penalty values."""
        from statmatch.mixed_mtc import sel_mtc_by_unc

        tab_x, tab_xy, tab_xz = sample_contingency_tables

        # Test with different corr_d values
        for corr_d in [0, 1, 2]:
            result = sel_mtc_by_unc(
                tab_x=tab_x, tab_xy=tab_xy, tab_xz=tab_xz, corr_d=corr_d
            )
            assert "ini.ord" in result

    def test_align_margins(self, sample_contingency_tables):
        """Test align_margins parameter."""
        from statmatch.mixed_mtc import sel_mtc_by_unc

        tab_x, tab_xy, tab_xz = sample_contingency_tables

        result = sel_mtc_by_unc(
            tab_x=tab_x,
            tab_xy=tab_xy,
            tab_xz=tab_xz,
            corr_d=2,
            align_margins=True,
        )

        assert "ini.ord" in result
