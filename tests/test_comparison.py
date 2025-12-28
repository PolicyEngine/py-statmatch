"""Test comparison functions against R's StatMatch implementation."""

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


class TestCompCont:
    """Test suite for comp_cont function."""

    @pytest.fixture
    def sample_continuous_data(self):
        """Create sample continuous data for comparison."""
        np.random.seed(42)

        # Dataset A
        data_a = pd.DataFrame(
            {
                "x": np.random.normal(10, 3, 100),
                "w": np.random.uniform(0.5, 2.0, 100),
            }
        )

        # Dataset B - slightly different distribution
        data_b = pd.DataFrame(
            {
                "x": np.random.normal(11, 3.5, 120),
                "w": np.random.uniform(0.5, 2.0, 120),
            }
        )

        return data_a, data_b

    def test_comp_cont_basic(self, sample_continuous_data):
        """Test basic comp_cont functionality."""
        from statmatch.comparison import comp_cont

        data_a, data_b = sample_continuous_data

        result = comp_cont(
            data_a=data_a,
            data_b=data_b,
            xlab_a="x",
        )

        # Check that result has expected keys
        assert "summary" in result
        assert "diff_qs" in result
        assert "dist_ecdf" in result
        assert "dist_discr" in result

        # Check summary has stats for both datasets
        assert "A" in result["summary"].index or 0 in result["summary"].index
        assert "B" in result["summary"].index or 1 in result["summary"].index

        # Check that distances are non-negative
        assert result["dist_ecdf"]["ks_dist"] >= 0
        assert result["dist_ecdf"]["kuiper_dist"] >= 0
        assert result["dist_ecdf"]["av_abs_diff"] >= 0

        assert result["dist_discr"]["tvd"] >= 0
        assert result["dist_discr"]["overlap"] >= 0
        assert result["dist_discr"]["hellinger"] >= 0

    def test_comp_cont_with_weights(self, sample_continuous_data):
        """Test comp_cont with weights."""
        from statmatch.comparison import comp_cont

        data_a, data_b = sample_continuous_data

        result = comp_cont(
            data_a=data_a,
            data_b=data_b,
            xlab_a="x",
            w_a="w",
            w_b="w",
        )

        # Check result structure
        assert "summary" in result
        assert "diff_qs" in result

    def test_comp_cont_different_variable_names(self, sample_continuous_data):
        """Test comp_cont with different variable names in A and B."""
        from statmatch.comparison import comp_cont

        data_a, data_b = sample_continuous_data

        # Rename variable in data_b
        data_b = data_b.rename(columns={"x": "y"})

        result = comp_cont(
            data_a=data_a,
            data_b=data_b,
            xlab_a="x",
            xlab_b="y",
        )

        assert "summary" in result

    def test_comp_cont_with_reference(self, sample_continuous_data):
        """Test comp_cont with ref=True (data_b as reference)."""
        from statmatch.comparison import comp_cont

        data_a, data_b = sample_continuous_data

        result = comp_cont(
            data_a=data_a,
            data_b=data_b,
            xlab_a="x",
            ref=True,
        )

        assert "summary" in result

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_comp_cont_against_r(self, sample_continuous_data):
        """Test comp_cont against R implementation."""
        from statmatch.comparison import comp_cont

        data_a, data_b = sample_continuous_data

        # Python implementation
        result_py = comp_cont(
            data_a=data_a,
            data_b=data_b,
            xlab_a="x",
        )

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            data_a_r = ro.conversion.py2rpy(data_a)
            data_b_r = ro.conversion.py2rpy(data_b)

            result_r = statmatch_r.comp_cont(
                data_A=data_a_r,
                data_B=data_b_r,
                xlab_A="x",
            )

        # Extract R results
        dist_ecdf_r = dict(
            zip(result_r.rx2("dist.ecdf").names, result_r.rx2("dist.ecdf"))
        )
        dist_discr_r = dict(
            zip(result_r.rx2("dist.discr").names, result_r.rx2("dist.discr"))
        )

        # Compare KS distance
        np.testing.assert_allclose(
            result_py["dist_ecdf"]["ks_dist"],
            dist_ecdf_r["KSdist"],
            rtol=1e-5,
            err_msg="KS distance mismatch",
        )

        # Compare total variation distance
        np.testing.assert_allclose(
            result_py["dist_discr"]["tvd"],
            dist_discr_r["tvd"],
            rtol=1e-5,
            err_msg="TVD mismatch",
        )

        # Compare Hellinger distance
        np.testing.assert_allclose(
            result_py["dist_discr"]["hellinger"],
            dist_discr_r["Hellinger"],
            rtol=1e-5,
            err_msg="Hellinger distance mismatch",
        )


class TestCompProp:
    """Test suite for comp_prop function."""

    @pytest.fixture
    def sample_proportions(self):
        """Create sample proportion data."""
        # Proportions from sample 1
        p1 = np.array([0.2, 0.3, 0.35, 0.15])

        # Proportions from sample 2
        p2 = np.array([0.25, 0.25, 0.30, 0.20])

        return p1, p2

    def test_comp_prop_basic(self, sample_proportions):
        """Test basic comp_prop functionality."""
        from statmatch.comparison import comp_prop

        p1, p2 = sample_proportions

        result = comp_prop(p1=p1, p2=p2, n1=100, n2=150)

        # Check that result has expected keys
        assert "meas" in result
        assert "chi_sq" in result
        assert "p_exp" in result

        # Check measures
        assert "tvd" in result["meas"]
        assert "overlap" in result["meas"]
        assert "bhatt" in result["meas"]
        assert "hell" in result["meas"]

        # Check values are in expected ranges
        assert 0 <= result["meas"]["tvd"] <= 1
        assert 0 <= result["meas"]["overlap"] <= 1
        assert 0 <= result["meas"]["bhatt"] <= 1
        assert 0 <= result["meas"]["hell"] <= 1

        # Check overlap + tvd = 1
        np.testing.assert_allclose(
            result["meas"]["tvd"] + result["meas"]["overlap"],
            1.0,
            rtol=1e-10,
        )

    def test_comp_prop_with_reference(self, sample_proportions):
        """Test comp_prop with reference distribution."""
        from statmatch.comparison import comp_prop

        p1, p2 = sample_proportions

        result = comp_prop(p1=p1, p2=p2, n1=100, ref=True)

        # When ref=True, n2 is not required
        assert "meas" in result
        assert "chi_sq" in result

        # p_exp should equal p2 when ref=True
        np.testing.assert_array_almost_equal(result["p_exp"], p2)

    def test_comp_prop_requires_n2_when_not_ref(self, sample_proportions):
        """Test that n2 is required when ref=False."""
        from statmatch.comparison import comp_prop

        p1, p2 = sample_proportions

        with pytest.raises(ValueError, match="n2"):
            comp_prop(p1=p1, p2=p2, n1=100, ref=False)

    def test_comp_prop_normalizes_counts(self):
        """Test that comp_prop normalizes counts to proportions."""
        from statmatch.comparison import comp_prop

        # Pass counts instead of proportions
        p1 = np.array([20, 30, 35, 15])  # sum = 100
        p2 = np.array([25, 25, 30, 20])  # sum = 100

        result = comp_prop(p1=p1, p2=p2, n1=100, n2=100)

        # Should still work
        assert "meas" in result
        assert 0 <= result["meas"]["tvd"] <= 1

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_comp_prop_against_r(self, sample_proportions):
        """Test comp_prop against R implementation."""
        from statmatch.comparison import comp_prop

        p1, p2 = sample_proportions

        # Python implementation
        result_py = comp_prop(p1=p1, p2=p2, n1=100, n2=150)

        # R implementation
        p1_r = ro.FloatVector(p1)
        p2_r = ro.FloatVector(p2)

        result_r = statmatch_r.comp_prop(p1=p1_r, p2=p2_r, n1=100, n2=150)

        # Extract R results
        meas_r = dict(zip(result_r.rx2("meas").names, result_r.rx2("meas")))
        chi_sq_r = dict(
            zip(result_r.rx2("chi.sq").names, result_r.rx2("chi.sq"))
        )

        # Compare measures
        np.testing.assert_allclose(
            result_py["meas"]["tvd"],
            meas_r["tvd"],
            rtol=1e-10,
            err_msg="TVD mismatch",
        )

        np.testing.assert_allclose(
            result_py["meas"]["overlap"],
            meas_r["overlap"],
            rtol=1e-10,
            err_msg="Overlap mismatch",
        )

        np.testing.assert_allclose(
            result_py["meas"]["bhatt"],
            meas_r["Bhatt"],
            rtol=1e-10,
            err_msg="Bhattacharyya mismatch",
        )

        np.testing.assert_allclose(
            result_py["meas"]["hell"],
            meas_r["Hell"],
            rtol=1e-10,
            err_msg="Hellinger mismatch",
        )

        # Compare chi-squared
        np.testing.assert_allclose(
            result_py["chi_sq"]["pearson"],
            chi_sq_r["Pearson"],
            rtol=1e-10,
            err_msg="Pearson chi-sq mismatch",
        )

        np.testing.assert_allclose(
            result_py["chi_sq"]["df"],
            chi_sq_r["df"],
            rtol=1e-10,
            err_msg="df mismatch",
        )


class TestPwAssoc:
    """Test suite for pw_assoc function."""

    @pytest.fixture
    def sample_categorical_data(self):
        """Create sample categorical data."""
        np.random.seed(42)
        n = 200

        data = pd.DataFrame(
            {
                "y": pd.Categorical(
                    np.random.choice(["low", "medium", "high"], n)
                ),
                "x1": pd.Categorical(np.random.choice(["A", "B", "C"], n)),
                "x2": pd.Categorical(np.random.choice(["yes", "no"], n)),
                "x3": pd.Categorical(
                    np.random.choice(["urban", "rural", "suburban"], n)
                ),
                "weight": np.random.uniform(0.5, 2.0, n),
            }
        )

        return data

    def test_pw_assoc_basic(self, sample_categorical_data):
        """Test basic pw_assoc functionality."""
        from statmatch.comparison import pw_assoc

        data = sample_categorical_data

        result = pw_assoc(formula="y ~ x1 + x2 + x3", data=data)

        # Check that result is a dict (when out_df=False)
        assert isinstance(result, dict)

        # Check expected keys
        assert "V" in result
        assert "bcV" in result
        assert "mi" in result
        assert "norm_mi" in result
        assert "lambda_" in result
        assert "tau" in result
        assert "U" in result
        assert "AIC" in result
        assert "BIC" in result
        assert "npar" in result

        # Check that each measure has values for all predictors
        assert len(result["V"]) == 3
        assert "x1" in result["V"]
        assert "x2" in result["V"]
        assert "x3" in result["V"]

        # Check value ranges
        for var in ["x1", "x2", "x3"]:
            assert 0 <= result["V"][var] <= 1
            assert 0 <= result["bcV"][var] <= 1
            assert 0 <= result["lambda_"][var] <= 1
            assert 0 <= result["tau"][var] <= 1
            assert 0 <= result["U"][var] <= 1

    def test_pw_assoc_with_weights(self, sample_categorical_data):
        """Test pw_assoc with weights."""
        from statmatch.comparison import pw_assoc

        data = sample_categorical_data

        result = pw_assoc(
            formula="y ~ x1 + x2 + x3",
            data=data,
            weights="weight",
        )

        # Check that result has expected structure
        assert "V" in result
        assert len(result["V"]) == 3

    def test_pw_assoc_out_df(self, sample_categorical_data):
        """Test pw_assoc with out_df=True."""
        from statmatch.comparison import pw_assoc

        data = sample_categorical_data

        result = pw_assoc(
            formula="y ~ x1 + x2 + x3",
            data=data,
            out_df=True,
        )

        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check columns
        assert "V" in result.columns
        assert "bcV" in result.columns
        assert "mi" in result.columns
        assert "norm_mi" in result.columns
        assert "lambda_" in result.columns
        assert "tau" in result.columns
        assert "U" in result.columns
        assert "AIC" in result.columns
        assert "BIC" in result.columns
        assert "npar" in result.columns

        # Check rows
        assert "x1" in result.index
        assert "x2" in result.index
        assert "x3" in result.index

    def test_pw_assoc_single_predictor(self, sample_categorical_data):
        """Test pw_assoc with single predictor."""
        from statmatch.comparison import pw_assoc

        data = sample_categorical_data

        result = pw_assoc(formula="y ~ x1", data=data)

        assert len(result["V"]) == 1
        assert "x1" in result["V"]

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_pw_assoc_against_r(self, sample_categorical_data):
        """Test pw_assoc against R implementation."""
        from statmatch.comparison import pw_assoc

        data = sample_categorical_data

        # Python implementation
        result_py = pw_assoc(formula="y ~ x1 + x2 + x3", data=data)

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            data_r = ro.conversion.py2rpy(data)

            result_r = statmatch_r.pw_assoc(
                formula=ro.Formula("y ~ x1 + x2 + x3"),
                data=data_r,
            )

        # Extract R results
        v_r = dict(zip(result_r.rx2("V").names, result_r.rx2("V")))
        bcv_r = dict(zip(result_r.rx2("bcV").names, result_r.rx2("bcV")))
        lambda_r = dict(
            zip(result_r.rx2("lambda").names, result_r.rx2("lambda"))
        )
        tau_r = dict(zip(result_r.rx2("tau").names, result_r.rx2("tau")))

        # Compare Cramer's V
        for var in ["x1", "x2", "x3"]:
            np.testing.assert_allclose(
                result_py["V"][var],
                v_r[var],
                rtol=1e-5,
                err_msg=f"V mismatch for {var}",
            )

            np.testing.assert_allclose(
                result_py["bcV"][var],
                bcv_r[var],
                rtol=1e-5,
                err_msg=f"bcV mismatch for {var}",
            )

            np.testing.assert_allclose(
                result_py["lambda_"][var],
                lambda_r[var],
                rtol=1e-5,
                err_msg=f"lambda mismatch for {var}",
            )

            np.testing.assert_allclose(
                result_py["tau"][var],
                tau_r[var],
                rtol=1e-5,
                err_msg=f"tau mismatch for {var}",
            )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_pw_assoc_with_weights_against_r(self, sample_categorical_data):
        """Test pw_assoc with weights against R implementation."""
        from statmatch.comparison import pw_assoc

        data = sample_categorical_data

        # Python implementation
        result_py = pw_assoc(
            formula="y ~ x1 + x2 + x3",
            data=data,
            weights="weight",
        )

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            data_r = ro.conversion.py2rpy(data)

            result_r = statmatch_r.pw_assoc(
                formula=ro.Formula("y ~ x1 + x2 + x3"),
                data=data_r,
                weights="weight",
            )

        # Extract R results
        v_r = dict(zip(result_r.rx2("V").names, result_r.rx2("V")))

        # Compare Cramer's V
        for var in ["x1", "x2", "x3"]:
            np.testing.assert_allclose(
                result_py["V"][var],
                v_r[var],
                rtol=1e-5,
                err_msg=f"Weighted V mismatch for {var}",
            )
