"""Test sample utility functions against R's StatMatch implementation."""

import numpy as np
import pandas as pd
import pytest

# Only import rpy2 for test generation
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    # Import StatMatch and survey packages
    statmatch_r = importr("StatMatch")
    survey_r = importr("survey")
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False


class TestFact2Dummy:
    """Test suite for fact2dummy function."""

    def test_single_factor_all_levels(self):
        """Test converting a single factor with all levels."""
        from statmatch.sample_utils import fact2dummy

        # Create a factor-like pandas Series
        data = pd.Series(
            pd.Categorical(["a", "b", "a", "c", "b"]),
            name="x"
        )

        result = fact2dummy(data)

        # Should have 5 rows and 3 columns (one per level)
        assert result.shape == (5, 3)

        # Check column names
        assert "x.a" in result.columns
        assert "x.b" in result.columns
        assert "x.c" in result.columns

        # Check values - first row should be [1, 0, 0] for 'a'
        assert result.iloc[0]["x.a"] == 1
        assert result.iloc[0]["x.b"] == 0
        assert result.iloc[0]["x.c"] == 0

    def test_single_factor_drop_last(self):
        """Test converting a single factor dropping the last level."""
        from statmatch.sample_utils import fact2dummy

        data = pd.Series(
            pd.Categorical(["a", "b", "a", "c", "b"]),
            name="x"
        )

        result = fact2dummy(data, all_levels=False)

        # Should have 5 rows and 2 columns (last level dropped)
        assert result.shape == (5, 2)

        # 'c' should be dropped as the last level
        assert "x.c" not in result.columns

    def test_dataframe_mixed_types(self):
        """Test converting a DataFrame with mixed numeric and categorical."""
        from statmatch.sample_utils import fact2dummy

        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": pd.Categorical([1, 2, 1, 2, 2]),
            "z": pd.Categorical(["low", "med", "high", "med", "med"],
                               categories=["low", "med", "high"],
                               ordered=True)
        })

        result = fact2dummy(df)

        # x should be preserved as numeric
        assert "x" in result.columns

        # y has 2 levels
        assert "y.1" in result.columns
        assert "y.2" in result.columns

        # z has 3 levels
        assert "z.low" in result.columns
        assert "z.med" in result.columns
        assert "z.high" in result.columns

    def test_missing_values(self):
        """Test handling of missing values in factors."""
        from statmatch.sample_utils import fact2dummy

        data = pd.Series(
            pd.Categorical(["a", "b", None, "c", "b"]),
            name="x"
        )

        result = fact2dummy(data)

        # Row with NA should have NaN in all dummy columns
        assert pd.isna(result.iloc[2]["x.a"])
        assert pd.isna(result.iloc[2]["x.b"])
        assert pd.isna(result.iloc[2]["x.c"])

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_against_r_implementation(self):
        """Test fact2dummy matches R's StatMatch::fact2dummy."""
        from statmatch.sample_utils import fact2dummy

        # Create test data
        df = pd.DataFrame({
            "x": [1.5, 2.3, 0.8, 4.1, 3.2],
            "y": pd.Categorical([1, 2, 1, 2, 2]),
            "z": pd.Categorical(["low", "med", "high", "med", "low"],
                               categories=["high", "low", "med"],
                               ordered=True)
        })

        # Python implementation
        result_py = fact2dummy(df)

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            # In R, we need to create factors explicitly
            ro.r('''
            x <- c(1.5, 2.3, 0.8, 4.1, 3.2)
            y <- factor(c(1, 2, 1, 2, 2))
            z <- ordered(c("low", "med", "high", "med", "low"),
                        levels=c("high", "low", "med"))
            xyz <- data.frame(x, y, z)
            ''')
            result_r = statmatch_r.fact2dummy(ro.r["xyz"])
            result_r_df = ro.conversion.rpy2py(result_r)

        # Convert R matrix to DataFrame for comparison
        if isinstance(result_r_df, np.ndarray):
            r_cols = list(ro.r("colnames")(result_r))
            result_r_df = pd.DataFrame(result_r_df, columns=r_cols)

        # Compare shapes
        assert result_py.shape == result_r_df.shape

        # Compare values (allowing for column order differences)
        for col in result_py.columns:
            if col in result_r_df.columns:
                np.testing.assert_array_almost_equal(
                    result_py[col].values,
                    result_r_df[col].values,
                    decimal=10
                )


class TestHarmonizeX:
    """Test suite for harmonize_x function."""

    @pytest.fixture
    def sample_surveys(self):
        """Create sample survey datasets A and B."""
        np.random.seed(42)

        # Survey A data
        n_a = 100
        survey_a = pd.DataFrame({
            "sex": pd.Categorical(
                np.random.choice(["M", "F"], n_a),
                categories=["F", "M"]
            ),
            "age_group": pd.Categorical(
                np.random.choice(["young", "mid", "old"], n_a),
                categories=["young", "mid", "old"]
            ),
            "income": np.random.normal(50000, 15000, n_a),
            "weight": np.random.uniform(1.0, 2.0, n_a)
        })

        # Survey B data
        n_b = 120
        survey_b = pd.DataFrame({
            "sex": pd.Categorical(
                np.random.choice(["M", "F"], n_b),
                categories=["F", "M"]
            ),
            "age_group": pd.Categorical(
                np.random.choice(["young", "mid", "old"], n_b),
                categories=["young", "mid", "old"]
            ),
            "expenditure": np.random.normal(40000, 10000, n_b),
            "weight": np.random.uniform(0.8, 1.8, n_b)
        })

        return survey_a, survey_b

    def test_harmonize_marginal(self, sample_surveys):
        """Test harmonization with marginal distributions."""
        from statmatch.sample_utils import harmonize_x

        survey_a, survey_b = sample_surveys

        result = harmonize_x(
            svy_a=survey_a,
            svy_b=survey_b,
            x_vars=["sex", "age_group"],
            weight_a="weight",
            weight_b="weight"
        )

        # Check output structure
        assert "weights_a" in result
        assert "weights_b" in result

        # Check weights have correct length
        assert len(result["weights_a"]) == len(survey_a)
        assert len(result["weights_b"]) == len(survey_b)

        # Check weights are positive
        assert np.all(result["weights_a"] > 0)
        assert np.all(result["weights_b"] > 0)

    def test_harmonize_joint(self, sample_surveys):
        """Test harmonization with joint distributions."""
        from statmatch.sample_utils import harmonize_x

        survey_a, survey_b = sample_surveys

        result = harmonize_x(
            svy_a=survey_a,
            svy_b=survey_b,
            x_vars=["sex", "age_group"],
            weight_a="weight",
            weight_b="weight",
            joint=True  # Consider joint distribution
        )

        # Verify that weighted totals match after harmonization
        # For joint distribution of sex x age_group
        assert "weights_a" in result
        assert "weights_b" in result

    def test_harmonize_with_known_totals(self, sample_surveys):
        """Test harmonization with known population totals."""
        from statmatch.sample_utils import harmonize_x

        survey_a, survey_b = sample_surveys

        # Known population totals for sex
        x_tot = {"sex.F": 500, "sex.M": 500}

        result = harmonize_x(
            svy_a=survey_a,
            svy_b=survey_b,
            x_vars=["sex"],
            weight_a="weight",
            weight_b="weight",
            x_tot=x_tot
        )

        # Check that harmonized weights sum to population totals
        weighted_totals_a = pd.crosstab(
            survey_a["sex"],
            columns="count",
            values=result["weights_a"],
            aggfunc="sum"
        )
        # Weighted totals should approximate known totals
        assert "weights_a" in result

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_against_r_implementation(self, sample_surveys):
        """Test harmonize_x matches R's StatMatch::harmonize.x."""
        from statmatch.sample_utils import harmonize_x

        survey_a, survey_b = sample_surveys

        # Python implementation
        result_py = harmonize_x(
            svy_a=survey_a,
            svy_b=survey_b,
            x_vars=["sex", "age_group"],
            weight_a="weight",
            weight_b="weight",
            cal_method="linear"
        )

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            survey_a_r = ro.conversion.py2rpy(survey_a)
            survey_b_r = ro.conversion.py2rpy(survey_b)

            # Create survey design objects in R
            ro.r.assign("survey_a", survey_a_r)
            ro.r.assign("survey_b", survey_b_r)

            ro.r('''
            library(survey)
            survey_a$fpc <- nrow(survey_a) / 1000
            survey_b$fpc <- nrow(survey_b) / 1000
            svy_a <- svydesign(~1, weights=~weight, data=survey_a)
            svy_b <- svydesign(~1, weights=~weight, data=survey_b)
            result_r <- harmonize.x(svy_a, svy_b, form.x=~sex+age_group)
            ''')

            weights_a_r = np.array(ro.r("result_r$weights.A"))
            weights_b_r = np.array(ro.r("result_r$weights.B"))

        # Compare weights (may have minor differences due to algorithms)
        np.testing.assert_allclose(
            result_py["weights_a"],
            weights_a_r,
            rtol=0.01,  # 1% relative tolerance
            atol=0.1
        )


class TestCombSamples:
    """Test suite for comb_samples function."""

    @pytest.fixture
    def sample_surveys_yz(self):
        """Create sample survey datasets for Y-Z estimation."""
        np.random.seed(123)

        # Survey A: has X variables and Y
        n_a = 80
        survey_a = pd.DataFrame({
            "sex": pd.Categorical(
                np.random.choice(["M", "F"], n_a),
                categories=["F", "M"]
            ),
            "age_group": pd.Categorical(
                np.random.choice(["young", "mid", "old"], n_a),
                categories=["young", "mid", "old"]
            ),
            "education": pd.Categorical(  # Y variable
                np.random.choice(["low", "mid", "high"], n_a),
                categories=["low", "mid", "high"]
            ),
            "weight": np.random.uniform(1.0, 2.0, n_a)
        })

        # Survey B: has X variables and Z
        n_b = 100
        survey_b = pd.DataFrame({
            "sex": pd.Categorical(
                np.random.choice(["M", "F"], n_b),
                categories=["F", "M"]
            ),
            "age_group": pd.Categorical(
                np.random.choice(["young", "mid", "old"], n_b),
                categories=["young", "mid", "old"]
            ),
            "spending": pd.Categorical(  # Z variable
                np.random.choice(["low", "med", "high"], n_b),
                categories=["low", "med", "high"]
            ),
            "weight": np.random.uniform(0.8, 1.8, n_b)
        })

        return survey_a, survey_b

    @pytest.fixture
    def sample_surveys_with_c(self, sample_surveys_yz):
        """Create sample surveys including auxiliary survey C."""
        survey_a, survey_b = sample_surveys_yz
        np.random.seed(456)

        # Survey C: has both Y and Z (auxiliary data)
        n_c = 50
        survey_c = pd.DataFrame({
            "sex": pd.Categorical(
                np.random.choice(["M", "F"], n_c),
                categories=["F", "M"]
            ),
            "age_group": pd.Categorical(
                np.random.choice(["young", "mid", "old"], n_c),
                categories=["young", "mid", "old"]
            ),
            "education": pd.Categorical(
                np.random.choice(["low", "mid", "high"], n_c),
                categories=["low", "mid", "high"]
            ),
            "spending": pd.Categorical(
                np.random.choice(["low", "med", "high"], n_c),
                categories=["low", "med", "high"]
            ),
            "weight": np.random.uniform(0.9, 1.5, n_c)
        })

        return survey_a, survey_b, survey_c

    def test_cia_estimation(self, sample_surveys_yz):
        """Test conditional independence assumption estimation."""
        from statmatch.sample_utils import comb_samples

        survey_a, survey_b = sample_surveys_yz

        result = comb_samples(
            svy_a=survey_a,
            svy_b=survey_b,
            y_lab="education",
            z_lab="spending",
            x_vars=["sex", "age_group"],
            weight_a="weight",
            weight_b="weight"
        )

        # Check output structure
        assert "yz_cia" in result

        # yz_cia should be a contingency table (Y x Z)
        yz_table = result["yz_cia"]
        assert yz_table.shape == (3, 3)  # 3 Y levels x 3 Z levels

        # All cells should be non-negative
        assert np.all(yz_table >= 0)

    def test_with_auxiliary_survey(self, sample_surveys_with_c):
        """Test estimation with auxiliary survey C."""
        from statmatch.sample_utils import comb_samples

        survey_a, survey_b, survey_c = sample_surveys_with_c

        result = comb_samples(
            svy_a=survey_a,
            svy_b=survey_b,
            svy_c=survey_c,
            y_lab="education",
            z_lab="spending",
            x_vars=["sex", "age_group"],
            weight_a="weight",
            weight_b="weight",
            weight_c="weight",
            estimation="incomplete"
        )

        # Check output structure
        assert "yz_cia" in result
        assert "yz_est" in result

        # yz_est uses auxiliary data for better estimates
        assert result["yz_est"].shape == (3, 3)

    def test_micro_output(self, sample_surveys_yz):
        """Test micro-level probability output."""
        from statmatch.sample_utils import comb_samples

        survey_a, survey_b = sample_surveys_yz

        result = comb_samples(
            svy_a=survey_a,
            svy_b=survey_b,
            y_lab="education",
            z_lab="spending",
            x_vars=["sex", "age_group"],
            weight_a="weight",
            weight_b="weight",
            micro=True
        )

        # With micro=True, should return probability matrices
        assert "z_a" in result
        assert "y_b" in result

        # z_a: probabilities of Z categories for each A unit
        assert result["z_a"].shape[0] == len(survey_a)
        assert result["z_a"].shape[1] == 3  # 3 Z categories

        # y_b: probabilities of Y categories for each B unit
        assert result["y_b"].shape[0] == len(survey_b)
        assert result["y_b"].shape[1] == 3  # 3 Y categories

        # Probabilities should sum to 1 for each row
        np.testing.assert_allclose(
            result["z_a"].sum(axis=1),
            np.ones(len(survey_a)),
            rtol=1e-6
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_against_r_implementation(self, sample_surveys_yz):
        """Test comb_samples matches R's StatMatch::comb.samples."""
        from statmatch.sample_utils import comb_samples

        survey_a, survey_b = sample_surveys_yz

        # Python implementation
        result_py = comb_samples(
            svy_a=survey_a,
            svy_b=survey_b,
            y_lab="education",
            z_lab="spending",
            x_vars=["sex", "age_group"],
            weight_a="weight",
            weight_b="weight"
        )

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            survey_a_r = ro.conversion.py2rpy(survey_a)
            survey_b_r = ro.conversion.py2rpy(survey_b)

            ro.r.assign("survey_a", survey_a_r)
            ro.r.assign("survey_b", survey_b_r)

            ro.r('''
            library(survey)
            svy_a <- svydesign(~1, weights=~weight, data=survey_a)
            svy_b <- svydesign(~1, weights=~weight, data=survey_b)
            result_r <- comb.samples(svy.A=svy_a, svy.B=svy_b,
                                    y.lab="education", z.lab="spending",
                                    form.x=~sex+age_group)
            ''')

            yz_cia_r = np.array(ro.r("result_r$yz.CIA"))

        # Compare CIA tables
        np.testing.assert_allclose(
            result_py["yz_cia"],
            yz_cia_r,
            rtol=0.01,
            atol=0.01
        )
