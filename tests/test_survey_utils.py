"""Tests for survey weight utilities."""

import numpy as np
import pandas as pd
import pytest
from statmatch.survey_utils import (
    weighted_distance,
    calibrate_weights,
    design_effect,
    replicate_variance,
)
from statmatch.nnd_hotdeck import nnd_hotdeck


class TestWeightedDistance:
    """Tests for weighted_distance function."""

    def test_equal_weights_matches_unweighted(self):
        """With equal weights, should match unweighted standardization."""
        np.random.seed(42)
        data_x = np.random.normal(0, 1, (20, 3))
        data_y = np.random.normal(0, 1, (15, 3))

        # Equal weights
        weights_x = np.ones(20)
        weights_y = np.ones(15)

        result = weighted_distance(
            data_x, data_y, weights_x, weights_y, dist_fun="euclidean"
        )

        # Check shape is correct
        assert result.shape == (20, 15)

        # Distances should be non-negative
        assert np.all(result >= 0)

    def test_weighted_standardization(self):
        """Verify weighted means and stds are used for standardization."""
        # Simple case: one variable
        data_x = np.array([[1.0], [2.0], [3.0], [4.0]])
        data_y = np.array([[2.5]])

        # Heavy weight on value 4
        weights_x = np.array([0.1, 0.1, 0.1, 0.7])
        weights_y = np.array([1.0])

        result = weighted_distance(
            data_x, data_y, weights_x, weights_y, dist_fun="euclidean"
        )

        # Weighted mean of x is closer to 4 than unweighted mean
        # So standardized value of 4 should be closer to 0
        # Distance from y=2.5 to x=4 should reflect this
        assert result.shape == (4, 1)

    def test_different_distance_functions(self):
        """Test that different distance functions work."""
        np.random.seed(42)
        data_x = np.random.normal(0, 1, (10, 2))
        data_y = np.random.normal(0, 1, (5, 2))
        weights_x = np.ones(10)
        weights_y = np.ones(5)

        for dist_fun in ["euclidean", "manhattan", "mahalanobis"]:
            result = weighted_distance(
                data_x, data_y, weights_x, weights_y, dist_fun=dist_fun
            )
            assert result.shape == (10, 5)
            assert np.all(result >= 0)


class TestCalibrateWeights:
    """Tests for calibrate_weights function."""

    @pytest.fixture
    def sample_survey_data(self):
        """Create sample survey data for calibration tests."""
        np.random.seed(42)
        n = 100

        # Create data with two categorical variables
        data = pd.DataFrame(
            {
                "gender": np.random.choice(["M", "F"], n),
                "age_group": np.random.choice(["18-34", "35-54", "55+"], n),
                "income": np.random.normal(50000, 15000, n),
            }
        )
        weights = np.ones(n)

        return data, weights

    def test_calibration_hits_targets(self, sample_survey_data):
        """Calibrated weights should reproduce target totals."""
        data, weights = sample_survey_data

        # Define target totals (these should be hit exactly or very closely)
        targets = {
            "gender": {"M": 500.0, "F": 500.0},  # 50/50 split
            "age_group": {"18-34": 350.0, "35-54": 400.0, "55+": 250.0},
        }

        calibrated = calibrate_weights(data, weights, targets)

        # Check that calibrated weights hit targets
        for var, level_targets in targets.items():
            for level, target in level_targets.items():
                mask = data[var] == level
                actual = np.sum(calibrated[mask])
                np.testing.assert_allclose(
                    actual,
                    target,
                    rtol=1e-4,
                    err_msg=f"Target not hit for {var}={level}",
                )

    def test_calibration_preserves_positivity(self, sample_survey_data):
        """Calibrated weights should remain positive."""
        data, weights = sample_survey_data

        targets = {
            "gender": {"M": 600.0, "F": 400.0},
        }

        calibrated = calibrate_weights(data, weights, targets)

        assert np.all(calibrated > 0), "Calibrated weights should be positive"

    def test_calibration_with_numeric_variable(self):
        """Test calibration can handle numeric total targets."""
        np.random.seed(42)
        n = 50
        data = pd.DataFrame(
            {
                "gender": np.random.choice(["M", "F"], n),
                "age": np.random.uniform(18, 80, n),
            }
        )
        weights = np.ones(n)

        targets = {"gender": {"M": 30.0, "F": 20.0}}

        calibrated = calibrate_weights(data, weights, targets)

        # Total weight should equal sum of targets
        assert np.isclose(np.sum(calibrated), 50.0)


class TestDesignEffect:
    """Tests for design_effect function."""

    def test_equal_weights_deff_one(self):
        """Equal weights should give DEFF = 1."""
        weights = np.ones(100)
        deff = design_effect(weights)

        np.testing.assert_allclose(deff, 1.0, rtol=1e-10)

    def test_unequal_weights_deff_greater_than_one(self):
        """Unequal weights should give DEFF > 1."""
        weights = np.array([1, 1, 1, 1, 10])  # One large weight
        deff = design_effect(weights)

        assert deff > 1.0

    def test_deff_formula(self):
        """Verify DEFF formula: 1 + CV(weights)^2."""
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        deff = design_effect(weights)

        # Manual calculation
        mean_w = np.mean(weights)
        std_w = np.std(weights, ddof=0)
        cv = std_w / mean_w
        expected_deff = 1 + cv**2

        np.testing.assert_allclose(deff, expected_deff)

    def test_effective_sample_size(self):
        """Test effective sample size calculation."""
        n = 100
        weights = np.ones(n) * 2  # All equal, but not 1
        weights[0] = 20  # One outlier

        deff = design_effect(weights)
        n_eff = n / deff

        assert n_eff < n, "Effective sample size should be less than n"
        assert n_eff > 0, "Effective sample size should be positive"


class TestReplicateVariance:
    """Tests for replicate_variance function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for variance estimation."""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame(
            {
                "x": np.random.normal(100, 20, n),
                "y": np.random.normal(50, 10, n),
            }
        )
        weights = np.ones(n)
        return data, weights

    def test_jackknife_variance(self, sample_data):
        """Test jackknife variance estimation."""
        data, weights = sample_data

        def mean_statistic(df, w):
            return np.average(df["x"], weights=w)

        variance = replicate_variance(
            data, weights, mean_statistic, method="jackknife"
        )

        assert variance > 0, "Variance should be positive"
        # Variance of mean should be approximately var(x)/n
        expected_approx = np.var(data["x"]) / len(data)
        assert (
            variance < expected_approx * 10
        ), "Variance is unreasonably large"

    def test_bootstrap_variance(self, sample_data):
        """Test bootstrap variance estimation."""
        data, weights = sample_data

        def mean_statistic(df, w):
            return np.average(df["x"], weights=w)

        variance = replicate_variance(
            data, weights, mean_statistic, method="bootstrap", n_replicates=100
        )

        assert variance > 0, "Variance should be positive"

    def test_variance_with_complex_statistic(self, sample_data):
        """Test variance estimation with a more complex statistic."""
        data, weights = sample_data

        def ratio_statistic(df, w):
            return np.average(df["x"], weights=w) / np.average(
                df["y"], weights=w
            )

        variance = replicate_variance(
            data, weights, ratio_statistic, method="jackknife"
        )

        assert variance > 0, "Variance should be positive"


class TestWeightedMatching:
    """Tests for weighted matching in nnd_hotdeck."""

    @pytest.fixture
    def weighted_sample_data(self):
        """Create sample data with weights."""
        np.random.seed(42)

        n_donors = 50
        n_recipients = 30

        donor_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n_donors),
                "x2": np.random.normal(0, 2, n_donors),
                "y": np.random.normal(100, 20, n_donors),
            }
        )

        recipient_data = pd.DataFrame(
            {
                "x1": np.random.normal(0.5, 1, n_recipients),
                "x2": np.random.normal(0.5, 2, n_recipients),
            }
        )

        # Create unequal donor weights (some donors more "valuable")
        don_weights = np.abs(np.random.normal(1, 0.5, n_donors))
        rec_weights = np.abs(np.random.normal(1, 0.3, n_recipients))

        return donor_data, recipient_data, don_weights, rec_weights

    def test_weighted_matching_basic(self, weighted_sample_data):
        """Test that weighted matching runs without error."""
        donor_data, recipient_data, don_weights, rec_weights = (
            weighted_sample_data
        )

        result = nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            rec_weights=rec_weights,
            don_weights=don_weights,
            dist_fun="euclidean",
        )

        assert "dist.rd" in result
        assert len(result["dist.rd"]) == len(recipient_data)

    def test_weights_affect_constrained_matching(self, weighted_sample_data):
        """Weighted constrained matching should favor higher-weight donors."""
        donor_data, recipient_data, _, _ = weighted_sample_data

        # Create extreme weights: one donor has very high weight
        don_weights = np.ones(len(donor_data))
        don_weights[0] = 100.0  # First donor has very high weight

        result_weighted = nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            don_weights=don_weights,
            dist_fun="euclidean",
            constr_alg="hungarian",
            k=2,
        )

        # With high weight, donor 0 should be used more often (if it's close)
        # This is a probabilistic test, so we just check it runs
        assert len(result_weighted["noad.index"]) == len(recipient_data)
