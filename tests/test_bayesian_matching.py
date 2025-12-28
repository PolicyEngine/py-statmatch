"""Tests for Bayesian uncertainty quantification in statistical matching."""

import numpy as np
import pandas as pd
import pytest
from statmatch.bayesian_matching import (
    bayesian_match,
    posterior_predictive,
    credible_interval,
    cia_posterior_test,
)


class TestBayesianMatch:
    """Test suite for Bayesian matching function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample donor and recipient datasets."""
        np.random.seed(42)

        # Donor dataset with matching variables X and donation variables Z
        n_donors = 100
        donor_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n_donors),
                "x2": np.random.normal(0, 2, n_donors),
                "z1": np.random.normal(100, 20, n_donors),
                "z2": np.random.normal(50, 10, n_donors),
            }
        )

        # Recipient dataset with matching variables X but no Z
        n_recipients = 50
        recipient_data = pd.DataFrame(
            {
                "x1": np.random.normal(0.2, 1, n_recipients),
                "x2": np.random.normal(0.2, 2, n_recipients),
            }
        )

        return donor_data, recipient_data

    def test_posterior_samples_shape(self, sample_data):
        """Test that posterior samples have correct shape."""
        donor_data, recipient_data = sample_data

        result = bayesian_match(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            z_vars=["z1"],
            n_samples=500,
        )

        # Should have n_samples posterior samples for each recipient
        assert result["posterior_samples"].shape == (
            len(recipient_data),
            500,
        )

    def test_posterior_samples_multiple_z_vars(self, sample_data):
        """Test posterior samples shape with multiple z variables."""
        donor_data, recipient_data = sample_data

        result = bayesian_match(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            z_vars=["z1", "z2"],
            n_samples=100,
        )

        # Should have dictionary with samples for each z variable
        assert "z1" in result["posterior_samples"]
        assert "z2" in result["posterior_samples"]
        assert result["posterior_samples"]["z1"].shape == (
            len(recipient_data),
            100,
        )
        assert result["posterior_samples"]["z2"].shape == (
            len(recipient_data),
            100,
        )

    def test_point_estimates_returned(self, sample_data):
        """Test that point estimates are returned."""
        donor_data, recipient_data = sample_data

        result = bayesian_match(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            z_vars=["z1"],
            n_samples=100,
        )

        assert "point_estimates" in result
        assert len(result["point_estimates"]) == len(recipient_data)

    def test_credible_intervals_returned(self, sample_data):
        """Test that credible intervals are returned."""
        donor_data, recipient_data = sample_data

        result = bayesian_match(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            z_vars=["z1"],
            n_samples=100,
        )

        assert "credible_intervals" in result
        # Should have lower and upper bounds for each recipient
        ci = result["credible_intervals"]
        assert "lower" in ci
        assert "upper" in ci
        assert len(ci["lower"]) == len(recipient_data)
        assert len(ci["upper"]) == len(recipient_data)

    def test_uniform_prior(self, sample_data):
        """Test Bayesian matching with uniform prior."""
        donor_data, recipient_data = sample_data

        result = bayesian_match(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            z_vars=["z1"],
            n_samples=100,
            prior="uniform",
        )

        # Should run without error and return valid samples
        assert result["posterior_samples"].shape[0] == len(recipient_data)

    def test_distance_weighted_prior(self, sample_data):
        """Test Bayesian matching with distance-weighted prior."""
        donor_data, recipient_data = sample_data

        result = bayesian_match(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            z_vars=["z1"],
            n_samples=100,
            prior="distance_weighted",
        )

        # Should run without error and return valid samples
        assert result["posterior_samples"].shape[0] == len(recipient_data)

    def test_more_samples_tighter_intervals(self, sample_data):
        """Test that more samples lead to more stable intervals."""
        donor_data, recipient_data = sample_data

        # Run with fewer samples
        np.random.seed(42)
        result_few = bayesian_match(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            z_vars=["z1"],
            n_samples=50,
        )

        # Run with more samples
        np.random.seed(42)
        result_many = bayesian_match(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            z_vars=["z1"],
            n_samples=1000,
        )

        # Intervals from more samples should be approximately stable
        # (they may not be tighter, but point estimates should be similar)
        point_few = result_few["point_estimates"]
        point_many = result_many["point_estimates"]

        # Correlation between point estimates should be high
        corr = np.corrcoef(point_few, point_many)[0, 1]
        assert corr > 0.8

    def test_return_posterior_false(self, sample_data):
        """Test that posterior samples can be suppressed."""
        donor_data, recipient_data = sample_data

        result = bayesian_match(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            z_vars=["z1"],
            n_samples=100,
            return_posterior=False,
        )

        # Should still have point estimates and intervals
        assert "point_estimates" in result
        assert "credible_intervals" in result
        # But no full posterior samples
        assert result.get("posterior_samples") is None


class TestPosteriorPredictive:
    """Test suite for posterior predictive sampling."""

    @pytest.fixture
    def matched_donors(self):
        """Create sample matched donors data."""
        np.random.seed(42)
        n_donors = 50
        n_recipients = 20

        # Create donor data
        donor_data = pd.DataFrame(
            {
                "z1": np.random.normal(100, 20, n_donors),
                "z2": np.random.normal(50, 10, n_donors),
            }
        )

        # Create matching weights (how similar each donor is to each recipient)
        weights = np.random.dirichlet(np.ones(n_donors), size=n_recipients)

        return donor_data, weights

    def test_posterior_predictive_shape(self, matched_donors):
        """Test posterior predictive samples have correct shape."""
        donor_data, weights = matched_donors

        samples = posterior_predictive(
            donor_data=donor_data,
            z_vars=["z1"],
            weights=weights,
            n_samples=100,
        )

        # Should have n_recipients x n_samples
        assert samples.shape == (weights.shape[0], 100)

    def test_posterior_predictive_multiple_vars(self, matched_donors):
        """Test posterior predictive with multiple z variables."""
        donor_data, weights = matched_donors

        samples = posterior_predictive(
            donor_data=donor_data,
            z_vars=["z1", "z2"],
            weights=weights,
            n_samples=100,
        )

        # Should return dict with samples for each variable
        assert "z1" in samples
        assert "z2" in samples
        assert samples["z1"].shape == (weights.shape[0], 100)
        assert samples["z2"].shape == (weights.shape[0], 100)

    def test_posterior_predictive_values_reasonable(self, matched_donors):
        """Test that posterior predictive values are reasonable."""
        donor_data, weights = matched_donors

        samples = posterior_predictive(
            donor_data=donor_data,
            z_vars=["z1"],
            weights=weights,
            n_samples=1000,
        )

        # Mean of samples should be close to weighted mean of donor values
        for i in range(weights.shape[0]):
            expected_mean = np.sum(weights[i] * donor_data["z1"].values)
            sample_mean = np.mean(samples[i])
            # Allow some sampling variance
            assert abs(sample_mean - expected_mean) < 20


class TestCredibleInterval:
    """Test suite for credible interval computation."""

    def test_credible_interval_shape(self):
        """Test credible interval returns correct shape."""
        np.random.seed(42)
        samples = np.random.normal(0, 1, (10, 1000))

        lower, upper, point = credible_interval(samples, level=0.95)

        assert len(lower) == 10
        assert len(upper) == 10
        assert len(point) == 10

    def test_credible_interval_coverage(self):
        """Test credible intervals have proper coverage."""
        np.random.seed(42)
        # Generate many samples from known distribution
        true_mean = 100
        true_std = 20
        n_recipients = 100
        n_samples = 2000

        samples = np.random.normal(
            true_mean, true_std, (n_recipients, n_samples)
        )

        lower, upper, point = credible_interval(samples, level=0.95)

        # For each recipient, check if true mean is within interval
        # (this is a frequentist check, but useful for validation)
        coverage = np.mean((lower <= true_mean) & (upper >= true_mean))

        # Should be approximately 95% (allow some tolerance)
        assert coverage >= 0.85

    def test_credible_interval_narrower_with_less_variance(self):
        """Test that credible intervals are narrower with less variance."""
        np.random.seed(42)
        n_samples = 1000

        # High variance samples
        high_var = np.random.normal(0, 10, (1, n_samples))
        # Low variance samples
        low_var = np.random.normal(0, 1, (1, n_samples))

        _, upper_high, _ = credible_interval(high_var, level=0.95)
        lower_high, _, _ = credible_interval(high_var, level=0.95)

        _, upper_low, _ = credible_interval(low_var, level=0.95)
        lower_low, _, _ = credible_interval(low_var, level=0.95)

        width_high = upper_high[0] - lower_high[0]
        width_low = upper_low[0] - lower_low[0]

        assert width_low < width_high

    def test_different_credible_levels(self):
        """Test different credible levels (90%, 95%)."""
        np.random.seed(42)
        samples = np.random.normal(0, 1, (10, 1000))

        lower_90, upper_90, _ = credible_interval(samples, level=0.90)
        lower_95, upper_95, _ = credible_interval(samples, level=0.95)

        # 95% intervals should be wider than 90% intervals
        width_90 = np.mean(upper_90 - lower_90)
        width_95 = np.mean(upper_95 - lower_95)

        assert width_95 > width_90


class TestCIAPosteriorTest:
    """Test suite for CIA posterior test."""

    @pytest.fixture
    def sample_data_with_correlations(self):
        """Create sample data with known correlation structure."""
        np.random.seed(42)
        n = 200

        # Create correlated data
        x = np.random.normal(0, 1, n)
        y = 0.5 * x + np.random.normal(0, 1, n)
        z = 0.3 * x + 0.4 * y + np.random.normal(0, 1, n)

        data = pd.DataFrame({"x": x, "y": y, "z": z})
        return data

    def test_cia_posterior_returns_probability(
        self, sample_data_with_correlations
    ):
        """Test that CIA test returns posterior probability."""
        data = sample_data_with_correlations

        prob = cia_posterior_test(
            data=data,
            x_vars=["x"],
            y_vars=["y"],
            z_vars=["z"],
            n_samples=500,
        )

        # Should return probability between 0 and 1
        assert 0 <= prob <= 1

    def test_cia_high_when_conditional_independence_holds(self):
        """Test CIA probability is high when conditional independence holds."""
        np.random.seed(42)
        n = 500

        # Create data where Y and Z are conditionally independent given X
        x = np.random.normal(0, 1, n)
        y = 0.8 * x + np.random.normal(0, 0.5, n)
        z = 0.8 * x + np.random.normal(0, 0.5, n)  # Depends only on X

        data = pd.DataFrame({"x": x, "y": y, "z": z})

        prob = cia_posterior_test(
            data=data,
            x_vars=["x"],
            y_vars=["y"],
            z_vars=["z"],
            n_samples=500,
        )

        # Should have high probability (data satisfies CIA)
        # Note: with finite samples, won't be exactly 1
        assert prob > 0.3

    def test_cia_low_when_conditional_dependence_exists(self):
        """Test CIA probability is low when conditional dependence exists."""
        np.random.seed(42)
        n = 500

        # Create data where Y and Z are dependent even given X
        x = np.random.normal(0, 1, n)
        y = 0.3 * x + np.random.normal(0, 1, n)
        z = 0.3 * x + 0.8 * y + np.random.normal(0, 0.5, n)  # Depends on Y!

        data = pd.DataFrame({"x": x, "y": y, "z": z})

        prob = cia_posterior_test(
            data=data,
            x_vars=["x"],
            y_vars=["y"],
            z_vars=["z"],
            n_samples=500,
        )

        # Should have lower probability (CIA is violated)
        assert prob < 0.7
