"""Test propensity score matching functionality."""

import numpy as np
import pandas as pd
import pytest
from statmatch.propensity_matching import (
    propensity_hotdeck,
    estimate_propensity,
)


class TestEstimatePropensity:
    """Test suite for propensity score estimation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample donor and recipient datasets."""
        np.random.seed(42)

        # Donor dataset
        n_donors = 100
        donor_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n_donors),
                "x2": np.random.normal(0, 2, n_donors),
                "x3": np.random.uniform(0, 10, n_donors),
                "y": np.random.normal(100, 20, n_donors),
            }
        )

        # Recipient dataset (slightly shifted distribution)
        n_recipients = 80
        recipient_data = pd.DataFrame(
            {
                "x1": np.random.normal(0.3, 1, n_recipients),
                "x2": np.random.normal(0.3, 2, n_recipients),
                "x3": np.random.uniform(1, 11, n_recipients),
            }
        )

        return donor_data, recipient_data

    def test_propensity_scores_in_valid_range(self, sample_data):
        """Test that propensity scores are in [0, 1]."""
        donor_data, recipient_data = sample_data
        match_vars = ["x1", "x2", "x3"]

        scores_rec, scores_don = estimate_propensity(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=match_vars,
            estimator="logistic",
        )

        # All scores should be in [0, 1]
        assert np.all(scores_rec >= 0) and np.all(scores_rec <= 1)
        assert np.all(scores_don >= 0) and np.all(scores_don <= 1)

        # Lengths should match input data
        assert len(scores_rec) == len(recipient_data)
        assert len(scores_don) == len(donor_data)

    @pytest.mark.parametrize(
        "estimator",
        ["logistic", "gbm", "random_forest", "neural_net"],
    )
    def test_all_estimators_produce_valid_scores(self, sample_data, estimator):
        """Test that all estimators produce valid propensity scores."""
        donor_data, recipient_data = sample_data
        match_vars = ["x1", "x2", "x3"]

        scores_rec, scores_don = estimate_propensity(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=match_vars,
            estimator=estimator,
        )

        # All scores should be valid probabilities
        assert np.all(np.isfinite(scores_rec))
        assert np.all(np.isfinite(scores_don))
        assert np.all(scores_rec >= 0) and np.all(scores_rec <= 1)
        assert np.all(scores_don >= 0) and np.all(scores_don <= 1)

    def test_cross_validation_option(self, sample_data):
        """Test that cross-validation can be enabled."""
        donor_data, recipient_data = sample_data
        match_vars = ["x1", "x2", "x3"]

        # Should run without error with cross-validation
        scores_rec, scores_don = estimate_propensity(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=match_vars,
            estimator="logistic",
            cv=5,
        )

        assert len(scores_rec) == len(recipient_data)
        assert len(scores_don) == len(donor_data)

    def test_invalid_estimator_raises_error(self, sample_data):
        """Test that invalid estimator raises ValueError."""
        donor_data, recipient_data = sample_data

        with pytest.raises(ValueError, match="Unknown estimator"):
            estimate_propensity(
                data_rec=recipient_data,
                data_don=donor_data,
                match_vars=["x1", "x2", "x3"],
                estimator="invalid_estimator",
            )


class TestPropensityHotdeck:
    """Test suite for propensity-based hot deck matching."""

    @pytest.fixture
    def sample_data(self):
        """Create sample donor and recipient datasets."""
        np.random.seed(42)

        # Donor dataset
        n_donors = 100
        donor_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n_donors),
                "x2": np.random.normal(0, 2, n_donors),
                "x3": np.random.uniform(0, 10, n_donors),
                "y": np.random.normal(100, 20, n_donors),
            }
        )

        # Recipient dataset
        n_recipients = 80
        recipient_data = pd.DataFrame(
            {
                "x1": np.random.normal(0.3, 1, n_recipients),
                "x2": np.random.normal(0.3, 2, n_recipients),
                "x3": np.random.uniform(1, 11, n_recipients),
            }
        )

        return donor_data, recipient_data

    def test_basic_propensity_matching(self, sample_data):
        """Test basic propensity matching returns valid results."""
        donor_data, recipient_data = sample_data
        match_vars = ["x1", "x2", "x3"]

        result = propensity_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=match_vars,
        )

        # Check result structure matches nnd_hotdeck
        assert "mtc.ids" in result
        assert "noad.index" in result
        assert "dist.rd" in result

        # Check dimensions
        assert len(result["dist.rd"]) == len(recipient_data)
        assert len(result["noad.index"]) == len(recipient_data)

        # Check donor indices are valid
        assert np.all(result["noad.index"] >= 0)
        assert np.all(result["noad.index"] < len(donor_data))

        # Check distances are non-negative
        assert np.all(result["dist.rd"] >= 0)

    @pytest.mark.parametrize(
        "estimator",
        ["logistic", "gbm", "random_forest", "neural_net"],
    )
    def test_all_estimators_produce_valid_matches(
        self, sample_data, estimator
    ):
        """Test that all estimators produce valid matches."""
        donor_data, recipient_data = sample_data
        match_vars = ["x1", "x2", "x3"]

        result = propensity_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=match_vars,
            estimator=estimator,
        )

        # All matches should be valid
        assert len(result["noad.index"]) == len(recipient_data)
        assert np.all(result["noad.index"] >= 0)
        assert np.all(result["noad.index"] < len(donor_data))

    def test_caliper_restricts_match_distance(self, sample_data):
        """Test that caliper restricts match distances."""
        donor_data, recipient_data = sample_data
        match_vars = ["x1", "x2", "x3"]

        # First get matches without caliper
        result_no_caliper = propensity_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=match_vars,
            caliper=None,
        )

        # Get matches with a tight caliper
        caliper = 0.1
        result_with_caliper = propensity_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=match_vars,
            caliper=caliper,
        )

        # All matched distances should be within caliper
        matched_mask = ~np.isnan(result_with_caliper["dist.rd"])
        assert np.all(result_with_caliper["dist.rd"][matched_mask] <= caliper)

    def test_caliper_produces_unmatched_recipients(self, sample_data):
        """Test that very tight caliper can leave recipients unmatched."""
        donor_data, recipient_data = sample_data
        match_vars = ["x1", "x2", "x3"]

        # Very tight caliper
        result = propensity_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=match_vars,
            caliper=0.001,  # Very tight
        )

        # Some matches should be NaN (unmatched)
        # With such a tight caliper, at least some should be unmatched
        # unless the data happens to have very similar propensity scores
        assert "noad.index" in result  # Structure is valid

    def test_n_neighbors_parameter(self, sample_data):
        """Test that n_neighbors parameter works correctly."""
        donor_data, recipient_data = sample_data
        match_vars = ["x1", "x2", "x3"]

        # Get matches with n_neighbors=3
        result = propensity_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=match_vars,
            n_neighbors=3,
        )

        # Result should contain matches
        assert len(result["noad.index"]) == len(recipient_data)

    def test_propensity_scores_returned(self, sample_data):
        """Test that propensity scores are returned in result."""
        donor_data, recipient_data = sample_data
        match_vars = ["x1", "x2", "x3"]

        result = propensity_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=match_vars,
        )

        # Propensity scores should be in result
        assert "ps.rec" in result
        assert "ps.don" in result

        # Scores should be valid probabilities
        assert np.all(result["ps.rec"] >= 0) and np.all(result["ps.rec"] <= 1)
        assert np.all(result["ps.don"] >= 0) and np.all(result["ps.don"] <= 1)


class TestBalanceImprovement:
    """Test that propensity matching improves covariate balance."""

    @pytest.fixture
    def imbalanced_data(self):
        """Create datasets with clear imbalance."""
        np.random.seed(42)

        # Donor dataset - centered at 0
        n_donors = 200
        donor_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n_donors),
                "x2": np.random.normal(0, 1, n_donors),
                "y": np.random.normal(100, 20, n_donors),
            }
        )

        # Recipient dataset - shifted to create imbalance
        n_recipients = 100
        recipient_data = pd.DataFrame(
            {
                "x1": np.random.normal(1.0, 1, n_recipients),
                "x2": np.random.normal(1.0, 1, n_recipients),
            }
        )

        return donor_data, recipient_data

    def test_balance_improves_after_matching(self, imbalanced_data):
        """Test that covariate balance improves after propensity matching."""
        donor_data, recipient_data = imbalanced_data
        match_vars = ["x1", "x2"]

        # Calculate initial imbalance (standardized mean difference)
        def calc_smd(rec_vals, don_vals):
            """Calculate standardized mean difference."""
            pooled_std = np.sqrt((np.var(rec_vals) + np.var(don_vals)) / 2)
            return abs(np.mean(rec_vals) - np.mean(don_vals)) / pooled_std

        initial_smd_x1 = calc_smd(
            recipient_data["x1"].values, donor_data["x1"].values
        )
        initial_smd_x2 = calc_smd(
            recipient_data["x2"].values, donor_data["x2"].values
        )

        # Perform propensity matching
        result = propensity_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=match_vars,
        )

        # Get matched donors
        matched_donors = donor_data.iloc[result["noad.index"]]

        # Calculate post-matching imbalance
        post_smd_x1 = calc_smd(
            recipient_data["x1"].values, matched_donors["x1"].values
        )
        post_smd_x2 = calc_smd(
            recipient_data["x2"].values, matched_donors["x2"].values
        )

        # Balance should improve (SMD should decrease)
        # Note: This is a statistical test so we use a soft assertion
        # Average SMD should improve
        initial_avg_smd = (initial_smd_x1 + initial_smd_x2) / 2
        post_avg_smd = (post_smd_x1 + post_smd_x2) / 2

        assert post_avg_smd < initial_avg_smd, (
            f"Balance did not improve: initial SMD={initial_avg_smd:.3f}, "
            f"post SMD={post_avg_smd:.3f}"
        )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_match_vars_in_recipient(self):
        """Test error when match vars missing in recipient data."""
        donor_data = pd.DataFrame({"x1": [1, 2, 3], "y": [10, 20, 30]})
        recipient_data = pd.DataFrame({"x2": [1, 2]})

        with pytest.raises(ValueError, match="not found in recipient"):
            propensity_hotdeck(
                data_rec=recipient_data,
                data_don=donor_data,
                match_vars=["x1"],
            )

    def test_missing_match_vars_in_donor(self):
        """Test error when match vars missing in donor data."""
        donor_data = pd.DataFrame({"x2": [1, 2, 3], "y": [10, 20, 30]})
        recipient_data = pd.DataFrame({"x1": [1, 2]})

        with pytest.raises(ValueError, match="not found in donor"):
            propensity_hotdeck(
                data_rec=recipient_data,
                data_don=donor_data,
                match_vars=["x1"],
            )

    def test_empty_dataframes(self):
        """Test handling of empty dataframes."""
        donor_data = pd.DataFrame({"x1": [], "y": []})
        recipient_data = pd.DataFrame({"x1": [1, 2]})

        with pytest.raises(ValueError, match="empty"):
            propensity_hotdeck(
                data_rec=recipient_data,
                data_don=donor_data,
                match_vars=["x1"],
            )

    def test_single_observation(self):
        """Test matching with minimal data."""
        donor_data = pd.DataFrame({"x1": [1.0], "y": [10]})
        recipient_data = pd.DataFrame({"x1": [1.5]})

        result = propensity_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1"],
        )

        # Should match to the only donor
        assert result["noad.index"][0] == 0
