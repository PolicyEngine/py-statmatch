"""Test Multiple Imputation (MI) matching functions."""

import numpy as np
import pandas as pd
import pytest

from statmatch.mi_matching import (
    mi_nnd_hotdeck,
    mi_create_fused,
    combine_mi_estimates,
    mi_summary,
)


class TestMINNDHotdeck:
    """Test suite for mi_nnd_hotdeck function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample donor and recipient datasets."""
        np.random.seed(42)

        n_donors = 50
        donor_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n_donors),
                "x2": np.random.normal(0, 2, n_donors),
                "y": np.random.normal(100, 20, n_donors),
                "donation_class": np.random.choice(["A", "B"], n_donors),
            }
        )

        n_recipients = 30
        recipient_data = pd.DataFrame(
            {
                "x1": np.random.normal(0.5, 1, n_recipients),
                "x2": np.random.normal(0.5, 2, n_recipients),
                "donation_class": np.random.choice(["A", "B"], n_recipients),
            }
        )

        return donor_data, recipient_data

    def test_produces_m_datasets(self, sample_data):
        """Test that MI matching produces m imputed datasets."""
        donor_data, recipient_data = sample_data

        result = mi_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            m=5,
        )

        assert len(result) == 5
        for i, match_result in enumerate(result):
            assert "mtc.ids" in match_result
            assert "noad.index" in match_result
            assert "dist.rd" in match_result
            assert len(match_result["noad.index"]) == len(recipient_data)

    def test_default_m_is_5(self, sample_data):
        """Test that default number of imputations is 5."""
        donor_data, recipient_data = sample_data

        result = mi_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
        )

        assert len(result) == 5

    def test_imputations_differ(self, sample_data):
        """Test that different imputations produce different matches."""
        donor_data, recipient_data = sample_data

        result = mi_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            m=5,
        )

        # At least some imputations should have different donor indices
        all_same = all(
            np.array_equal(result[0]["noad.index"], result[i]["noad.index"])
            for i in range(1, 5)
        )
        assert not all_same

    def test_with_donation_classes(self, sample_data):
        """Test MI matching respects donation classes."""
        donor_data, recipient_data = sample_data

        result = mi_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            don_class="donation_class",
            m=3,
        )

        assert len(result) == 3

        # Check that each imputation respects donation classes
        for match_result in result:
            for i, rec_class in enumerate(recipient_data["donation_class"]):
                donor_idx = match_result["noad.index"][i]
                donor_class = donor_data.iloc[donor_idx]["donation_class"]
                assert rec_class == donor_class

    def test_bootstrap_method(self, sample_data):
        """Test MI matching with bootstrap resampling method."""
        donor_data, recipient_data = sample_data

        result = mi_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            m=5,
            method="bootstrap",
        )

        assert len(result) == 5

    def test_noise_method(self, sample_data):
        """Test MI matching with noise injection method."""
        donor_data, recipient_data = sample_data

        result = mi_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            m=5,
            method="noise",
        )

        assert len(result) == 5

    def test_reproducibility_with_seed(self, sample_data):
        """Test that results are reproducible with same seed."""
        donor_data, recipient_data = sample_data

        result1 = mi_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            m=3,
            seed=123,
        )

        result2 = mi_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            m=3,
            seed=123,
        )

        for i in range(3):
            np.testing.assert_array_equal(
                result1[i]["noad.index"], result2[i]["noad.index"]
            )


class TestMICreateFused:
    """Test suite for mi_create_fused function."""

    @pytest.fixture
    def sample_data_with_matches(self):
        """Create sample data with MI matching results."""
        np.random.seed(42)

        n_donors = 50
        donor_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n_donors),
                "x2": np.random.normal(0, 2, n_donors),
                "y": np.random.normal(100, 20, n_donors),
                "z": np.random.choice(["A", "B", "C"], n_donors),
            }
        )

        n_recipients = 30
        recipient_data = pd.DataFrame(
            {
                "x1": np.random.normal(0.5, 1, n_recipients),
                "x2": np.random.normal(0.5, 2, n_recipients),
            }
        )

        # Run MI matching
        mi_results = mi_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            m=5,
            seed=42,
        )

        return donor_data, recipient_data, mi_results

    def test_creates_m_fused_datasets(self, sample_data_with_matches):
        """Test that mi_create_fused creates m fused datasets."""
        donor_data, recipient_data, mi_results = sample_data_with_matches

        fused_datasets = mi_create_fused(
            data_rec=recipient_data,
            data_don=donor_data,
            mi_results=mi_results,
            z_vars=["y", "z"],
        )

        assert len(fused_datasets) == 5
        for fused in fused_datasets:
            assert "y" in fused.columns
            assert "z" in fused.columns
            assert len(fused) == len(recipient_data)

    def test_fused_datasets_preserve_recipient_data(
        self, sample_data_with_matches
    ):
        """Test that fused datasets preserve recipient data."""
        donor_data, recipient_data, mi_results = sample_data_with_matches

        fused_datasets = mi_create_fused(
            data_rec=recipient_data,
            data_don=donor_data,
            mi_results=mi_results,
            z_vars=["y"],
        )

        for fused in fused_datasets:
            for col in recipient_data.columns:
                pd.testing.assert_series_equal(
                    fused[col].reset_index(drop=True),
                    recipient_data[col].reset_index(drop=True),
                    check_names=False,
                )


class TestCombineMIEstimates:
    """Test suite for combine_mi_estimates (Rubin's rules)."""

    def test_combine_simple_estimates(self):
        """Test combining simple point estimates."""
        # 5 imputations with point estimates and variances
        estimates = np.array([10.0, 11.0, 9.5, 10.5, 10.0])
        variances = np.array([1.0, 1.2, 0.9, 1.1, 1.0])

        result = combine_mi_estimates(estimates, variances)

        # Combined estimate is mean of estimates
        expected_estimate = np.mean(estimates)
        assert np.isclose(result["estimate"], expected_estimate)

        # Within-imputation variance is mean of variances
        expected_within = np.mean(variances)
        assert np.isclose(result["within_variance"], expected_within)

        # Between-imputation variance is variance of estimates
        expected_between = np.var(estimates, ddof=1)
        assert np.isclose(result["between_variance"], expected_between)

        # Total variance = W + (1 + 1/m) * B
        m = len(estimates)
        expected_total = expected_within + (1 + 1 / m) * expected_between
        assert np.isclose(result["total_variance"], expected_total)

    def test_combine_with_known_values(self):
        """Test Rubin's rules with known values for verification."""
        # Example from Rubin (1987)
        estimates = np.array([5.0, 6.0, 5.5])
        variances = np.array([2.0, 2.0, 2.0])

        result = combine_mi_estimates(estimates, variances)

        # Check calculations
        m = 3
        qbar = np.mean(estimates)  # 5.5
        ubar = np.mean(variances)  # 2.0
        b = np.var(estimates, ddof=1)  # 0.25
        total = ubar + (1 + 1 / m) * b  # 2.0 + 1.333 * 0.25 = 2.333

        assert np.isclose(result["estimate"], qbar)
        assert np.isclose(result["total_variance"], total)

    def test_degrees_of_freedom(self):
        """Test degrees of freedom calculation."""
        estimates = np.array([10.0, 11.0, 9.5, 10.5, 10.0])
        variances = np.array([1.0, 1.2, 0.9, 1.1, 1.0])

        result = combine_mi_estimates(estimates, variances)

        # df = (m - 1) * (1 + W / ((1 + 1/m) * B))^2
        m = len(estimates)
        W = np.mean(variances)
        B = np.var(estimates, ddof=1)
        r = (1 + 1 / m) * B / W  # fraction of missing information
        expected_df = (m - 1) * (1 + 1 / r) ** 2

        assert "df" in result
        assert result["df"] > 0

    def test_returns_standard_error(self):
        """Test that result includes standard error."""
        estimates = np.array([10.0, 11.0, 9.5])
        variances = np.array([1.0, 1.2, 0.9])

        result = combine_mi_estimates(estimates, variances)

        assert "std_error" in result
        assert np.isclose(
            result["std_error"], np.sqrt(result["total_variance"])
        )


class TestMISummary:
    """Test suite for mi_summary function."""

    @pytest.fixture
    def sample_fused_datasets(self):
        """Create sample fused datasets for testing."""
        np.random.seed(42)

        n = 100
        fused_datasets = []

        for i in range(5):
            df = pd.DataFrame(
                {
                    "x1": np.random.normal(0, 1, n),
                    "y": np.random.normal(100 + i, 20, n),
                    "z": np.random.choice(["A", "B", "C"], n),
                }
            )
            fused_datasets.append(df)

        return fused_datasets

    def test_summary_returns_dataframe(self, sample_fused_datasets):
        """Test that mi_summary returns a DataFrame."""
        result = mi_summary(sample_fused_datasets, var="y")

        assert isinstance(result, pd.DataFrame)

    def test_summary_contains_expected_columns(self, sample_fused_datasets):
        """Test that summary contains expected columns."""
        result = mi_summary(sample_fused_datasets, var="y")

        expected_cols = [
            "estimate",
            "std_error",
            "ci_lower",
            "ci_upper",
            "within_var",
            "between_var",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_summary_mean_statistic(self, sample_fused_datasets):
        """Test summary for mean statistic."""
        result = mi_summary(sample_fused_datasets, var="y", statistic="mean")

        # The combined estimate should be close to the average of means
        means = [df["y"].mean() for df in sample_fused_datasets]
        expected_mean = np.mean(means)

        assert np.isclose(
            result.loc["mean", "estimate"], expected_mean, rtol=0.01
        )

    def test_summary_with_groupby(self, sample_fused_datasets):
        """Test summary with groupby variable."""
        result = mi_summary(
            sample_fused_datasets, var="y", statistic="mean", groupby="z"
        )

        # Should have rows for each group
        assert len(result) == 3  # A, B, C

    def test_confidence_interval_coverage(self, sample_fused_datasets):
        """Test that CI is computed correctly."""
        result = mi_summary(sample_fused_datasets, var="y", conf_level=0.95)

        # CI should be symmetric around estimate
        ci_width_lower = (
            result.loc["mean", "estimate"] - result.loc["mean", "ci_lower"]
        )
        ci_width_upper = (
            result.loc["mean", "ci_upper"] - result.loc["mean", "estimate"]
        )

        assert np.isclose(ci_width_lower, ci_width_upper, rtol=0.01)

    def test_summary_multiple_statistics(self, sample_fused_datasets):
        """Test summary with multiple statistics."""
        result = mi_summary(
            sample_fused_datasets,
            var="y",
            statistic=["mean", "var"],
        )

        assert "mean" in result.index
        assert "var" in result.index
