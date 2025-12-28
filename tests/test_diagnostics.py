"""Tests for diagnostics module."""

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import tempfile
import os

from statmatch.nnd_hotdeck import nnd_hotdeck
from statmatch.diagnostics import (
    match_diagnostics,
    MatchDiagnostics,
    standardized_mean_diff,
    variance_ratio,
    ks_test_balance,
    love_plot,
)


class TestStandardizedMeanDiff:
    """Test suite for standardized_mean_diff function."""

    def test_smd_identical_means(self):
        """Test SMD is zero when means are identical."""
        x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        smd = standardized_mean_diff(x1, x2)
        assert np.isclose(smd, 0.0, atol=1e-10)

    def test_smd_different_means(self):
        """Test SMD calculation with different means."""
        # Known values for verification
        x1 = np.array([0.0, 0.0, 0.0, 0.0])  # mean=0, std=0
        x2 = np.array([1.0, 1.0, 1.0, 1.0])  # mean=1, std=0
        # With zero variance, should handle gracefully (returns inf or nan)
        smd = standardized_mean_diff(x1, x2)
        # When means differ but variance is 0, result is -inf (negative because
        # mean1 < mean2)
        assert np.isinf(smd) or np.isnan(smd)

    def test_smd_known_calculation(self):
        """Test SMD with known calculation."""
        # Create arrays with known statistics
        np.random.seed(42)
        x1 = np.random.normal(0, 1, 1000)
        x2 = np.random.normal(0.5, 1, 1000)
        smd = standardized_mean_diff(x1, x2)
        # SMD should be approximately -0.5 (0 - 0.5) / pooled_std
        assert -0.7 < smd < -0.3

    def test_smd_symmetry(self):
        """Test SMD is antisymmetric when swapping arguments."""
        np.random.seed(42)
        x1 = np.random.normal(0, 1, 100)
        x2 = np.random.normal(1, 1.5, 100)
        smd1 = standardized_mean_diff(x1, x2)
        smd2 = standardized_mean_diff(x2, x1)
        assert np.isclose(smd1, -smd2, rtol=1e-10)


class TestVarianceRatio:
    """Test suite for variance_ratio function."""

    def test_variance_ratio_equal_variance(self):
        """Test variance ratio is 1 for equal variance."""
        np.random.seed(42)
        x1 = np.random.normal(0, 1, 100)
        x2 = np.random.normal(0, 1, 100)
        vr = variance_ratio(x1, x2)
        # Should be close to 1 (within sampling variation)
        assert 0.7 < vr < 1.5

    def test_variance_ratio_known_value(self):
        """Test variance ratio with known variance."""
        x1 = np.array([1, 2, 3, 4, 5])  # var = 2.5
        x2 = np.array([1, 3, 5, 7, 9])  # var = 10
        vr = variance_ratio(x1, x2)
        assert np.isclose(vr, 0.25, rtol=0.1)

    def test_variance_ratio_always_positive(self):
        """Test variance ratio is always positive."""
        np.random.seed(42)
        x1 = np.random.normal(0, 1, 50)
        x2 = np.random.normal(0, 2, 50)
        vr = variance_ratio(x1, x2)
        assert vr > 0


class TestKSTestBalance:
    """Test suite for ks_test_balance function."""

    def test_ks_identical_distributions(self):
        """Test KS test for identical distributions."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        stat, pval = ks_test_balance(x, x)
        assert stat == 0.0
        assert pval == 1.0

    def test_ks_different_distributions(self):
        """Test KS test for different distributions."""
        np.random.seed(42)
        x1 = np.random.normal(0, 1, 100)
        x2 = np.random.normal(2, 1, 100)
        stat, pval = ks_test_balance(x1, x2)
        assert stat > 0
        assert pval < 0.05  # Should reject null hypothesis

    def test_ks_returns_tuple(self):
        """Test KS test returns tuple of (statistic, pvalue)."""
        np.random.seed(42)
        x1 = np.random.normal(0, 1, 50)
        x2 = np.random.normal(0, 1, 50)
        result = ks_test_balance(x1, x2)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestLovePlot:
    """Test suite for love_plot function."""

    def test_love_plot_returns_figure(self):
        """Test love_plot returns a matplotlib figure."""
        smd_before = {"var1": 0.3, "var2": -0.2, "var3": 0.5}
        smd_after = {"var1": 0.05, "var2": -0.03, "var3": 0.08}
        fig = love_plot(smd_before, smd_after)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_love_plot_shows_threshold(self):
        """Test love_plot includes threshold line at 0.1."""
        smd_before = {"x1": 0.4, "x2": 0.2}
        smd_after = {"x1": 0.08, "x2": 0.05}
        fig = love_plot(smd_before, smd_after, threshold=0.1)
        # Should have axes with lines
        ax = fig.axes[0]
        lines = [c for c in ax.get_children() if isinstance(c, plt.Line2D)]
        assert len(lines) > 0
        plt.close(fig)

    def test_love_plot_single_variable(self):
        """Test love_plot with single variable."""
        smd_before = {"income": 0.5}
        smd_after = {"income": 0.02}
        fig = love_plot(smd_before, smd_after)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestMatchDiagnostics:
    """Test suite for MatchDiagnostics class."""

    @pytest.fixture
    def sample_match_data(self):
        """Create sample matching data and perform matching."""
        np.random.seed(42)

        # Donor data
        n_don = 100
        data_don = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n_don),
                "x2": np.random.normal(0, 2, n_don),
                "z": np.random.normal(50, 10, n_don),  # Variable to donate
            }
        )

        # Recipient data
        n_rec = 50
        data_rec = pd.DataFrame(
            {
                "x1": np.random.normal(0.2, 1, n_rec),
                "x2": np.random.normal(0.3, 2, n_rec),
            }
        )

        match_vars = ["x1", "x2"]

        # Perform matching
        result = nnd_hotdeck(
            data_rec=data_rec,
            data_don=data_don,
            match_vars=match_vars,
            dist_fun="euclidean",
        )

        return result, data_rec, data_don, match_vars

    def test_balance_table_returns_dataframe(self, sample_match_data):
        """Test balance_table returns a DataFrame."""
        result, data_rec, data_don, match_vars = sample_match_data
        diag = match_diagnostics(
            result=result,
            data_rec=data_rec,
            data_don=data_don,
            match_vars=match_vars,
        )
        balance = diag.balance_table()
        assert isinstance(balance, pd.DataFrame)
        assert (
            "smd_before" in balance.columns or "SMD Before" in balance.columns
        )

    def test_balance_table_has_all_match_vars(self, sample_match_data):
        """Test balance_table includes all matching variables."""
        result, data_rec, data_don, match_vars = sample_match_data
        diag = match_diagnostics(
            result=result,
            data_rec=data_rec,
            data_don=data_don,
            match_vars=match_vars,
        )
        balance = diag.balance_table()
        for var in match_vars:
            assert var in balance.index or var in balance["variable"].values

    def test_distance_distribution_returns_figure(self, sample_match_data):
        """Test distance_distribution returns a matplotlib figure."""
        result, data_rec, data_don, match_vars = sample_match_data
        diag = match_diagnostics(
            result=result,
            data_rec=data_rec,
            data_don=data_don,
            match_vars=match_vars,
        )
        fig = diag.distance_distribution()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_donor_usage_returns_series_or_dict(self, sample_match_data):
        """Test donor_usage returns usage counts."""
        result, data_rec, data_don, match_vars = sample_match_data
        diag = match_diagnostics(
            result=result,
            data_rec=data_rec,
            data_don=data_don,
            match_vars=match_vars,
        )
        usage = diag.donor_usage()
        assert isinstance(usage, (pd.Series, dict, np.ndarray))

    def test_summary_returns_string(self, sample_match_data):
        """Test summary returns a string."""
        result, data_rec, data_don, match_vars = sample_match_data
        diag = match_diagnostics(
            result=result,
            data_rec=data_rec,
            data_don=data_don,
            match_vars=match_vars,
        )
        summary = diag.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_to_dict_returns_dict(self, sample_match_data):
        """Test to_dict returns a dictionary."""
        result, data_rec, data_don, match_vars = sample_match_data
        diag = match_diagnostics(
            result=result,
            data_rec=data_rec,
            data_don=data_don,
            match_vars=match_vars,
        )
        result_dict = diag.to_dict()
        assert isinstance(result_dict, dict)
        assert "balance" in result_dict or "smd" in result_dict

    def test_to_html_creates_file(self, sample_match_data):
        """Test to_html creates an HTML file."""
        result, data_rec, data_don, match_vars = sample_match_data
        diag = match_diagnostics(
            result=result,
            data_rec=data_rec,
            data_don=data_don,
            match_vars=match_vars,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = os.path.join(tmpdir, "diagnostics.html")
            diag.to_html(html_path)
            assert os.path.exists(html_path)
            # Verify it's valid HTML
            with open(html_path, "r") as f:
                content = f.read()
            assert "<html" in content.lower()

    def test_overlap_plot_returns_figure(self, sample_match_data):
        """Test overlap_plot returns a matplotlib figure."""
        result, data_rec, data_don, match_vars = sample_match_data
        diag = match_diagnostics(
            result=result,
            data_rec=data_rec,
            data_don=data_don,
            match_vars=match_vars,
        )
        fig = diag.overlap_plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_cia_diagnostics_returns_dict(self, sample_match_data):
        """Test cia_diagnostics returns diagnostic info."""
        result, data_rec, data_don, match_vars = sample_match_data
        diag = match_diagnostics(
            result=result,
            data_rec=data_rec,
            data_don=data_don,
            match_vars=match_vars,
            z_vars=["z"],
        )
        cia = diag.cia_diagnostics()
        assert isinstance(cia, dict)


class TestMatchDiagnosticsFunction:
    """Test the match_diagnostics factory function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)

        n_don = 100
        data_don = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n_don),
                "x2": np.random.normal(0, 2, n_don),
                "z": np.random.normal(50, 10, n_don),
            }
        )

        n_rec = 50
        data_rec = pd.DataFrame(
            {
                "x1": np.random.normal(0.2, 1, n_rec),
                "x2": np.random.normal(0.3, 2, n_rec),
            }
        )

        match_vars = ["x1", "x2"]

        result = nnd_hotdeck(
            data_rec=data_rec,
            data_don=data_don,
            match_vars=match_vars,
            dist_fun="euclidean",
        )

        return result, data_rec, data_don, match_vars

    def test_match_diagnostics_returns_object(self, sample_data):
        """Test match_diagnostics returns MatchDiagnostics object."""
        result, data_rec, data_don, match_vars = sample_data
        diag = match_diagnostics(
            result=result,
            data_rec=data_rec,
            data_don=data_don,
            match_vars=match_vars,
        )
        assert isinstance(diag, MatchDiagnostics)

    def test_diagnostics_with_z_vars(self, sample_data):
        """Test diagnostics with z_vars specified."""
        result, data_rec, data_don, match_vars = sample_data
        diag = match_diagnostics(
            result=result,
            data_rec=data_rec,
            data_don=data_don,
            match_vars=match_vars,
            z_vars=["z"],
        )
        assert isinstance(diag, MatchDiagnostics)
