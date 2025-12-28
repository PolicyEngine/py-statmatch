"""Tests for plotting functions."""

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from statmatch.plotting import plot_bounds, plot_cont, plot_tab


class TestPlotBounds:
    """Test suite for plot_bounds function."""

    @pytest.fixture
    def frechet_bounds_output_with_x(self):
        """Create sample output from Frechet.bounds.cat with X variables."""
        # Simulates output when X variables are provided
        return {
            "low.u": pd.DataFrame(
                [[0.05, 0.03], [0.02, 0.08]],
                index=["Y1", "Y2"],
                columns=["Z1", "Z2"],
            ),
            "up.u": pd.DataFrame(
                [[0.25, 0.15], [0.18, 0.30]],
                index=["Y1", "Y2"],
                columns=["Z1", "Z2"],
            ),
            "low.cx": pd.DataFrame(
                [[0.08, 0.05], [0.04, 0.12]],
                index=["Y1", "Y2"],
                columns=["Z1", "Z2"],
            ),
            "up.cx": pd.DataFrame(
                [[0.20, 0.12], [0.15, 0.25]],
                index=["Y1", "Y2"],
                columns=["Z1", "Z2"],
            ),
            "CIA": pd.DataFrame(
                [[0.15, 0.08], [0.10, 0.18]],
                index=["Y1", "Y2"],
                columns=["Z1", "Z2"],
            ),
            "uncertainty": {"av.u": 0.18, "av.cx": 0.10},
        }

    @pytest.fixture
    def frechet_bounds_output_no_x(self):
        """Create sample output from Frechet.bounds.cat without X variables."""
        # Simulates output when X variables are NOT provided
        return {
            "low.u": pd.DataFrame(
                [[0.05, 0.03], [0.02, 0.08]],
                index=["Y1", "Y2"],
                columns=["Z1", "Z2"],
            ),
            "up.u": pd.DataFrame(
                [[0.25, 0.15], [0.18, 0.30]],
                index=["Y1", "Y2"],
                columns=["Z1", "Z2"],
            ),
            "IA": pd.DataFrame(
                [[0.12, 0.08], [0.08, 0.15]],
                index=["Y1", "Y2"],
                columns=["Z1", "Z2"],
            ),
            "uncertainty": 0.18,
        }

    def test_plot_bounds_returns_figure(self, frechet_bounds_output_with_x):
        """Test that plot_bounds returns a matplotlib figure."""
        fig = plot_bounds(frechet_bounds_output_with_x)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_bounds_with_x_variables(self, frechet_bounds_output_with_x):
        """Test plot_bounds with X variables (shows both bounds)."""
        fig = plot_bounds(frechet_bounds_output_with_x)

        # Should have axes
        assert len(fig.axes) > 0

        # The plot should show data (check that there are artists)
        ax = fig.axes[0]
        assert len(ax.get_children()) > 0

        plt.close(fig)

    def test_plot_bounds_without_x_variables(self, frechet_bounds_output_no_x):
        """Test plot_bounds without X variables (shows only unconditional)."""
        fig = plot_bounds(frechet_bounds_output_no_x)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0

        plt.close(fig)

    def test_plot_bounds_shows_uncertainty_range(
        self, frechet_bounds_output_with_x
    ):
        """Test that the plot displays uncertainty ranges correctly."""
        fig = plot_bounds(frechet_bounds_output_with_x)

        # Check that the figure contains line elements (for bounds)
        ax = fig.axes[0]
        lines = [
            child
            for child in ax.get_children()
            if isinstance(child, plt.Line2D)
        ]
        # Should have some lines (dotted for unconditional, solid for cond.)
        assert len(lines) > 0

        plt.close(fig)


class TestPlotCont:
    """Test suite for plot_cont function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample continuous data for two datasets."""
        np.random.seed(42)
        data_a = pd.DataFrame(
            {
                "income": np.random.normal(50000, 15000, 100),
                "age": np.random.normal(40, 12, 100),
                "weight": np.random.uniform(0.5, 1.5, 100),
            }
        )
        data_b = pd.DataFrame(
            {
                "income": np.random.normal(55000, 18000, 80),
                "age": np.random.normal(42, 10, 80),
                "weight": np.random.uniform(0.8, 1.2, 80),
            }
        )
        return data_a, data_b

    def test_plot_cont_returns_figure(self, sample_data):
        """Test that plot_cont returns a matplotlib figure."""
        data_a, data_b = sample_data
        fig = plot_cont(data_a, data_b, xlab_a="income")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_cont_density_type(self, sample_data):
        """Test density plot type."""
        data_a, data_b = sample_data
        fig = plot_cont(data_a, data_b, xlab_a="income", type="density")
        assert isinstance(fig, plt.Figure)

        ax = fig.axes[0]
        # Density plots should have lines
        lines = [c for c in ax.get_children() if isinstance(c, plt.Line2D)]
        assert len(lines) >= 2  # At least two lines (A and B)

        plt.close(fig)

    def test_plot_cont_hist_type(self, sample_data):
        """Test histogram plot type."""
        data_a, data_b = sample_data
        fig = plot_cont(data_a, data_b, xlab_a="income", type="hist")
        assert isinstance(fig, plt.Figure)

        ax = fig.axes[0]
        # Histogram should have patches (bars)
        patches = ax.patches
        assert len(patches) > 0

        plt.close(fig)

    def test_plot_cont_ecdf_type(self, sample_data):
        """Test empirical CDF plot type."""
        data_a, data_b = sample_data
        fig = plot_cont(data_a, data_b, xlab_a="income", type="ecdf")
        assert isinstance(fig, plt.Figure)

        ax = fig.axes[0]
        lines = [c for c in ax.get_children() if isinstance(c, plt.Line2D)]
        assert len(lines) >= 2

        plt.close(fig)

    def test_plot_cont_qqplot_type(self, sample_data):
        """Test QQ plot type."""
        data_a, data_b = sample_data
        fig = plot_cont(data_a, data_b, xlab_a="income", type="qqplot")
        assert isinstance(fig, plt.Figure)

        ax = fig.axes[0]
        # QQ plot should have scatter points or line
        assert len(ax.get_children()) > 0

        plt.close(fig)

    def test_plot_cont_qqshift_type(self, sample_data):
        """Test QQ shift plot type."""
        data_a, data_b = sample_data
        fig = plot_cont(data_a, data_b, xlab_a="income", type="qqshift")
        assert isinstance(fig, plt.Figure)

        ax = fig.axes[0]
        assert len(ax.get_children()) > 0

        plt.close(fig)

    def test_plot_cont_different_variable_names(self, sample_data):
        """Test with different variable names in A and B."""
        data_a, data_b = sample_data
        data_b = data_b.rename(columns={"income": "earnings"})
        fig = plot_cont(
            data_a, data_b, xlab_a="income", xlab_b="earnings", type="density"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_cont_with_weights(self, sample_data):
        """Test with sample weights."""
        data_a, data_b = sample_data
        fig = plot_cont(
            data_a,
            data_b,
            xlab_a="income",
            w_a="weight",
            w_b="weight",
            type="density",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_cont_ref_option(self, sample_data):
        """Test with ref=True (use B as reference for binning)."""
        data_a, data_b = sample_data
        fig = plot_cont(data_a, data_b, xlab_a="income", type="hist", ref=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_cont_invalid_type_raises(self, sample_data):
        """Test that invalid type raises ValueError."""
        data_a, data_b = sample_data
        with pytest.raises(ValueError, match="type"):
            plot_cont(data_a, data_b, xlab_a="income", type="invalid")


class TestPlotTab:
    """Test suite for plot_tab function."""

    @pytest.fixture
    def sample_categorical_data(self):
        """Create sample categorical data for two datasets."""
        np.random.seed(42)
        data_a = pd.DataFrame(
            {
                "education": np.random.choice(
                    ["HS", "College", "Graduate"], 100, p=[0.4, 0.4, 0.2]
                ),
                "region": np.random.choice(
                    ["North", "South", "East", "West"], 100
                ),
                "weight": np.random.uniform(0.5, 1.5, 100),
            }
        )
        data_b = pd.DataFrame(
            {
                "education": np.random.choice(
                    ["HS", "College", "Graduate"], 80, p=[0.3, 0.45, 0.25]
                ),
                "region": np.random.choice(
                    ["North", "South", "East", "West"], 80
                ),
                "weight": np.random.uniform(0.8, 1.2, 80),
            }
        )
        return data_a, data_b

    def test_plot_tab_returns_figure(self, sample_categorical_data):
        """Test that plot_tab returns a matplotlib figure."""
        data_a, data_b = sample_categorical_data
        fig = plot_tab(data_a, data_b, xlab_a="education")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_tab_single_variable(self, sample_categorical_data):
        """Test with single categorical variable."""
        data_a, data_b = sample_categorical_data
        fig = plot_tab(data_a, data_b, xlab_a="education")

        ax = fig.axes[0]
        # Should have bars
        assert len(ax.patches) > 0

        plt.close(fig)

    def test_plot_tab_multiple_variables(self, sample_categorical_data):
        """Test with multiple categorical variables."""
        data_a, data_b = sample_categorical_data
        fig = plot_tab(data_a, data_b, xlab_a=["education", "region"])

        ax = fig.axes[0]
        assert len(ax.patches) > 0

        plt.close(fig)

    def test_plot_tab_different_variable_names(self, sample_categorical_data):
        """Test with different variable names in A and B."""
        data_a, data_b = sample_categorical_data
        data_b = data_b.rename(columns={"education": "edu_level"})
        fig = plot_tab(data_a, data_b, xlab_a="education", xlab_b="edu_level")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_tab_with_weights(self, sample_categorical_data):
        """Test with sample weights."""
        data_a, data_b = sample_categorical_data
        fig = plot_tab(
            data_a, data_b, xlab_a="education", w_a="weight", w_b="weight"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_tab_shows_tvd(self, sample_categorical_data):
        """Test that total variation distance is displayed."""
        data_a, data_b = sample_categorical_data
        fig = plot_tab(data_a, data_b, xlab_a="education")

        ax = fig.axes[0]
        # Check xlabel contains TVD info
        xlabel = ax.get_xlabel()
        assert "tvd" in xlabel.lower() or "%" in xlabel

        plt.close(fig)

    def test_plot_tab_bar_heights_sum_to_one(self, sample_categorical_data):
        """Test that bar heights represent relative frequencies."""
        data_a, data_b = sample_categorical_data
        fig = plot_tab(data_a, data_b, xlab_a="education")

        ax = fig.axes[0]
        # Get bar heights - matplotlib groups bars by sample
        # First half are sample A, second half are sample B
        all_heights = [patch.get_height() for patch in ax.patches]
        n_cats = len(all_heights) // 2
        heights_a = all_heights[:n_cats]
        heights_b = all_heights[n_cats:]

        # Each should sum to approximately 1.0 (relative frequencies)
        # Allow tolerance for floating point
        assert abs(sum(heights_a) - 1.0) < 0.01
        assert abs(sum(heights_b) - 1.0) < 0.01

        plt.close(fig)

    def test_plot_tab_mismatched_variable_count_raises(
        self, sample_categorical_data
    ):
        """Test that mismatched variable count raises error."""
        data_a, data_b = sample_categorical_data
        with pytest.raises(ValueError):
            plot_tab(
                data_a,
                data_b,
                xlab_a=["education", "region"],
                xlab_b=["education"],
            )
