"""Plotting functions for visualizing statistical matching results.

This module provides Python implementations of R's StatMatch plotting functions:
- plot_bounds: Visualize Frechet bounds for contingency tables
- plot_cont: Compare continuous variable distributions
- plot_tab: Compare categorical variable distributions
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


def plot_bounds(
    out_fb: Dict,
) -> plt.Figure:
    """
    Plot uncertainty bounds from Frechet.bounds.cat output.

    Creates a graphical representation of the uncertainty bounds for a
    contingency table of Y vs. Z. Dotted lines indicate unconditional
    bounds (without X), solid lines indicate conditional bounds (with X).

    Parameters
    ----------
    out_fb : Dict
        Output from Frechet.bounds.cat containing:
        - 'low.u', 'up.u': Unconditional lower/upper bounds
        - 'low.cx', 'up.cx': Conditional lower/upper bounds (if X provided)
        - 'CIA': Estimates under conditional independence (if X provided)
        - 'IA': Estimates under independence (if X not provided)
        - 'uncertainty': Average width of bounds

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    # Check if X variables were used (conditional case)
    has_x = "low.cx" in out_fb and "up.cx" in out_fb

    # Get bounds data
    low_u = out_fb["low.u"]
    up_u = out_fb["up.u"]

    # Flatten the tables to get all cells
    if isinstance(low_u, pd.DataFrame):
        cells_y = low_u.index.tolist()
        cells_z = low_u.columns.tolist()
        n_cells = low_u.size
    else:
        # Handle array input
        low_u = pd.DataFrame(low_u)
        up_u = pd.DataFrame(up_u)
        cells_y = list(range(low_u.shape[0]))
        cells_z = list(range(low_u.shape[1]))
        n_cells = low_u.size

    # Create cell labels
    cell_labels = []
    for y in cells_y:
        for z in cells_z:
            cell_labels.append(f"{y}*{z}")

    # Flatten bounds
    low_u_flat = low_u.values.flatten()
    up_u_flat = up_u.values.flatten()

    if has_x:
        low_cx = out_fb["low.cx"]
        up_cx = out_fb["up.cx"]
        cia = out_fb["CIA"]

        if isinstance(low_cx, pd.DataFrame):
            low_cx_flat = low_cx.values.flatten()
            up_cx_flat = up_cx.values.flatten()
            cia_flat = cia.values.flatten()
        else:
            low_cx_flat = np.array(low_cx).flatten()
            up_cx_flat = np.array(up_cx).flatten()
            cia_flat = np.array(cia).flatten()
    else:
        # Get independence assumption estimates
        ia = out_fb.get("IA", out_fb.get("low.u"))
        if isinstance(ia, pd.DataFrame):
            ia_flat = ia.values.flatten()
        else:
            ia_flat = np.array(ia).flatten()

    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, n_cells * 1.5), 6))

    x_positions = np.arange(n_cells)
    bar_width = 0.6

    for i in range(n_cells):
        x = x_positions[i]

        # Unconditional bounds (dotted line)
        u_low = low_u_flat[i]
        u_up = up_u_flat[i]
        u_width = u_up - u_low

        # Draw unconditional bound line (dotted)
        ax.plot(
            [x - bar_width / 4, x + bar_width / 4],
            [u_low, u_low],
            "k:",
            linewidth=1.5,
        )
        ax.plot(
            [x - bar_width / 4, x + bar_width / 4],
            [u_up, u_up],
            "k:",
            linewidth=1.5,
        )
        ax.plot([x, x], [u_low, u_up], "k:", linewidth=1.5)

        # Add unconditional width annotation
        ax.annotate(
            f"({u_width:.3f})",
            xy=(x, u_low - 0.02),
            ha="center",
            va="top",
            fontsize=8,
        )

        if has_x:
            # Conditional bounds (solid line)
            cx_low = low_cx_flat[i]
            cx_up = up_cx_flat[i]
            cx_width = cx_up - cx_low

            # Draw conditional bound line (solid)
            ax.plot(
                [x - bar_width / 4, x + bar_width / 4],
                [cx_low, cx_low],
                "k-",
                linewidth=2,
            )
            ax.plot(
                [x - bar_width / 4, x + bar_width / 4],
                [cx_up, cx_up],
                "k-",
                linewidth=2,
            )
            ax.plot([x, x], [cx_low, cx_up], "k-", linewidth=2)

            # Add conditional width annotation
            ax.annotate(
                f"{cx_width:.3f}",
                xy=(x, cx_low - 0.01),
                ha="center",
                va="top",
                fontsize=8,
            )

            # Add CIA estimate at top
            ax.annotate(
                f"{cia_flat[i]:.3f}",
                xy=(x, cx_up + 0.01),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        else:
            # Add IA estimate at top
            ax.annotate(
                f"{ia_flat[i]:.3f}",
                xy=(x, u_up + 0.01),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Set axis labels and title
    ax.set_xticks(x_positions)
    ax.set_xticklabels(cell_labels, rotation=45, ha="right")
    ax.set_xlabel("Y * Z cells")
    ax.set_ylabel("Relative Frequency")

    if has_x:
        ax.set_title(
            "Frechet Bounds (dotted: unconditional, solid: conditional on X)"
        )
        # Add legend
        legend_elements = [
            Line2D([0], [0], linestyle=":", color="k", label="Unconditional"),
            Line2D([0], [0], linestyle="-", color="k", label="Conditional|X"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")
    else:
        ax.set_title("Frechet Bounds (unconditional)")

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_cont(
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
    xlab_a: str,
    xlab_b: Optional[str] = None,
    w_a: Optional[str] = None,
    w_b: Optional[str] = None,
    type: str = "density",
    ref: bool = False,
) -> plt.Figure:
    """
    Compare continuous variable distributions between two datasets.

    Parameters
    ----------
    data_a : pd.DataFrame
        First dataset (typically recipient data).
    data_b : pd.DataFrame
        Second dataset (typically donor data).
    xlab_a : str
        Column name of the variable to compare in data_a.
    xlab_b : Optional[str], default=None
        Column name of the variable to compare in data_b.
        If None, uses xlab_a.
    w_a : Optional[str], default=None
        Column name for sample weights in data_a.
    w_b : Optional[str], default=None
        Column name for sample weights in data_b.
    type : str, default="density"
        Type of plot. Options:
        - "density": Kernel density estimate
        - "hist": Histogram
        - "ecdf": Empirical cumulative distribution function
        - "qqplot": Quantile-quantile plot
        - "qqshift": Quantile shift plot
    ref : bool, default=False
        If True, use data_b as reference for histogram binning.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.

    Raises
    ------
    ValueError
        If type is not one of the valid options.
    """
    valid_types = ["density", "hist", "ecdf", "qqplot", "qqshift"]
    if type not in valid_types:
        raise ValueError(f"type must be one of {valid_types}, got '{type}'")

    if xlab_b is None:
        xlab_b = xlab_a

    # Extract data
    x_a = data_a[xlab_a].dropna().values
    x_b = data_b[xlab_b].dropna().values

    # Get weights if specified
    if w_a is not None:
        weights_a = data_a.loc[data_a[xlab_a].notna(), w_a].values
    else:
        weights_a = None

    if w_b is not None:
        weights_b = data_b.loc[data_b[xlab_b].notna(), w_b].values
    else:
        weights_b = None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    if type == "density":
        _plot_density(ax, x_a, x_b, weights_a, weights_b)
    elif type == "hist":
        _plot_histogram(ax, x_a, x_b, weights_a, weights_b, ref)
    elif type == "ecdf":
        _plot_ecdf(ax, x_a, x_b, weights_a, weights_b)
    elif type == "qqplot":
        _plot_qqplot(ax, x_a, x_b, weights_a, weights_b)
    elif type == "qqshift":
        _plot_qqshift(ax, x_a, x_b, weights_a, weights_b)

    # Set labels
    ax.set_xlabel(xlab_a if xlab_a == xlab_b else f"{xlab_a} / {xlab_b}")
    ax.legend(["Sample A", "Sample B"])

    plt.tight_layout()
    return fig


def _plot_density(
    ax: plt.Axes,
    x_a: np.ndarray,
    x_b: np.ndarray,
    weights_a: Optional[np.ndarray],
    weights_b: Optional[np.ndarray],
) -> None:
    """Plot kernel density estimates."""
    from scipy import stats

    # Determine x range
    x_min = min(x_a.min(), x_b.min())
    x_max = max(x_a.max(), x_b.max())
    x_range = np.linspace(x_min, x_max, 200)

    # Compute KDE for A
    if weights_a is not None:
        kde_a = stats.gaussian_kde(x_a, weights=weights_a)
    else:
        kde_a = stats.gaussian_kde(x_a)
    y_a = kde_a(x_range)

    # Compute KDE for B
    if weights_b is not None:
        kde_b = stats.gaussian_kde(x_b, weights=weights_b)
    else:
        kde_b = stats.gaussian_kde(x_b)
    y_b = kde_b(x_range)

    ax.plot(x_range, y_a, label="Sample A")
    ax.plot(x_range, y_b, label="Sample B")
    ax.set_ylabel("Density")
    ax.set_title("Kernel Density Comparison")


def _plot_histogram(
    ax: plt.Axes,
    x_a: np.ndarray,
    x_b: np.ndarray,
    weights_a: Optional[np.ndarray],
    weights_b: Optional[np.ndarray],
    ref: bool,
) -> None:
    """Plot histograms."""
    # Determine bins using Freedman-Diaconis rule
    combined = np.concatenate([x_a, x_b])

    if ref:
        # Use B for binning reference
        iqr = np.percentile(x_b, 75) - np.percentile(x_b, 25)
        bin_width = 2 * iqr / (len(x_b) ** (1 / 3))
    else:
        # Use combined data for binning
        iqr = np.percentile(combined, 75) - np.percentile(combined, 25)
        bin_width = 2 * iqr / (len(combined) ** (1 / 3))

    if bin_width > 0:
        n_bins = int(np.ceil((combined.max() - combined.min()) / bin_width))
        n_bins = max(10, min(n_bins, 100))  # Limit bins
    else:
        n_bins = 30

    bins = np.linspace(combined.min(), combined.max(), n_bins + 1)

    ax.hist(
        x_a,
        bins=bins,
        weights=weights_a,
        density=True,
        alpha=0.5,
        label="Sample A",
    )
    ax.hist(
        x_b,
        bins=bins,
        weights=weights_b,
        density=True,
        alpha=0.5,
        label="Sample B",
    )
    ax.set_ylabel("Density")
    ax.set_title("Histogram Comparison")


def _plot_ecdf(
    ax: plt.Axes,
    x_a: np.ndarray,
    x_b: np.ndarray,
    weights_a: Optional[np.ndarray],
    weights_b: Optional[np.ndarray],
) -> None:
    """Plot empirical CDFs."""

    def weighted_ecdf(x, weights=None):
        """Compute weighted ECDF."""
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]

        if weights is not None:
            sorted_weights = weights[sorted_indices]
            cumsum = np.cumsum(sorted_weights)
            ecdf_y = cumsum / cumsum[-1]
        else:
            n = len(x)
            ecdf_y = np.arange(1, n + 1) / n

        return sorted_x, ecdf_y

    x_a_sorted, ecdf_a = weighted_ecdf(x_a, weights_a)
    x_b_sorted, ecdf_b = weighted_ecdf(x_b, weights_b)

    ax.step(x_a_sorted, ecdf_a, where="post", label="Sample A")
    ax.step(x_b_sorted, ecdf_b, where="post", label="Sample B")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Empirical CDF Comparison")


def _plot_qqplot(
    ax: plt.Axes,
    x_a: np.ndarray,
    x_b: np.ndarray,
    weights_a: Optional[np.ndarray],
    weights_b: Optional[np.ndarray],
) -> None:
    """Plot Q-Q plot comparing distributions A and B."""

    def weighted_quantiles(x, weights, probs):
        """Compute weighted quantiles."""
        if weights is None:
            return np.percentile(x, probs * 100)

        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = weights[sorted_indices]
        cum_weights = np.cumsum(sorted_w)
        cum_weights /= cum_weights[-1]

        return np.interp(probs, cum_weights, sorted_x)

    # Generate probability points
    probs = np.linspace(0.01, 0.99, 99)

    q_a = weighted_quantiles(x_a, weights_a, probs)
    q_b = weighted_quantiles(x_b, weights_b, probs)

    ax.scatter(q_a, q_b, alpha=0.6, s=20)

    # Add diagonal reference line
    min_val = min(q_a.min(), q_b.min())
    max_val = max(q_a.max(), q_b.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="y=x")

    ax.set_xlabel("Quantiles of Sample A")
    ax.set_ylabel("Quantiles of Sample B")
    ax.set_title("Q-Q Plot")


def _plot_qqshift(
    ax: plt.Axes,
    x_a: np.ndarray,
    x_b: np.ndarray,
    weights_a: Optional[np.ndarray],
    weights_b: Optional[np.ndarray],
) -> None:
    """Plot Q-Q shift (difference in quantiles)."""

    def weighted_quantiles(x, weights, probs):
        """Compute weighted quantiles."""
        if weights is None:
            return np.percentile(x, probs * 100)

        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = weights[sorted_indices]
        cum_weights = np.cumsum(sorted_w)
        cum_weights /= cum_weights[-1]

        return np.interp(probs, cum_weights, sorted_x)

    # Generate probability points
    probs = np.linspace(0.01, 0.99, 99)

    q_a = weighted_quantiles(x_a, weights_a, probs)
    q_b = weighted_quantiles(x_b, weights_b, probs)

    # Compute shift (difference)
    shift = q_b - q_a

    ax.plot(probs * 100, shift, "b-", linewidth=2)
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Quantile Shift (B - A)")
    ax.set_title("Q-Q Shift Plot")


def plot_tab(
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
    xlab_a: Union[str, List[str]],
    xlab_b: Optional[Union[str, List[str]]] = None,
    w_a: Optional[str] = None,
    w_b: Optional[str] = None,
) -> plt.Figure:
    """
    Compare categorical variable distributions between two datasets.

    Creates a grouped bar chart comparing the relative frequencies of
    categorical variables, along with the total variation distance (TVD).

    Parameters
    ----------
    data_a : pd.DataFrame
        First dataset (typically recipient data).
    data_b : pd.DataFrame
        Second dataset (typically donor data).
    xlab_a : Union[str, List[str]]
        Column name(s) of the variable(s) to compare in data_a.
    xlab_b : Optional[Union[str, List[str]]], default=None
        Column name(s) of the variable(s) to compare in data_b.
        If None, uses xlab_a.
    w_a : Optional[str], default=None
        Column name for sample weights in data_a.
    w_b : Optional[str], default=None
        Column name for sample weights in data_b.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.

    Raises
    ------
    ValueError
        If xlab_a and xlab_b have different lengths.
    """
    # Convert to list if string
    if isinstance(xlab_a, str):
        xlab_a = [xlab_a]
    if xlab_b is None:
        xlab_b = xlab_a
    elif isinstance(xlab_b, str):
        xlab_b = [xlab_b]

    # Validate matching variable counts
    if len(xlab_a) != len(xlab_b):
        raise ValueError(
            f"Different number of variables: xlab_a has {len(xlab_a)}, "
            f"xlab_b has {len(xlab_b)}"
        )

    # Compute cross-tabulation for A
    if w_a is not None:
        # Weighted frequency table
        tab_a = data_a.groupby(xlab_a)[w_a].sum()
    else:
        # Unweighted frequency table
        tab_a = data_a.groupby(xlab_a).size()

    # Normalize to proportions
    tab_a = tab_a / tab_a.sum()

    # Compute cross-tabulation for B
    if w_b is not None:
        tab_b = data_b.groupby(xlab_b)[w_b].sum()
    else:
        tab_b = data_b.groupby(xlab_b).size()

    # Normalize to proportions
    tab_b = tab_b / tab_b.sum()

    # Convert to DataFrames for alignment
    df_a = pd.DataFrame({"Freq": tab_a, "Sample": "A"})
    df_b = pd.DataFrame({"Freq": tab_b, "Sample": "B"})

    # Create consistent category labels
    if len(xlab_a) > 1:
        # Multiple variables - create combined labels
        df_a["x"] = [
            (
                "*".join(str(v) for v in idx)
                if isinstance(idx, tuple)
                else str(idx)
            )
            for idx in df_a.index
        ]
        df_b["x"] = [
            (
                "*".join(str(v) for v in idx)
                if isinstance(idx, tuple)
                else str(idx)
            )
            for idx in df_b.index
        ]
    else:
        df_a["x"] = df_a.index.astype(str)
        df_b["x"] = df_b.index.astype(str)

    # Get all unique categories
    all_cats = sorted(set(df_a["x"].tolist() + df_b["x"].tolist()))

    # Fill missing categories with 0
    df_a = df_a.set_index("x").reindex(all_cats).fillna(0).reset_index()
    df_b = df_b.set_index("x").reindex(all_cats).fillna(0).reset_index()
    df_a["Sample"] = "A"
    df_b["Sample"] = "B"

    # Compute Total Variation Distance
    tvd = 0.5 * np.sum(np.abs(df_a["Freq"].values - df_b["Freq"].values))
    tvd_label = f"tvd = {tvd * 100:.2f}%"

    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, len(all_cats) * 0.8), 6))

    x = np.arange(len(all_cats))
    width = 0.35

    bars_a = ax.bar(
        x - width / 2,
        df_a["Freq"],
        width,
        label="A",
        alpha=0.8,
    )
    bars_b = ax.bar(
        x + width / 2,
        df_b["Freq"],
        width,
        label="B",
        alpha=0.8,
    )

    # Set labels
    var_label = "*".join(xlab_a) if xlab_a == xlab_b else f"{xlab_a}/{xlab_b}"
    ax.set_xlabel(f"{var_label}, {tvd_label}")
    ax.set_ylabel("Rel. freq.")
    ax.set_title("Distribution Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(all_cats, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    return fig
