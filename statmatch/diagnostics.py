"""Diagnostics module for evaluating statistical matching quality.

This module provides comprehensive diagnostics for assessing the quality
of statistical matching results, including balance diagnostics, overlap
visualization, and tests for the conditional independence assumption (CIA).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def standardized_mean_diff(
    x1: np.ndarray,
    x2: np.ndarray,
) -> float:
    """
    Compute the standardized mean difference (SMD) between two arrays.

    SMD = (mean_x1 - mean_x2) / pooled_std

    A common rule of thumb is that |SMD| < 0.1 indicates good balance.

    Parameters
    ----------
    x1 : np.ndarray
        First array (e.g., recipient group).
    x2 : np.ndarray
        Second array (e.g., matched donor group).

    Returns
    -------
    float
        The standardized mean difference.
    """
    x1 = np.asarray(x1).flatten()
    x2 = np.asarray(x2).flatten()

    mean1 = np.mean(x1)
    mean2 = np.mean(x2)

    var1 = np.var(x1, ddof=1)
    var2 = np.var(x2, ddof=1)

    # Pooled standard deviation
    pooled_var = (var1 + var2) / 2
    pooled_std = np.sqrt(pooled_var)

    if pooled_std == 0 or np.isnan(pooled_std):
        return np.nan if mean1 == mean2 else np.inf * np.sign(mean1 - mean2)

    return (mean1 - mean2) / pooled_std


def variance_ratio(
    x1: np.ndarray,
    x2: np.ndarray,
) -> float:
    """
    Compute the variance ratio between two arrays.

    A variance ratio close to 1 indicates similar spread in both groups.

    Parameters
    ----------
    x1 : np.ndarray
        First array.
    x2 : np.ndarray
        Second array.

    Returns
    -------
    float
        The variance ratio (var_x1 / var_x2).
    """
    x1 = np.asarray(x1).flatten()
    x2 = np.asarray(x2).flatten()

    var1 = np.var(x1, ddof=1)
    var2 = np.var(x2, ddof=1)

    if var2 == 0:
        return np.inf if var1 > 0 else np.nan

    return var1 / var2


def ks_test_balance(
    x1: np.ndarray,
    x2: np.ndarray,
) -> Tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test for distribution similarity.

    Parameters
    ----------
    x1 : np.ndarray
        First array.
    x2 : np.ndarray
        Second array.

    Returns
    -------
    Tuple[float, float]
        The KS statistic and p-value.
    """
    x1 = np.asarray(x1).flatten()
    x2 = np.asarray(x2).flatten()

    result = stats.ks_2samp(x1, x2)
    return result.statistic, result.pvalue


def love_plot(
    smd_before: Dict[str, float],
    smd_after: Dict[str, float],
    threshold: float = 0.1,
    title: str = "Love Plot: Standardized Mean Differences",
) -> plt.Figure:
    """
    Create a Love plot showing SMD before and after matching.

    A Love plot is a common visualization in matching literature that
    displays standardized mean differences for covariates before and
    after matching.

    Parameters
    ----------
    smd_before : Dict[str, float]
        Dictionary mapping variable names to SMD before matching.
    smd_after : Dict[str, float]
        Dictionary mapping variable names to SMD after matching.
    threshold : float, default=0.1
        Threshold for acceptable SMD (drawn as vertical line).
    title : str, default="Love Plot: Standardized Mean Differences"
        Title for the plot.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    variables = list(smd_before.keys())
    n_vars = len(variables)

    fig, ax = plt.subplots(figsize=(10, max(4, n_vars * 0.5)))

    y_positions = np.arange(n_vars)

    # Plot before and after SMDs
    before_values = [abs(smd_before[v]) for v in variables]
    after_values = [abs(smd_after[v]) for v in variables]

    ax.scatter(
        before_values,
        y_positions,
        marker="o",
        s=80,
        color="red",
        label="Before Matching",
        alpha=0.7,
    )
    ax.scatter(
        after_values,
        y_positions,
        marker="s",
        s=80,
        color="blue",
        label="After Matching",
        alpha=0.7,
    )

    # Draw lines connecting before and after
    for i, var in enumerate(variables):
        ax.plot(
            [abs(smd_before[var]), abs(smd_after[var])],
            [i, i],
            color="gray",
            linestyle="-",
            alpha=0.5,
            linewidth=1,
        )

    # Draw threshold lines
    ax.axvline(
        x=threshold,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Threshold ({threshold})",
    )
    ax.axvline(x=-threshold, color="black", linestyle="--", linewidth=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(variables)
    ax.set_xlabel("|Standardized Mean Difference|")
    ax.set_ylabel("Variable")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(left=0)

    plt.tight_layout()
    return fig


class MatchDiagnostics:
    """
    Comprehensive match quality diagnostics for statistical matching.

    This class provides methods for assessing the quality of a statistical
    matching result, including balance diagnostics, overlap visualization,
    distance distribution analysis, and tests for the conditional
    independence assumption.

    Parameters
    ----------
    result : Dict
        Matching result from nnd_hotdeck containing:
        - 'noad.index': Array of donor indices
        - 'dist.rd': Array of distances
        - 'mtc.ids': DataFrame with matched IDs
    data_rec : pd.DataFrame
        Recipient dataset.
    data_don : pd.DataFrame
        Donor dataset.
    match_vars : List[str]
        List of matching variable names.
    z_vars : Optional[List[str]], default=None
        List of donated variable names (for CIA diagnostics).
    """

    def __init__(
        self,
        result: Dict,
        data_rec: pd.DataFrame,
        data_don: pd.DataFrame,
        match_vars: List[str],
        z_vars: Optional[List[str]] = None,
    ):
        self.result = result
        self.data_rec = data_rec
        self.data_don = data_don
        self.match_vars = match_vars
        self.z_vars = z_vars or []

        # Extract matched donor data
        self.donor_indices = result["noad.index"]
        self.distances = result["dist.rd"]
        self.matched_donors = data_don.iloc[self.donor_indices]

    def balance_table(self) -> pd.DataFrame:
        """
        Create a balance table with standardized mean differences.

        Returns a DataFrame with SMD and variance ratio for each matching
        variable, comparing recipients to the full donor pool (before)
        and recipients to matched donors (after).

        Returns
        -------
        pd.DataFrame
            Balance table with columns for variable, mean recipient,
            mean donor (before/after), SMD (before/after), and
            variance ratio (before/after).
        """
        rows = []
        for var in self.match_vars:
            rec_vals = self.data_rec[var].values
            don_vals = self.data_don[var].values
            matched_vals = self.matched_donors[var].values

            smd_before = standardized_mean_diff(rec_vals, don_vals)
            smd_after = standardized_mean_diff(rec_vals, matched_vals)
            vr_before = variance_ratio(rec_vals, don_vals)
            vr_after = variance_ratio(rec_vals, matched_vals)

            rows.append(
                {
                    "variable": var,
                    "mean_rec": np.mean(rec_vals),
                    "mean_don_before": np.mean(don_vals),
                    "mean_don_after": np.mean(matched_vals),
                    "smd_before": smd_before,
                    "smd_after": smd_after,
                    "var_ratio_before": vr_before,
                    "var_ratio_after": vr_after,
                }
            )

        df = pd.DataFrame(rows)
        df = df.set_index("variable")
        return df

    def overlap_plot(
        self,
        var: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create an overlap plot showing propensity/distance distribution.

        If var is specified, shows the distribution of that variable
        for recipients vs matched donors. Otherwise, shows the
        distribution of matching distances.

        Parameters
        ----------
        var : Optional[str], default=None
            Variable to plot. If None, plots matching distances.

        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if var is None:
            # Plot distance distribution overlap using all match vars
            # Create a combined distance measure
            rec_vals = self.data_rec[self.match_vars].values
            don_vals = self.matched_donors[self.match_vars].values

            # Use first principal component if multiple vars
            if len(self.match_vars) > 1:
                from sklearn.decomposition import PCA

                combined = np.vstack([rec_vals, don_vals])
                pca = PCA(n_components=1)
                pca.fit(combined)
                rec_proj = pca.transform(rec_vals).flatten()
                don_proj = pca.transform(don_vals).flatten()

                ax.hist(
                    rec_proj,
                    bins=30,
                    alpha=0.5,
                    density=True,
                    label="Recipients",
                )
                ax.hist(
                    don_proj,
                    bins=30,
                    alpha=0.5,
                    density=True,
                    label="Matched Donors",
                )
                ax.set_xlabel("First Principal Component")
            else:
                var = self.match_vars[0]
                rec_vals = self.data_rec[var].values
                don_vals = self.matched_donors[var].values
                ax.hist(
                    rec_vals,
                    bins=30,
                    alpha=0.5,
                    density=True,
                    label="Recipients",
                )
                ax.hist(
                    don_vals,
                    bins=30,
                    alpha=0.5,
                    density=True,
                    label="Matched Donors",
                )
                ax.set_xlabel(var)
        else:
            rec_vals = self.data_rec[var].values
            don_vals = self.matched_donors[var].values
            ax.hist(
                rec_vals,
                bins=30,
                alpha=0.5,
                density=True,
                label="Recipients",
            )
            ax.hist(
                don_vals,
                bins=30,
                alpha=0.5,
                density=True,
                label="Matched Donors",
            )
            ax.set_xlabel(var)

        ax.set_ylabel("Density")
        ax.set_title("Overlap of Recipient and Matched Donor Distributions")
        ax.legend()

        plt.tight_layout()
        return fig

    def distance_distribution(self) -> plt.Figure:
        """
        Create a plot showing the distribution of match distances.

        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(self.distances, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(
            np.mean(self.distances),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(self.distances):.4f}",
        )
        ax.axvline(
            np.median(self.distances),
            color="orange",
            linestyle="-.",
            label=f"Median: {np.median(self.distances):.4f}",
        )

        ax.set_xlabel("Match Distance")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Match Distances")
        ax.legend()

        plt.tight_layout()
        return fig

    def donor_usage(self) -> pd.Series:
        """
        Calculate how often each donor was used.

        Returns
        -------
        pd.Series
            Series with donor indices as index and usage counts as values.
        """
        usage_counts = pd.Series(self.donor_indices).value_counts()
        usage_counts = usage_counts.sort_index()
        return usage_counts

    def cia_diagnostics(self) -> Dict:
        """
        Provide diagnostics for the conditional independence assumption.

        The CIA assumes that Y (recipient-only vars) and Z (donor-only vars)
        are independent conditional on X (matching vars). This method
        provides tests and measures to assess this assumption.

        Returns
        -------
        Dict
            Dictionary with CIA diagnostic results including:
            - 'z_var_stats': Statistics for donated variables
            - 'correlation_with_distance': Correlation between z vars
              and match distance
            - 'match_quality_by_z': Quality metrics stratified by z
        """
        diagnostics = {
            "z_var_stats": {},
            "correlation_with_distance": {},
            "match_quality_summary": {},
        }

        if not self.z_vars:
            diagnostics["warning"] = (
                "No z_vars specified. Cannot compute full CIA diagnostics."
            )
            return diagnostics

        for z_var in self.z_vars:
            if z_var not in self.data_don.columns:
                continue

            z_vals = self.matched_donors[z_var].values

            # Basic statistics of donated variable
            diagnostics["z_var_stats"][z_var] = {
                "mean": float(np.mean(z_vals)),
                "std": float(np.std(z_vals)),
                "min": float(np.min(z_vals)),
                "max": float(np.max(z_vals)),
            }

            # Correlation between z and match distance
            # Higher correlation might indicate CIA violation
            corr, pval = stats.pearsonr(z_vals, self.distances)
            diagnostics["correlation_with_distance"][z_var] = {
                "correlation": float(corr),
                "p_value": float(pval),
            }

        # Overall match quality summary
        diagnostics["match_quality_summary"] = {
            "mean_distance": float(np.mean(self.distances)),
            "median_distance": float(np.median(self.distances)),
            "max_distance": float(np.max(self.distances)),
            "unique_donors_used": int(len(np.unique(self.donor_indices))),
            "total_donors": int(len(self.data_don)),
            "donor_utilization": float(
                len(np.unique(self.donor_indices)) / len(self.data_don)
            ),
        }

        return diagnostics

    def summary(self) -> str:
        """
        Generate a text summary of all diagnostics.

        Returns
        -------
        str
            Multi-line string summarizing match quality diagnostics.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("STATISTICAL MATCHING DIAGNOSTICS SUMMARY")
        lines.append("=" * 60)
        lines.append("")

        # Basic info
        lines.append("MATCHING INFO")
        lines.append("-" * 40)
        lines.append(f"Number of recipients: {len(self.data_rec)}")
        lines.append(f"Number of donors: {len(self.data_don)}")
        lines.append(f"Matching variables: {', '.join(self.match_vars)}")
        if self.z_vars:
            lines.append(f"Donated variables: {', '.join(self.z_vars)}")
        lines.append("")

        # Distance summary
        lines.append("DISTANCE SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Mean distance:   {np.mean(self.distances):.6f}")
        lines.append(f"Median distance: {np.median(self.distances):.6f}")
        lines.append(f"Std distance:    {np.std(self.distances):.6f}")
        lines.append(f"Max distance:    {np.max(self.distances):.6f}")
        lines.append("")

        # Donor usage
        usage = self.donor_usage()
        lines.append("DONOR USAGE")
        lines.append("-" * 40)
        lines.append(f"Unique donors used: {len(usage)}")
        lines.append(f"Total donors:       {len(self.data_don)}")
        lines.append(
            f"Utilization rate:   {100 * len(usage) / len(self.data_don):.1f}%"
        )
        if len(usage) > 0:
            lines.append(f"Max uses per donor: {usage.max()}")
            lines.append(f"Mean uses per donor: {usage.mean():.2f}")
        lines.append("")

        # Balance table
        lines.append("BALANCE DIAGNOSTICS")
        lines.append("-" * 40)
        balance = self.balance_table()
        lines.append("Variable              SMD Before  SMD After  Status")
        for var in balance.index:
            smd_before = balance.loc[var, "smd_before"]
            smd_after = balance.loc[var, "smd_after"]
            # Status based on |SMD| < 0.1 rule
            status = "OK" if abs(smd_after) < 0.1 else "CAUTION"
            if abs(smd_after) >= 0.25:
                status = "POOR"
            lines.append(
                f"{var:20s}  {smd_before:+.4f}     {smd_after:+.4f}     {status}"
            )
        lines.append("")
        lines.append("Note: |SMD| < 0.1 is considered good balance")
        lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """
        Return all diagnostic metrics as a dictionary.

        Returns
        -------
        Dict
            Dictionary containing all diagnostic metrics.
        """
        balance = self.balance_table()
        usage = self.donor_usage()
        cia = self.cia_diagnostics()

        # Build SMD dictionaries for love plot
        smd_before = {}
        smd_after = {}
        for var in balance.index:
            smd_before[var] = balance.loc[var, "smd_before"]
            smd_after[var] = balance.loc[var, "smd_after"]

        return {
            "n_recipients": len(self.data_rec),
            "n_donors": len(self.data_don),
            "match_vars": self.match_vars,
            "z_vars": self.z_vars,
            "balance": balance.to_dict(),
            "smd_before": smd_before,
            "smd_after": smd_after,
            "distances": {
                "mean": float(np.mean(self.distances)),
                "median": float(np.median(self.distances)),
                "std": float(np.std(self.distances)),
                "min": float(np.min(self.distances)),
                "max": float(np.max(self.distances)),
            },
            "donor_usage": {
                "unique_donors": int(len(usage)),
                "max_uses": int(usage.max()) if len(usage) > 0 else 0,
                "mean_uses": float(usage.mean()) if len(usage) > 0 else 0.0,
                "utilization_rate": float(len(usage) / len(self.data_don)),
            },
            "cia_diagnostics": cia,
        }

    def to_html(self, path: str) -> None:
        """
        Export diagnostics as an HTML report.

        Parameters
        ----------
        path : str
            Path to save the HTML file.
        """
        metrics = self.to_dict()
        balance_df = self.balance_table()

        html_parts = []
        html_parts.append("<!DOCTYPE html>")
        html_parts.append("<html>")
        html_parts.append("<head>")
        html_parts.append(
            "<title>Statistical Matching Diagnostics Report</title>"
        )
        html_parts.append("<style>")
        html_parts.append(
            """
            body {
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
            h2 { color: #34495e; margin-top: 30px; }
            table {
                border-collapse: collapse;
                width: 100%;
                background-color: white;
                margin: 10px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: right;
            }
            th { background-color: #3498db; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .metric-box {
                display: inline-block;
                background-color: white;
                padding: 15px;
                margin: 5px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
            .metric-label { font-size: 12px; color: #7f8c8d; }
            .status-ok { color: green; }
            .status-caution { color: orange; }
            .status-poor { color: red; }
            .summary-text { white-space: pre-wrap; font-family: monospace; }
        """
        )
        html_parts.append("</style>")
        html_parts.append("</head>")
        html_parts.append("<body>")

        # Header
        html_parts.append("<h1>Statistical Matching Diagnostics Report</h1>")

        # Overview metrics
        html_parts.append("<h2>Overview</h2>")
        html_parts.append('<div class="metric-box">')
        html_parts.append(
            f'<div class="metric-value">{metrics["n_recipients"]}</div>'
        )
        html_parts.append('<div class="metric-label">Recipients</div>')
        html_parts.append("</div>")
        html_parts.append('<div class="metric-box">')
        html_parts.append(
            f'<div class="metric-value">{metrics["n_donors"]}</div>'
        )
        html_parts.append('<div class="metric-label">Donors</div>')
        html_parts.append("</div>")
        html_parts.append('<div class="metric-box">')
        html_parts.append(
            f'<div class="metric-value">'
            f'{metrics["donor_usage"]["unique_donors"]}</div>'
        )
        html_parts.append('<div class="metric-label">Donors Used</div>')
        html_parts.append("</div>")
        html_parts.append('<div class="metric-box">')
        html_parts.append(
            f'<div class="metric-value">'
            f'{metrics["distances"]["mean"]:.4f}</div>'
        )
        html_parts.append('<div class="metric-label">Mean Distance</div>')
        html_parts.append("</div>")

        # Balance table
        html_parts.append("<h2>Balance Diagnostics</h2>")
        html_parts.append("<table>")
        html_parts.append(
            "<tr><th>Variable</th><th>Mean (Rec)</th><th>Mean (Don Before)"
            "</th><th>Mean (Don After)</th><th>SMD Before</th>"
            "<th>SMD After</th><th>Status</th></tr>"
        )
        for var in balance_df.index:
            row = balance_df.loc[var]
            smd_after = row["smd_after"]
            if abs(smd_after) < 0.1:
                status_class = "status-ok"
                status_text = "OK"
            elif abs(smd_after) < 0.25:
                status_class = "status-caution"
                status_text = "CAUTION"
            else:
                status_class = "status-poor"
                status_text = "POOR"

            html_parts.append(
                f"<tr><td>{var}</td>"
                f"<td>{row['mean_rec']:.4f}</td>"
                f"<td>{row['mean_don_before']:.4f}</td>"
                f"<td>{row['mean_don_after']:.4f}</td>"
                f"<td>{row['smd_before']:+.4f}</td>"
                f"<td>{row['smd_after']:+.4f}</td>"
                f'<td class="{status_class}">{status_text}</td></tr>'
            )
        html_parts.append("</table>")
        html_parts.append(
            "<p><em>Note: |SMD| < 0.1 is considered good balance. "
            "|SMD| >= 0.25 may indicate poor matching.</em></p>"
        )

        # Distance summary
        html_parts.append("<h2>Distance Summary</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")
        for key, val in metrics["distances"].items():
            html_parts.append(
                f"<tr><td>{key.title()}</td><td>{val:.6f}</td></tr>"
            )
        html_parts.append("</table>")

        # Donor usage
        html_parts.append("<h2>Donor Usage</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")
        html_parts.append(
            f"<tr><td>Unique Donors Used</td>"
            f"<td>{metrics['donor_usage']['unique_donors']}</td></tr>"
        )
        html_parts.append(
            f"<tr><td>Total Donors</td><td>{metrics['n_donors']}</td></tr>"
        )
        html_parts.append(
            f"<tr><td>Utilization Rate</td>"
            f"<td>{100*metrics['donor_usage']['utilization_rate']:.1f}%</td></tr>"
        )
        html_parts.append(
            f"<tr><td>Max Uses per Donor</td>"
            f"<td>{metrics['donor_usage']['max_uses']}</td></tr>"
        )
        html_parts.append(
            f"<tr><td>Mean Uses per Donor</td>"
            f"<td>{metrics['donor_usage']['mean_uses']:.2f}</td></tr>"
        )
        html_parts.append("</table>")

        html_parts.append("</body>")
        html_parts.append("</html>")

        with open(path, "w") as f:
            f.write("\n".join(html_parts))


def match_diagnostics(
    result: Dict,
    data_rec: pd.DataFrame,
    data_don: pd.DataFrame,
    match_vars: List[str],
    z_vars: Optional[List[str]] = None,
) -> MatchDiagnostics:
    """
    Create comprehensive match quality diagnostics.

    This is the main entry point for creating diagnostic objects.

    Parameters
    ----------
    result : Dict
        Matching result from nnd_hotdeck containing:
        - 'noad.index': Array of donor indices
        - 'dist.rd': Array of distances
        - 'mtc.ids': DataFrame with matched IDs
    data_rec : pd.DataFrame
        Recipient dataset.
    data_don : pd.DataFrame
        Donor dataset.
    match_vars : List[str]
        List of matching variable names.
    z_vars : Optional[List[str]], default=None
        List of donated variable names (for CIA diagnostics).

    Returns
    -------
    MatchDiagnostics
        Object with methods for various diagnostic analyses.

    Examples
    --------
    >>> result = nnd_hotdeck(data_rec, data_don, match_vars=['x1', 'x2'])
    >>> diag = match_diagnostics(result, data_rec, data_don, ['x1', 'x2'])
    >>> print(diag.summary())
    >>> balance = diag.balance_table()
    >>> fig = diag.love_plot()
    """
    return MatchDiagnostics(
        result=result,
        data_rec=data_rec,
        data_don=data_don,
        match_vars=match_vars,
        z_vars=z_vars,
    )
