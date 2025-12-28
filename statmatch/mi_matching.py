"""Multiple Imputation (MI) for statistical matching."""

from typing import Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats

from .nnd_hotdeck import nnd_hotdeck
from .create_fused import create_fused


def mi_nnd_hotdeck(
    data_rec: pd.DataFrame,
    data_don: pd.DataFrame,
    match_vars: List[str],
    m: int = 5,
    don_class: Optional[str] = None,
    dist_fun: str = "euclidean",
    method: str = "bootstrap",
    noise_scale: float = 0.1,
    seed: Optional[int] = None,
    **kwargs,
) -> List[Dict[str, Union[np.ndarray, pd.DataFrame]]]:
    """
    Multiple imputation wrapper around nnd_hotdeck.

    Generates m imputed datasets by introducing stochastic variation in the
    matching process. This accounts for uncertainty in the matching procedure.

    Parameters
    ----------
    data_rec : pd.DataFrame
        The recipient data set.
    data_don : pd.DataFrame
        The donor data set.
    match_vars : List[str]
        List of variable names to use for matching.
    m : int, default=5
        Number of imputations to generate.
    don_class : Optional[str], default=None
        Variable name defining donation classes.
    dist_fun : str, default="euclidean"
        Distance function to use.
    method : str, default="bootstrap"
        Method for introducing variation:
        - "bootstrap": Bootstrap resample donors for each imputation
        - "noise": Add noise to distances before matching
    noise_scale : float, default=0.1
        Scale of noise to add (relative to distance standard deviation).
        Only used when method="noise".
    seed : Optional[int], default=None
        Random seed for reproducibility.
    **kwargs
        Additional arguments passed to nnd_hotdeck.

    Returns
    -------
    List[Dict[str, Union[np.ndarray, pd.DataFrame]]]
        List of m matching results, each with same structure as nnd_hotdeck.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from statmatch import mi_nnd_hotdeck
    >>>
    >>> donor = pd.DataFrame({
    ...     'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     'y': [10, 20, 30, 40, 50]
    ... })
    >>> recipient = pd.DataFrame({
    ...     'x1': [1.5, 2.5, 3.5]
    ... })
    >>>
    >>> mi_results = mi_nnd_hotdeck(
    ...     recipient, donor, match_vars=['x1'], m=5, seed=42
    ... )
    >>> len(mi_results)
    5
    """
    if seed is not None:
        np.random.seed(seed)

    results = []

    for _ in range(m):
        if method == "bootstrap":
            # Bootstrap resample donors
            result = _mi_bootstrap_match(
                data_rec,
                data_don,
                match_vars,
                don_class,
                dist_fun,
                **kwargs,
            )
        elif method == "noise":
            # Add noise to distances
            result = _mi_noise_match(
                data_rec,
                data_don,
                match_vars,
                don_class,
                dist_fun,
                noise_scale,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'bootstrap' or 'noise'."
            )

        results.append(result)

    return results


def _mi_bootstrap_match(
    data_rec: pd.DataFrame,
    data_don: pd.DataFrame,
    match_vars: List[str],
    don_class: Optional[str],
    dist_fun: str,
    **kwargs,
) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """Perform matching with bootstrap resampled donors."""
    n_don = len(data_don)

    if don_class is not None:
        # Bootstrap within each donation class
        bootstrap_indices = []
        classes = data_don[don_class].unique()

        for cls in classes:
            cls_mask = data_don[don_class] == cls
            cls_indices = np.where(cls_mask)[0]
            n_cls = len(cls_indices)
            # Bootstrap resample within class
            resampled = np.random.choice(cls_indices, size=n_cls, replace=True)
            bootstrap_indices.extend(resampled)

        bootstrap_indices = np.array(bootstrap_indices)
        # Sort to maintain some structure
        bootstrap_indices = np.sort(bootstrap_indices)
    else:
        # Simple bootstrap of all donors
        bootstrap_indices = np.random.choice(n_don, size=n_don, replace=True)

    # Create bootstrap sample of donors
    data_don_boot = data_don.iloc[bootstrap_indices].reset_index(drop=True)

    # Perform matching on bootstrap sample
    result = nnd_hotdeck(
        data_rec=data_rec,
        data_don=data_don_boot,
        match_vars=match_vars,
        don_class=don_class,
        dist_fun=dist_fun,
        **kwargs,
    )

    # Map bootstrap indices back to original donor indices
    original_indices = bootstrap_indices[result["noad.index"]]

    # Update result to reference original donor data
    result["noad.index"] = original_indices
    result["mtc.ids"] = pd.DataFrame(
        {
            "rec.id": data_rec.index,
            "don.id": data_don.index[original_indices],
        }
    )

    return result


def _mi_noise_match(
    data_rec: pd.DataFrame,
    data_don: pd.DataFrame,
    match_vars: List[str],
    don_class: Optional[str],
    dist_fun: str,
    noise_scale: float,
    **kwargs,
) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """Perform matching with noise added to distances."""
    from scipy.spatial.distance import cdist

    n_rec = len(data_rec)
    n_don = len(data_don)

    # Prepare matching data
    rec_data = data_rec[match_vars].values
    don_data = data_don[match_vars].values

    # Initialize results
    donor_indices = np.zeros(n_rec, dtype=int)
    distances = np.zeros(n_rec)

    if don_class:
        rec_classes = data_rec[don_class].values
        don_classes = data_don[don_class].values

        for i in range(n_rec):
            cls = rec_classes[i]
            don_mask = don_classes == cls

            if not np.any(don_mask):
                raise ValueError(f"No donors found for class '{cls}'")

            rec_data_i = rec_data[i : i + 1]
            don_data_cls = don_data[don_mask]

            # Compute distances
            if dist_fun.lower() == "manhattan":
                dist_vec = cdist(rec_data_i, don_data_cls, "cityblock")[0]
            else:
                dist_vec = cdist(rec_data_i, don_data_cls, dist_fun.lower())[0]

            # Add noise to distances
            noise = np.random.normal(
                0, noise_scale * np.std(dist_vec), len(dist_vec)
            )
            dist_vec_noisy = np.maximum(0, dist_vec + noise)

            # Find best match with noisy distances
            best_local = np.argmin(dist_vec_noisy)
            don_idx_global = np.where(don_mask)[0]
            donor_indices[i] = don_idx_global[best_local]
            distances[i] = dist_vec[best_local]  # Report actual distance
    else:
        # Compute full distance matrix
        if dist_fun.lower() == "manhattan":
            dist_matrix = cdist(rec_data, don_data, "cityblock")
        else:
            dist_matrix = cdist(rec_data, don_data, dist_fun.lower())

        # Add noise to distances
        noise = np.random.normal(
            0, noise_scale * np.std(dist_matrix), dist_matrix.shape
        )
        dist_matrix_noisy = np.maximum(0, dist_matrix + noise)

        # Find best matches
        donor_indices = np.argmin(dist_matrix_noisy, axis=1)
        distances = dist_matrix[np.arange(n_rec), donor_indices]

    # Create results
    mtc_ids = pd.DataFrame(
        {"rec.id": data_rec.index, "don.id": data_don.index[donor_indices]}
    )

    return {
        "mtc.ids": mtc_ids,
        "noad.index": donor_indices,
        "dist.rd": distances,
    }


def mi_create_fused(
    data_rec: pd.DataFrame,
    data_don: pd.DataFrame,
    mi_results: List[Dict[str, Union[np.ndarray, pd.DataFrame]]],
    z_vars: List[str],
    dup_x: bool = False,
    match_vars: Optional[List[str]] = None,
) -> List[pd.DataFrame]:
    """
    Create m fused datasets from MI matching results.

    Parameters
    ----------
    data_rec : pd.DataFrame
        The recipient data set.
    data_don : pd.DataFrame
        The donor data set.
    mi_results : List[Dict]
        List of matching results from mi_nnd_hotdeck.
    z_vars : List[str]
        Names of variables to donate from data_don.
    dup_x : bool, default=False
        When True, also donate matching variables with ".don" suffix.
    match_vars : Optional[List[str]], default=None
        Names of matching variables. Required when dup_x=True.

    Returns
    -------
    List[pd.DataFrame]
        List of m fused datasets.

    Examples
    --------
    >>> mi_results = mi_nnd_hotdeck(recipient, donor, ['x1'], m=5)
    >>> fused_datasets = mi_create_fused(
    ...     recipient, donor, mi_results, z_vars=['y']
    ... )
    >>> len(fused_datasets)
    5
    """
    fused_datasets = []

    for result in mi_results:
        fused = create_fused(
            data_rec=data_rec,
            data_don=data_don,
            mtc_ids=result["mtc.ids"],
            z_vars=z_vars,
            dup_x=dup_x,
            match_vars=match_vars,
        )
        fused_datasets.append(fused)

    return fused_datasets


def combine_mi_estimates(
    estimates: np.ndarray,
    variances: np.ndarray,
) -> Dict[str, float]:
    """
    Combine MI estimates using Rubin's combining rules.

    Parameters
    ----------
    estimates : np.ndarray
        Array of m point estimates (one per imputation).
    variances : np.ndarray
        Array of m variance estimates (one per imputation).

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'estimate': Combined point estimate
        - 'within_variance': Within-imputation variance (W)
        - 'between_variance': Between-imputation variance (B)
        - 'total_variance': Total variance (W + (1 + 1/m) * B)
        - 'std_error': Standard error (sqrt of total variance)
        - 'df': Degrees of freedom
        - 'fmi': Fraction of missing information

    Notes
    -----
    Implements Rubin's (1987) combining rules for multiple imputation.
    The total variance accounts for both within-imputation sampling
    variance and between-imputation variance due to missing data.

    References
    ----------
    Rubin, D.B. (1987). Multiple Imputation for Nonresponse in Surveys.
    New York: John Wiley & Sons.
    """
    estimates = np.asarray(estimates)
    variances = np.asarray(variances)

    m = len(estimates)

    # Combined point estimate (mean of estimates)
    qbar = np.mean(estimates)

    # Within-imputation variance (mean of variances)
    ubar = np.mean(variances)

    # Between-imputation variance (variance of estimates)
    b = np.var(estimates, ddof=1)

    # Total variance: W + (1 + 1/m) * B
    total_var = ubar + (1 + 1 / m) * b

    # Standard error
    std_error = np.sqrt(total_var)

    # Fraction of missing information
    r = (1 + 1 / m) * b / ubar if ubar > 0 else 0
    fmi = r / (1 + r) if r > 0 else 0

    # Degrees of freedom (Barnard-Rubin adjustment)
    if b > 0 and ubar > 0:
        df = (m - 1) * (1 + 1 / r) ** 2
    else:
        df = float("inf")

    return {
        "estimate": qbar,
        "within_variance": ubar,
        "between_variance": b,
        "total_variance": total_var,
        "std_error": std_error,
        "df": df,
        "fmi": fmi,
    }


def mi_summary(
    fused_datasets: List[pd.DataFrame],
    var: str,
    statistic: Union[str, List[str]] = "mean",
    groupby: Optional[str] = None,
    conf_level: float = 0.95,
    custom_func: Optional[Callable[[pd.Series], float]] = None,
    custom_var_func: Optional[Callable[[pd.Series], float]] = None,
) -> pd.DataFrame:
    """
    Summarize MI results with proper confidence intervals.

    Parameters
    ----------
    fused_datasets : List[pd.DataFrame]
        List of m fused datasets from mi_create_fused.
    var : str
        Variable name to summarize.
    statistic : Union[str, List[str]], default="mean"
        Statistic(s) to compute. Options: "mean", "var", "sum", "median"
        or a list of these.
    groupby : Optional[str], default=None
        Variable to group by before computing statistics.
    conf_level : float, default=0.95
        Confidence level for intervals.
    custom_func : Optional[Callable], default=None
        Custom function to compute statistic.
    custom_var_func : Optional[Callable], default=None
        Custom function to compute variance of statistic.

    Returns
    -------
    pd.DataFrame
        Summary statistics with columns:
        - estimate: Combined point estimate
        - std_error: Standard error
        - ci_lower: Lower confidence interval bound
        - ci_upper: Upper confidence interval bound
        - within_var: Within-imputation variance
        - between_var: Between-imputation variance

    Examples
    --------
    >>> fused = mi_create_fused(rec, don, mi_results, z_vars=['y'])
    >>> summary = mi_summary(fused, var='y', statistic='mean')
    """
    if isinstance(statistic, str):
        statistics = [statistic]
    else:
        statistics = statistic

    results = []
    index_names = []

    if groupby is not None:
        # Group-level summary
        groups = fused_datasets[0][groupby].unique()

        for group in groups:
            for stat in statistics:
                estimates, variances = _compute_stat_by_imputation(
                    fused_datasets, var, stat, groupby, group
                )

                combined = combine_mi_estimates(estimates, variances)
                ci_lower, ci_upper = _compute_ci(combined, conf_level)

                results.append(
                    {
                        "estimate": combined["estimate"],
                        "std_error": combined["std_error"],
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "within_var": combined["within_variance"],
                        "between_var": combined["between_variance"],
                    }
                )
                index_names.append(f"{group}_{stat}")
    else:
        # Overall summary
        for stat in statistics:
            estimates, variances = _compute_stat_by_imputation(
                fused_datasets, var, stat
            )

            combined = combine_mi_estimates(estimates, variances)
            ci_lower, ci_upper = _compute_ci(combined, conf_level)

            results.append(
                {
                    "estimate": combined["estimate"],
                    "std_error": combined["std_error"],
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "within_var": combined["within_variance"],
                    "between_var": combined["between_variance"],
                }
            )
            index_names.append(stat)

    return pd.DataFrame(results, index=index_names)


def _compute_stat_by_imputation(
    fused_datasets: List[pd.DataFrame],
    var: str,
    statistic: str,
    groupby: Optional[str] = None,
    group_value: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute statistic and its variance for each imputation."""
    m = len(fused_datasets)
    estimates = np.zeros(m)
    variances = np.zeros(m)

    for i, df in enumerate(fused_datasets):
        if groupby is not None:
            data = df.loc[df[groupby] == group_value, var]
        else:
            data = df[var]

        n = len(data)

        if statistic == "mean":
            estimates[i] = data.mean()
            variances[i] = data.var() / n
        elif statistic == "var":
            estimates[i] = data.var()
            # Variance of sample variance
            # Var(s^2) = E[(X - mu)^4]/n - (n-3)/(n(n-1)) * sigma^4
            # Approximate with 2*sigma^4/(n-1)
            variances[i] = 2 * (data.var() ** 2) / (n - 1)
        elif statistic == "sum":
            estimates[i] = data.sum()
            variances[i] = n * data.var()
        elif statistic == "median":
            estimates[i] = data.median()
            # Variance of median approximation
            variances[i] = (np.pi / 2) * data.var() / n
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

    return estimates, variances


def _compute_ci(
    combined: Dict[str, float],
    conf_level: float,
) -> tuple[float, float]:
    """Compute confidence interval from combined estimates."""
    df = combined["df"]
    estimate = combined["estimate"]
    std_error = combined["std_error"]

    # Use t-distribution if df is finite, otherwise normal
    if np.isfinite(df) and df > 0:
        t_crit = stats.t.ppf((1 + conf_level) / 2, df)
    else:
        t_crit = stats.norm.ppf((1 + conf_level) / 2)

    ci_lower = estimate - t_crit * std_error
    ci_upper = estimate + t_crit * std_error

    return ci_lower, ci_upper
