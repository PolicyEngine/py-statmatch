"""Survey weight utilities for statistical matching.

This module provides functions for working with survey weights in
statistical matching contexts:
- weighted_distance: Weight-aware distance computation
- calibrate_weights: Post-matching weight calibration (raking)
- design_effect: Compute design effect (DEFF) from weights
- replicate_variance: Variance estimation using replicate weights
"""

from typing import Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def weighted_distance(
    data_x: np.ndarray,
    data_y: np.ndarray,
    weights_x: np.ndarray,
    weights_y: np.ndarray,
    dist_fun: str = "euclidean",
) -> np.ndarray:
    """
    Compute distance matrix using weighted standardization.

    Variables are standardized using weighted means and standard deviations
    before computing distances. This ensures that the distance calculation
    accounts for the survey design through the weights.

    Parameters
    ----------
    data_x : np.ndarray
        Data matrix for first set of observations (n_x x n_vars).
    data_y : np.ndarray
        Data matrix for second set of observations (n_y x n_vars).
    weights_x : np.ndarray
        Survey weights for observations in data_x (n_x,).
    weights_y : np.ndarray
        Survey weights for observations in data_y (n_y,).
    dist_fun : str, default="euclidean"
        Distance function: "euclidean", "manhattan", or "mahalanobis".

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_x, n_y).

    Notes
    -----
    The standardization uses pooled weighted statistics from both datasets:
    - Weighted mean: sum(w * x) / sum(w)
    - Weighted std: sqrt(sum(w * (x - mean)^2) / sum(w))

    Examples
    --------
    >>> import numpy as np
    >>> from statmatch.survey_utils import weighted_distance
    >>> data_x = np.array([[1, 2], [3, 4], [5, 6]])
    >>> data_y = np.array([[2, 3], [4, 5]])
    >>> weights_x = np.array([1, 2, 1])
    >>> weights_y = np.array([1, 1])
    >>> dist = weighted_distance(data_x, data_y, weights_x, weights_y)
    """
    data_x = np.asarray(data_x, dtype=float)
    data_y = np.asarray(data_y, dtype=float)
    weights_x = np.asarray(weights_x, dtype=float)
    weights_y = np.asarray(weights_y, dtype=float)

    # Compute pooled weighted mean and std for standardization
    total_weight = np.sum(weights_x) + np.sum(weights_y)
    n_vars = data_x.shape[1]

    # Compute weighted means
    weighted_sum_x = np.sum(data_x * weights_x[:, np.newaxis], axis=0)
    weighted_sum_y = np.sum(data_y * weights_y[:, np.newaxis], axis=0)
    weighted_mean = (weighted_sum_x + weighted_sum_y) / total_weight

    # Compute weighted variance
    dev_x = data_x - weighted_mean
    dev_y = data_y - weighted_mean
    weighted_sq_dev_x = np.sum(dev_x**2 * weights_x[:, np.newaxis], axis=0)
    weighted_sq_dev_y = np.sum(dev_y**2 * weights_y[:, np.newaxis], axis=0)
    weighted_var = (weighted_sq_dev_x + weighted_sq_dev_y) / total_weight
    weighted_std = np.sqrt(weighted_var)

    # Avoid division by zero
    weighted_std = np.where(weighted_std == 0, 1.0, weighted_std)

    # Standardize data
    data_x_std = (data_x - weighted_mean) / weighted_std
    data_y_std = (data_y - weighted_mean) / weighted_std

    # Compute distance matrix
    if dist_fun.lower() == "manhattan":
        dist_matrix = cdist(data_x_std, data_y_std, metric="cityblock")
    elif dist_fun.lower() == "mahalanobis":
        # For weighted Mahalanobis, compute weighted covariance
        combined_data = np.vstack([data_x_std, data_y_std])
        combined_weights = np.concatenate([weights_x, weights_y])

        # Weighted covariance matrix
        weighted_cov = _weighted_covariance(combined_data, combined_weights)
        # Add regularization for numerical stability
        weighted_cov += np.eye(n_vars) * 1e-6
        inv_cov = np.linalg.inv(weighted_cov)

        n_x, n_y = len(data_x), len(data_y)
        dist_matrix = np.zeros((n_x, n_y))
        for i in range(n_x):
            for j in range(n_y):
                diff = data_x_std[i] - data_y_std[j]
                dist_matrix[i, j] = np.sqrt(diff @ inv_cov @ diff)
    else:
        # Default to Euclidean
        dist_matrix = cdist(data_x_std, data_y_std, metric="euclidean")

    return dist_matrix


def _weighted_covariance(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted covariance matrix."""
    weights = weights / np.sum(weights)  # Normalize weights
    mean = np.sum(data * weights[:, np.newaxis], axis=0)
    centered = data - mean
    cov = (centered.T * weights) @ centered
    return cov


def calibrate_weights(
    data: pd.DataFrame,
    weights: np.ndarray,
    targets: Dict[str, Dict[str, float]],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Calibrate survey weights to hit known marginal totals using raking.

    This function adjusts survey weights so that the weighted totals for
    specified variables match known population totals. It uses iterative
    proportional fitting (IPF), also known as raking.

    Parameters
    ----------
    data : pd.DataFrame
        Survey data containing the calibration variables.
    weights : np.ndarray
        Initial survey weights.
    targets : Dict[str, Dict[str, float]]
        Target totals for calibration variables. Format:
        {variable_name: {level: total, ...}, ...}
    max_iter : int, default=100
        Maximum number of raking iterations.
    tol : float, default=1e-6
        Convergence tolerance for raking.

    Returns
    -------
    np.ndarray
        Calibrated weights.

    Notes
    -----
    Raking iteratively adjusts weights to match marginal totals. Each
    iteration cycles through all calibration variables, adjusting weights
    so that the weighted total for each level matches the target.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from statmatch.survey_utils import calibrate_weights
    >>> data = pd.DataFrame({
    ...     "gender": ["M", "F", "M", "F", "M"],
    ...     "region": ["N", "S", "N", "S", "N"]
    ... })
    >>> weights = np.ones(5)
    >>> targets = {
    ...     "gender": {"M": 60.0, "F": 40.0},
    ...     "region": {"N": 55.0, "S": 45.0}
    ... }
    >>> calibrated = calibrate_weights(data, weights, targets)
    """
    weights = np.asarray(weights, dtype=float).copy()

    for iteration in range(max_iter):
        old_weights = weights.copy()

        for var, level_targets in targets.items():
            for level, target in level_targets.items():
                mask = data[var] == level

                if not np.any(mask):
                    continue

                current_total = np.sum(weights[mask])

                if current_total > 0:
                    adjustment = target / current_total
                    weights[mask] *= adjustment

        # Check convergence
        max_change = np.max(np.abs(weights - old_weights))
        if max_change < tol:
            break

    return weights


def design_effect(weights: np.ndarray) -> float:
    """
    Compute the design effect (DEFF) from survey weights.

    The design effect measures the loss of precision due to complex
    sampling design (in this case, unequal weighting). A DEFF of 1
    indicates no loss; larger values indicate greater efficiency loss.

    Parameters
    ----------
    weights : np.ndarray
        Survey weights.

    Returns
    -------
    float
        Design effect (DEFF). Always >= 1.

    Notes
    -----
    The formula used is: DEFF = 1 + CV(weights)^2

    where CV is the coefficient of variation (std/mean). This is derived
    from the Kish design effect approximation for weighted samples.

    The effective sample size is: n_eff = n / DEFF

    Examples
    --------
    >>> import numpy as np
    >>> from statmatch.survey_utils import design_effect
    >>> # Equal weights -> DEFF = 1
    >>> deff = design_effect(np.ones(100))
    >>> print(f"DEFF with equal weights: {deff:.2f}")
    DEFF with equal weights: 1.00
    >>> # Unequal weights -> DEFF > 1
    >>> deff = design_effect(np.array([1, 1, 1, 1, 10]))
    >>> print(f"DEFF with unequal weights: {deff:.2f}")
    DEFF with unequal weights: 2.60
    """
    weights = np.asarray(weights, dtype=float)

    mean_w = np.mean(weights)
    std_w = np.std(weights, ddof=0)  # Population std

    if mean_w == 0:
        return 1.0

    cv = std_w / mean_w
    deff = 1 + cv**2

    return deff


def replicate_variance(
    data: pd.DataFrame,
    weights: np.ndarray,
    statistic_fn: Callable[[pd.DataFrame, np.ndarray], float],
    method: str = "jackknife",
    n_replicates: int = 200,
    seed: Optional[int] = None,
) -> float:
    """
    Estimate variance of a statistic using replicate weights.

    This function estimates the sampling variance of a statistic using
    either the jackknife (leave-one-out) method or bootstrap resampling.

    Parameters
    ----------
    data : pd.DataFrame
        Survey data.
    weights : np.ndarray
        Survey weights.
    statistic_fn : Callable[[pd.DataFrame, np.ndarray], float]
        Function that computes the statistic of interest. Takes the data
        DataFrame and weights array as arguments, returns a scalar.
    method : str, default="jackknife"
        Variance estimation method: "jackknife" or "bootstrap".
    n_replicates : int, default=200
        Number of bootstrap replicates (ignored for jackknife).
    seed : int, optional
        Random seed for reproducibility (bootstrap only).

    Returns
    -------
    float
        Estimated variance of the statistic.

    Notes
    -----
    Jackknife:
        Drops one observation at a time and rescales weights. Variance is
        computed as: ((n-1)/n) * sum((theta_i - theta)^2)

    Bootstrap:
        Resamples observations with replacement, adjusting weights.
        Variance is the variance of the bootstrap estimates.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from statmatch.survey_utils import replicate_variance
    >>> data = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    >>> weights = np.ones(5)
    >>> def mean_x(df, w):
    ...     return np.average(df["x"], weights=w)
    >>> var = replicate_variance(data, weights, mean_x, method="jackknife")
    """
    weights = np.asarray(weights, dtype=float)
    n = len(data)

    if method.lower() == "jackknife":
        # Compute full-sample estimate
        theta_full = statistic_fn(data, weights)

        # Compute jackknife replicates
        theta_j = np.zeros(n)

        for i in range(n):
            # Create jackknife weights (drop observation i, rescale)
            jk_weights = weights.copy()
            jk_weights[i] = 0

            # Rescale to maintain sum
            if np.sum(jk_weights) > 0:
                jk_weights = jk_weights * np.sum(weights) / np.sum(jk_weights)

            # Compute statistic without observation i
            theta_j[i] = statistic_fn(data, jk_weights)

        # Jackknife variance estimate
        variance = ((n - 1) / n) * np.sum((theta_j - np.mean(theta_j)) ** 2)

    elif method.lower() == "bootstrap":
        if seed is not None:
            np.random.seed(seed)

        # Compute full-sample estimate
        theta_full = statistic_fn(data, weights)

        # Compute bootstrap replicates
        theta_b = np.zeros(n_replicates)

        for b in range(n_replicates):
            # Resample indices with replacement
            indices = np.random.choice(n, size=n, replace=True)

            # Create bootstrap sample
            boot_data = data.iloc[indices].reset_index(drop=True)
            boot_weights = weights[indices]

            # Compute statistic on bootstrap sample
            theta_b[b] = statistic_fn(boot_data, boot_weights)

        # Bootstrap variance estimate
        variance = np.var(theta_b, ddof=1)

    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'jackknife' or 'bootstrap'."
        )

    return variance
