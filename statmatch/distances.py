"""Distance functions for statistical matching.

This module provides distance computation functions that are compatible
with R's StatMatch package:
- gower_dist: Gower's distance for mixed-type data
- mahalanobis_dist: Mahalanobis distance for continuous data
- maximum_dist: Maximum (L-infinity) distance for continuous data
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def gower_dist(
    data_x: Union[pd.DataFrame, np.ndarray],
    data_y: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    rngs: Optional[List[float]] = None,
    kr_corr: bool = True,
    var_weights: Optional[List[float]] = None,
    robcb: Optional[str] = None,
) -> np.ndarray:
    """
    Compute Gower's distance between observations.

    This function computes the Gower's distance (dissimilarity) between units
    in a dataset or between observations in two distinct datasets. It handles
    mixed data types (numeric, categorical, ordered categorical).

    Parameters
    ----------
    data_x : pd.DataFrame or np.ndarray
        Matrix or data frame containing variables for distance computation.
        Numeric columns are treated as interval-scaled variables.
        Categorical columns are treated as nominal variables.
        Ordered categorical columns are treated as ordinal variables.
    data_y : pd.DataFrame or np.ndarray, optional
        Matrix or data frame with same structure as data_x. If not provided,
        distances are computed within data_x.
    rngs : list of float, optional
        Vector of ranges for scaling numeric variables. Length must match
        number of variables. If None, ranges are computed from data.
    kr_corr : bool, default=True
        If True, apply Kaufman-Rousseeuw correction for ordinal variables.
    var_weights : list of float, optional
        Weights for variables. Automatically normalized to sum to 1.
    robcb : str, optional
        Robust range computation method: "boxp" or "asyboxp".

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_x, n_y) where n_x is number of rows
        in data_x and n_y is number of rows in data_y.

    References
    ----------
    Gower, J. C. (1971). "A general coefficient of similarity and some of
    its properties". Biometrics, 27, 623-637.

    Kaufman, L. and Rousseeuw, P.J. (1990). Finding Groups in Data: An
    Introduction to Cluster Analysis. Wiley, New York.
    """
    # Convert to DataFrame if numpy array
    if isinstance(data_x, np.ndarray):
        data_x = pd.DataFrame(data_x)

    if data_y is None:
        data_y = data_x
    elif isinstance(data_y, np.ndarray):
        data_y = pd.DataFrame(data_y, columns=data_x.columns)

    n_x = len(data_x)
    n_y = len(data_y)
    n_vars = data_x.shape[1]

    # Initialize weights
    if var_weights is None:
        weights = np.ones(n_vars)
    else:
        weights = np.array(var_weights)

    # Normalize weights to sum to 1
    weights = weights / weights.sum()

    # Compute ranges if not provided
    if rngs is None:
        rngs = []
        for col in data_x.columns:
            if _is_numeric(data_x[col]) and _is_numeric(data_y[col]):
                if robcb is not None:
                    # Robust range computation
                    combined = np.concatenate(
                        [data_x[col].values, data_y[col].values]
                    )
                    combined = combined[~np.isnan(combined)]
                    if robcb == "boxp":
                        q1, q3 = np.percentile(combined, [25, 75])
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        trimmed = combined[
                            (combined >= lower) & (combined <= upper)
                        ]
                        if len(trimmed) > 0:
                            rngs.append(trimmed.max() - trimmed.min())
                        else:
                            rngs.append(combined.max() - combined.min())
                    else:  # asyboxp
                        # Asymmetric boxplot-based range
                        q1, q2, q3 = np.percentile(combined, [25, 50, 75])
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        trimmed = combined[
                            (combined >= lower) & (combined <= upper)
                        ]
                        if len(trimmed) > 0:
                            rngs.append(trimmed.max() - trimmed.min())
                        else:
                            rngs.append(combined.max() - combined.min())
                else:
                    combined = np.concatenate(
                        [data_x[col].values, data_y[col].values]
                    )
                    combined = combined[~np.isnan(combined)]
                    if len(combined) > 0:
                        rngs.append(combined.max() - combined.min())
                    else:
                        rngs.append(1.0)
            else:
                # Non-numeric: range is 1 for nominal variables
                rngs.append(1.0)
        rngs = np.array(rngs)
    else:
        rngs = np.array(rngs)

    # Handle zero ranges
    rngs = np.where(rngs == 0, 1.0, rngs)

    # Initialize distance matrix
    dist_matrix = np.zeros((n_x, n_y))
    weight_matrix = np.zeros((n_x, n_y))

    for k, col in enumerate(data_x.columns):
        x_vals = data_x[col].values
        y_vals = data_y[col].values

        if _is_numeric(data_x[col]) and _is_numeric(data_y[col]):
            # Numeric variable: Manhattan distance normalized by range
            x_mat = x_vals.reshape(-1, 1)
            y_mat = y_vals.reshape(1, -1)
            contrib = np.abs(x_mat - y_mat) / rngs[k]
            # Cap contributions at 1.0 (as R does)
            contrib = np.minimum(contrib, 1.0)

            # Create delta matrix (1 where both values are present)
            x_na = np.isnan(x_vals).reshape(-1, 1)
            y_na = np.isnan(y_vals).reshape(1, -1)
            delta = (~(x_na | y_na)).astype(float)

        elif _is_ordered_categorical(data_x[col]) and _is_ordered_categorical(
            data_y[col]
        ):
            # Ordered categorical: treat as numeric ranks
            x_codes = data_x[col].cat.codes.values.astype(float)
            y_codes = data_y[col].cat.codes.values.astype(float)

            # Handle missing values (code = -1)
            x_codes[x_codes < 0] = np.nan
            y_codes[y_codes < 0] = np.nan

            n_levels = len(data_x[col].cat.categories)

            if kr_corr and n_levels > 1:
                # Kaufman-Rousseeuw correction: scale to [0, 1]
                x_scaled = x_codes / (n_levels - 1)
                y_scaled = y_codes / (n_levels - 1)
            else:
                x_scaled = x_codes
                y_scaled = y_codes
                rngs[k] = n_levels - 1 if n_levels > 1 else 1

            x_mat = x_scaled.reshape(-1, 1)
            y_mat = y_scaled.reshape(1, -1)
            contrib = np.abs(x_mat - y_mat)
            if not kr_corr:
                contrib = contrib / rngs[k]

            # Delta matrix
            x_na = np.isnan(x_codes).reshape(-1, 1)
            y_na = np.isnan(y_codes).reshape(1, -1)
            delta = (~(x_na | y_na)).astype(float)

        else:
            # Categorical (nominal): 0 if equal, 1 if different
            x_mat = x_vals.reshape(-1, 1)
            y_mat = y_vals.reshape(1, -1)
            contrib = (x_mat != y_mat).astype(float)

            # Handle missing values
            x_na = pd.isna(x_vals).reshape(-1, 1)
            y_na = pd.isna(y_vals).reshape(1, -1)
            delta = (~(x_na | y_na)).astype(float)

        # Accumulate weighted distances
        dist_matrix += weights[k] * contrib * delta
        weight_matrix += weights[k] * delta

    # Normalize by total weight
    # Avoid division by zero
    weight_matrix = np.where(weight_matrix == 0, 1.0, weight_matrix)
    result = dist_matrix / weight_matrix

    return result


def mahalanobis_dist(
    data_x: Union[pd.DataFrame, np.ndarray],
    data_y: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    vc: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute Mahalanobis distance between observations.

    This function computes the Mahalanobis distance among units in a dataset
    or between observations in two distinct datasets.

    Parameters
    ----------
    data_x : pd.DataFrame or np.ndarray
        Matrix or data frame with continuous variables only.
        Missing values (NA) are not allowed.
    data_y : pd.DataFrame or np.ndarray, optional
        Matrix or data frame with same variables as data_x.
        If not provided, distances are computed within data_x.
    vc : np.ndarray, optional
        Covariance matrix for distance computation. If None, estimated
        from the data.

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_x, n_y).

    References
    ----------
    Mahalanobis, P C (1936) "On the generalised distance in statistics".
    Proceedings of the National Institute of Sciences of India 2, pp. 49-55.
    """
    # Convert to numpy arrays
    if isinstance(data_x, pd.DataFrame):
        xx = data_x.values.astype(float)
    else:
        xx = np.asarray(data_x, dtype=float)

    if data_y is None:
        yy = xx.copy()
    elif isinstance(data_y, pd.DataFrame):
        yy = data_y.values.astype(float)
    else:
        yy = np.asarray(data_y, dtype=float)

    # Estimate covariance matrix if not provided
    if vc is None:
        if data_y is None:
            mm = xx
        else:
            mm = np.vstack([xx, yy])
        vc = np.cov(mm.T)

    # Ensure vc is 2D
    if vc.ndim == 0:
        vc = vc.reshape(1, 1)

    # Compute inverse of covariance matrix
    inv_vc = np.linalg.inv(vc)

    n_x = xx.shape[0]
    n_y = yy.shape[0]

    # Compute Mahalanobis distance for each pair
    md = np.zeros((n_x, n_y))
    for j in range(n_y):
        diff = xx - yy[j]
        md[:, j] = np.sum(diff @ inv_vc * diff, axis=1)

    # Return square root of Mahalanobis distance
    return np.sqrt(md)


def maximum_dist(
    data_x: Union[pd.DataFrame, np.ndarray],
    data_y: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    rank: bool = False,
) -> np.ndarray:
    """
    Compute maximum (L-infinity) distance between observations.

    This function computes the L-infinity distance (also known as Chebyshev
    or minimax distance). The distance between two records is the maximum
    of the absolute differences among the observed variables.

    Parameters
    ----------
    data_x : pd.DataFrame or np.ndarray
        Matrix or data frame with continuous variables only.
        Missing values (NA) are not allowed.
    data_y : pd.DataFrame or np.ndarray, optional
        Matrix or data frame with same variables as data_x.
        If not provided, distances are computed within data_x.
    rank : bool, default=False
        If True, original values are replaced by their ranks divided by
        (n+1), removing the effect of different scales.

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_x, n_y).

    References
    ----------
    Kovar, J.G., MacMillan, J. and Whitridge, P. (1988). "Overview and
    strategy for the Generalized Edit and Imputation System". Statistics
    Canada, Methodology Branch Working Paper No. BSMD 88-007 E/F.
    """
    # Convert to numpy arrays
    if isinstance(data_x, pd.DataFrame):
        xx = data_x.values.astype(float)
    else:
        xx = np.asarray(data_x, dtype=float)

    if data_y is None:
        yy = xx.copy()
        same_data = True
    elif isinstance(data_y, pd.DataFrame):
        yy = data_y.values.astype(float)
        same_data = False
    else:
        yy = np.asarray(data_y, dtype=float)
        same_data = False

    n_x = xx.shape[0]
    n_y = yy.shape[0]

    if rank:
        if same_data:
            # Rank within the same data
            n = n_x
            rx = np.zeros_like(xx)
            for k in range(xx.shape[1]):
                rx[:, k] = rankdata(xx[:, k], method="average") / (n + 1)
            ry = rx
        else:
            # Rank across combined data
            n_total = n_x + n_y
            combined = np.vstack([xx, yy])
            ranked = np.zeros_like(combined)
            for k in range(combined.shape[1]):
                ranked[:, k] = (
                    rankdata(combined[:, k], method="average") / (n_total + 1)
                )
            rx = ranked[:n_x]
            ry = ranked[n_x:]
    else:
        rx = xx
        ry = yy

    # Compute maximum absolute difference for each pair
    mdist = np.zeros((n_x, n_y))
    for i in range(n_x):
        diff = np.abs(rx[i] - ry)
        mdist[i] = np.max(diff, axis=1)

    return mdist


def _is_numeric(series: pd.Series) -> bool:
    """Check if a pandas Series is numeric."""
    return pd.api.types.is_numeric_dtype(series)


def _is_ordered_categorical(series: pd.Series) -> bool:
    """Check if a pandas Series is an ordered categorical."""
    return hasattr(series, "cat") and series.cat.ordered
