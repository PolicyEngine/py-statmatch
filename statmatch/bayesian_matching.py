"""Bayesian uncertainty quantification for statistical matching.

This module provides Bayesian methods for quantifying uncertainty in
statistical matching, including posterior inference for matched values
and tests for the Conditional Independence Assumption (CIA).
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr


def bayesian_match(
    data_rec: pd.DataFrame,
    data_don: pd.DataFrame,
    match_vars: List[str],
    z_vars: Union[str, List[str]],
    n_samples: int = 1000,
    prior: str = "uniform",
    return_posterior: bool = True,
    credible_level: float = 0.95,
    dist_fun: str = "euclidean",
    k_neighbors: Optional[int] = None,
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Perform Bayesian matching with posterior inference.

    This function matches recipients to donors using a Bayesian approach,
    returning posterior distributions over the imputed values rather than
    just point estimates.

    Parameters
    ----------
    data_rec : pd.DataFrame
        The recipient dataset containing matching variables.
    data_don : pd.DataFrame
        The donor dataset containing matching variables and z variables.
    match_vars : List[str]
        List of variable names to use for matching.
    z_vars : str or List[str]
        Variable(s) to impute from donors to recipients.
    n_samples : int, default=1000
        Number of posterior samples to draw.
    prior : str, default='uniform'
        Prior distribution over donors:
        - 'uniform': Equal probability for all donors
        - 'distance_weighted': Probability inversely proportional to distance
    return_posterior : bool, default=True
        If True, return full posterior samples. If False, return only
        point estimates and credible intervals.
    credible_level : float, default=0.95
        Credible interval level (e.g., 0.95 for 95% intervals).
    dist_fun : str, default='euclidean'
        Distance function for computing donor similarities.
    k_neighbors : int, optional
        If specified, only consider k nearest neighbors for each recipient.

    Returns
    -------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing:
        - 'posterior_samples': Array or dict of posterior samples
          (shape: n_recipients x n_samples for single z_var,
          or dict of arrays for multiple z_vars)
        - 'point_estimates': Point estimates (posterior means)
        - 'credible_intervals': Dict with 'lower' and 'upper' bounds
        - 'donor_weights': Posterior weights for each donor
    """
    # Validate inputs
    if not all(var in data_rec.columns for var in match_vars):
        missing = [var for var in match_vars if var not in data_rec.columns]
        raise ValueError(
            f"Match variables {missing} not found in recipient data"
        )

    if not all(var in data_don.columns for var in match_vars):
        missing = [var for var in match_vars if var not in data_don.columns]
        raise ValueError(f"Match variables {missing} not found in donor data")

    # Normalize z_vars to list
    if isinstance(z_vars, str):
        z_vars = [z_vars]
        single_z = True
    else:
        single_z = len(z_vars) == 1

    if not all(var in data_don.columns for var in z_vars):
        missing = [var for var in z_vars if var not in data_don.columns]
        raise ValueError(f"Z variables {missing} not found in donor data")

    n_rec = len(data_rec)
    n_don = len(data_don)

    # Extract matching variable data
    rec_data = data_rec[match_vars].values.astype(float)
    don_data = data_don[match_vars].values.astype(float)

    # Compute distance matrix
    dist_matrix = _compute_distances(rec_data, don_data, dist_fun)

    # Compute donor weights based on prior
    weights = _compute_donor_weights(dist_matrix, prior, k_neighbors)

    # Sample from posterior
    if single_z:
        z_values = data_don[z_vars[0]].values.astype(float)
        posterior_samples = _sample_posterior(z_values, weights, n_samples)
        point_estimates = np.sum(weights * z_values, axis=1)

        lower, upper, _ = credible_interval(
            posterior_samples, level=credible_level
        )
    else:
        posterior_samples = {}
        point_estimates = {}
        lower = {}
        upper = {}

        for z_var in z_vars:
            z_values = data_don[z_var].values.astype(float)
            posterior_samples[z_var] = _sample_posterior(
                z_values, weights, n_samples
            )
            point_estimates[z_var] = np.sum(weights * z_values, axis=1)

            l, u, _ = credible_interval(
                posterior_samples[z_var], level=credible_level
            )
            lower[z_var] = l
            upper[z_var] = u

    # Build result dictionary
    result = {
        "point_estimates": (
            point_estimates if single_z else point_estimates[z_vars[0]]
        ),
        "credible_intervals": {"lower": lower, "upper": upper},
        "donor_weights": weights,
    }

    if return_posterior:
        result["posterior_samples"] = posterior_samples
    else:
        result["posterior_samples"] = None

    # Flatten credible intervals for single z variable
    if single_z:
        result["credible_intervals"] = {"lower": lower, "upper": upper}

    return result


def posterior_predictive(
    donor_data: pd.DataFrame,
    z_vars: Union[str, List[str]],
    weights: np.ndarray,
    n_samples: int = 1000,
    method: str = "resample",
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Sample from the posterior predictive distribution.

    Given matched donors and their weights, sample z values accounting
    for uncertainty in the matching.

    Parameters
    ----------
    donor_data : pd.DataFrame
        Donor dataset containing z variables.
    z_vars : str or List[str]
        Variable(s) to sample.
    weights : np.ndarray
        Matching weights for each recipient-donor pair.
        Shape: (n_recipients, n_donors).
    n_samples : int, default=1000
        Number of samples to draw.
    method : str, default='resample'
        Sampling method:
        - 'resample': Sample donors proportional to weights
        - 'gaussian': Use Gaussian approximation based on weighted moments

    Returns
    -------
    np.ndarray or Dict[str, np.ndarray]
        Posterior predictive samples.
        If single z_var: shape (n_recipients, n_samples)
        If multiple z_vars: dict mapping variable names to sample arrays
    """
    if isinstance(z_vars, str):
        z_vars = [z_vars]
        single_z = True
    else:
        single_z = len(z_vars) == 1

    n_rec = weights.shape[0]
    n_don = weights.shape[1]

    if single_z:
        z_values = donor_data[z_vars[0]].values.astype(float)
        samples = _sample_posterior(z_values, weights, n_samples, method)
        return samples
    else:
        samples = {}
        for z_var in z_vars:
            z_values = donor_data[z_var].values.astype(float)
            samples[z_var] = _sample_posterior(
                z_values, weights, n_samples, method
            )
        return samples


def credible_interval(
    samples: np.ndarray,
    level: float = 0.95,
) -> tuple:
    """
    Compute credible intervals from posterior samples.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples. Shape: (n_units, n_samples).
    level : float, default=0.95
        Credible level (e.g., 0.95 for 95% interval).

    Returns
    -------
    tuple
        (lower, upper, point_estimate) where each is an array of length n_units.
        Point estimate is the posterior mean.
    """
    alpha = 1 - level
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2

    lower = np.percentile(samples, lower_quantile * 100, axis=1)
    upper = np.percentile(samples, upper_quantile * 100, axis=1)
    point = np.mean(samples, axis=1)

    return lower, upper, point


def cia_posterior_test(
    data: pd.DataFrame,
    x_vars: List[str],
    y_vars: List[str],
    z_vars: List[str],
    n_samples: int = 1000,
    threshold: float = 0.05,
) -> float:
    """
    Bayesian test of the Conditional Independence Assumption (CIA).

    The CIA states that Y and Z are independent given X:
    P(Y, Z | X) = P(Y | X) * P(Z | X)

    This function estimates the posterior probability that CIA holds
    by testing whether the partial correlation between Y and Z
    given X is zero.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing all variables.
    x_vars : List[str]
        Conditioning variables (common to both surveys).
    y_vars : List[str]
        Variables from first survey.
    z_vars : List[str]
        Variables from second survey.
    n_samples : int, default=1000
        Number of posterior samples for Bayesian bootstrap.
    threshold : float, default=0.05
        Threshold for partial correlation to be considered "small".

    Returns
    -------
    float
        Posterior probability that CIA holds (partial correlations are small).
    """
    n = len(data)

    # Extract variable matrices
    X = data[x_vars].values.astype(float)
    Y = data[y_vars].values.astype(float)
    Z = data[z_vars].values.astype(float)

    # If Y or Z are 1D, reshape
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    # Compute partial correlations using Bayesian bootstrap
    cia_holds_count = 0

    for _ in range(n_samples):
        # Bayesian bootstrap weights (Dirichlet)
        weights = np.random.dirichlet(np.ones(n))

        # Compute residuals of Y and Z given X using weighted regression
        Y_resid = _weighted_residuals(X, Y, weights)
        Z_resid = _weighted_residuals(X, Z, weights)

        # Compute weighted correlation between residuals
        partial_corrs = []
        for i in range(Y_resid.shape[1]):
            for j in range(Z_resid.shape[1]):
                corr = _weighted_correlation(
                    Y_resid[:, i], Z_resid[:, j], weights
                )
                partial_corrs.append(abs(corr))

        # Check if all partial correlations are below threshold
        max_partial_corr = max(partial_corrs)
        if max_partial_corr < threshold:
            cia_holds_count += 1

    return cia_holds_count / n_samples


def _compute_distances(
    rec_data: np.ndarray,
    don_data: np.ndarray,
    dist_fun: str,
) -> np.ndarray:
    """Compute distance matrix between recipients and donors."""
    if dist_fun.lower() == "manhattan":
        return cdist(rec_data, don_data, metric="cityblock")
    elif dist_fun.lower() in ["euclidean", "chebyshev", "cosine"]:
        return cdist(rec_data, don_data, metric=dist_fun.lower())
    elif dist_fun.lower() == "mahalanobis":
        combined = np.vstack([rec_data, don_data])
        cov = np.cov(combined.T)
        cov += np.eye(cov.shape[0]) * 1e-6
        inv_cov = np.linalg.inv(cov)

        n_rec = rec_data.shape[0]
        n_don = don_data.shape[0]
        dist_matrix = np.zeros((n_rec, n_don))

        for i in range(n_rec):
            for j in range(n_don):
                diff = rec_data[i] - don_data[j]
                dist_matrix[i, j] = np.sqrt(diff @ inv_cov @ diff)

        return dist_matrix
    else:
        return cdist(rec_data, don_data, metric="euclidean")


def _compute_donor_weights(
    dist_matrix: np.ndarray,
    prior: str,
    k_neighbors: Optional[int] = None,
) -> np.ndarray:
    """
    Compute donor weights for each recipient based on distances.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Distance matrix (n_recipients x n_donors).
    prior : str
        Prior type ('uniform' or 'distance_weighted').
    k_neighbors : int, optional
        If specified, only consider k nearest neighbors.

    Returns
    -------
    np.ndarray
        Weight matrix (n_recipients x n_donors).
    """
    n_rec, n_don = dist_matrix.shape

    if prior == "uniform":
        # Equal weights for all donors
        weights = np.ones((n_rec, n_don)) / n_don

        if k_neighbors is not None:
            # Zero out non-neighbors
            for i in range(n_rec):
                sorted_idx = np.argsort(dist_matrix[i])
                mask = np.zeros(n_don)
                mask[sorted_idx[:k_neighbors]] = 1
                weights[i] *= mask
                weights[i] /= weights[i].sum()

    elif prior == "distance_weighted":
        # Inverse distance weighting with softmax-like normalization
        # Use exponential kernel: w_ij = exp(-d_ij / scale)
        # Scale by median distance for numerical stability
        scale = np.median(dist_matrix[dist_matrix > 0])
        if scale == 0:
            scale = 1.0

        weights = np.exp(-dist_matrix / scale)

        if k_neighbors is not None:
            # Zero out non-neighbors
            for i in range(n_rec):
                sorted_idx = np.argsort(dist_matrix[i])
                mask = np.zeros(n_don)
                mask[sorted_idx[:k_neighbors]] = 1
                weights[i] *= mask

        # Normalize rows to sum to 1
        weights = weights / weights.sum(axis=1, keepdims=True)

    else:
        raise ValueError(
            f"Unknown prior: {prior}. Use 'uniform' or 'distance_weighted'"
        )

    return weights


def _sample_posterior(
    z_values: np.ndarray,
    weights: np.ndarray,
    n_samples: int,
    method: str = "resample",
) -> np.ndarray:
    """
    Sample from posterior distribution of z values.

    Parameters
    ----------
    z_values : np.ndarray
        Donor z values (n_donors,).
    weights : np.ndarray
        Donor weights (n_recipients, n_donors).
    n_samples : int
        Number of samples to draw.
    method : str
        Sampling method ('resample' or 'gaussian').

    Returns
    -------
    np.ndarray
        Posterior samples (n_recipients, n_samples).
    """
    n_rec = weights.shape[0]
    n_don = len(z_values)

    if method == "gaussian":
        # Gaussian approximation using weighted moments
        means = np.sum(weights * z_values, axis=1)
        variances = np.sum(
            weights * (z_values - means.reshape(-1, 1)) ** 2, axis=1
        )
        stds = np.sqrt(variances)

        samples = np.zeros((n_rec, n_samples))
        for i in range(n_rec):
            samples[i] = np.random.normal(means[i], stds[i], n_samples)

    else:  # resample
        # Sample donor indices according to weights
        samples = np.zeros((n_rec, n_samples))

        for i in range(n_rec):
            # Sample with Bayesian bootstrap perturbation
            for s in range(n_samples):
                # Perturb weights using Dirichlet
                perturbed_weights = weights[i] * np.random.exponential(
                    1, n_don
                )
                perturbed_weights /= perturbed_weights.sum()

                # Sample a donor according to perturbed weights
                donor_idx = np.random.choice(n_don, p=perturbed_weights)
                samples[i, s] = z_values[donor_idx]

    return samples


def _weighted_residuals(
    X: np.ndarray,
    Y: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Compute weighted regression residuals of Y on X.

    Parameters
    ----------
    X : np.ndarray
        Predictor matrix (n, p).
    Y : np.ndarray
        Response matrix (n, q).
    weights : np.ndarray
        Observation weights (n,).

    Returns
    -------
    np.ndarray
        Residuals (n, q).
    """
    n = X.shape[0]

    # Add intercept
    X_aug = np.column_stack([np.ones(n), X])

    # Weighted least squares using sqrt(W) formulation for stability
    # This avoids creating a full n x n diagonal matrix
    sqrt_w = np.sqrt(weights).reshape(-1, 1)
    X_weighted = X_aug * sqrt_w
    Y_weighted = Y * sqrt_w

    try:
        # Use lstsq for numerical stability
        beta, _, _, _ = np.linalg.lstsq(X_weighted, Y_weighted, rcond=None)
        Y_pred = X_aug @ beta
        residuals = Y - Y_pred
    except np.linalg.LinAlgError:
        # If regression fails, use centered Y
        residuals = Y - np.average(Y, weights=weights, axis=0)

    return residuals


def _weighted_correlation(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Compute weighted Pearson correlation.

    Parameters
    ----------
    x : np.ndarray
        First variable (n,).
    y : np.ndarray
        Second variable (n,).
    weights : np.ndarray
        Observation weights (n,).

    Returns
    -------
    float
        Weighted correlation coefficient.
    """
    # Weighted means
    mean_x = np.average(x, weights=weights)
    mean_y = np.average(y, weights=weights)

    # Weighted covariance
    cov_xy = np.average((x - mean_x) * (y - mean_y), weights=weights)

    # Weighted standard deviations
    std_x = np.sqrt(np.average((x - mean_x) ** 2, weights=weights))
    std_y = np.sqrt(np.average((y - mean_y) ** 2, weights=weights))

    if std_x < 1e-10 or std_y < 1e-10:
        return 0.0

    return cov_xy / (std_x * std_y)
