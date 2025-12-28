"""Implementation of Optimal Transport (OT) based matching."""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def ot_hotdeck(
    data_rec: pd.DataFrame,
    data_don: pd.DataFrame,
    match_vars: List[str],
    dist_fun: str = "euclidean",
    reg: float = 0.01,
    method: str = "sinkhorn",
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> Dict[str, Union[np.ndarray, pd.DataFrame, float]]:
    """
    Implement Optimal Transport-based Hot Deck matching.

    This function finds globally optimal matches between recipients and donors
    by solving an optimal transport problem. Unlike nearest neighbor matching,
    OT considers the global structure of both datasets to minimize total
    transport cost.

    Parameters
    ----------
    data_rec : pd.DataFrame
        The recipient data set.
    data_don : pd.DataFrame
        The donor data set.
    match_vars : List[str]
        List of variable names to use for matching.
    dist_fun : str, default="euclidean"
        Distance function to use for computing the cost matrix.
        Options: "euclidean", "manhattan", "cosine".
    reg : float, default=0.01
        Entropy regularization parameter for Sinkhorn algorithm.
        Lower values give solutions closer to exact OT but may be slower.
        Ignored if method="emd".
    method : str, default="sinkhorn"
        OT solving method:
        - "sinkhorn": Entropy-regularized OT using Sinkhorn algorithm.
          Fast and differentiable, but approximate.
        - "emd": Exact Earth Mover's Distance using Hungarian algorithm.
          Gives exact solution but slower for large datasets.
    max_iter : int, default=1000
        Maximum iterations for Sinkhorn algorithm.
    tol : float, default=1e-9
        Convergence tolerance for Sinkhorn algorithm.

    Returns
    -------
    Dict[str, Union[np.ndarray, pd.DataFrame, float]]
        Dictionary containing:
        - 'mtc.ids': DataFrame with recipient and donor IDs
        - 'noad.index': Array of donor indices for each recipient (0-based)
        - 'dist.rd': Array of distances between matched recipients and donors
        - 'transport_plan': The optimal transport plan matrix (n_rec x n_don)
        - 'total_cost': Total transport cost

    Notes
    -----
    The optimal transport problem minimizes:

        min_{T} sum_{i,j} T_{ij} * C_{ij}

    subject to:
        T >= 0
        sum_j T_{ij} = a_i  (recipient marginal)
        sum_i T_{ij} = b_j  (donor marginal)

    where C is the cost matrix and T is the transport plan.

    For entropy-regularized OT (Sinkhorn), we add -reg * H(T) to the objective,
    where H(T) is the entropy of the transport plan.

    Examples
    --------
    >>> donor_data = pd.DataFrame({'x': [0, 1, 2], 'y': [10, 20, 30]})
    >>> recipient_data = pd.DataFrame({'x': [0.1, 1.1, 2.1]})
    >>> result = ot_hotdeck(recipient_data, donor_data, match_vars=['x'])
    >>> result['noad.index']
    array([0, 1, 2])
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

    if method not in ["sinkhorn", "emd"]:
        raise ValueError(
            f"Unknown method '{method}'. Use 'sinkhorn' or 'emd'."
        )

    # Extract matching variable data
    rec_data = data_rec[match_vars].values.astype(float)
    don_data = data_don[match_vars].values.astype(float)

    n_rec = len(rec_data)
    n_don = len(don_data)

    # Compute cost matrix
    cost_matrix = _compute_cost_matrix(rec_data, don_data, dist_fun)

    # Uniform marginals
    a = np.ones(n_rec) / n_rec  # recipient marginal
    b = np.ones(n_don) / n_don  # donor marginal

    # Solve optimal transport
    if method == "emd":
        transport_plan = _solve_emd(cost_matrix, a, b)
    else:  # sinkhorn
        transport_plan = _sinkhorn(
            cost_matrix, a, b, reg, max_iter=max_iter, tol=tol
        )

    # Extract assignments from transport plan
    # For each recipient, find the donor with maximum transport mass
    donor_indices = np.argmax(transport_plan, axis=1)

    # Compute distances for matched pairs
    distances = np.array(
        [cost_matrix[i, donor_indices[i]] for i in range(n_rec)]
    )

    # Compute total transport cost
    total_cost = np.sum(transport_plan * cost_matrix)

    # Create results dictionary
    mtc_ids = pd.DataFrame(
        {"rec.id": data_rec.index, "don.id": data_don.index[donor_indices]}
    )

    results = {
        "mtc.ids": mtc_ids,
        "noad.index": donor_indices,
        "dist.rd": distances,
        "transport_plan": transport_plan,
        "total_cost": total_cost,
    }

    return results


def wasserstein_dist(
    x: np.ndarray,
    y: np.ndarray,
    p: int = 1,
    reg: float = 0.0,
) -> float:
    """
    Compute the p-Wasserstein distance between two distributions.

    The Wasserstein distance (also known as Earth Mover's Distance for p=1)
    measures the minimum "work" required to transform one distribution into
    another.

    Parameters
    ----------
    x : np.ndarray
        First dataset, shape (n_samples_x, n_features).
    y : np.ndarray
        Second dataset, shape (n_samples_y, n_features).
    p : int, default=1
        Order of the Wasserstein distance.
        - p=1: Earth Mover's Distance (W1)
        - p=2: Quadratic Wasserstein distance (W2)
    reg : float, default=0.0
        Entropy regularization parameter. If 0, computes exact Wasserstein.
        If > 0, uses Sinkhorn algorithm for approximation.

    Returns
    -------
    float
        The p-Wasserstein distance between x and y.

    Notes
    -----
    The p-Wasserstein distance is defined as:

        W_p(mu, nu) = (min_{T} sum_{i,j} T_{ij} * d(x_i, y_j)^p)^{1/p}

    where d is the ground distance (Euclidean by default).

    Examples
    --------
    >>> x = np.array([[0], [1], [2]])
    >>> y = np.array([[1], [2], [3]])
    >>> wasserstein_dist(x, y, p=1)
    1.0
    """
    # Ensure 2D arrays
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_x = len(x)
    n_y = len(y)

    # Compute ground distance matrix
    ground_dist = cdist(x, y, metric="euclidean")

    # For p-Wasserstein, we use d^p as costs
    cost_matrix = ground_dist**p

    # Uniform marginals
    a = np.ones(n_x) / n_x
    b = np.ones(n_y) / n_y

    # Solve OT problem
    if reg > 0:
        transport_plan = _sinkhorn(cost_matrix, a, b, reg)
    else:
        transport_plan = _solve_emd(cost_matrix, a, b)

    # Compute Wasserstein distance
    total_cost = np.sum(transport_plan * cost_matrix)

    # Return W_p = (total_cost)^{1/p}
    return total_cost ** (1.0 / p)


def _compute_cost_matrix(
    rec_data: np.ndarray,
    don_data: np.ndarray,
    dist_fun: str,
) -> np.ndarray:
    """
    Compute the cost matrix between recipients and donors.

    Parameters
    ----------
    rec_data : np.ndarray
        Recipient data matrix (n_rec x n_vars).
    don_data : np.ndarray
        Donor data matrix (n_don x n_vars).
    dist_fun : str
        Distance function name.

    Returns
    -------
    np.ndarray
        Cost matrix of shape (n_rec, n_don).
    """
    if dist_fun.lower() == "manhattan":
        return cdist(rec_data, don_data, metric="cityblock")
    elif dist_fun.lower() == "cosine":
        return cdist(rec_data, don_data, metric="cosine")
    elif dist_fun.lower() == "euclidean":
        return cdist(rec_data, don_data, metric="euclidean")
    else:
        # Default to Euclidean
        return cdist(rec_data, don_data, metric="euclidean")


def _solve_emd(
    cost_matrix: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Solve exact Earth Mover's Distance using Hungarian algorithm.

    This solves the linear assignment problem for optimal transport
    with uniform marginals.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Cost matrix of shape (n_rec, n_don).
    a : np.ndarray
        Source (recipient) distribution, shape (n_rec,).
    b : np.ndarray
        Target (donor) distribution, shape (n_don,).

    Returns
    -------
    np.ndarray
        Optimal transport plan of shape (n_rec, n_don).
    """
    n_rec, n_don = cost_matrix.shape

    # For uniform marginals, we can use Hungarian algorithm
    # Need to handle case where n_rec != n_don

    if n_rec == n_don:
        # Standard assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        transport_plan = np.zeros((n_rec, n_don))
        transport_plan[row_ind, col_ind] = 1.0 / n_rec
    elif n_rec < n_don:
        # More donors than recipients
        # Each recipient gets mass 1/n_rec, each donor gets mass 1/n_don
        # We need fractional assignment

        # Use network simplex or iterative assignment
        transport_plan = _solve_unbalanced_emd(cost_matrix, a, b)
    else:
        # More recipients than donors
        transport_plan = _solve_unbalanced_emd(cost_matrix, a, b)

    return transport_plan


def _solve_unbalanced_emd(
    cost_matrix: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Solve EMD for unbalanced case (different number of sources and targets).

    Uses a discretization approach: replicate rows/columns to make the
    problem balanced, then solve with Hungarian algorithm.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Cost matrix of shape (n_rec, n_don).
    a : np.ndarray
        Source (recipient) distribution.
    b : np.ndarray
        Target (donor) distribution.

    Returns
    -------
    np.ndarray
        Optimal transport plan of shape (n_rec, n_don).
    """
    n_rec, n_don = cost_matrix.shape

    # Scale to integers for replication
    # Use LCM approach for exact representation
    lcm = np.lcm(n_rec, n_don)
    rep_rec = lcm // n_rec  # times to replicate each recipient
    rep_don = lcm // n_don  # times to replicate each donor

    # Create expanded cost matrix
    expanded_cost = np.repeat(cost_matrix, rep_don, axis=1)
    expanded_cost = np.repeat(expanded_cost, rep_rec, axis=0)

    # Solve balanced assignment
    row_ind, col_ind = linear_sum_assignment(expanded_cost)

    # Aggregate back to original dimensions
    transport_plan = np.zeros((n_rec, n_don))
    for r, c in zip(row_ind, col_ind):
        orig_r = r // rep_rec
        orig_c = c // rep_don
        transport_plan[orig_r, orig_c] += 1.0 / lcm

    return transport_plan


def _sinkhorn(
    cost_matrix: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    reg: float,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Solve entropy-regularized optimal transport using Sinkhorn algorithm.

    The Sinkhorn algorithm iteratively scales rows and columns of the
    kernel matrix K = exp(-C/reg) to satisfy the marginal constraints.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Cost matrix of shape (n_rec, n_don).
    a : np.ndarray
        Source (recipient) distribution, shape (n_rec,).
    b : np.ndarray
        Target (donor) distribution, shape (n_don,).
    reg : float
        Entropy regularization parameter.
    max_iter : int, default=1000
        Maximum number of Sinkhorn iterations.
    tol : float, default=1e-9
        Convergence tolerance.

    Returns
    -------
    np.ndarray
        Optimal transport plan of shape (n_rec, n_don).

    Notes
    -----
    The Sinkhorn algorithm solves:

        min_{T} <T, C> - reg * H(T)
        s.t. T @ 1 = a, T.T @ 1 = b, T >= 0

    where H(T) = -sum_{ij} T_{ij} * (log(T_{ij}) - 1) is the entropy.
    """
    n_rec, n_don = cost_matrix.shape

    # Stabilized log-domain Sinkhorn
    # K = exp(-C / reg)
    log_K = -cost_matrix / reg

    # Initialize dual variables
    log_u = np.zeros(n_rec)
    log_v = np.zeros(n_don)

    for iteration in range(max_iter):
        log_u_prev = log_u.copy()

        # Update u: u = a / (K @ v)
        # In log domain: log_u = log(a) - logsumexp(log_K + log_v, axis=1)
        log_Kv = _logsumexp(log_K + log_v[np.newaxis, :], axis=1)
        log_u = np.log(a + 1e-300) - log_Kv

        # Update v: v = b / (K.T @ u)
        # In log domain: log_v = log(b) - logsumexp(log_K.T + log_u, axis=1)
        log_Ku = _logsumexp(log_K.T + log_u[np.newaxis, :], axis=1)
        log_v = np.log(b + 1e-300) - log_Ku

        # Check convergence
        if np.max(np.abs(log_u - log_u_prev)) < tol:
            break

    # Compute transport plan: T = diag(u) @ K @ diag(v)
    # In log domain: log_T = log_u[:, None] + log_K + log_v[None, :]
    log_T = log_u[:, np.newaxis] + log_K + log_v[np.newaxis, :]
    transport_plan = np.exp(log_T)

    # Normalize to ensure exact marginals
    transport_plan = transport_plan / transport_plan.sum() * (a.sum())

    return transport_plan


def _logsumexp(x: np.ndarray, axis: int = None) -> np.ndarray:
    """
    Compute log(sum(exp(x))) in a numerically stable way.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axis : int, optional
        Axis along which to sum.

    Returns
    -------
    np.ndarray
        The log-sum-exp of x along the specified axis.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    result = x_max + np.log(
        np.sum(np.exp(x - x_max), axis=axis, keepdims=True)
    )
    return np.squeeze(result, axis=axis)
