"""Implementation of NND.hotdeck (Nearest Neighbor Distance Hot Deck) matching."""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def nnd_hotdeck(
    data_rec: pd.DataFrame,
    data_don: pd.DataFrame,
    match_vars: List[str],
    don_class: Optional[str] = None,
    dist_fun: str = "euclidean",
    cut_don: Optional[str] = None,
    k: Optional[int] = None,
    w_don: Optional[Union[np.ndarray, List[float]]] = None,
    w_rec: Optional[Union[np.ndarray, List[float]]] = None,
    constr_alg: Optional[str] = None,
) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """
    Implement the Nearest Neighbor Distance Hot Deck (NND.hotdeck) method.

    This function finds the nearest neighbor donors for each recipient based on
    matching variables, optionally within donation classes.

    Parameters
    ----------
    data_rec : pd.DataFrame
        The recipient data set.
    data_don : pd.DataFrame
        The donor data set.
    match_vars : List[str]
        List of variable names to use for matching.
    don_class : Optional[str], default=None
        Variable name defining donation classes. If specified, matches are
        constrained within the same class.
    dist_fun : str, default="euclidean"
        Distance function to use. Options include: "euclidean", "manhattan",
        "mahalanobis", "minimax", "cosine", etc.
    cut_don : Optional[str], default=None
        Variable name to use for cutting the donor pool (not implemented yet).
    k : Optional[int], default=None
        Maximum number of times each donor can be used (for constrained matching).
    w_don : Optional[Union[np.ndarray, List[float]]], default=None
        Weights for donor units.
    w_rec : Optional[Union[np.ndarray, List[float]]], default=None
        Weights for recipient units.
    constr_alg : Optional[str], default=None
        Algorithm for constrained matching. Options: "lpsolve", "hungarian".

    Returns
    -------
    Dict[str, Union[np.ndarray, pd.DataFrame]]
        Dictionary containing:
        - 'mtc.ids': DataFrame with recipient and donor IDs
        - 'noad.index': Array of donor indices for each recipient (0-based)
        - 'dist.rd': Array of distances between matched recipients and donors
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

    if don_class and don_class not in data_rec.columns:
        raise ValueError(
            f"Donation class variable '{don_class}' not found in recipient data"
        )

    if don_class and don_class not in data_don.columns:
        raise ValueError(
            f"Donation class variable '{don_class}' not found in donor data"
        )

    # Initialize results
    n_rec = len(data_rec)
    n_don = len(data_don)

    # Arrays to store results
    donor_indices = np.zeros(n_rec, dtype=int)
    distances = np.zeros(n_rec)

    # Handle donation classes
    if don_class:
        # Get unique classes
        rec_classes = data_rec[don_class].values
        don_classes = data_don[don_class].values

        # Process each recipient maintaining original order
        for i in range(n_rec):
            cls = rec_classes[i]

            # Get indices for this class
            don_mask = don_classes == cls

            if not np.any(don_mask):
                raise ValueError(f"No donors found for class '{cls}'")

            # Get data for this recipient and donors of same class
            rec_data_i = data_rec.iloc[[i]][match_vars].values
            don_data_cls = data_don.loc[don_mask, match_vars].values

            # Find match for this recipient
            cls_indices, cls_distances = _find_matches(
                rec_data_i,
                don_data_cls,
                dist_fun,
                k if k is None else 1,
                constr_alg,
            )

            # Convert local indices to global indices
            don_idx_global = np.where(don_mask)[0]
            donor_indices[i] = don_idx_global[cls_indices[0]]
            distances[i] = cls_distances[0]
    else:
        # No donation classes - match all recipients to all donors
        rec_data = data_rec[match_vars].values
        don_data = data_don[match_vars].values

        donor_indices, distances = _find_matches(
            rec_data, don_data, dist_fun, k, constr_alg
        )

    # Create results dictionary
    mtc_ids = pd.DataFrame(
        {"rec.id": data_rec.index, "don.id": data_don.index[donor_indices]}
    )

    results = {
        "mtc.ids": mtc_ids,
        "noad.index": donor_indices,
        "dist.rd": distances,
    }

    return results


def _find_matches(
    rec_data: np.ndarray,
    don_data: np.ndarray,
    dist_fun: str,
    k: Optional[int] = None,
    constr_alg: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find nearest neighbor matches between recipients and donors.

    Parameters
    ----------
    rec_data : np.ndarray
        Recipient data matrix (n_rec x n_vars).
    don_data : np.ndarray
        Donor data matrix (n_don x n_vars).
    dist_fun : str
        Distance function name.
    k : Optional[int]
        Maximum times each donor can be used.
    constr_alg : Optional[str]
        Constrained matching algorithm.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Donor indices and distances for each recipient.
    """
    n_rec = rec_data.shape[0]
    n_don = don_data.shape[0]

    # Handle missing values by imputing with column means
    if np.any(np.isnan(rec_data)) or np.any(np.isnan(don_data)):
        # Compute column means ignoring NaN
        col_means = np.nanmean(np.vstack([rec_data, don_data]), axis=0)

        # Fill NaN values
        rec_data = rec_data.copy()
        don_data = don_data.copy()

        for i in range(rec_data.shape[1]):
            rec_data[np.isnan(rec_data[:, i]), i] = col_means[i]
            don_data[np.isnan(don_data[:, i]), i] = col_means[i]

    # Compute distance matrix
    if dist_fun.lower() == "manhattan":
        dist_matrix = cdist(rec_data, don_data, metric="cityblock")
    elif dist_fun.lower() in ["euclidean", "chebyshev", "cosine"]:
        dist_matrix = cdist(rec_data, don_data, metric=dist_fun.lower())
    elif dist_fun.lower() == "minimax":
        # Minimax distance is the same as Chebyshev
        dist_matrix = cdist(rec_data, don_data, metric="chebyshev")
    elif dist_fun.lower() == "mahalanobis":
        # For Mahalanobis, we need the covariance matrix
        combined_data = np.vstack([rec_data, don_data])
        cov_matrix = np.cov(combined_data.T)
        # Add small value to diagonal for numerical stability
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
        inv_cov = np.linalg.inv(cov_matrix)

        dist_matrix = np.zeros((n_rec, n_don))
        for i in range(n_rec):
            for j in range(n_don):
                diff = rec_data[i] - don_data[j]
                dist_matrix[i, j] = np.sqrt(diff @ inv_cov @ diff)
    else:
        # Default to Euclidean
        dist_matrix = cdist(rec_data, don_data, metric="euclidean")

    # Find matches
    if k is not None and constr_alg is not None:
        # Constrained matching
        donor_indices, distances = _constrained_matching(
            dist_matrix, k, constr_alg
        )
    else:
        # Unconstrained matching - find nearest neighbor for each recipient
        donor_indices = np.argmin(dist_matrix, axis=1)
        distances = dist_matrix[np.arange(n_rec), donor_indices]

    return donor_indices, distances


def _constrained_matching(
    dist_matrix: np.ndarray, k: int, algorithm: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform constrained matching where each donor is used at most k times.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Distance matrix (n_rec x n_don).
    k : int
        Maximum times each donor can be used.
    algorithm : str
        Algorithm to use ('lpsolve' or 'hungarian').

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Donor indices and distances for each recipient.
    """
    n_rec, n_don = dist_matrix.shape

    if algorithm.lower() in ["lpsolve", "hungarian"]:
        # Create expanded distance matrix
        # Each donor appears k times
        expanded_dist = np.tile(dist_matrix, (1, k))

        # Use Hungarian algorithm for assignment
        row_ind, col_ind = linear_sum_assignment(expanded_dist)

        # Convert column indices back to original donor indices
        donor_indices = col_ind % n_don

        # Get corresponding distances
        distances = dist_matrix[row_ind, donor_indices]
    else:
        # Fallback to greedy algorithm
        donor_indices = np.zeros(n_rec, dtype=int)
        distances = np.zeros(n_rec)
        donor_usage = np.zeros(n_don, dtype=int)

        # Sort recipients by their minimum distance to any donor
        min_dists = np.min(dist_matrix, axis=1)
        rec_order = np.argsort(min_dists)

        for rec_idx in rec_order:
            # Find donors that haven't exceeded usage limit
            available_donors = donor_usage < k

            if not np.any(available_donors):
                raise ValueError(
                    "Not enough donor capacity for all recipients"
                )

            # Find nearest available donor
            available_dists = dist_matrix[rec_idx].copy()
            available_dists[~available_donors] = np.inf

            best_donor = np.argmin(available_dists)
            donor_indices[rec_idx] = best_donor
            distances[rec_idx] = available_dists[best_donor]
            donor_usage[best_donor] += 1

    return donor_indices, distances
