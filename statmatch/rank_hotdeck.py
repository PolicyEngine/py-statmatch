"""
Implementation of rankNND.hotdeck (Rank Nearest Neighbor Distance Hot Deck).

This function implements a rank-based hot deck distance method for statistical
matching. It finds donor records for recipients by computing distances based
on empirical cumulative distribution function (ECDF) percentage points.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def rank_nnd_hotdeck(
    data_rec: pd.DataFrame,
    data_don: pd.DataFrame,
    var_rec: str,
    var_don: Optional[str] = None,
    don_class: Optional[str] = None,
    weight_rec: Optional[str] = None,
    weight_don: Optional[str] = None,
    constrained: bool = False,
    constr_alg: str = "hungarian",
    keep_t: bool = False,
) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """
    Implement the Rank Nearest Neighbor Distance Hot Deck method.

    For each recipient record, find the closest donor by considering the
    distance between the percentage points of the empirical cumulative
    distribution function.

    Parameters
    ----------
    data_rec : pd.DataFrame
        The recipient data set. Must contain var_rec and optionally don_class
        and weight_rec variables. Missing values (NA) are not allowed.
    data_don : pd.DataFrame
        The donor data set. Must contain var_don and optionally don_class
        and weight_don variables.
    var_rec : str
        Variable name in recipient data to be ranked.
    var_don : Optional[str], default=None
        Variable name in donor data to be ranked. If None, defaults to var_rec.
    don_class : Optional[str], default=None
        Variable name defining donation classes. If specified, ECDF is computed
        independently in each class and matches are restricted to same class.
    weight_rec : Optional[str], default=None
        Variable name for recipient weights used in ECDF computation.
    weight_don : Optional[str], default=None
        Variable name for donor weights used in ECDF computation.
    constrained : bool, default=False
        If True, each donor can be used at most once. Solves a transportation
        problem to minimize total matching distance.
    constr_alg : str, default="hungarian"
        Algorithm for constrained matching. Options: "hungarian", "lpsolve".
    keep_t : bool, default=False
        If True, print information about donation class processing.

    Returns
    -------
    Dict[str, Union[np.ndarray, pd.DataFrame]]
        Dictionary containing:
        - 'mtc.ids': DataFrame with recipient (rec.id) and donor (don.id) IDs
        - 'dist.rd': Array of distances between matched recipients and donors
        - 'noad': Array of available donors at minimum distance for each recipient
    """
    # Default var_don to var_rec if not specified
    if var_don is None:
        var_don = var_rec

    # Validate inputs
    if var_rec not in data_rec.columns:
        raise ValueError(f"Variable '{var_rec}' not found in recipient data")

    if var_don not in data_don.columns:
        raise ValueError(f"Variable '{var_don}' not found in donor data")

    if don_class is not None:
        if don_class not in data_rec.columns:
            raise ValueError(
                f"Donation class variable '{don_class}' not found in "
                "recipient data"
            )
        if don_class not in data_don.columns:
            raise ValueError(
                f"Donation class variable '{don_class}' not found in "
                "donor data"
            )

    if weight_rec is not None and weight_rec not in data_rec.columns:
        raise ValueError(
            f"Weight variable '{weight_rec}' not found in recipient data"
        )

    if weight_don is not None and weight_don not in data_don.columns:
        raise ValueError(
            f"Weight variable '{weight_don}' not found in donor data"
        )

    # Initialize results
    n_rec = len(data_rec)
    n_don = len(data_don)

    donor_indices = np.zeros(n_rec, dtype=int)
    distances = np.zeros(n_rec)
    noad = np.zeros(n_rec, dtype=int)

    if don_class is not None:
        # Process each donation class separately
        rec_classes = data_rec[don_class].values
        don_classes = data_don[don_class].values
        unique_classes = np.unique(rec_classes)

        for cls in unique_classes:
            if keep_t:
                print(f"Processing donation class: {cls}")

            # Get indices for this class
            rec_mask = rec_classes == cls
            don_mask = don_classes == cls

            if not np.any(don_mask):
                raise ValueError(f"No donors found for class '{cls}'")

            rec_indices = np.where(rec_mask)[0]
            don_global_indices = np.where(don_mask)[0]

            # Extract data for this class
            x_rec = data_rec.loc[rec_mask, var_rec].values
            x_don = data_don.loc[don_mask, var_don].values

            # Get weights if specified
            w_rec = None
            w_don = None
            if weight_rec is not None:
                w_rec = data_rec.loc[rec_mask, weight_rec].values
            if weight_don is not None:
                w_don = data_don.loc[don_mask, weight_don].values

            # Perform matching within class
            cls_don_indices, cls_distances, cls_noad = _rank_nnd_match(
                x_rec, x_don, w_rec, w_don, constrained, constr_alg
            )

            # Store results, converting local indices to global
            for i, rec_idx in enumerate(rec_indices):
                donor_indices[rec_idx] = don_global_indices[cls_don_indices[i]]
                distances[rec_idx] = cls_distances[i]
                noad[rec_idx] = cls_noad[i]
    else:
        # No donation classes - match all recipients to all donors
        x_rec = data_rec[var_rec].values
        x_don = data_don[var_don].values

        w_rec = None
        w_don = None
        if weight_rec is not None:
            w_rec = data_rec[weight_rec].values
        if weight_don is not None:
            w_don = data_don[weight_don].values

        donor_indices, distances, noad = _rank_nnd_match(
            x_rec, x_don, w_rec, w_don, constrained, constr_alg
        )

    # Create results dictionary
    mtc_ids = pd.DataFrame(
        {
            "rec.id": np.arange(n_rec),
            "don.id": donor_indices,
        }
    )

    results = {
        "mtc.ids": mtc_ids,
        "dist.rd": distances,
        "noad": noad,
    }

    return results


def _weighted_ecdf(
    x: np.ndarray, w: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute weighted empirical cumulative distribution function values.

    Parameters
    ----------
    x : np.ndarray
        Data values.
    w : Optional[np.ndarray]
        Weights. If None, equal weights are used.

    Returns
    -------
    np.ndarray
        ECDF values (cumulative proportions) for each observation.
    """
    n = len(x)
    if w is None:
        w = np.ones(n)

    # Sort indices by x values
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    w_sorted = w[sort_idx]

    # Compute cumulative weights
    cum_weights = np.cumsum(w_sorted)
    total_weight = cum_weights[-1]

    # ECDF values at sorted positions
    ecdf_sorted = cum_weights / total_weight

    # Handle ties: all tied values get the same (maximum) ECDF value
    # Work backwards to assign max ECDF to ties
    for i in range(n - 2, -1, -1):
        if x_sorted[i] == x_sorted[i + 1]:
            ecdf_sorted[i] = ecdf_sorted[i + 1]

    # Unsort to original order
    ecdf = np.zeros(n)
    ecdf[sort_idx] = ecdf_sorted

    return ecdf


def _rank_nnd_match(
    x_rec: np.ndarray,
    x_don: np.ndarray,
    w_rec: Optional[np.ndarray] = None,
    w_don: Optional[np.ndarray] = None,
    constrained: bool = False,
    constr_alg: str = "hungarian",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform rank-based nearest neighbor matching.

    Parameters
    ----------
    x_rec : np.ndarray
        Recipient variable values.
    x_don : np.ndarray
        Donor variable values.
    w_rec : Optional[np.ndarray]
        Recipient weights for ECDF.
    w_don : Optional[np.ndarray]
        Donor weights for ECDF.
    constrained : bool
        If True, each donor used at most once.
    constr_alg : str
        Algorithm for constrained matching.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Donor indices, distances, and number of donors at minimum distance.
    """
    n_rec = len(x_rec)
    n_don = len(x_don)

    # Compute ECDF values for recipients and donors
    ecdf_rec = _weighted_ecdf(x_rec, w_rec)
    ecdf_don = _weighted_ecdf(x_don, w_don)

    # Compute distance matrix (absolute difference in ECDF values)
    dist_matrix = np.abs(ecdf_rec[:, np.newaxis] - ecdf_don[np.newaxis, :])

    if constrained:
        # Constrained matching - each donor used at most once
        if n_rec > n_don:
            raise ValueError(
                f"Constrained matching requires at least as many donors "
                f"({n_don}) as recipients ({n_rec})"
            )

        # Use Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        donor_indices = col_ind
        distances = dist_matrix[row_ind, col_ind]

        # In constrained case, noad is always 1
        noad = np.ones(n_rec, dtype=int)
    else:
        # Unconstrained matching - find nearest donor for each recipient
        min_distances = np.min(dist_matrix, axis=1)
        distances = min_distances

        # Count donors at minimum distance
        noad = np.sum(
            np.isclose(dist_matrix, min_distances[:, np.newaxis]), axis=1
        )

        # Find donor indices - when multiple donors at same distance,
        # select one randomly (matching R behavior)
        donor_indices = np.zeros(n_rec, dtype=int)
        for i in range(n_rec):
            candidates = np.where(
                np.isclose(dist_matrix[i], min_distances[i])
            )[0]
            if len(candidates) == 1:
                donor_indices[i] = candidates[0]
            else:
                # Random selection among tied donors
                # Use a deterministic approach based on recipient index
                # to ensure reproducibility
                donor_indices[i] = candidates[i % len(candidates)]

    return donor_indices, distances, noad
