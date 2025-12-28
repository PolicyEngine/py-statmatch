"""Implementation of RANDwNND.hotdeck (Random Distance Hot Deck) matching."""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def rand_hotdeck(
    data_rec: pd.DataFrame,
    data_don: pd.DataFrame,
    match_vars: Optional[List[str]] = None,
    don_class: Optional[str] = None,
    dist_fun: str = "manhattan",
    cut_don: str = "rot",
    k: Optional[Union[int, float]] = None,
    weight_don: Optional[str] = None,
    keep_t: bool = False,
) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """
    Implement the Random Distance Hot Deck (RANDwNND.hotdeck) method.

    For each recipient record, a subset of the closest donors is retained
    and then a donor is selected at random from this subset.

    Parameters
    ----------
    data_rec : pd.DataFrame
        The recipient data set.
    data_don : pd.DataFrame
        The donor data set.
    match_vars : Optional[List[str]], default=None
        List of variable names to use for matching. If None, random
        selection without distance calculation.
    don_class : Optional[str], default=None
        Variable name defining donation classes. If specified, matches are
        constrained within the same class.
    dist_fun : str, default="manhattan"
        Distance function to use. Options: "euclidean", "manhattan",
        "mahalanobis", "minimax", "gower".
    cut_don : str, default="rot"
        Method for selecting donor subset:
        - "rot": Use ceil(sqrt(n_donors)) closest donors
        - "span": Use ceil(n_donors * k) closest donors (k is proportion)
        - "exact": Use exactly k closest donors
        - "min": Use only minimum distance donor(s)
        - "k.dist": Use donors within distance k
    k : Optional[Union[int, float]], default=None
        Parameter for cut_don. Required for "span", "exact", "k.dist".
    weight_don : Optional[str], default=None
        Variable name for donor weights. If provided, donors are selected
        with probability proportional to their weights.
    keep_t : bool, default=False
        If True, print information about donation class processing.

    Returns
    -------
    Dict[str, Union[np.ndarray, pd.DataFrame]]
        Dictionary containing:
        - 'mtc.ids': DataFrame with recipient and donor IDs
        - 'sum.dist': DataFrame with distance summary statistics
        - 'noad': Array with number of available donors for each recipient
    """
    # Validate cut_don parameter
    valid_cut_don = ["rot", "span", "exact", "min", "k.dist"]
    if cut_don not in valid_cut_don:
        raise ValueError(
            f"Invalid cut_don '{cut_don}'. Must be one of {valid_cut_don}"
        )

    # Validate k parameter for methods that require it
    if cut_don in ["span", "exact", "k.dist"] and k is None:
        raise ValueError(f"k is required when cut_don='{cut_don}'")

    # Validate match_vars
    if match_vars is not None:
        for var in match_vars:
            if var not in data_rec.columns:
                raise ValueError(
                    f"Match variable '{var}' not found in recipient data"
                )
            if var not in data_don.columns:
                raise ValueError(
                    f"Match variable '{var}' not found in donor data"
                )

    # Validate don_class
    if don_class is not None:
        if don_class not in data_rec.columns:
            raise ValueError(
                f"Donation class '{don_class}' not found in recipient data"
            )
        if don_class not in data_don.columns:
            raise ValueError(
                f"Donation class '{don_class}' not found in donor data"
            )

    # Validate weight_don
    if weight_don is not None and weight_don not in data_don.columns:
        raise ValueError(
            f"Weight variable '{weight_don}' not found in donor data"
        )

    n_rec = len(data_rec)
    n_don = len(data_don)

    # Handle donation classes
    if don_class is None:
        # No donation classes - process all together
        if match_vars is None:
            # Random selection without matching
            return _random_selection(
                data_rec, data_don, weight_don, n_rec, n_don
            )
        else:
            # Distance-based matching
            return _rand_nnd_match(
                data_rec,
                data_don,
                match_vars,
                dist_fun,
                cut_don,
                k,
                weight_don,
            )
    else:
        # Process by donation class
        return _process_by_class(
            data_rec,
            data_don,
            match_vars,
            don_class,
            dist_fun,
            cut_don,
            k,
            weight_don,
            keep_t,
        )


def _random_selection(
    data_rec: pd.DataFrame,
    data_don: pd.DataFrame,
    weight_don: Optional[str],
    n_rec: int,
    n_don: int,
) -> Dict:
    """Random selection of donors without distance calculation."""
    if weight_don is None:
        # Equal probability selection
        donor_indices = np.random.choice(n_don, size=n_rec, replace=True)
    else:
        # Weighted selection
        weights = data_don[weight_don].values
        probs = weights / weights.sum()
        donor_indices = np.random.choice(
            n_don, size=n_rec, replace=True, p=probs
        )

    mtc_ids = pd.DataFrame(
        {
            "rec.id": np.arange(n_rec),
            "don.id": donor_indices,
        }
    )

    return {
        "mtc.ids": mtc_ids,
        "sum.dist": None,
        "noad": np.full(n_rec, n_don),
    }


def _rand_nnd_match(
    data_rec: pd.DataFrame,
    data_don: pd.DataFrame,
    match_vars: List[str],
    dist_fun: str,
    cut_don: str,
    k: Optional[Union[int, float]],
    weight_don: Optional[str],
) -> Dict:
    """Perform random distance-based matching."""
    n_rec = len(data_rec)
    n_don = len(data_don)

    # Extract matching variable data
    rec_data = data_rec[match_vars].values.astype(float)
    don_data = data_don[match_vars].values.astype(float)

    # Compute distance matrix
    dist_matrix = _compute_distances(rec_data, don_data, dist_fun)

    # Get weights
    if weight_don is None:
        weights = np.ones(n_don)
    else:
        weights = data_don[weight_don].values

    # Determine k based on cut_don
    effective_k = _get_effective_k(cut_don, k, n_don)

    # Process each recipient
    donor_indices = np.zeros(n_rec, dtype=int)
    noad = np.zeros(n_rec, dtype=int)
    min_dist = np.zeros(n_rec)
    max_dist = np.zeros(n_rec)
    sd_dist = np.zeros(n_rec)
    cut_dist = np.zeros(n_rec)
    dist_rd = np.zeros(n_rec)

    for i in range(n_rec):
        distances = dist_matrix[i, :]

        # Find subset of closest donors
        subset_indices, subset_distances, subset_weights, cut_d = (
            _get_donor_subset(distances, weights, cut_don, k, effective_k)
        )

        noad[i] = len(subset_indices)

        if noad[i] == 0:
            donor_indices[i] = -1
            dist_rd[i] = np.nan
        elif noad[i] == 1:
            donor_indices[i] = subset_indices[0]
            dist_rd[i] = subset_distances[0]
        else:
            # Random selection from subset (possibly weighted)
            probs = subset_weights / subset_weights.sum()
            choice_idx = np.random.choice(len(subset_indices), p=probs)
            donor_indices[i] = subset_indices[choice_idx]
            dist_rd[i] = subset_distances[choice_idx]

        min_dist[i] = np.min(distances)
        max_dist[i] = np.max(distances)
        sd_dist[i] = np.std(distances)
        cut_dist[i] = cut_d

    mtc_ids = pd.DataFrame(
        {
            "rec.id": np.arange(n_rec),
            "don.id": donor_indices,
        }
    )

    sum_dist = pd.DataFrame(
        {
            "min": min_dist,
            "max": max_dist,
            "sd": sd_dist,
            "cut": cut_dist,
            "dist.rd": dist_rd,
        }
    )

    return {
        "mtc.ids": mtc_ids,
        "sum.dist": sum_dist,
        "noad": noad,
    }


def _process_by_class(
    data_rec: pd.DataFrame,
    data_don: pd.DataFrame,
    match_vars: Optional[List[str]],
    don_class: str,
    dist_fun: str,
    cut_don: str,
    k: Optional[Union[int, float]],
    weight_don: Optional[str],
    keep_t: bool,
) -> Dict:
    """Process matching by donation class."""
    n_rec = len(data_rec)

    # Get unique classes
    rec_classes = data_rec[don_class].values
    don_classes = data_don[don_class].values
    unique_classes = np.unique(rec_classes)

    # Check that all recipient classes have donors
    for cls in unique_classes:
        if not np.any(don_classes == cls):
            raise ValueError(
                f"Donation class '{cls}' has no donors in donor data"
            )

    # Initialize result arrays
    donor_indices = np.zeros(n_rec, dtype=int)
    noad = np.zeros(n_rec, dtype=int)
    min_dist = np.zeros(n_rec)
    max_dist = np.zeros(n_rec)
    sd_dist = np.zeros(n_rec)
    cut_dist = np.zeros(n_rec)
    dist_rd = np.zeros(n_rec)

    # Process each class
    for cls in unique_classes:
        if keep_t:
            print(f"Selecting donors for donation class: {cls}")

        rec_mask = rec_classes == cls
        don_mask = don_classes == cls

        rec_idx = np.where(rec_mask)[0]
        don_idx = np.where(don_mask)[0]

        rec_subset = data_rec.iloc[rec_idx]
        don_subset = data_don.iloc[don_idx]

        n_rec_cls = len(rec_subset)
        n_don_cls = len(don_subset)

        if match_vars is None:
            # Random selection within class
            if weight_don is None:
                local_indices = np.random.choice(
                    n_don_cls, size=n_rec_cls, replace=True
                )
            else:
                weights = don_subset[weight_don].values
                probs = weights / weights.sum()
                local_indices = np.random.choice(
                    n_don_cls, size=n_rec_cls, replace=True, p=probs
                )

            for j, rec_i in enumerate(rec_idx):
                donor_indices[rec_i] = don_idx[local_indices[j]]
                noad[rec_i] = n_don_cls
        else:
            # Distance-based matching within class
            rec_data = rec_subset[match_vars].values.astype(float)
            don_data = don_subset[match_vars].values.astype(float)

            dist_matrix = _compute_distances(rec_data, don_data, dist_fun)

            if weight_don is None:
                weights = np.ones(n_don_cls)
            else:
                weights = don_subset[weight_don].values

            effective_k = _get_effective_k(cut_don, k, n_don_cls)

            for j, rec_i in enumerate(rec_idx):
                distances = dist_matrix[j, :]

                subset_indices, subset_distances, subset_weights, cut_d = (
                    _get_donor_subset(
                        distances, weights, cut_don, k, effective_k
                    )
                )

                noad[rec_i] = len(subset_indices)
                min_dist[rec_i] = np.min(distances)
                max_dist[rec_i] = np.max(distances)
                sd_dist[rec_i] = np.std(distances)
                cut_dist[rec_i] = cut_d

                if noad[rec_i] == 0:
                    donor_indices[rec_i] = -1
                    dist_rd[rec_i] = np.nan
                elif noad[rec_i] == 1:
                    donor_indices[rec_i] = don_idx[subset_indices[0]]
                    dist_rd[rec_i] = subset_distances[0]
                else:
                    probs = subset_weights / subset_weights.sum()
                    choice_idx = np.random.choice(len(subset_indices), p=probs)
                    donor_indices[rec_i] = don_idx[subset_indices[choice_idx]]
                    dist_rd[rec_i] = subset_distances[choice_idx]

    mtc_ids = pd.DataFrame(
        {
            "rec.id": np.arange(n_rec),
            "don.id": donor_indices,
        }
    )

    if match_vars is None:
        sum_dist = None
    else:
        sum_dist = pd.DataFrame(
            {
                "min": min_dist,
                "max": max_dist,
                "sd": sd_dist,
                "cut": cut_dist,
                "dist.rd": dist_rd,
            }
        )

    return {
        "mtc.ids": mtc_ids,
        "sum.dist": sum_dist,
        "noad": noad,
    }


def _compute_distances(
    rec_data: np.ndarray,
    don_data: np.ndarray,
    dist_fun: str,
) -> np.ndarray:
    """Compute distance matrix between recipients and donors."""
    dist_fun_lower = dist_fun.lower()

    if dist_fun_lower == "manhattan":
        return cdist(rec_data, don_data, metric="cityblock")
    elif dist_fun_lower == "euclidean":
        return cdist(rec_data, don_data, metric="euclidean")
    elif dist_fun_lower == "minimax":
        return cdist(rec_data, don_data, metric="chebyshev")
    elif dist_fun_lower == "mahalanobis":
        combined_data = np.vstack([rec_data, don_data])
        cov_matrix = np.cov(combined_data.T)
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
        inv_cov = np.linalg.inv(cov_matrix)
        return cdist(rec_data, don_data, metric="mahalanobis", VI=inv_cov)
    elif dist_fun_lower == "gower":
        # Simplified Gower distance for numeric data
        # Normalize each variable to [0, 1] range
        combined = np.vstack([rec_data, don_data])
        ranges = np.ptp(combined, axis=0)
        ranges[ranges == 0] = 1  # Avoid division by zero
        rec_norm = rec_data / ranges
        don_norm = don_data / ranges
        return (
            cdist(rec_norm, don_norm, metric="cityblock") / rec_data.shape[1]
        )
    else:
        # Default to Euclidean
        return cdist(rec_data, don_data, metric="euclidean")


def _get_effective_k(
    cut_don: str,
    k: Optional[Union[int, float]],
    n_don: int,
) -> int:
    """Calculate effective k based on cut_don method."""
    if cut_don == "rot":
        return int(np.ceil(np.sqrt(n_don)))
    elif cut_don == "span":
        return int(np.ceil(n_don * k))
    elif cut_don == "exact":
        return int(k)
    elif cut_don == "min":
        return 1  # Will be handled specially
    elif cut_don == "k.dist":
        return n_don  # Will be handled specially
    return n_don


def _get_donor_subset(
    distances: np.ndarray,
    weights: np.ndarray,
    cut_don: str,
    k: Optional[Union[int, float]],
    effective_k: int,
) -> tuple:
    """Get subset of closest donors based on cut_don method."""
    n_don = len(distances)

    if cut_don == "min":
        # Only donors with minimum distance
        min_d = np.min(distances)
        mask = distances == min_d
        subset_indices = np.where(mask)[0]
        subset_distances = distances[mask]
        subset_weights = weights[mask]
        cut_d = min_d
    elif cut_don == "k.dist":
        # Donors within distance k
        mask = distances <= k
        subset_indices = np.where(mask)[0]
        subset_distances = distances[mask]
        subset_weights = weights[mask]
        cut_d = k
    else:
        # Get k closest donors
        actual_k = min(effective_k, n_don)
        sorted_indices = np.argsort(distances)[:actual_k]
        subset_indices = sorted_indices
        subset_distances = distances[sorted_indices]
        subset_weights = weights[sorted_indices]
        cut_d = distances[sorted_indices[-1]] if len(sorted_indices) > 0 else 0

    return subset_indices, subset_distances, subset_weights, cut_d
