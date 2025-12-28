"""Propensity score matching for statistical matching."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler


def estimate_propensity(
    data_rec: pd.DataFrame,
    data_don: pd.DataFrame,
    match_vars: List[str],
    estimator: str = "logistic",
    cv: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate propensity scores for donor and recipient datasets.

    Trains a model to predict whether an observation is from the donor
    dataset (1) or recipient dataset (0). The propensity score is the
    predicted probability of being a donor.

    Parameters
    ----------
    data_rec : pd.DataFrame
        The recipient data set.
    data_don : pd.DataFrame
        The donor data set.
    match_vars : List[str]
        List of variable names to use for propensity estimation.
    estimator : str, default="logistic"
        The estimator to use. Options:
        - "logistic": Logistic regression (interpretable, fast)
        - "gbm": Gradient boosting machine (handles non-linearity)
        - "random_forest": Random forest classifier
        - "neural_net": Multi-layer perceptron classifier
    cv : Optional[int], default=None
        Number of cross-validation folds. If None, no cross-validation
        is used and predictions are made on the training data.
    random_state : Optional[int], default=None
        Random state for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Propensity scores for recipients and donors respectively.
        Each array contains probabilities in [0, 1].

    Raises
    ------
    ValueError
        If an unknown estimator is specified.
    """
    # Validate estimator
    valid_estimators = ["logistic", "gbm", "random_forest", "neural_net"]
    if estimator not in valid_estimators:
        raise ValueError(
            f"Unknown estimator '{estimator}'. "
            f"Valid options are: {valid_estimators}"
        )

    # Combine data with labels
    X_rec = data_rec[match_vars].values
    X_don = data_don[match_vars].values

    X = np.vstack([X_rec, X_don])
    y = np.concatenate([np.zeros(len(X_rec)), np.ones(len(X_don))])

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create estimator
    model = _create_estimator(estimator, random_state)

    # Fit and predict
    if cv is not None and cv > 1:
        # Cross-validated predictions
        proba = cross_val_predict(
            model, X_scaled, y, cv=cv, method="predict_proba"
        )
        # Get probability of being donor (class 1)
        scores = proba[:, 1]
    else:
        # Fit on all data and predict
        model.fit(X_scaled, y)
        scores = model.predict_proba(X_scaled)[:, 1]

    # Split back to recipient and donor scores
    n_rec = len(X_rec)
    scores_rec = scores[:n_rec]
    scores_don = scores[n_rec:]

    return scores_rec, scores_don


def _create_estimator(estimator: str, random_state: Optional[int] = None):
    """
    Create a scikit-learn estimator based on the name.

    Parameters
    ----------
    estimator : str
        Name of the estimator.
    random_state : Optional[int]
        Random state for reproducibility.

    Returns
    -------
    Estimator
        A scikit-learn classifier.
    """
    if estimator == "logistic":
        return LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver="lbfgs",
        )
    elif estimator == "gbm":
        return GradientBoostingClassifier(
            random_state=random_state,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
        )
    elif estimator == "random_forest":
        return RandomForestClassifier(
            random_state=random_state,
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=5,
        )
    elif estimator == "neural_net":
        return MLPClassifier(
            random_state=random_state,
            hidden_layer_sizes=(50, 25),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
        )
    else:
        raise ValueError(f"Unknown estimator: {estimator}")


def propensity_hotdeck(
    data_rec: pd.DataFrame,
    data_don: pd.DataFrame,
    match_vars: List[str],
    estimator: str = "logistic",
    caliper: Optional[float] = None,
    n_neighbors: int = 1,
    random_state: Optional[int] = None,
) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """
    Perform propensity score matching for statistical matching.

    This function estimates propensity scores using ML models and matches
    recipients to donors based on similarity in propensity scores rather
    than raw covariates.

    Parameters
    ----------
    data_rec : pd.DataFrame
        The recipient data set.
    data_don : pd.DataFrame
        The donor data set.
    match_vars : List[str]
        List of variable names to use for propensity estimation.
    estimator : str, default="logistic"
        The estimator to use for propensity score estimation.
        Options: "logistic", "gbm", "random_forest", "neural_net"
    caliper : Optional[float], default=None
        Maximum distance in propensity score for a match. If specified,
        matches beyond this distance are marked as unmatched (NaN).
    n_neighbors : int, default=1
        Number of nearest neighbors to consider for matching.
        Currently only n_neighbors=1 returns a single match per recipient.
    random_state : Optional[int], default=None
        Random state for reproducibility.

    Returns
    -------
    Dict[str, Union[np.ndarray, pd.DataFrame]]
        Dictionary containing:
        - 'mtc.ids': DataFrame with recipient and donor IDs
        - 'noad.index': Array of donor indices for each recipient (0-based)
        - 'dist.rd': Array of propensity score distances between matches
        - 'ps.rec': Propensity scores for recipients
        - 'ps.don': Propensity scores for donors

    Raises
    ------
    ValueError
        If match variables are not found in the data or if data is empty.
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

    if len(data_don) == 0:
        raise ValueError("Donor data is empty")

    if len(data_rec) == 0:
        raise ValueError("Recipient data is empty")

    n_rec = len(data_rec)
    n_don = len(data_don)

    # Estimate propensity scores
    ps_rec, ps_don = estimate_propensity(
        data_rec=data_rec,
        data_don=data_don,
        match_vars=match_vars,
        estimator=estimator,
        random_state=random_state,
    )

    # Compute distance matrix based on propensity scores
    # Distance is the absolute difference in propensity scores
    ps_rec_2d = ps_rec.reshape(-1, 1)
    ps_don_2d = ps_don.reshape(1, -1)
    dist_matrix = np.abs(ps_rec_2d - ps_don_2d)

    # Find nearest neighbors
    if n_neighbors == 1:
        donor_indices = np.argmin(dist_matrix, axis=1)
        distances = dist_matrix[np.arange(n_rec), donor_indices]
    else:
        # Get k nearest neighbors and randomly select one
        sorted_indices = np.argsort(dist_matrix, axis=1)[:, :n_neighbors]
        rng = np.random.default_rng(random_state)
        selected = rng.integers(0, n_neighbors, size=n_rec)
        donor_indices = sorted_indices[np.arange(n_rec), selected]
        distances = dist_matrix[np.arange(n_rec), donor_indices]

    # Apply caliper if specified
    if caliper is not None:
        # Mark matches beyond caliper as unmatched
        unmatched = distances > caliper
        distances = distances.astype(float)  # Ensure float for NaN
        distances[unmatched] = np.nan
        donor_indices = donor_indices.astype(float)
        donor_indices[unmatched] = np.nan

    # Convert indices back to int for matched cases
    if caliper is not None:
        matched_mask = ~np.isnan(donor_indices)
        donor_indices_int = np.full(n_rec, -1, dtype=int)
        donor_indices_int[matched_mask] = donor_indices[matched_mask].astype(
            int
        )
    else:
        donor_indices_int = donor_indices.astype(int)

    # Create results dictionary
    # Handle case where some may be unmatched
    if caliper is not None:
        don_ids = np.where(
            donor_indices_int >= 0,
            data_don.index[donor_indices_int],
            np.nan,
        )
    else:
        don_ids = data_don.index[donor_indices_int]

    mtc_ids = pd.DataFrame({"rec.id": data_rec.index, "don.id": don_ids})

    results = {
        "mtc.ids": mtc_ids,
        "noad.index": donor_indices_int,
        "dist.rd": distances,
        "ps.rec": ps_rec,
        "ps.don": ps_don,
    }

    return results
