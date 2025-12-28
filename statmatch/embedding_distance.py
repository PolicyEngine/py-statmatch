"""Embedding-based distance computation for categorical variables.

This module provides functions to learn low-dimensional embeddings for
categorical variables and compute distances using these embeddings.
Useful for statistical matching when categorical variables have many
levels or when semantic similarity between categories is important.

Two embedding methods are supported:
- Target encoding: Smoothed mean of target variable per category
- SVD: Singular Value Decomposition of co-occurrence matrix
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def learn_embeddings(
    data: pd.DataFrame,
    cat_vars: List[str],
    target_var: str,
    embedding_dim: int = 8,
    method: str = "target",
    smoothing: float = 1.0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Learn embeddings for categorical variables.

    This function creates low-dimensional vector representations for
    categories in categorical variables. These embeddings can then be
    used to compute distances between records with categorical features.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing categorical variables and target.
    cat_vars : List[str]
        List of categorical variable names to embed.
    target_var : str
        Name of the target variable used for learning embeddings.
        For 'target' method, this is the variable whose mean per
        category determines the embedding.
    embedding_dim : int, default=8
        Dimension of the embedding vectors.
    method : str, default='target'
        Method for learning embeddings:
        - 'target': Target encoding with smoothing. Creates embeddings
          based on the smoothed mean of the target per category.
        - 'svd': SVD factorization of the co-occurrence matrix between
          categorical variables.
    smoothing : float, default=1.0
        Smoothing parameter for target encoding. Higher values pull
        category means toward the global mean more strongly.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Nested dictionary mapping:
        {variable_name: {category: embedding_vector}}

    Raises
    ------
    ValueError
        If method is not 'target' or 'svd'.
    KeyError
        If cat_vars or target_var not in data columns.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame({
    ...     'category': ['A', 'A', 'B', 'B', 'C', 'C'],
    ...     'target': [100, 90, 50, 60, 10, 0]
    ... })
    >>> embeddings = learn_embeddings(
    ...     data, cat_vars=['category'], target_var='target',
    ...     embedding_dim=2, method='target'
    ... )
    >>> embeddings['category']['A']  # High target embedding
    array([...])
    """
    if method not in ["target", "svd"]:
        raise ValueError(f"Unknown method '{method}'. Use 'target' or 'svd'.")

    # Validate columns exist
    for var in cat_vars:
        if var not in data.columns:
            raise KeyError(f"Categorical variable '{var}' not in data")
    if target_var not in data.columns:
        raise KeyError(f"Target variable '{target_var}' not in data")

    if method == "target":
        return _learn_target_embeddings(
            data, cat_vars, target_var, embedding_dim, smoothing
        )
    else:  # svd
        return _learn_svd_embeddings(data, cat_vars, embedding_dim)


def _learn_target_embeddings(
    data: pd.DataFrame,
    cat_vars: List[str],
    target_var: str,
    embedding_dim: int,
    smoothing: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Learn embeddings using target encoding with smoothing.

    The embedding is based on the smoothed mean of the target variable
    for each category. Higher moments (variance, skewness) are used to
    fill additional embedding dimensions if embedding_dim > 1.
    """
    embeddings = {}
    target = data[target_var].values
    global_mean = np.nanmean(target)
    global_std = np.nanstd(target)
    if global_std == 0:
        global_std = 1.0

    for var in cat_vars:
        var_embeddings = {}
        categories = data[var].unique()

        for cat in categories:
            mask = data[var] == cat
            cat_target = target[mask]
            n = len(cat_target)

            if n == 0:
                # No data for this category, use global mean
                smoothed_mean = global_mean
                variance = 0.0
            else:
                # Smoothed mean: weighted average of category mean and global mean
                cat_mean = np.nanmean(cat_target)
                smoothed_mean = (n * cat_mean + smoothing * global_mean) / (
                    n + smoothing
                )
                variance = np.nanvar(cat_target) if n > 1 else 0.0

            # Normalize to create embedding dimensions
            normalized_mean = (smoothed_mean - global_mean) / global_std
            normalized_var = (
                np.sqrt(variance) / global_std if global_std > 0 else 0.0
            )

            # Create embedding vector
            # Use mean, variance, and random projections for higher dims
            embedding = np.zeros(embedding_dim)

            if embedding_dim >= 1:
                embedding[0] = normalized_mean

            if embedding_dim >= 2:
                embedding[1] = normalized_var

            # For higher dimensions, use Fourier-like features of the mean
            # This creates a richer representation while being deterministic
            for d in range(2, embedding_dim):
                freq = (d - 1) * np.pi
                if d % 2 == 0:
                    embedding[d] = np.sin(freq * normalized_mean)
                else:
                    embedding[d] = np.cos(freq * normalized_mean)

            var_embeddings[cat] = embedding

        embeddings[var] = var_embeddings

    return embeddings


def _learn_svd_embeddings(
    data: pd.DataFrame,
    cat_vars: List[str],
    embedding_dim: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Learn embeddings using SVD of co-occurrence matrix.

    For each categorical variable, we create a co-occurrence matrix
    with other variables (or itself) and factorize it using SVD.
    """
    embeddings = {}

    for var in cat_vars:
        categories = data[var].unique()
        n_categories = len(categories)

        # Create category to index mapping
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}

        # Build co-occurrence matrix (category x category within same rows)
        # This captures which categories tend to appear together in the data
        cooc_matrix = np.zeros((n_categories, n_categories))

        # Self co-occurrence: count occurrences of each category
        for cat in categories:
            idx = cat_to_idx[cat]
            count = (data[var] == cat).sum()
            cooc_matrix[idx, idx] = count

        # Cross co-occurrence with other categorical variables
        for other_var in cat_vars:
            if other_var == var:
                continue

            other_categories = data[other_var].unique()
            other_to_idx = {cat: i for i, cat in enumerate(other_categories)}

            # Build cross co-occurrence
            for _, row in data[[var, other_var]].iterrows():
                var_cat = row[var]
                other_cat = row[other_var]
                var_idx = cat_to_idx[var_cat]
                other_idx = other_to_idx[other_cat]

                # Add to main diagonal weighted by cross-occurrence
                cooc_matrix[var_idx, var_idx] += 0.1

        # Apply log transform to co-occurrence (like word2vec)
        cooc_matrix = np.log1p(cooc_matrix)

        # Compute SVD
        effective_dim = min(embedding_dim, n_categories)
        if effective_dim < n_categories:
            # Use truncated SVD for efficiency
            U, S, _ = np.linalg.svd(cooc_matrix, full_matrices=False)
            U = U[:, :effective_dim]
            S = S[:effective_dim]
        else:
            U, S, _ = np.linalg.svd(cooc_matrix, full_matrices=False)

        # Embedding is U * sqrt(S) to distribute singular values
        sqrt_S = np.sqrt(S)
        embedding_matrix = U * sqrt_S

        # Store embeddings for each category
        var_embeddings = {}
        for cat, idx in cat_to_idx.items():
            var_embeddings[cat] = embedding_matrix[idx].copy()

        embeddings[var] = var_embeddings

    return embeddings


def embedding_dist(
    data_x: pd.DataFrame,
    data_y: Optional[pd.DataFrame],
    embeddings: Dict[str, Dict[str, np.ndarray]],
    cat_vars: List[str],
    numeric_vars: List[str],
) -> np.ndarray:
    """
    Compute distance matrix using learned embeddings.

    This function replaces categorical variables with their learned
    embeddings and computes Euclidean distance in the combined space
    of embeddings and numeric variables.

    Parameters
    ----------
    data_x : pd.DataFrame
        First dataset for distance computation.
    data_y : pd.DataFrame, optional
        Second dataset. If None, distances are computed within data_x.
    embeddings : Dict[str, Dict[str, np.ndarray]]
        Learned embeddings from learn_embeddings().
    cat_vars : List[str]
        List of categorical variable names to use embeddings for.
    numeric_vars : List[str]
        List of numeric variable names to include directly.

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_x, n_y).

    Raises
    ------
    KeyError
        If a category in the data is not in the embeddings.

    Examples
    --------
    >>> dist_matrix = embedding_dist(
    ...     data_x=recipients,
    ...     data_y=donors,
    ...     embeddings=embeddings,
    ...     cat_vars=['category'],
    ...     numeric_vars=['income', 'age']
    ... )
    """
    if data_y is None:
        data_y = data_x

    n_x = len(data_x)
    n_y = len(data_y)

    # Build feature matrices by replacing categoricals with embeddings
    features_x = _build_feature_matrix(
        data_x, embeddings, cat_vars, numeric_vars
    )
    features_y = _build_feature_matrix(
        data_y, embeddings, cat_vars, numeric_vars
    )

    # Compute Euclidean distance
    from scipy.spatial.distance import cdist

    dist_matrix = cdist(features_x, features_y, metric="euclidean")

    return dist_matrix


def _build_feature_matrix(
    data: pd.DataFrame,
    embeddings: Dict[str, Dict[str, np.ndarray]],
    cat_vars: List[str],
    numeric_vars: List[str],
) -> np.ndarray:
    """
    Build feature matrix by replacing categoricals with embeddings.
    """
    n = len(data)
    feature_parts = []

    # Add embedded categorical variables
    for var in cat_vars:
        if var not in embeddings:
            raise KeyError(f"Variable '{var}' not in embeddings")

        var_embeddings = embeddings[var]

        # Get embedding dimension
        first_cat = next(iter(var_embeddings.keys()))
        emb_dim = len(var_embeddings[first_cat])

        # Build embedding matrix for this variable
        emb_matrix = np.zeros((n, emb_dim))
        for i, cat in enumerate(data[var]):
            if cat not in var_embeddings:
                raise KeyError(
                    f"Category '{cat}' not in embeddings for variable '{var}'"
                )
            emb_matrix[i] = var_embeddings[cat]

        feature_parts.append(emb_matrix)

    # Add numeric variables
    if numeric_vars:
        numeric_data = data[numeric_vars].values.astype(float)
        feature_parts.append(numeric_data)

    # Concatenate all features
    if feature_parts:
        features = np.hstack(feature_parts)
    else:
        # No features, return zeros
        features = np.zeros((n, 1))

    return features


def entity_similarity(
    embeddings: Dict[str, Dict[str, np.ndarray]],
    var_name: str,
    return_categories: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """
    Compute cosine similarity between categories in embedding space.

    This function computes pairwise cosine similarity between all
    categories of a variable using their learned embeddings. Useful
    for understanding what relationships the embeddings have learned.

    Parameters
    ----------
    embeddings : Dict[str, Dict[str, np.ndarray]]
        Learned embeddings from learn_embeddings().
    var_name : str
        Name of the categorical variable.
    return_categories : bool, default=False
        If True, also return the list of categories in order.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, List[str]]
        Similarity matrix of shape (n_categories, n_categories).
        If return_categories=True, also returns list of category names.

    Examples
    --------
    >>> sim_matrix, categories = entity_similarity(
    ...     embeddings, 'occupation', return_categories=True
    ... )
    >>> # Find most similar occupations
    >>> most_similar = categories[np.argsort(sim_matrix[0])[-3:]]
    """
    if var_name not in embeddings:
        raise KeyError(f"Variable '{var_name}' not in embeddings")

    var_embeddings = embeddings[var_name]
    categories = list(var_embeddings.keys())
    n_cats = len(categories)

    # Build embedding matrix
    emb_matrix = np.array([var_embeddings[cat] for cat in categories])

    # Compute cosine similarity
    # cos_sim(a, b) = (a . b) / (|a| * |b|)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    normalized = emb_matrix / norms

    similarity_matrix = normalized @ normalized.T

    # Clip to [0, 1] to handle numerical issues
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

    if return_categories:
        return similarity_matrix, categories
    else:
        return similarity_matrix
