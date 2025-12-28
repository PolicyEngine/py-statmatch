"""Sample utility functions for statistical matching.

This module provides utility functions for working with survey samples,
including:
- fact2dummy: Convert categorical variables to dummy variables
- harmonize_x: Harmonize marginal/joint distributions across surveys
- comb_samples: Combine samples from different surveys
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def fact2dummy(
    data: Union[pd.Series, pd.DataFrame],
    all_levels: bool = True,
    lab: str = "x",
) -> pd.DataFrame:
    """
    Transform categorical variables into dummy (indicator) variables.

    This function substitutes categorical columns (columns with dtype
    'category' or object containing categorical-like data) with the
    corresponding dummy variables.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        A pandas Series (categorical) or DataFrame containing one or more
        categorical columns to be converted to dummy variables. Numeric
        columns are preserved as-is.
    all_levels : bool, default=True
        When True, creates dummy variables for all factor levels.
        When False, drops the dummy for the last level (reference coding).
    lab : str, default="x"
        The name prefix for dummy variables when data is a Series.
        Used as: "<lab>.<level>".

    Returns
    -------
    pd.DataFrame
        A DataFrame with dummy variables replacing the original categorical
        variables. Numeric columns are preserved unchanged.

    Notes
    -----
    - If a factor includes missing values (NA), all associated dummies
      will report NA for that observation.
    - This is a Python implementation of R's StatMatch::fact2dummy function.

    Examples
    --------
    >>> import pandas as pd
    >>> from statmatch.sample_utils import fact2dummy

    >>> # Single categorical Series
    >>> s = pd.Series(pd.Categorical(["a", "b", "a"]), name="x")
    >>> fact2dummy(s)
       x.a  x.b
    0  1.0  0.0
    1  0.0  1.0
    2  1.0  0.0

    >>> # DataFrame with mixed types
    >>> df = pd.DataFrame({
    ...     "num": [1.0, 2.0, 3.0],
    ...     "cat": pd.Categorical(["low", "high", "low"])
    ... })
    >>> fact2dummy(df)
       num  cat.high  cat.low
    0  1.0       0.0      1.0
    1  2.0       1.0      0.0
    2  3.0       0.0      1.0
    """
    if isinstance(data, pd.Series):
        return _series_to_dummies(data, all_levels, lab)
    elif isinstance(data, pd.DataFrame):
        return _dataframe_to_dummies(data, all_levels)
    else:
        raise TypeError(
            f"data must be a pandas Series or DataFrame, got {type(data)}"
        )


def _series_to_dummies(
    series: pd.Series, all_levels: bool, lab: str
) -> pd.DataFrame:
    """Convert a categorical Series to dummy variables."""
    # Ensure we have the original series, not a CategoricalAccessor
    original_series = series

    # Use the series name if available, otherwise use lab
    name = (
        original_series.name
        if hasattr(original_series, "name") and original_series.name is not None
        else lab
    )

    # Convert to Categorical if not already
    if hasattr(series, "cat"):
        # It's a categorical series - get the underlying categorical
        cat_data = series.cat
        categories = list(cat_data.categories)
        codes = np.array(cat_data.codes)
        n_obs = len(series)
    elif hasattr(series, "dtype") and series.dtype.name == "category":
        cat_data = series.cat
        categories = list(cat_data.categories)
        codes = np.array(cat_data.codes)
        n_obs = len(series)
    elif isinstance(series, pd.Categorical):
        categories = list(series.categories)
        codes = np.array(series.codes)
        n_obs = len(codes)
    else:
        # Convert to categorical
        cat_series = pd.Categorical(series)
        categories = list(cat_series.categories)
        codes = np.array(cat_series.codes)
        n_obs = len(codes)

    # Drop last level if not all_levels
    if not all_levels:
        categories = categories[:-1]

    # Create dummy DataFrame
    n_cats = len(categories)
    dummy_matrix = np.zeros((n_obs, n_cats), dtype=float)

    for i in range(n_cats):
        dummy_matrix[:, i] = (codes == i).astype(float)

    # Handle missing values (code = -1)
    missing_mask = codes == -1
    if np.any(missing_mask):
        dummy_matrix[missing_mask, :] = np.nan

    # Create column names
    col_names = [f"{name}.{cat}" for cat in categories]

    return pd.DataFrame(dummy_matrix, columns=col_names)


def _dataframe_to_dummies(df: pd.DataFrame, all_levels: bool) -> pd.DataFrame:
    """Convert categorical columns in a DataFrame to dummy variables."""
    result_cols = []
    result_data = {}

    for col in df.columns:
        series = df[col]

        # Check if column is categorical or can be treated as such
        is_categorical = (
            series.dtype.name == "category"
            or (
                hasattr(series.dtype, "name")
                and "ordered" in str(series.dtype)
            )
        )

        if is_categorical:
            # Convert to dummies
            dummies = _series_to_dummies(series, all_levels, col)
            for dummy_col in dummies.columns:
                result_data[dummy_col] = dummies[dummy_col].values
                result_cols.append(dummy_col)
        elif np.issubdtype(series.dtype, np.number):
            # Keep numeric columns as-is
            result_data[col] = series.values
            result_cols.append(col)
        elif series.dtype == object:
            # Check if it's string-like (potential categorical)
            unique_vals = series.dropna().unique()
            if len(unique_vals) < len(series) / 2:
                # Likely categorical, convert
                cat_series = pd.Series(
                    pd.Categorical(series), name=col
                )
                dummies = _series_to_dummies(cat_series, all_levels, col)
                for dummy_col in dummies.columns:
                    result_data[dummy_col] = dummies[dummy_col].values
                    result_cols.append(dummy_col)
            else:
                # Keep as-is
                result_data[col] = series.values
                result_cols.append(col)
        else:
            # Keep other types as-is
            result_data[col] = series.values
            result_cols.append(col)

    return pd.DataFrame(result_data, columns=result_cols)


def harmonize_x(
    svy_a: pd.DataFrame,
    svy_b: pd.DataFrame,
    x_vars: List[str],
    weight_a: str = "weight",
    weight_b: str = "weight",
    x_tot: Optional[Dict[str, float]] = None,
    cal_method: str = "linear",
    joint: bool = False,
    max_iter: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Harmonize marginal or joint distributions of variables across surveys.

    This function calibrates survey weights so that weighted distributions
    of common variables match between two surveys (or match known population
    totals).

    Parameters
    ----------
    svy_a : pd.DataFrame
        Survey A data containing the common variables and weights.
    svy_b : pd.DataFrame
        Survey B data containing the common variables and weights.
    x_vars : List[str]
        List of variable names common to both surveys to harmonize.
    weight_a : str, default="weight"
        Name of the weight column in svy_a.
    weight_b : str, default="weight"
        Name of the weight column in svy_b.
    x_tot : Optional[Dict[str, float]], default=None
        Known population totals for the X variables. If None, totals are
        estimated by combining both surveys.
    cal_method : str, default="linear"
        Calibration method: "linear", "raking", or "poststratify".
    joint : bool, default=False
        If True, harmonize the joint distribution of all x_vars.
        If False, harmonize marginal distributions separately.
    max_iter : int, default=50
        Maximum number of iterations for calibration algorithms.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'weights_a': Calibrated weights for survey A
        - 'weights_b': Calibrated weights for survey B

    Notes
    -----
    This is a Python implementation of R's StatMatch::harmonize.x function.
    The calibration is performed using iterative proportional fitting (raking)
    or linear calibration methods.

    Examples
    --------
    >>> from statmatch.sample_utils import harmonize_x
    >>> result = harmonize_x(
    ...     svy_a=survey_a_df,
    ...     svy_b=survey_b_df,
    ...     x_vars=["sex", "age_group"],
    ...     weight_a="weight",
    ...     weight_b="weight"
    ... )
    >>> calibrated_weights_a = result["weights_a"]
    """
    # Validate inputs
    for var in x_vars:
        if var not in svy_a.columns:
            raise ValueError(f"Variable '{var}' not found in svy_a")
        if var not in svy_b.columns:
            raise ValueError(f"Variable '{var}' not found in svy_b")

    if weight_a not in svy_a.columns:
        raise ValueError(f"Weight column '{weight_a}' not found in svy_a")
    if weight_b not in svy_b.columns:
        raise ValueError(f"Weight column '{weight_b}' not found in svy_b")

    # Get initial weights
    w_a = svy_a[weight_a].values.copy().astype(float)
    w_b = svy_b[weight_b].values.copy().astype(float)

    # Create dummy variables for categorical variables
    design_a = _create_design_matrix(svy_a, x_vars, joint)
    design_b = _create_design_matrix(svy_b, x_vars, joint)

    # Calculate target totals
    if x_tot is not None:
        # Use provided population totals
        target_totals = np.array(
            [x_tot.get(col, 0.0) for col in design_a.columns]
        )
    else:
        # Estimate totals by combining both surveys
        # Use simple pooling of weighted estimates
        total_a = (design_a.values.T @ w_a)
        total_b = (design_b.values.T @ w_b)
        # Average the two estimates (simple approach)
        target_totals = (total_a + total_b) / 2

    # Perform calibration
    if cal_method == "linear":
        weights_a = _linear_calibration(design_a.values, w_a, target_totals)
        weights_b = _linear_calibration(design_b.values, w_b, target_totals)
    elif cal_method == "raking":
        weights_a = _raking_calibration(
            design_a.values, w_a, target_totals, max_iter
        )
        weights_b = _raking_calibration(
            design_b.values, w_b, target_totals, max_iter
        )
    elif cal_method == "poststratify":
        weights_a = _poststratify(
            svy_a, x_vars, w_a, target_totals, design_a.columns
        )
        weights_b = _poststratify(
            svy_b, x_vars, w_b, target_totals, design_b.columns
        )
    else:
        raise ValueError(
            f"Unknown calibration method: {cal_method}. "
            f"Use 'linear', 'raking', or 'poststratify'."
        )

    return {
        "weights_a": weights_a,
        "weights_b": weights_b,
    }


def _create_design_matrix(
    df: pd.DataFrame, x_vars: List[str], joint: bool
) -> pd.DataFrame:
    """Create design matrix for calibration."""
    if joint:
        # Create joint categories
        if len(x_vars) == 1:
            subset = df[x_vars].copy()
        else:
            # Create interaction variable
            subset = pd.DataFrame()
            joint_col = df[x_vars[0]].astype(str)
            for var in x_vars[1:]:
                joint_col = joint_col + ":" + df[var].astype(str)
            subset["joint"] = pd.Categorical(joint_col)
        return fact2dummy(subset, all_levels=True)
    else:
        # Create marginal dummies for each variable
        result_parts = []
        for var in x_vars:
            var_data = df[[var]].copy()
            if df[var].dtype.name != "category":
                var_data[var] = pd.Categorical(df[var])
            dummies = fact2dummy(var_data, all_levels=True)
            result_parts.append(dummies)
        return pd.concat(result_parts, axis=1)


def _linear_calibration(
    X: np.ndarray, w: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Perform linear calibration (GREG-like adjustment)."""
    # Current totals
    current = X.T @ w

    # Compute adjustment factor using linear calibration
    # g = 1 + (target - current) @ inv(X.T @ diag(w) @ X) @ X.T
    XtW = X.T * w
    XtWX = XtW @ X

    # Add small regularization for numerical stability
    XtWX += np.eye(XtWX.shape[0]) * 1e-8

    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse
        XtWX_inv = np.linalg.pinv(XtWX)

    diff = target - current
    adjustment = X @ (XtWX_inv @ diff)

    # New weights
    g = 1 + adjustment / np.maximum(w, 1e-10)
    new_weights = w * g

    # Ensure positive weights
    new_weights = np.maximum(new_weights, 1e-10)

    return new_weights


def _raking_calibration(
    X: np.ndarray, w: np.ndarray, target: np.ndarray, max_iter: int
) -> np.ndarray:
    """Perform raking (iterative proportional fitting)."""
    weights = w.copy()
    n_vars = X.shape[1]

    for _ in range(max_iter):
        old_weights = weights.copy()

        for j in range(n_vars):
            # Current total for variable j
            current_j = np.sum(weights * X[:, j])

            if current_j > 0 and target[j] > 0:
                # Adjustment factor for this variable
                factor = target[j] / current_j

                # Apply adjustment only to units with this category
                mask = X[:, j] > 0
                weights[mask] *= factor

        # Check convergence
        if np.max(np.abs(weights - old_weights)) < 1e-6:
            break

    return weights


def _poststratify(
    df: pd.DataFrame,
    x_vars: List[str],
    w: np.ndarray,
    target: np.ndarray,
    col_names: List[str],
) -> np.ndarray:
    """Perform post-stratification adjustment."""
    # Create strata from x_vars
    strata = df[x_vars[0]].astype(str)
    for var in x_vars[1:]:
        strata = strata + ":" + df[var].astype(str)

    weights = w.copy()
    unique_strata = strata.unique()

    for stratum in unique_strata:
        mask = strata == stratum

        # Find corresponding target
        current_total = np.sum(w[mask])
        if current_total > 0:
            # Find target total for this stratum
            # This is simplified - in practice need to match column names
            target_total = np.mean(target) * np.sum(mask) / len(df)
            factor = target_total / current_total
            weights[mask] *= factor

    return weights


def comb_samples(
    svy_a: pd.DataFrame,
    svy_b: pd.DataFrame,
    y_lab: str,
    z_lab: str,
    x_vars: List[str],
    weight_a: str = "weight",
    weight_b: str = "weight",
    svy_c: Optional[pd.DataFrame] = None,
    weight_c: Optional[str] = None,
    estimation: Optional[str] = None,
    micro: bool = False,
) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """
    Combine samples from two surveys to estimate joint distribution of Y and Z.

    This function estimates the joint distribution of two categorical variables
    Y and Z that are observed in separate surveys (Y in survey A, Z in survey B),
    using common matching variables X observed in both surveys.

    Parameters
    ----------
    svy_a : pd.DataFrame
        Survey A data containing X variables and Y variable.
    svy_b : pd.DataFrame
        Survey B data containing X variables and Z variable.
    y_lab : str
        Name of the Y variable in survey A.
    z_lab : str
        Name of the Z variable in survey B.
    x_vars : List[str]
        List of common X variable names for matching.
    weight_a : str, default="weight"
        Name of the weight column in svy_a.
    weight_b : str, default="weight"
        Name of the weight column in svy_b.
    svy_c : Optional[pd.DataFrame], default=None
        Optional auxiliary survey C containing both Y and Z.
    weight_c : Optional[str], default=None
        Name of the weight column in svy_c.
    estimation : Optional[str], default=None
        Estimation method when svy_c is provided: "incomplete" or "synthetic".
    micro : bool, default=False
        If True, return predicted probabilities of Z for A units and
        Y for B units.

    Returns
    -------
    Dict[str, Union[np.ndarray, pd.DataFrame]]
        Dictionary containing:
        - 'yz_cia': Contingency table of Y vs Z under conditional independence
        - 'yz_est': Estimated Y vs Z table (when svy_c is provided)
        - 'z_a': Predicted Z probabilities for A units (when micro=True)
        - 'y_b': Predicted Y probabilities for B units (when micro=True)

    Notes
    -----
    This is a Python implementation of R's StatMatch::comb.samples function.
    Under the conditional independence assumption (CIA), given X:
        P(Y, Z | X) = P(Y | X) * P(Z | X)

    Examples
    --------
    >>> from statmatch.sample_utils import comb_samples
    >>> result = comb_samples(
    ...     svy_a=survey_a,
    ...     svy_b=survey_b,
    ...     y_lab="education",
    ...     z_lab="spending",
    ...     x_vars=["sex", "age_group"],
    ...     weight_a="weight",
    ...     weight_b="weight"
    ... )
    >>> yz_table = result["yz_cia"]
    """
    # Validate inputs
    if y_lab not in svy_a.columns:
        raise ValueError(f"Y variable '{y_lab}' not found in svy_a")
    if z_lab not in svy_b.columns:
        raise ValueError(f"Z variable '{z_lab}' not found in svy_b")

    for var in x_vars:
        if var not in svy_a.columns:
            raise ValueError(f"X variable '{var}' not found in svy_a")
        if var not in svy_b.columns:
            raise ValueError(f"X variable '{var}' not found in svy_b")

    # Get weights
    w_a = svy_a[weight_a].values.astype(float)
    w_b = svy_b[weight_b].values.astype(float)

    # Get Y and Z categories
    y_cats = _get_categories(svy_a[y_lab])
    z_cats = _get_categories(svy_b[z_lab])

    # Compute P(Y|X) and P(Z|X) for each X stratum
    x_strata_a = _create_strata(svy_a, x_vars)
    x_strata_b = _create_strata(svy_b, x_vars)

    # All unique X strata
    all_strata = set(x_strata_a.unique()) | set(x_strata_b.unique())

    # Initialize results
    n_y = len(y_cats)
    n_z = len(z_cats)
    yz_cia = np.zeros((n_y, n_z))

    # Storage for micro-level predictions
    if micro:
        z_a_probs = np.zeros((len(svy_a), n_z))
        y_b_probs = np.zeros((len(svy_b), n_y))

    # For each stratum, compute conditional distributions
    for stratum in all_strata:
        # P(Y|X=stratum) from survey A
        mask_a = x_strata_a == stratum
        if np.sum(mask_a) > 0:
            p_y_given_x = _compute_conditional_dist(
                svy_a.loc[mask_a, y_lab], w_a[mask_a], y_cats
            )
        else:
            p_y_given_x = np.ones(n_y) / n_y  # Uniform if no data

        # P(Z|X=stratum) from survey B
        mask_b = x_strata_b == stratum
        if np.sum(mask_b) > 0:
            p_z_given_x = _compute_conditional_dist(
                svy_b.loc[mask_b, z_lab], w_b[mask_b], z_cats
            )
        else:
            p_z_given_x = np.ones(n_z) / n_z  # Uniform if no data

        # P(X=stratum) - use combined estimate from both surveys
        p_x_a = np.sum(w_a[mask_a]) / np.sum(w_a) if np.sum(mask_a) > 0 else 0
        p_x_b = np.sum(w_b[mask_b]) / np.sum(w_b) if np.sum(mask_b) > 0 else 0
        p_x = (p_x_a + p_x_b) / 2

        # Under CIA: P(Y,Z|X) = P(Y|X) * P(Z|X)
        # Contribution to P(Y,Z): P(Y,Z|X) * P(X)
        yz_cia += p_x * np.outer(p_y_given_x, p_z_given_x)

        # Store micro predictions
        if micro:
            if np.sum(mask_a) > 0:
                z_a_probs[mask_a] = p_z_given_x
            if np.sum(mask_b) > 0:
                y_b_probs[mask_b] = p_y_given_x

    # Normalize to sum to 1
    yz_cia = yz_cia / np.sum(yz_cia) if np.sum(yz_cia) > 0 else yz_cia

    # Create result DataFrame with proper labels
    yz_cia_df = pd.DataFrame(
        yz_cia, index=y_cats, columns=z_cats
    )

    result = {"yz_cia": yz_cia_df}

    # Handle auxiliary survey C
    if svy_c is not None:
        if weight_c is None:
            weight_c = "weight"

        w_c = svy_c[weight_c].values.astype(float)

        # Compute empirical P(Y,Z) from survey C
        yz_est = _compute_joint_dist(
            svy_c[y_lab], svy_c[z_lab], w_c, y_cats, z_cats
        )

        if estimation == "incomplete":
            # Incomplete Two-Way Stratification (ITWS)
            # Adjust CIA estimate using empirical margins from C
            yz_est = _itws_adjustment(yz_cia, yz_est)
        elif estimation == "synthetic":
            # Synthetic Two-Way Stratification (STWS)
            # Use C directly with calibration
            pass  # yz_est is already computed

        result["yz_est"] = pd.DataFrame(yz_est, index=y_cats, columns=z_cats)

    if micro:
        result["z_a"] = pd.DataFrame(z_a_probs, columns=z_cats)
        result["y_b"] = pd.DataFrame(y_b_probs, columns=y_cats)

    return result


def _get_categories(series: pd.Series) -> List:
    """Get ordered list of categories from a series."""
    if hasattr(series, "cat"):
        return list(series.cat.categories)
    elif series.dtype.name == "category":
        return list(series.cat.categories)
    else:
        return sorted(series.dropna().unique())


def _create_strata(df: pd.DataFrame, x_vars: List[str]) -> pd.Series:
    """Create stratum labels from X variables."""
    strata = df[x_vars[0]].astype(str)
    for var in x_vars[1:]:
        strata = strata + ":" + df[var].astype(str)
    return strata


def _compute_conditional_dist(
    series: pd.Series, weights: np.ndarray, categories: List
) -> np.ndarray:
    """Compute weighted conditional distribution."""
    n_cats = len(categories)
    probs = np.zeros(n_cats)

    for i, cat in enumerate(categories):
        mask = series.values == cat
        probs[i] = np.sum(weights[mask])

    total = np.sum(probs)
    if total > 0:
        probs = probs / total

    return probs


def _compute_joint_dist(
    y_series: pd.Series,
    z_series: pd.Series,
    weights: np.ndarray,
    y_cats: List,
    z_cats: List,
) -> np.ndarray:
    """Compute weighted joint distribution of Y and Z."""
    n_y = len(y_cats)
    n_z = len(z_cats)
    joint = np.zeros((n_y, n_z))

    for i, y_cat in enumerate(y_cats):
        for j, z_cat in enumerate(z_cats):
            mask = (y_series.values == y_cat) & (z_series.values == z_cat)
            joint[i, j] = np.sum(weights[mask])

    total = np.sum(joint)
    if total > 0:
        joint = joint / total

    return joint


def _itws_adjustment(yz_cia: np.ndarray, yz_emp: np.ndarray) -> np.ndarray:
    """
    Incomplete Two-Way Stratification adjustment.

    Adjusts the CIA estimate using empirical margins from auxiliary data.
    """
    # Use iterative proportional fitting to match margins
    result = yz_cia.copy()

    # Target margins from empirical data
    y_margins = np.sum(yz_emp, axis=1)
    z_margins = np.sum(yz_emp, axis=0)

    # IPF iterations
    for _ in range(50):
        # Adjust to Y margins
        current_y = np.sum(result, axis=1)
        for i in range(len(y_margins)):
            if current_y[i] > 0:
                result[i, :] *= y_margins[i] / current_y[i]

        # Adjust to Z margins
        current_z = np.sum(result, axis=0)
        for j in range(len(z_margins)):
            if current_z[j] > 0:
                result[:, j] *= z_margins[j] / current_z[j]

    return result
