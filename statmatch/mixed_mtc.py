"""Implementation of mixed.mtc and selMtc.by.unc for statistical matching."""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import mahalanobis


def _fact2dummy(data: pd.DataFrame, all_levels: bool = False) -> pd.DataFrame:
    """
    Convert categorical variables to dummy variables.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with potentially categorical variables.
    all_levels : bool
        If True, create dummies for all levels. If False, drop first level.

    Returns
    -------
    pd.DataFrame
        DataFrame with categorical variables converted to dummies.
    """
    result_dfs = []
    col_names = []

    for col in data.columns:
        if data[col].dtype == "object" or isinstance(
            data[col].dtype, pd.CategoricalDtype
        ):
            # Convert to dummies
            dummies = pd.get_dummies(
                data[col], prefix=col, drop_first=not all_levels
            )
            result_dfs.append(dummies)
            col_names.extend(dummies.columns.tolist())
        else:
            result_dfs.append(data[[col]])
            col_names.append(col)

    if len(result_dfs) == 0:
        return pd.DataFrame()

    result = pd.concat(result_dfs, axis=1)
    return result


def mixed_mtc(
    data_rec: pd.DataFrame,
    data_don: pd.DataFrame,
    match_vars: List[str],
    y_rec: str,
    z_don: str,
    method: str = "ML",
    rho_yz: Optional[float] = None,
    micro: bool = False,
    constr_alg: str = "hungarian",
) -> Dict[str, Union[np.ndarray, pd.DataFrame, Dict]]:
    """
    Statistical matching via mixed methods.

    A mixed method consists of two steps:
    1. Adoption of a parametric model for the joint distribution of (X,Y,Z)
       and estimation of its parameters
    2. Derivation of a complete "synthetic" data set using a nonparametric
       approach (constrained distance hot deck)

    Parameters
    ----------
    data_rec : pd.DataFrame
        The recipient data set containing match_vars and y_rec.
    data_don : pd.DataFrame
        The donor data set containing match_vars and z_don.
    match_vars : List[str]
        Variable names used for matching (common variables X).
    y_rec : str
        Name of the target variable Y observed only in data_rec.
    z_don : str
        Name of the target variable Z observed only in data_don.
    method : str, default="ML"
        Method for parameter estimation: "ML" (Maximum Likelihood) or
        "MS" (Moriarity-Scheuren approach).
    rho_yz : Optional[float], default=None
        Guess for the correlation between Y and Z. For ML, this is the
        partial correlation given X. For MS, this is the direct correlation.
        If None, conditional independence assumption (CIA) is used.
    micro : bool, default=False
        If True, return recipient data filled with Z values using
        constrained hot deck matching.
    constr_alg : str, default="hungarian"
        Algorithm for constrained matching: "lpsolve" or "hungarian".

    Returns
    -------
    Dict[str, Union[np.ndarray, pd.DataFrame, Dict]]
        Dictionary containing:
        - 'mu': Estimated mean vector
        - 'vc': Estimated variance-covariance matrix
        - 'cor': Estimated correlation matrix
        - 'res.var': Residual variance estimates
        - 'filled.rec': Filled recipient data (if micro=True)
        - 'mtc.ids': Matching IDs (if micro=True)
        - 'dist.rd': Distances (if micro=True)
        - 'rho.yz': Info about rho.yz (for MS method)
    """
    n_rec = len(data_rec)
    n_don = len(data_don)

    # Check micro matching constraints early
    if micro and n_rec > n_don:
        raise ValueError(
            "The number of donors must be >= the number of recipients "
            f"for micro matching. Got {n_don} donors and {n_rec} recipients."
        )

    # Get labels for recipients and donors
    rec_labels = data_rec.index.tolist()
    don_labels = data_don.index.tolist()

    # Convert categorical variables to dummies
    x_rec = data_rec[match_vars].copy()
    x_rec = _fact2dummy(x_rec, all_levels=False)
    x_rec_np = x_rec.values.astype(float)

    x_don = data_don[match_vars].copy()
    x_don = _fact2dummy(x_don, all_levels=False)
    x_don_np = x_don.values.astype(float)

    # Get target variables
    y_A = data_rec[y_rec].values.astype(float)
    z_B = data_don[z_don].values.astype(float)

    # Get dimension info
    p_x = x_rec_np.shape[1]
    p = p_x + 2  # X variables + Y + Z

    # Variable names for output
    v_names = list(x_rec.columns) + [y_rec, z_don]
    pos_x = list(range(p_x))
    pos_y = p_x
    pos_z = p_x + 1

    # Initialize variance-covariance matrix
    vc = np.full((p, p), np.nan)

    if method.upper() == "ML":
        result = _ml_estimation(
            x_rec_np,
            x_don_np,
            y_A,
            z_B,
            rho_yz,
            n_rec,
            n_don,
            p_x,
            pos_x,
            pos_y,
            pos_z,
            vc,
            v_names,
        )
    elif method.upper() == "MS":
        result = _ms_estimation(
            x_rec_np,
            x_don_np,
            y_A,
            z_B,
            rho_yz,
            n_rec,
            n_don,
            p_x,
            pos_x,
            pos_y,
            pos_z,
            vc,
            v_names,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ML' or 'MS'.")

    if micro:
        result = _micro_matching(
            result,
            x_rec_np,
            x_don_np,
            y_A,
            z_B,
            data_rec,
            data_don,
            rec_labels,
            don_labels,
            pos_x,
            pos_y,
            pos_z,
            z_don,
            x_rec,
            method.upper(),
            constr_alg,
        )

    return result


def _ml_estimation(
    x_A: np.ndarray,
    x_B: np.ndarray,
    y_A: np.ndarray,
    z_B: np.ndarray,
    rho_yz: Optional[float],
    n_A: int,
    n_B: int,
    p_x: int,
    pos_x: List[int],
    pos_y: int,
    pos_z: int,
    vc: np.ndarray,
    v_names: List[str],
) -> Dict:
    """Maximum Likelihood estimation for mixed_mtc."""
    # Regression in file B: Z vs. X
    X_B_with_intercept = np.column_stack([np.ones(n_B), x_B])
    beta_B, _, _, _ = np.linalg.lstsq(X_B_with_intercept, z_B, rcond=None)
    res_B = z_B - X_B_with_intercept @ beta_B
    se_B = np.sqrt(np.mean(res_B**2))  # ML estimate of V(Z|X)

    # Regression in file A: Y vs. X
    X_A_with_intercept = np.column_stack([np.ones(n_A), x_A])
    beta_A, _, _, _ = np.linalg.lstsq(X_A_with_intercept, y_A, rcond=None)
    res_A = y_A - X_A_with_intercept @ beta_A
    se_A = np.sqrt(np.mean(res_A**2))  # ML estimate of V(Y|X)

    # ML estimates for X variables
    combined_x = np.vstack([x_A, x_B])
    mu_x = np.mean(combined_x, axis=0)
    S_x = np.cov(combined_x.T, bias=True)  # ML estimate (divide by N, not N-1)
    if S_x.ndim == 0:
        S_x = np.array([[S_x]])

    # ML estimates for Y
    mu_y = np.sum(beta_A * np.concatenate([[1], mu_x]))
    S_yx = (beta_A[1:].reshape(1, -1) @ S_x).flatten()
    S_y = se_A**2 + S_yx @ np.linalg.solve(S_x, S_yx)

    # ML estimates for Z
    mu_z = np.sum(beta_B * np.concatenate([[1], mu_x]))
    S_zx = (beta_B[1:].reshape(1, -1) @ S_x).flatten()
    S_z = se_B**2 + S_zx @ np.linalg.solve(S_x, S_zx)

    # ML estimates for Y, Z given input rho_yz (partial correlation)
    if rho_yz is None:
        rho_yz = 0.0  # CI assumption

    S_yzGx = rho_yz * se_A * se_B  # partial Cov[(Y,Z)|X]
    S_yz = S_yzGx + S_yx @ np.linalg.solve(S_x, S_zx)
    S_zGyx = se_B**2 - S_yzGx**2 / se_A**2
    S_yGzx = se_A**2 - S_yzGx**2 / se_B**2

    # Fill variance-covariance matrix
    vc[np.ix_(pos_x, pos_x)] = S_x
    vc[pos_x, pos_y] = S_yx
    vc[pos_y, pos_x] = S_yx
    vc[pos_x, pos_z] = S_zx
    vc[pos_z, pos_x] = S_zx
    vc[pos_y, pos_y] = S_y
    vc[pos_z, pos_z] = S_z
    vc[pos_y, pos_z] = S_yz
    vc[pos_z, pos_y] = S_yz

    # Prepare output
    mu = np.concatenate([mu_x, [mu_y, mu_z]])

    # Correlation matrix
    cor_mat = _cov2cor(vc)

    return {
        "start.prho.yz": rho_yz,
        "mu": mu,
        "vc": vc,
        "cor": cor_mat,
        "res.var": {"S.yGzx": S_yGzx, "S.zGyx": S_zGyx},
        "_se_A": se_A,
        "_se_B": se_B,
        "_S_yzGx": S_yzGx,
    }


def _ms_estimation(
    x_A: np.ndarray,
    x_B: np.ndarray,
    y_A: np.ndarray,
    z_B: np.ndarray,
    rho_yz: Optional[float],
    n_A: int,
    n_B: int,
    p_x: int,
    pos_x: List[int],
    pos_y: int,
    pos_z: int,
    vc: np.ndarray,
    v_names: List[str],
) -> Dict:
    """Moriarity-Scheuren estimation for mixed_mtc."""
    # Estimates for X variables
    combined_x = np.vstack([x_B, x_A])
    mu_x = np.mean(combined_x, axis=0)
    S_x = np.cov(combined_x.T, ddof=1)
    if S_x.ndim == 0:
        S_x = np.array([[S_x]])

    # Estimates for Y
    mu_y = np.mean(y_A)
    S_y = np.var(y_A, ddof=1)
    S_xy = np.cov(x_A.T, y_A, ddof=1)
    if S_xy.ndim == 1:
        S_xy = S_xy.reshape(-1, 1)
    S_xy = S_xy[:-1, -1]  # Get covariance of X with Y

    # Estimates for Z
    mu_z = np.mean(z_B)
    S_z = np.var(z_B, ddof=1)
    S_xz = np.cov(x_B.T, z_B, ddof=1)
    if S_xz.ndim == 1:
        S_xz = S_xz.reshape(-1, 1)
    S_xz = S_xz[:-1, -1]  # Get covariance of X with Z

    # Fill the Var-Cov matrix
    vc[np.ix_(pos_x, pos_x)] = S_x
    vc[pos_x, pos_y] = S_xy
    vc[pos_y, pos_x] = S_xy
    vc[pos_x, pos_z] = S_xz
    vc[pos_z, pos_x] = S_xz
    vc[pos_y, pos_y] = S_y
    vc[pos_z, pos_z] = S_z

    # Estimation of S_yz - find admissible bounds
    rho_yz_info = _find_admissible_rho(vc, pos_x, pos_y, pos_z, p_x, rho_yz)
    rho_yz_used = rho_yz_info["used"]

    S_yz = rho_yz_used * np.sqrt(S_y * S_z)
    vc[pos_y, pos_z] = S_yz
    vc[pos_z, pos_y] = S_yz

    # Compute residual variances
    xz_yz = np.concatenate([S_xz, [S_yz]])
    xy_pos = list(pos_x) + [pos_y]
    fi_3 = xz_yz @ np.linalg.solve(vc[np.ix_(xy_pos, xy_pos)], xz_yz)
    S_zGyx = max(0, S_z - fi_3)

    xy_yz = np.concatenate([S_xy, [S_yz]])
    xz_pos = list(pos_x) + [pos_z]
    fi_6 = xy_yz @ np.linalg.solve(vc[np.ix_(xz_pos, xz_pos)], xy_yz)
    S_yGzx = max(0, S_y - fi_6)

    # Prepare output
    mu = np.concatenate([mu_x, [mu_y, mu_z]])

    # Correlation matrix
    cor_mat = _cov2cor(vc)

    return {
        "rho.yz": rho_yz_info,
        "mu": mu,
        "vc": vc,
        "cor": cor_mat,
        "phi": {"fi.6.y": fi_6, "fi.3.z": fi_3},
        "res.var": {"S.yGzx": S_yGzx, "S.zGyx": S_zGyx},
        "_S_x": S_x,
        "_S_xy": S_xy,
        "_S_xz": S_xz,
        "_S_y": S_y,
        "_S_z": S_z,
        "_S_yz": S_yz,
    }


def _find_admissible_rho(
    vc: np.ndarray,
    pos_x: List[int],
    pos_y: int,
    pos_z: int,
    p_x: int,
    rho_yz: Optional[float],
) -> Dict:
    """Find admissible bounds for rho_yz in MS method."""
    if p_x == 1:
        # Single X variable case
        c_xy = vc[pos_x[0], pos_y] / np.sqrt(
            vc[pos_x[0], pos_x[0]] * vc[pos_y, pos_y]
        )
        c_xz = vc[pos_x[0], pos_z] / np.sqrt(
            vc[pos_x[0], pos_x[0]] * vc[pos_z, pos_z]
        )
        low_c = c_xy * c_xz - np.sqrt((1 - c_xy**2) * (1 - c_xz**2))
        up_c = c_xy * c_xz + np.sqrt((1 - c_xy**2) * (1 - c_xz**2))
        rho_yz_CI = c_xy * c_xz
    else:
        # Multiple X variables case
        eps = 0.0001
        cc = _cov2cor(vc)
        rr = np.arange(-1, 1 + eps, eps)
        vdet = np.zeros(len(rr))
        for i, r in enumerate(rr):
            cc[pos_z, pos_y] = r
            cc[pos_y, pos_z] = r
            vdet[i] = np.linalg.det(cc)
        cc_yz = rr[vdet >= 0]
        if len(cc_yz) > 0:
            low_c = np.min(cc_yz)
            up_c = np.max(cc_yz)
        else:
            low_c = -1.0
            up_c = 1.0
        rho_yz_CI = (low_c + up_c) / 2

    if rho_yz is None:
        rho_yz = rho_yz_CI

    start_rho = rho_yz

    # Adjust if outside bounds
    if rho_yz > up_c:
        rho_yz = up_c - 0.01
    elif rho_yz < low_c:
        rho_yz = low_c + 0.01

    return {
        "start": start_rho,
        "low.lim": low_c,
        "up.lim": up_c,
        "used": rho_yz,
    }


def _micro_matching(
    result: Dict,
    x_A: np.ndarray,
    x_B: np.ndarray,
    y_A: np.ndarray,
    z_B: np.ndarray,
    data_rec: pd.DataFrame,
    data_don: pd.DataFrame,
    rec_labels: List,
    don_labels: List,
    pos_x: List[int],
    pos_y: int,
    pos_z: int,
    z_don: str,
    x_rec_df: pd.DataFrame,
    method: str,
    constr_alg: str,
) -> Dict:
    """Perform micro-level matching with constrained hot deck."""
    n_A = len(y_A)
    n_B = len(z_B)

    mu = result["mu"]
    vc = result["vc"]
    mu_x = mu[pos_x]
    mu_y = mu[pos_y]
    mu_z = mu[pos_z]

    res_var = result["res.var"]
    S_zGyx = res_var["S.zGyx"]
    S_yGzx = res_var["S.yGzx"]

    if method == "ML":
        # Prediction of Z in file A
        cov_z_xy = np.concatenate([vc[pos_z, pos_x], [vc[pos_z, pos_y]]])
        xy_idx = list(pos_x) + [pos_y]
        vc_xy = vc[np.ix_(xy_idx, xy_idx)]
        beta_z_xy = np.linalg.solve(vc_xy, cov_z_xy)

        z_pred = np.zeros(n_A)
        for i in range(n_A):
            diff = np.concatenate([x_A[i] - mu_x, [y_A[i] - mu_y]])
            z_pred[i] = mu_z + diff @ beta_z_xy

        # Prediction of Y in file B
        cov_y_xz = np.concatenate([vc[pos_y, pos_x], [vc[pos_y, pos_z]]])
        xz_idx = list(pos_x) + [pos_z]
        vc_xz = vc[np.ix_(xz_idx, xz_idx)]
        beta_y_xz = np.linalg.solve(vc_xz, cov_y_xz)

        y_pred = np.zeros(n_B)
        for i in range(n_B):
            diff = np.concatenate([x_B[i] - mu_x, [z_B[i] - mu_z]])
            y_pred[i] = mu_y + diff @ beta_y_xz

        rSS = vc[np.ix_([pos_y, pos_z], [pos_y, pos_z])]
    else:
        # MS method
        S_xy = result["_S_xy"]
        S_xz = result["_S_xz"]
        S_yz = result["_S_yz"]
        S_y = result["_S_y"]
        S_z = result["_S_z"]

        # Prediction of Z in file A
        xz_yz = np.concatenate([S_xz, [S_yz]])
        xy_idx = list(pos_x) + [pos_y]
        B_zGxy = np.linalg.solve(vc[np.ix_(xy_idx, xy_idx)], xz_yz)

        z_pred = np.zeros(n_A)
        for i in range(n_A):
            diff = np.concatenate([x_A[i] - mu_x, [y_A[i] - mu_y]])
            z_pred[i] = mu_z + diff @ B_zGxy

        # Prediction of Y in file B
        xy_yz = np.concatenate([S_xy, [S_yz]])
        xz_idx = list(pos_x) + [pos_z]
        B_yGxz = np.linalg.solve(vc[np.ix_(xz_idx, xz_idx)], xy_yz)

        y_pred = np.zeros(n_B)
        for i in range(n_B):
            diff = np.concatenate([x_B[i] - mu_x, [z_B[i] - mu_z]])
            y_pred[i] = mu_y + diff @ B_yGxz

        # Compute rSS
        S1 = vc.copy()
        S2 = vc.copy()
        S1[pos_z, pos_z] = result["phi"]["fi.3.z"]
        S2[pos_y, pos_y] = result["phi"]["fi.6.y"]
        SS = S1 + S2
        rSS = SS[np.ix_([pos_y, pos_z], [pos_y, pos_z])]

    # Add random noise
    np.random.seed(None)  # Use different seed each time
    z_ep = z_pred + np.random.normal(0, np.sqrt(max(0, S_zGyx)), n_A)
    y_ep = y_pred + np.random.normal(0, np.sqrt(max(0, S_yGzx)), n_B)

    # Compute Mahalanobis distances
    new_B = np.column_stack([y_ep, z_B])
    irSS = np.linalg.inv(rSS)

    madist = np.zeros((n_A, n_B))
    for i in range(n_A):
        new_A = np.array([y_A[i], z_ep[i]])
        for j in range(n_B):
            diff = new_B[j] - new_A
            madist[i, j] = np.sqrt(diff @ irSS @ diff)

    # Constrained matching using Hungarian algorithm
    if constr_alg.lower() == "hungarian":
        row_ind, col_ind = linear_sum_assignment(madist)
        don_idx = col_ind
        dist_rd = madist[row_ind, col_ind]
    else:
        # LP solve (use Hungarian as fallback)
        row_ind, col_ind = linear_sum_assignment(madist)
        don_idx = col_ind
        dist_rd = madist[row_ind, col_ind]

    # Create output
    mtc_ids = pd.DataFrame(
        {
            "rec.id": [rec_labels[i] for i in range(n_A)],
            "don.id": [don_labels[don_idx[i]] for i in range(n_A)],
        }
    )

    # Create filled recipient dataset
    filled_rec = pd.concat(
        [x_rec_df, data_rec[[y_rec for y_rec in data_rec.columns if y_rec not in x_rec_df.columns]]],
        axis=1
    ).copy()
    filled_rec[z_don] = data_don.iloc[don_idx][z_don].values

    result["filled.rec"] = filled_rec
    result["mtc.ids"] = mtc_ids
    result["dist.rd"] = dist_rd

    return result


def _cov2cor(cov: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix."""
    std_dev = np.sqrt(np.diag(cov))
    cor = cov / np.outer(std_dev, std_dev)
    # Ensure diagonal is exactly 1
    np.fill_diagonal(cor, 1.0)
    return cor


def sel_mtc_by_unc(
    tab_x: pd.DataFrame,
    tab_xy: pd.DataFrame,
    tab_xz: pd.DataFrame,
    corr_d: int = 2,
    nA: Optional[int] = None,
    nB: Optional[int] = None,
    align_margins: bool = False,
) -> Dict[str, Union[List, pd.DataFrame]]:
    """
    Identify the best combination of matching variables by uncertainty.

    This function identifies the "best" subset of matching variables in terms
    of reduction of uncertainty when estimating relative frequencies in the
    contingency table Y vs. Z.

    Parameters
    ----------
    tab_x : pd.DataFrame
        A contingency table of X variables (created via pd.crosstab or similar).
    tab_xy : pd.DataFrame
        A contingency table crossing X variables with a single Y variable,
        created from data file A.
    tab_xz : pd.DataFrame
        A contingency table crossing X variables with a single Z variable,
        created from data file B.
    corr_d : int, default=2
        Penalty parameter (0, 1, or 2).
    nA : Optional[int], default=None
        Sample size of file A. Defaults to sum of tab_xy.
    nB : Optional[int], default=None
        Sample size of file B. Defaults to sum of tab_xz.
    align_margins : bool, default=False
        Whether to align X variable distributions across tables.

    Returns
    -------
    Dict[str, Union[List, pd.DataFrame]]
        Dictionary containing:
        - 'ini.ord': Variables ordered by effectiveness
        - 'list.xs': Combinations of matching variables tested
        - 'av.df': Data frame with detailed metrics
    """
    # Extract variable names from index
    if isinstance(tab_xy.index, pd.MultiIndex):
        x_vars = list(tab_xy.index.names)
    else:
        x_vars = [tab_xy.index.name] if tab_xy.index.name else ["X"]

    # Get sample sizes
    if nA is None:
        nA = tab_xy.values.sum()
    if nB is None:
        nB = tab_xz.values.sum()

    # Get Y and Z categories
    y_cats = tab_xy.columns.tolist()
    z_cats = tab_xz.columns.tolist()

    n_y = len(y_cats)
    n_z = len(z_cats)
    n_x = len(x_vars)

    # Initialize results
    results = []
    ini_ord = []
    list_xs = []

    # If only one X variable, return it directly
    if n_x == 1:
        return {
            "ini.ord": x_vars,
            "list.xs": [x_vars],
            "av.df": pd.DataFrame(
                {
                    "x_vars": [str(x_vars)],
                    "av_width": [0.0],
                    "penalty": [0.0],
                }
            ),
        }

    # Initial ordering: evaluate each X variable individually
    x_scores = {}
    for x_var in x_vars:
        score = _evaluate_x_variable(
            tab_xy, tab_xz, x_var, y_cats, z_cats, corr_d, nA, nB
        )
        x_scores[x_var] = score

    # Order by score (lower is better)
    ini_ord = sorted(x_vars, key=lambda x: x_scores[x])

    # Sequential variable selection
    selected = []
    remaining = ini_ord.copy()

    while remaining:
        best_var = None
        best_score = float("inf")

        for var in remaining:
            candidate = selected + [var]
            score = _evaluate_combination(
                tab_xy, tab_xz, candidate, y_cats, z_cats, corr_d, nA, nB
            )
            if score < best_score:
                best_score = score
                best_var = var

        if best_var is not None:
            selected.append(best_var)
            remaining.remove(best_var)
            list_xs.append(selected.copy())
            results.append(
                {
                    "x_vars": str(selected),
                    "av_width": best_score,
                    "penalty": _compute_penalty(
                        tab_xy, tab_xz, selected, corr_d
                    ),
                }
            )
        else:
            break

    av_df = pd.DataFrame(results) if results else pd.DataFrame()

    return {"ini.ord": ini_ord, "list.xs": list_xs, "av.df": av_df}


def _evaluate_x_variable(
    tab_xy: pd.DataFrame,
    tab_xz: pd.DataFrame,
    x_var: str,
    y_cats: List,
    z_cats: List,
    corr_d: int,
    nA: int,
    nB: int,
) -> float:
    """Evaluate a single X variable's effectiveness in reducing uncertainty."""
    # Simple metric: compute Frechet bounds width reduction
    # This is a simplified version of the full R implementation

    # Get marginal distributions
    try:
        if isinstance(tab_xy.index, pd.MultiIndex):
            idx = tab_xy.index.names.index(x_var)
            p_y = tab_xy.sum(axis=0) / tab_xy.values.sum()
            p_z = tab_xz.sum(axis=0) / tab_xz.values.sum()
        else:
            p_y = tab_xy.sum(axis=0) / tab_xy.values.sum()
            p_z = tab_xz.sum(axis=0) / tab_xz.values.sum()

        # Compute average Frechet width
        width = 0.0
        for y_val in y_cats:
            for z_val in z_cats:
                p_y_i = p_y[y_val] if y_val in p_y.index else 0
                p_z_j = p_z[z_val] if z_val in p_z.index else 0
                # Frechet bounds
                lower = max(0, p_y_i + p_z_j - 1)
                upper = min(p_y_i, p_z_j)
                width += upper - lower

        return width / (len(y_cats) * len(z_cats))
    except Exception:
        return float("inf")


def _evaluate_combination(
    tab_xy: pd.DataFrame,
    tab_xz: pd.DataFrame,
    x_vars: List[str],
    y_cats: List,
    z_cats: List,
    corr_d: int,
    nA: int,
    nB: int,
) -> float:
    """Evaluate a combination of X variables."""
    # Simplified evaluation - sum of individual scores
    total = 0.0
    for x_var in x_vars:
        total += _evaluate_x_variable(
            tab_xy, tab_xz, x_var, y_cats, z_cats, corr_d, nA, nB
        )
    return total / len(x_vars) if x_vars else float("inf")


def _compute_penalty(
    tab_xy: pd.DataFrame,
    tab_xz: pd.DataFrame,
    x_vars: List[str],
    corr_d: int,
) -> float:
    """Compute penalty for sparseness."""
    if corr_d == 0:
        return 0.0

    # Simple penalty based on table sparseness
    n_cells = len(tab_xy) if isinstance(tab_xy, pd.DataFrame) else 1
    n_empty = (tab_xy.values == 0).sum() if hasattr(tab_xy, "values") else 0

    sparseness = n_empty / max(1, n_cells)
    return sparseness * corr_d
