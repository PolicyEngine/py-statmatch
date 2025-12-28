"""Comparison functions for statistical matching quality assessment.

This module provides functions for comparing distributions between datasets,
which is useful for assessing the quality of statistical matching.
"""

from typing import Dict, List, Optional, Union
import re
import numpy as np
import pandas as pd
from scipy import stats


def comp_cont(
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
    xlab_a: str,
    xlab_b: Optional[str] = None,
    w_a: Optional[str] = None,
    w_b: Optional[str] = None,
    ref: bool = False,
) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
    """
    Compare distributions of a continuous variable between two datasets.

    This function estimates the 'closeness' of the distributions of the same
    continuous variable(s) estimated from different data sources.

    Parameters
    ----------
    data_a : pd.DataFrame
        First dataset containing the variable to compare.
    data_b : pd.DataFrame
        Second dataset containing the variable to compare.
    xlab_a : str
        Name of the variable in data_a.
    xlab_b : str, optional
        Name of the variable in data_b. Defaults to xlab_a.
    w_a : str, optional
        Name of weight variable in data_a.
    w_b : str, optional
        Name of weight variable in data_b.
    ref : bool, default=False
        If True, data_b distribution is treated as the reference.

    Returns
    -------
    dict
        Dictionary containing:
        - 'summary': DataFrame with descriptive statistics for both variables
        - 'diff_qs': Dict with average absolute and squared quantile diffs
        - 'dist_ecdf': Dict with ECDF distance measures (KS, Kuiper, avg)
        - 'dist_discr': Dict with discretized distribution distances
    """
    if xlab_b is None:
        xlab_b = xlab_a

    n_a = len(data_a)
    n_b = len(data_b)

    x_a = data_a[xlab_a].values.astype(float)
    x_b = data_b[xlab_b].values.astype(float)

    # Handle weights
    if w_a is not None:
        w_a_vals = data_a[w_a].values.astype(float)
        w_a_vals = w_a_vals / w_a_vals.sum() * n_a
    else:
        w_a_vals = np.ones(n_a)

    if w_b is not None:
        w_b_vals = data_b[w_b].values.astype(float)
        w_b_vals = w_b_vals / w_b_vals.sum() * n_b
    else:
        w_b_vals = np.ones(n_b)

    # Design effects
    deff_a = n_a * np.sum(w_a_vals**2) / (np.sum(w_a_vals) ** 2)
    deff_b = n_b * np.sum(w_b_vals**2) / (np.sum(w_b_vals) ** 2)

    # Compute ECDFs
    if ref:
        usx = np.unique(np.sort(x_b))
    else:
        usx = np.unique(np.sort(np.concatenate([x_a, x_b])))

    k = len(usx)
    ecdf_a = np.zeros(k)
    ecdf_b = np.zeros(k)

    for i in range(k):
        ecdf_a[i] = np.sum(w_a_vals[x_a <= usx[i]])
        ecdf_b[i] = np.sum(w_b_vals[x_b <= usx[i]])

    # Normalize to proportions
    ecdf_a = ecdf_a / ecdf_a[-1] if ecdf_a[-1] > 0 else ecdf_a
    ecdf_b = ecdf_b / ecdf_b[-1] if ecdf_b[-1] > 0 else ecdf_b

    # ECDF distances
    d_ks = np.max(np.abs(ecdf_a - ecdf_b))
    d_kuiper = np.max(ecdf_a - ecdf_b) + np.max(ecdf_b - ecdf_a)
    d_wass = np.mean(np.abs(ecdf_a - ecdf_b))

    dist_ecdf = {
        "ks_dist": d_ks,
        "kuiper_dist": d_kuiper,
        "av_abs_diff": d_wass,
    }

    # Quantile comparison
    n = min(n_a, n_b)
    if n <= 50:
        probs = np.array([0.25, 0.50, 0.75])
    elif n <= 150:
        probs = np.arange(0.2, 0.81, 0.2)
    elif n <= 250:
        probs = np.arange(0.1, 0.91, 0.1)
    else:
        probs = np.arange(0.05, 0.951, 0.05)

    if w_a is not None:
        q_a = _weighted_quantile(x_a, w_a_vals, probs)
    else:
        q_a = np.quantile(x_a, probs)

    if w_b is not None:
        q_b = _weighted_quantile(x_b, w_b_vals, probs)
    else:
        q_b = np.quantile(x_b, probs)

    d_qa = np.mean(np.abs(q_a - q_b))
    d_q2 = np.mean((q_a - q_b) ** 2)

    diff_qs = {
        "av_abs_d": d_qa,
        "av_sqrt_sq_d": np.sqrt(d_q2),
    }

    # Summary statistics
    summary_a = _compute_summary(x_a, w_a_vals if w_a else None)
    summary_b = _compute_summary(x_b, w_b_vals if w_b else None)

    summary = pd.DataFrame([summary_a, summary_b], index=["A", "B"])
    summary.columns = [
        "Min.",
        "1st Qu.",
        "Median",
        "Mean",
        "3rd Qu.",
        "Max.",
        "sd",
    ]

    # Discretized distribution distances
    # Compute histogram breaks using Freedman-Diaconis rule
    bks = _hist_breaks(
        x_b if ref else np.concatenate([x_a, x_b]),
        w_b_vals if ref else np.concatenate([w_a_vals, w_b_vals]),
        min(n_a / deff_a, n_b / deff_b),
    )

    # Extend breaks if needed
    if bks[0] > min(x_a):
        bks[0] = min(x_a)
    if bks[-1] < max(x_a):
        bks[-1] = max(x_a)

    # Discretize and compute proportions
    p_a = _compute_proportions(x_a, w_a_vals if w_a else None, bks)
    p_b = _compute_proportions(x_b, w_b_vals if w_b else None, bks)

    # Distance measures
    tvd = 0.5 * np.sum(np.abs(p_a - p_b))
    bhatt = np.sum(np.sqrt(p_a * p_b))
    hellinger = np.sqrt(1 - bhatt)

    dist_discr = {
        "tvd": tvd,
        "overlap": 1 - tvd,
        "hellinger": hellinger,
    }

    return {
        "summary": summary,
        "diff_qs": diff_qs,
        "dist_ecdf": dist_ecdf,
        "dist_discr": dist_discr,
    }


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probs: np.ndarray
) -> np.ndarray:
    """Compute weighted quantiles."""
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cumsum = np.cumsum(weights)
    cutpoints = probs * cumsum[-1]

    result = np.zeros(len(probs))
    for i, cp in enumerate(cutpoints):
        idx = np.searchsorted(cumsum, cp)
        if idx >= len(values):
            idx = len(values) - 1
        result[i] = values[idx]

    return result


def _compute_summary(x: np.ndarray, w: Optional[np.ndarray]) -> List[float]:
    """Compute summary statistics (min, q1, median, mean, q3, max, sd)."""
    if w is None:
        return [
            np.min(x),
            np.quantile(x, 0.25),
            np.median(x),
            np.mean(x),
            np.quantile(x, 0.75),
            np.max(x),
            np.std(x, ddof=1),
        ]
    else:
        quants = _weighted_quantile(x, w, np.array([0, 0.25, 0.5, 0.75, 1]))
        mean = np.average(x, weights=w)
        var = np.average((x - mean) ** 2, weights=w)
        # Adjust for sample variance
        n = len(w)
        sd = np.sqrt(var * n / (n - 1))
        return [
            quants[0],
            quants[1],
            quants[2],
            mean,
            quants[3],
            quants[4],
            sd,
        ]


def _hist_breaks(
    x: np.ndarray, w: Optional[np.ndarray], n: float
) -> np.ndarray:
    """Compute histogram breaks using Freedman-Diaconis rule."""
    xx = x[~np.isnan(x)]

    if w is None:
        q = np.quantile(xx, [0.25, 0.5, 0.75])
    else:
        ww = w[~np.isnan(x)]
        q = _weighted_quantile(xx, ww, np.array([0.25, 0.5, 0.75]))

    iqr = q[2] - q[0]

    # Freedman-Diaconis width
    width = 2 * iqr / (n ** (1 / 3))

    # Compute bounds
    low = max(np.min(xx), q[0] - 1.5 * 2 * (q[1] - q[0]))
    up = min(np.max(xx), q[2] + 1.5 * 2 * (q[2] - q[1]))

    if width <= 0:
        width = (np.max(xx) - np.min(xx)) / 10

    bins = int(np.floor((up - low) / width)) + 1
    if bins < 2:
        bins = 2
        width = (np.max(xx) - np.min(xx)) / bins

    span = bins * width
    dd = span - (up - low)
    low = low - dd / 2
    up = up + dd / 2

    return np.linspace(low, up, bins + 1)


def _compute_proportions(
    x: np.ndarray, w: Optional[np.ndarray], breaks: np.ndarray
) -> np.ndarray:
    """Compute proportions in each bin."""
    n_bins = len(breaks) - 1
    counts = np.zeros(n_bins)

    for i in range(n_bins):
        if i == 0:
            mask = (x >= breaks[i]) & (x <= breaks[i + 1])
        else:
            mask = (x > breaks[i]) & (x <= breaks[i + 1])

        if w is None:
            counts[i] = np.sum(mask)
        else:
            counts[i] = np.sum(w[mask])

    total = counts.sum()
    if total > 0:
        return counts / total
    return counts


def comp_prop(
    p1: np.ndarray,
    p2: np.ndarray,
    n1: int,
    n2: Optional[int] = None,
    ref: bool = False,
) -> Dict[str, Union[Dict[str, float], np.ndarray]]:
    """
    Compare two distributions of the same categorical variable.

    Parameters
    ----------
    p1 : np.ndarray
        Vector of relative or absolute frequencies from sample 1.
    p2 : np.ndarray
        Vector of relative or absolute frequencies from sample 2 (or reference).
    n1 : int
        Sample size for p1.
    n2 : int, optional
        Sample size for p2. Required when ref=False.
    ref : bool, default=False
        If True, p2 is treated as the reference distribution.

    Returns
    -------
    dict
        Dictionary containing:
        - 'meas': Dict with tvd, overlap, Bhattacharyya, Hellinger distances
        - 'chi_sq': Dict with Pearson chi-sq, df, critical value, design effect
        - 'p_exp': Expected proportions used in calculations
    """
    if not ref and n2 is None:
        raise ValueError(
            "If p2 is not the reference distribution (ref=False), "
            "please provide the n2 argument (sample size)"
        )

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    # Normalize to proportions if needed
    if p1.sum() > 1:
        p1 = p1 / p1.sum()
    if p2.sum() > 1:
        p2 = p2 / p2.sum()

    # Compute distance measures
    tvd = 0.5 * np.sum(np.abs(p1 - p2))
    overlap = 1 - tvd
    bhatt = np.sum(np.sqrt(p1 * p2))
    hell = np.sqrt(1 - bhatt)

    meas = {
        "tvd": tvd,
        "overlap": overlap,
        "bhatt": bhatt,
        "hell": hell,
    }

    # Chi-squared test
    if ref:
        # p2 is reference
        not0 = p2 > 0
        pp1 = p1[not0]
        pp2 = p2[not0]
        j = len(pp2)
        chi_p = n1 * np.sum((pp1 - pp2) ** 2 / pp2)
        p_exp = p2
    else:
        # Pool estimates
        w1 = n1 / (n1 + n2)
        p_exp = p1 * w1 + p2 * (1 - w1)
        not0 = p_exp > 0
        ppe = p_exp[not0]
        pp1 = p1[not0]
        pp2 = p2[not0]
        j = len(ppe)
        chi_1 = n1 * np.sum((pp1 - ppe) ** 2 / ppe)
        chi_2 = n2 * np.sum((pp2 - ppe) ** 2 / ppe)
        chi_p = chi_1 + chi_2

    df = j - 1
    q_05 = stats.chi2.ppf(0.95, df=df)
    delta_h0 = chi_p / q_05

    chi_sq = {
        "pearson": chi_p,
        "df": df,
        "q0_05": q_05,
        "delta_h0": delta_h0,
    }

    return {
        "meas": meas,
        "chi_sq": chi_sq,
        "p_exp": p_exp,
    }


def pw_assoc(
    formula: str,
    data: pd.DataFrame,
    weights: Optional[str] = None,
    out_df: bool = False,
) -> Union[Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Compute pairwise association measures between categorical variables.

    Computes association and Proportional Reduction in Error (PRE) measures
    between a categorical response variable and multiple categorical predictors.

    Parameters
    ----------
    formula : str
        Formula of the form "y ~ x1 + x2" where y is the response variable.
    data : pd.DataFrame
        DataFrame containing the variables.
    weights : str, optional
        Name of weight variable in data.
    out_df : bool, default=False
        If True, return DataFrame instead of dict.

    Returns
    -------
    dict or pd.DataFrame
        Association measures for each predictor:
        - V: Cramer's V
        - bcV: Bias-corrected Cramer's V
        - mi: Mutual information
        - norm_mi: Normalized mutual information
        - lambda_: Goodman-Kruskal lambda (Y|X)
        - tau: Goodman-Kruskal tau (Y|X)
        - U: Theil's uncertainty coefficient
        - AIC: Akaike Information Criterion
        - BIC: Bayesian Information Criterion
        - npar: Number of parameters
    """
    # Parse formula
    parts = formula.replace(" ", "").split("~")
    y_var = parts[0]
    x_vars = parts[1].split("+")

    n = len(data)

    # Handle weights
    if weights is not None:
        ww = data[weights].values.astype(float)
        ww = ww / ww.sum() * n
    else:
        ww = None

    # Initialize result containers
    results = {
        "V": {},
        "bcV": {},
        "mi": {},
        "norm_mi": {},
        "lambda_": {},
        "tau": {},
        "U": {},
        "AIC": {},
        "BIC": {},
        "npar": {},
    }

    for x_var in x_vars:
        # Remove missing values
        mask = ~(data[y_var].isna() | data[x_var].isna())
        y = data.loc[mask, y_var]
        x = data.loc[mask, x_var]

        if ww is not None:
            w_sub = ww[mask]
        else:
            w_sub = None

        # Compute contingency table
        if w_sub is not None:
            tab = _weighted_crosstab(y, x, w_sub)
        else:
            tab = pd.crosstab(y, x)

        tab_arr = tab.values.astype(float)

        # Cramer's V
        v, bcv = _cramers_v(tab_arr, n)
        results["V"][x_var] = v
        results["bcV"][x_var] = bcv

        # PRE measures
        prv = _prv_rc(tab_arr)
        results["lambda_"][x_var] = prv["lambda_rc"]
        results["tau"][x_var] = prv["tau_rc"]
        results["U"][x_var] = prv["u_rc"]
        results["mi"][x_var] = prv["mi"]
        results["norm_mi"][x_var] = prv["norm_mi"]

        # Information criteria
        ics = _ic_based(tab_arr, n)
        results["AIC"][x_var] = ics["AIC"]
        results["BIC"][x_var] = ics["BIC"]
        results["npar"][x_var] = ics["df"]

    if out_df:
        df = pd.DataFrame(results)
        df.index.name = "predictor"
        return df

    return results


def _weighted_crosstab(
    y: pd.Series, x: pd.Series, w: np.ndarray
) -> pd.DataFrame:
    """Compute weighted cross-tabulation."""
    y_cats = y.unique()
    x_cats = x.unique()

    tab = pd.DataFrame(0.0, index=y_cats, columns=x_cats)

    for i, (yi, xi, wi) in enumerate(zip(y, x, w)):
        tab.loc[yi, xi] += wi

    return tab


def _cramers_v(tab: np.ndarray, n: int) -> tuple:
    """Compute Cramer's V and bias-corrected Cramer's V."""
    nr, nc = tab.shape

    # Chi-squared statistic
    row_sums = tab.sum(axis=1, keepdims=True)
    col_sums = tab.sum(axis=0, keepdims=True)
    total = tab.sum()

    if total == 0:
        return 0.0, 0.0

    expected = row_sums * col_sums / total
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((tab - expected) ** 2 / expected)

    mm = min(nr - 1, nc - 1)
    if mm == 0:
        return 0.0, 0.0

    # Standard Cramer's V
    v = np.sqrt(chi2 / (n * mm))

    # Bias-corrected Cramer's V (Bergsma 2013)
    bb = chi2 / n - (nr - 1) * (nc - 1) / (n - 1)
    bb = max(0, bb)

    avr = nr - (nr - 1) ** 2 / (n - 1)
    avc = nc - (nc - 1) ** 2 / (n - 1)

    denom = min(avr - 1, avc - 1)
    if denom <= 0:
        bcv = 0.0
    else:
        bcv = np.sqrt(bb / denom)

    return v, bcv


def _prv_rc(tab: np.ndarray) -> Dict[str, float]:
    """
    Compute proportional reduction in variance measures.

    Returns lambda, tau, U (uncertainty coefficient), and mutual information.
    """
    tab = tab / tab.sum()
    r_s = tab.sum(axis=1)
    c_s = tab.sum(axis=0)

    # Goodman-Kruskal lambda (Y|X)
    v_r = 1 - r_s.max()
    ev_rgc = 1 - tab.max(axis=0).sum()
    lambda_rc = (v_r - ev_rgc) / v_r if v_r > 0 else 0

    # Goodman-Kruskal tau (Y|X)
    v_r = 1 - (r_s**2).sum()
    col_sums_sq = (tab**2).sum(axis=0)
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        ev_rgc = 1 - np.nansum(col_sums_sq / c_s)
    tau_rc = (v_r - ev_rgc) / v_r if v_r > 0 else 0

    # Entropy and mutual information
    def entropy(p):
        p = np.asarray(p).flatten()
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    h_r = entropy(r_s)
    h_c = entropy(c_s)
    h_rc = entropy(tab)

    mi = h_r + h_c - h_rc
    u_rc = mi / h_r if h_r > 0 else 0
    norm_mi = mi / min(h_r, h_c) if min(h_r, h_c) > 0 else 0

    return {
        "lambda_rc": lambda_rc,
        "tau_rc": tau_rc,
        "u_rc": u_rc,
        "mi": mi,
        "norm_mi": norm_mi,
    }


def _ic_based(tab: np.ndarray, n: int) -> Dict[str, float]:
    """Compute AIC and BIC for the contingency table model."""
    r_s = tab.sum(axis=1)
    c_s = tab.sum(axis=0)

    # Marginal model (just row proportions)
    nr = (r_s > 0).sum()
    df_0 = nr - 1

    # Avoid log(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ic_0 = np.nansum(r_s * np.log(r_s / n))

    # Conditional model (row given column)
    nc = (c_s > 0).sum()
    df_1 = df_0 * nc

    # Conditional proportions
    with np.errstate(divide="ignore", invalid="ignore"):
        cond_props = tab / c_s
        cond_props = np.nan_to_num(cond_props)
        log_cond = np.log(cond_props)
        log_cond = np.nan_to_num(log_cond, neginf=0)
        ic_1 = np.nansum(tab * log_cond)

    # AIC and BIC
    aic_0 = -2 * ic_0 + 2 * df_0
    bic_0 = -2 * ic_0 + np.log(n) * df_0

    aic_1 = -2 * ic_1 + 2 * df_1
    bic_1 = -2 * ic_1 + np.log(n) * df_1

    return {
        "AIC": aic_1,
        "BIC": bic_1,
        "df": df_1,
    }
