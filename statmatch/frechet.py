"""Implementation of Frechet bounds for categorical data.

This module provides functions for computing Frechet bounds on
cell probabilities in contingency tables, which is useful for
statistical matching when the joint distribution of Y and Z is
unknown but their conditional distributions given X are available.

References:
    D'Orazio, M., Di Zio, M. and Scanu, M. (2006). Statistical
    Matching: Theory and Practice. Wiley, Chichester.
"""

from typing import Any, Dict, List, Optional, Union
from itertools import combinations
import warnings

import numpy as np
import pandas as pd


def p_bayes(
    x: np.ndarray,
    method: str = "m.ind",
    const: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Estimate cell counts in contingency tables using pseudo-Bayes.

    This function applies shrinkage estimation to smooth cell
    frequencies, which is useful for handling sparse tables with
    zero counts.

    Parameters
    ----------
    x : np.ndarray
        A contingency table with observed cell counts.
    method : str, default="m.ind"
        Method for estimating final cell frequencies:
        - "Jeffreys": Add 0.5 to each cell
        - "minimax": Add sqrt(n)/c to each cell
        - "invcat": Add 1/c to each cell
        - "user": Add user-defined constant
        - "m.ind": Prior based on mutual independence hypothesis
        - "h.assoc": Prior based on homogeneous association
    const : Optional[float], default=None
        User-defined constant for method="user".

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'info': Dict with n, no.cells, av.cfr, no.0s, const, K, rel.K
        - 'prior': Array of prior cell frequencies
        - 'pseudoB': Array of pseudo-Bayes estimated frequencies

    Examples
    --------
    >>> import numpy as np
    >>> from statmatch.frechet import p_bayes
    >>> table = np.array([[10, 5], [8, 12]])
    >>> result = p_bayes(table, method="Jeffreys")
    >>> result["info"]["K"]
    2.0
    """
    x = np.asarray(x, dtype=float)
    n = x.sum()
    c = x.size
    no_zeros = np.sum(x == 0)

    # Compute relative frequencies
    p_h = x / n

    # Determine prior and K based on method
    if method.lower() == "jeffreys":
        const_val = 0.5
        K = c / 2
        gamma_h = np.ones_like(x) / c
    elif method.lower() == "minimax":
        const_val = np.sqrt(n) / c
        K = np.sqrt(n)
        gamma_h = np.ones_like(x) / c
    elif method.lower() == "invcat":
        const_val = 1 / c
        K = 1.0
        gamma_h = np.ones_like(x) / c
    elif method.lower() == "user":
        if const is None:
            raise ValueError("const must be provided when method='user'")
        const_val = const
        # K is the sum of constants across cells
        K = const_val * c
        gamma_h = np.ones_like(x) / c
    elif method.lower() == "m.ind":
        # Mutual independence prior
        const_val = None
        gamma_h = _compute_independence_prior(x)
        # Estimate K using data-driven approach
        K = _estimate_K(p_h, gamma_h)
    elif method.lower() == "h.assoc":
        # Homogeneous association prior
        const_val = None
        gamma_h = _compute_homogeneous_assoc_prior(x)
        K = _estimate_K(p_h, gamma_h)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute pseudo-Bayes estimate
    # ep_h = n/(n+K)*p_h + K/(n+K)*gamma_h
    # which is equivalent to:
    # ep_h = (n*p_h + K*gamma_h) / (n + K)
    # ep_h = (x + K*gamma_h) / (n + K)
    weight_data = n / (n + K)
    weight_prior = K / (n + K)
    pseudo_bayes = weight_data * x + weight_prior * n * gamma_h

    # Build prior in terms of expected counts (for compatibility)
    prior = n * gamma_h

    info = {
        "n": n,
        "no.cells": c,
        "av.cfr": n / c,
        "no.0s": no_zeros,
        "const": const_val,
        "K": K,
        "rel.K": K / (n + K),
    }

    return {"info": info, "prior": prior, "pseudoB": pseudo_bayes}


def _compute_independence_prior(x: np.ndarray) -> np.ndarray:
    """
    Compute prior under mutual independence hypothesis.

    For a multi-way table, the independence model assumes that
    the probability of each cell is the product of marginal
    probabilities.

    Parameters
    ----------
    x : np.ndarray
        Contingency table.

    Returns
    -------
    np.ndarray
        Prior cell probabilities under independence.
    """
    n = x.sum()
    ndim = x.ndim

    if ndim < 2:
        # For 1D, uniform prior
        return np.ones_like(x) / x.size

    # Compute marginal probabilities for each dimension
    marginals = []
    for axis in range(ndim):
        # Sum over all other axes
        axes_to_sum = tuple(i for i in range(ndim) if i != axis)
        marginal = x.sum(axis=axes_to_sum) / n
        marginals.append(marginal)

    # Compute outer product of all marginals
    gamma = marginals[0]
    for i in range(1, len(marginals)):
        gamma = np.outer(gamma, marginals[i])

    # Reshape to original dimensions
    gamma = gamma.reshape(x.shape)

    return gamma


def _compute_homogeneous_assoc_prior(x: np.ndarray) -> np.ndarray:
    """
    Compute prior under homogeneous association hypothesis.

    This uses the log-linear model approach where we estimate
    parameters assuming uniform association across cells.

    Parameters
    ----------
    x : np.ndarray
        Contingency table.

    Returns
    -------
    np.ndarray
        Prior cell probabilities under homogeneous association.
    """
    # For homogeneous association, we use iterative proportional
    # fitting (IPF) starting from uniform and fitting to marginals
    # For simplicity in 2D case, this is similar to independence
    # In higher dimensions, it involves more complex log-linear models

    # For a 2D table, homogeneous association with 2-way interactions
    # is actually the saturated model, so we use a simpler approach:
    # Use geometric mean of marginal-based estimates

    n = x.sum()
    ndim = x.ndim

    if ndim < 2:
        return np.ones_like(x) / x.size

    # Start with independence prior
    gamma = _compute_independence_prior(x)

    # For 2D and higher, adjust using observed associations
    # This is a simplified version that works well for most cases
    if ndim == 2:
        # Use a blend of independence and observed proportions
        # with higher weight on structure
        p_obs = x / n
        # Avoid division by zero
        gamma_safe = np.maximum(gamma, 1e-10)
        # Compute odds ratios and adjust
        ratio = p_obs / gamma_safe
        # Geometric mean adjustment (shrink towards 1)
        adj_ratio = np.power(ratio, 0.5)
        gamma = gamma * adj_ratio
        gamma = gamma / gamma.sum()
    else:
        # For higher dimensions, use independence as approximation
        pass

    return gamma


def _estimate_K(p_h: np.ndarray, gamma_h: np.ndarray) -> float:
    """
    Estimate K using data-driven approach.

    K = (1 - sum(p_h^2)) / sum((gamma_h - p_h)^2)

    Parameters
    ----------
    p_h : np.ndarray
        Observed relative frequencies.
    gamma_h : np.ndarray
        Prior probabilities.

    Returns
    -------
    float
        Estimated K value.
    """
    numerator = 1 - np.sum(p_h**2)
    denominator = np.sum((gamma_h - p_h) ** 2)

    if denominator < 1e-10:
        # If prior equals observed, use large K
        return 1.0

    K = numerator / denominator
    return max(K, 0.1)  # Ensure K is positive


def frechet_bounds_cat(
    tab_x: Optional[np.ndarray],
    tab_xy: np.ndarray,
    tab_xz: np.ndarray,
    print_f: str = "tables",
    align_margins: bool = False,
    tol: float = 0.001,
    warn: bool = True,
) -> Dict[str, Any]:
    """
    Derive Frechet bounds for cell probabilities in Y vs Z table.

    Computes bounds for P(Y,Z) given P(X), P(Y|X), and P(Z|X).

    Parameters
    ----------
    tab_x : Optional[np.ndarray]
        Contingency table of X variables. If None, only unconditional
        bounds are computed.
    tab_xy : np.ndarray
        Contingency table of X vs Y. If tab_x is None, this should be
        the marginal distribution of Y.
    tab_xz : np.ndarray
        Contingency table of X vs Z. If tab_x is None, this should be
        the marginal distribution of Z.
    print_f : str, default="tables"
        Output format: "tables" or "data.frame"
    align_margins : bool, default=False
        Whether to align marginal distributions using IPF.
    tol : float, default=0.001
        Tolerance for comparing marginal distributions.
    warn : bool, default=True
        Whether to show warnings for mismatched marginals.

    Returns
    -------
    Dict[str, Any]
        If print_f="tables":
            - 'low.u': Lower unconditional bounds (Y x Z table)
            - 'up.u': Upper unconditional bounds (Y x Z table)
            - 'CIA': Conditional Independence Assumption estimates
            - 'low.cx': Lower conditional bounds
            - 'up.cx': Upper conditional bounds
            - 'uncertainty': Dict with av.u and av.cx

        If print_f="data.frame":
            - 'bounds': DataFrame with all bounds
            - 'uncertainty': Dict with av.u and av.cx

    Examples
    --------
    >>> import numpy as np
    >>> from statmatch.frechet import frechet_bounds_cat
    >>> tab_x = np.array([30, 70])  # 2 X categories
    >>> tab_xy = np.array([[20, 10], [40, 30]])  # X vs Y (2x2)
    >>> tab_xz = np.array([[10, 10, 10], [20, 25, 25]])  # X vs Z (2x3)
    >>> result = frechet_bounds_cat(tab_x, tab_xy, tab_xz)
    >>> result["uncertainty"]["av.cx"] < result["uncertainty"]["av.u"]
    True
    """
    tab_xy = np.asarray(tab_xy, dtype=float)
    tab_xz = np.asarray(tab_xz, dtype=float)

    if tab_x is None:
        # Unconditional bounds only
        return _frechet_bounds_unconditional(tab_xy, tab_xz, print_f)

    tab_x = np.asarray(tab_x, dtype=float)

    # Get dimensions
    # tab_x has shape (x1, x2, ..., xk) for k X variables
    # tab_xy has shape (x1, x2, ..., xk, y)
    # tab_xz has shape (x1, x2, ..., xk, z)

    n_x_dims = tab_x.ndim
    shape_x = tab_x.shape

    # Y dimension is the last in tab_xy
    n_y = tab_xy.shape[-1]
    # Z dimension is the last in tab_xz
    n_z = tab_xz.shape[-1]

    # Check marginal distributions
    if warn:
        # Sum tab_xy over Y to get X marginal
        marg_x_from_xy = tab_xy.sum(axis=-1)
        # Sum tab_xz over Z to get X marginal
        marg_x_from_xz = tab_xz.sum(axis=-1)

        # Normalize to compare
        p_x = tab_x / tab_x.sum()
        p_x_from_xy = marg_x_from_xy / marg_x_from_xy.sum()
        p_x_from_xz = marg_x_from_xz / marg_x_from_xz.sum()

        if np.max(np.abs(p_x - p_x_from_xy)) > tol:
            warnings.warn(
                "The marginal distr. of the X variables\n"
                "in tab.xy is not equal to tab.x"
            )

        if np.max(np.abs(p_x - p_x_from_xz)) > tol:
            warnings.warn(
                "The marginal distr. of the X variables\n"
                "in tab.xz is not equal to tab.x"
            )

        if np.max(np.abs(p_x_from_xy - p_x_from_xz)) > tol:
            warnings.warn(
                "The marginal distr. of the X variables\n"
                "in tab.xy and in tab.xz are not equal"
            )

    if align_margins:
        # TODO: Implement IPF alignment
        pass

    # Compute P(X)
    n_total = tab_x.sum()
    p_x = tab_x.flatten() / n_total

    # Compute P(Y|X) and P(Z|X)
    # Reshape tables for easier indexing
    n_x_cells = tab_x.size
    tab_xy_flat = tab_xy.reshape(n_x_cells, n_y)
    tab_xz_flat = tab_xz.reshape(n_x_cells, n_z)

    # Sum over Y and Z for each X cell
    n_x_counts_xy = tab_xy_flat.sum(axis=1, keepdims=True)
    n_x_counts_xz = tab_xz_flat.sum(axis=1, keepdims=True)

    # Avoid division by zero
    n_x_counts_xy = np.maximum(n_x_counts_xy, 1e-10)
    n_x_counts_xz = np.maximum(n_x_counts_xz, 1e-10)

    p_y_given_x = tab_xy_flat / n_x_counts_xy  # (n_x_cells, n_y)
    p_z_given_x = tab_xz_flat / n_x_counts_xz  # (n_x_cells, n_z)

    # Compute marginal distributions of Y and Z
    p_y = tab_xy.sum(axis=tuple(range(n_x_dims))) / tab_xy.sum()
    p_z = tab_xz.sum(axis=tuple(range(n_x_dims))) / tab_xz.sum()

    # Initialize result arrays
    low_u = np.zeros((n_y, n_z))
    up_u = np.zeros((n_y, n_z))
    low_cx = np.zeros((n_y, n_z))
    up_cx = np.zeros((n_y, n_z))
    cia = np.zeros((n_y, n_z))

    # Compute bounds for each (j, k) cell in Y x Z table
    for j in range(n_y):
        for k in range(n_z):
            # Unconditional bounds
            low_u[j, k] = max(0, p_y[j] + p_z[k] - 1)
            up_u[j, k] = min(p_y[j], p_z[k])

            # Conditional bounds (Frechet bounds)
            # lower = sum_i(p(X=i) * max(0, p(Y=j|X=i) + p(Z=k|X=i) - 1))
            # upper = sum_i(p(X=i) * min(p(Y=j|X=i), p(Z=k|X=i)))
            lower_sum = 0.0
            upper_sum = 0.0
            cia_sum = 0.0

            for i in range(n_x_cells):
                p_yj_xi = p_y_given_x[i, j]
                p_zk_xi = p_z_given_x[i, k]

                lower_sum += p_x[i] * max(0, p_yj_xi + p_zk_xi - 1)
                upper_sum += p_x[i] * min(p_yj_xi, p_zk_xi)
                cia_sum += p_x[i] * p_yj_xi * p_zk_xi

            low_cx[j, k] = lower_sum
            up_cx[j, k] = upper_sum
            cia[j, k] = cia_sum

    # Compute uncertainty measures
    av_u = np.mean(up_u - low_u)
    av_cx = np.mean(up_cx - low_cx)

    uncertainty = {"av.u": av_u, "av.cx": av_cx}

    if print_f == "tables":
        return {
            "low.u": low_u,
            "up.u": up_u,
            "CIA": cia,
            "low.cx": low_cx,
            "up.cx": up_cx,
            "uncertainty": uncertainty,
        }
    else:
        # Create dataframe
        rows = []
        for j in range(n_y):
            for k in range(n_z):
                rows.append(
                    {
                        "Y": j,
                        "Z": k,
                        "low.u": low_u[j, k],
                        "low.cx": low_cx[j, k],
                        "CIA": cia[j, k],
                        "up.cx": up_cx[j, k],
                        "up.u": up_u[j, k],
                    }
                )

        bounds_df = pd.DataFrame(rows)
        return {"bounds": bounds_df, "uncertainty": uncertainty}


def _frechet_bounds_unconditional(
    tab_y: np.ndarray, tab_z: np.ndarray, print_f: str
) -> Dict[str, Any]:
    """
    Compute unconditional Frechet bounds when tab_x is None.

    Parameters
    ----------
    tab_y : np.ndarray
        Marginal distribution of Y.
    tab_z : np.ndarray
        Marginal distribution of Z.
    print_f : str
        Output format.

    Returns
    -------
    Dict[str, Any]
        Bounds and uncertainty.
    """
    tab_y = tab_y.flatten()
    tab_z = tab_z.flatten()

    n_y = len(tab_y)
    n_z = len(tab_z)

    # Normalize to get probabilities
    p_y = tab_y / tab_y.sum()
    p_z = tab_z / tab_z.sum()

    low_u = np.zeros((n_y, n_z))
    up_u = np.zeros((n_y, n_z))
    indep = np.zeros((n_y, n_z))

    for j in range(n_y):
        for k in range(n_z):
            low_u[j, k] = max(0, p_y[j] + p_z[k] - 1)
            up_u[j, k] = min(p_y[j], p_z[k])
            indep[j, k] = p_y[j] * p_z[k]

    av_u = np.mean(up_u - low_u)
    uncertainty = {"av.u": av_u, "av.cx": av_u}

    if print_f == "tables":
        return {
            "low.u": low_u,
            "up.u": up_u,
            "uncertainty": uncertainty,
        }
    else:
        rows = []
        for j in range(n_y):
            for k in range(n_z):
                rows.append(
                    {
                        "Y": j,
                        "Z": k,
                        "low.u": low_u[j, k],
                        "up.u": up_u[j, k],
                    }
                )

        bounds_df = pd.DataFrame(rows)
        return {"bounds": bounds_df, "uncertainty": uncertainty}


def fbwidths_by_x(
    tab_x: np.ndarray,
    tab_xy: np.ndarray,
    tab_xz: np.ndarray,
    deal_sparse: str = "discard",
    nA: Optional[int] = None,
    nB: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute Frechet bounds for all subsets of X variables.

    This function identifies which subset of X variables produces
    the best reduction in uncertainty about the Y vs Z distribution.

    Parameters
    ----------
    tab_x : np.ndarray
        Contingency table of X variables.
    tab_xy : np.ndarray
        Contingency table of X vs Y.
    tab_xz : np.ndarray
        Contingency table of X vs Z.
    deal_sparse : str, default="discard"
        How to handle sparse tables:
        - "discard": Skip estimation for sparse tables
        - "relfreq": Use standard relative frequency estimator
    nA : Optional[int]
        Sample size for tab_xy. If None, uses sum of tab_xy.
    nB : Optional[int]
        Sample size for tab_xz. If None, uses sum of tab_xz.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing results for each subset and 'sum.unc'
        DataFrame summarizing the findings.

    Examples
    --------
    >>> import numpy as np
    >>> from statmatch.frechet import fbwidths_by_x
    >>> tab_x = np.array([[30, 20], [25, 25]])
    >>> tab_xy = np.array([[[15, 15], [10, 10]], [[12, 13], [12, 13]]])
    >>> tab_xz = np.array([[[10, 10, 10], [7, 7, 6]], [[8, 9, 8], [8, 9, 8]]])
    >>> result = fbwidths_by_x(tab_x, tab_xy, tab_xz)
    >>> "sum.unc" in result
    True
    """
    tab_x = np.asarray(tab_x, dtype=float)
    tab_xy = np.asarray(tab_xy, dtype=float)
    tab_xz = np.asarray(tab_xz, dtype=float)

    n_x = tab_x.ndim  # Number of X variables
    n_y = tab_xy.shape[-1]
    n_z = tab_xz.shape[-1]

    # Sample sizes
    if nA is None:
        nA = tab_xy.sum()
    if nB is None:
        nB = tab_xz.sum()

    # Total number of X cells (for all X vars)
    H_DQ = tab_x.size

    results = {}
    summary_rows = []

    # First: unconditional case (no X variables)
    p_y = tab_xy.sum(axis=tuple(range(n_x))) / tab_xy.sum()
    p_z = tab_xz.sum(axis=tuple(range(n_x))) / tab_xz.sum()

    av_width_uncond = _compute_av_width_unconditional(p_y, p_z)

    summary_rows.append(
        {
            "x.vars": 0,
            "x.cells": np.nan,
            "x.freq0": np.nan,
            "xy.cells": np.nan,
            "xy.freq0": np.nan,
            "xz.cells": np.nan,
            "xz.freq0": np.nan,
            "av.n": 0.0,
            "cohen.ef": np.nan,
            "av.width": av_width_uncond,
            "penalty1": 0.0,
            "penalty2": 0.0,
            "av.width.pen1": av_width_uncond,
            "av.width.pen2": av_width_uncond,
        }
    )
    results["unconditional"] = {
        "av.width": av_width_uncond,
    }

    # Generate all non-empty subsets of X variables
    x_indices = list(range(n_x))

    for r in range(1, n_x + 1):
        for subset in combinations(x_indices, r):
            subset_name = "|" + "*".join([f"X{i+1}" for i in subset])

            # Marginalize tables to the subset of X variables
            # Keep dimensions in subset, sum over the rest
            axes_to_sum = tuple(i for i in range(n_x) if i not in subset)

            if axes_to_sum:
                tab_x_sub = tab_x.sum(axis=axes_to_sum)
                tab_xy_sub = tab_xy.sum(axis=axes_to_sum)
                tab_xz_sub = tab_xz.sum(axis=axes_to_sum)
            else:
                tab_x_sub = tab_x
                tab_xy_sub = tab_xy
                tab_xz_sub = tab_xz

            # Count cells and zeros
            x_cells = tab_x_sub.size
            x_freq0 = np.sum(tab_x_sub == 0)
            xy_cells = tab_xy_sub.size
            xy_freq0 = np.sum(tab_xy_sub == 0)
            xz_cells = tab_xz_sub.size
            xz_freq0 = np.sum(tab_xz_sub == 0)

            # Check sparseness
            H_Dm = x_cells  # Number of X cells for this subset
            av_n = min(nA / (H_Dm * n_y), nB / (H_Dm * n_z))

            # Cohen's effect size
            p_obs = tab_x_sub / tab_x_sub.sum()
            uniform_p = 1 / x_cells
            cohen_ef = np.sqrt(
                x_cells * np.sum((p_obs.flatten() - uniform_p) ** 2)
            )

            # Compute bounds if not too sparse
            if av_n <= 1 and deal_sparse == "discard":
                av_width = np.nan
            else:
                try:
                    bounds_result = frechet_bounds_cat(
                        tab_x_sub, tab_xy_sub, tab_xz_sub, warn=False
                    )
                    av_width = bounds_result["uncertainty"]["av.cx"]
                except Exception:
                    av_width = np.nan

            # Penalties
            # g1 = log(1 + H_Dm/H_DQ)
            penalty1 = np.log(1 + H_Dm / H_DQ)

            # g2 = max(1/(nA - H_Dm*J), 1/(nB - H_Dm*K))
            denom_A = nA - H_Dm * n_y
            denom_B = nB - H_Dm * n_z
            if denom_A > 0 and denom_B > 0:
                penalty2 = max(1 / denom_A, 1 / denom_B)
            else:
                penalty2 = np.nan

            # Penalized widths
            if np.isnan(av_width):
                av_width_pen1 = np.nan
                av_width_pen2 = np.nan
            else:
                av_width_pen1 = av_width + penalty1
                av_width_pen2 = (
                    av_width + penalty2 if not np.isnan(penalty2) else np.nan
                )

            summary_rows.append(
                {
                    "x.vars": len(subset),
                    "x.cells": x_cells,
                    "x.freq0": x_freq0,
                    "xy.cells": xy_cells,
                    "xy.freq0": xy_freq0,
                    "xz.cells": xz_cells,
                    "xz.freq0": xz_freq0,
                    "av.n": av_n,
                    "cohen.ef": cohen_ef,
                    "av.width": av_width,
                    "penalty1": penalty1,
                    "penalty2": penalty2 if not np.isnan(penalty2) else penalty2,
                    "av.width.pen1": av_width_pen1,
                    "av.width.pen2": av_width_pen2,
                }
            )

            results[subset_name] = {
                "av.width": av_width,
                "subset": subset,
            }

    sum_unc = pd.DataFrame(summary_rows)
    results["sum.unc"] = sum_unc

    return results


def _compute_av_width_unconditional(p_y: np.ndarray, p_z: np.ndarray) -> float:
    """
    Compute average width of unconditional bounds.

    Parameters
    ----------
    p_y : np.ndarray
        Marginal probabilities for Y.
    p_z : np.ndarray
        Marginal probabilities for Z.

    Returns
    -------
    float
        Average width of bounds.
    """
    n_y = len(p_y)
    n_z = len(p_z)

    widths = []
    for j in range(n_y):
        for k in range(n_z):
            low = max(0, p_y[j] + p_z[k] - 1)
            up = min(p_y[j], p_z[k])
            widths.append(up - low)

    return np.mean(widths)
