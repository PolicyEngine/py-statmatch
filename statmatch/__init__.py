"""
py-statmatch: Python implementation of R's StatMatch package.

Statistical matching and data fusion tools.
"""

from .nnd_hotdeck import nnd_hotdeck
from .create_fused import create_fused
from .rand_hotdeck import rand_hotdeck
from .rank_hotdeck import rank_nnd_hotdeck
from .distances import gower_dist, mahalanobis_dist, maximum_dist
from .plotting import plot_bounds, plot_cont, plot_tab
from .comparison import comp_cont, comp_prop, pw_assoc
from .sample_utils import comb_samples, fact2dummy, harmonize_x
from .mixed_mtc import mixed_mtc, sel_mtc_by_unc
from .frechet import frechet_bounds_cat, fbwidths_by_x, p_bayes
from .diagnostics import (
    MatchDiagnostics,
    ks_test_balance,
    love_plot,
    match_diagnostics,
    standardized_mean_diff,
    variance_ratio,
)
from .survey_utils import (
    calibrate_weights,
    design_effect,
    replicate_variance,
    weighted_distance,
)

__version__ = "0.2.0"
__all__ = [
    # Hot deck matching
    "nnd_hotdeck",
    "rand_hotdeck",
    "rank_nnd_hotdeck",
    # Fused dataset creation
    "create_fused",
    # Distance functions
    "gower_dist",
    "mahalanobis_dist",
    "maximum_dist",
    # Plotting
    "plot_bounds",
    "plot_cont",
    "plot_tab",
    # Comparison functions
    "comp_cont",
    "comp_prop",
    "pw_assoc",
    # Sample utilities
    "fact2dummy",
    "harmonize_x",
    "comb_samples",
    # Mixed matching
    "mixed_mtc",
    "sel_mtc_by_unc",
    # Frechet bounds
    "frechet_bounds_cat",
    "fbwidths_by_x",
    "p_bayes",
    # Diagnostics
    "MatchDiagnostics",
    "ks_test_balance",
    "love_plot",
    "match_diagnostics",
    "standardized_mean_diff",
    "variance_ratio",
    # Survey utilities
    "calibrate_weights",
    "design_effect",
    "replicate_variance",
    "weighted_distance",
]
