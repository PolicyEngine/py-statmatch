# Changelog

All notable changes to py-statmatch are documented here.

## [0.1.0] - 2024

### Added

- **Core Matching Functions**
  - `nnd_hotdeck`: Nearest Neighbor Distance Hot Deck matching
  - `rand_hotdeck`: Random selection from k-nearest donors (RANDwNND)
  - `rank_nnd_hotdeck`: ECDF-based rank matching
  - `create_fused`: Create synthetic fused datasets

- **Distance Functions**
  - `gower_dist`: Gower's distance for mixed-type data
  - `mahalanobis_dist`: Mahalanobis distance
  - `maximum_dist`: Chebyshev/L-infinity distance

- **Comparison Functions**
  - `comp_cont`: Compare continuous distributions
  - `comp_prop`: Compare categorical proportions
  - `pw_assoc`: Pairwise association measures

- **Frechet Bounds**
  - `frechet_bounds_cat`: Compute Frechet bounds for categorical data
  - `fbwidths_by_x`: Bounds for all X variable subsets
  - `p_bayes`: Pseudo-Bayes estimation for sparse tables

- **Plotting Functions**
  - `plot_bounds`: Visualize Frechet bounds
  - `plot_cont`: Compare continuous distributions
  - `plot_tab`: Compare contingency tables

- **Sample Utilities**
  - `comb_samples`: Combine survey samples
  - `harmonize_x`: Harmonize matching variables
  - `fact2dummy`: Factor to dummy variable conversion

- **Mixed Matching**
  - `mixed_mtc`: Mixed matching with multiple constraints
  - `sel_mtc_by_unc`: Select matches by uncertainty

### Notes

- All 21 functions produce results identical to R's StatMatch package
- Test suite includes R comparison tests (requires rpy2)
