# py-statmatch

Python implementation of R's StatMatch package for statistical matching and data fusion, **plus advanced methods not available in R**.

## Overview

`py-statmatch` provides tools for statistical matching (also known as data fusion or synthetic data matching) between different datasets. This package started as a Python port of the R package [StatMatch](https://cran.r-project.org/web/packages/StatMatch/), but now includes additional modern techniques.

## Features

### Core R StatMatch Functions (21 functions)
All functions produce **identical results** to R's StatMatch package.

- **Hot Deck Matching**
  - `nnd_hotdeck`: Nearest Neighbor Distance Hot Deck
  - `rand_hotdeck`: Random selection from k-nearest donors
  - `rank_nnd_hotdeck`: ECDF-based rank matching
  - `create_fused`: Create synthetic fused datasets

- **Distance Functions**
  - `gower_dist`: Gower's distance for mixed-type data
  - `mahalanobis_dist`: Covariance-adjusted distance
  - `maximum_dist`: Chebyshev/L-infinity distance

- **Frechet Bounds**
  - `frechet_bounds_cat`: Bounds for categorical data
  - `fbwidths_by_x`: Bounds for all X variable subsets
  - `p_bayes`: Pseudo-Bayes estimation

- **Comparison & Plotting**
  - `comp_cont`, `comp_prop`, `pw_assoc`
  - `plot_bounds`, `plot_cont`, `plot_tab`

- **Sample Utilities**
  - `comb_samples`, `harmonize_x`, `fact2dummy`
  - `mixed_mtc`, `sel_mtc_by_unc`

### Advanced Methods (Beyond R)

- **Multiple Imputation** (`mi_nnd_hotdeck`, `combine_mi_estimates`)
  - Generate m imputed datasets with proper uncertainty quantification
  - Rubin's combining rules for valid inference

- **ML-Based Propensity Matching** (`propensity_hotdeck`)
  - Gradient Boosting, Random Forest, Neural Network, Logistic
  - Caliper matching support

- **Optimal Transport** (`ot_hotdeck`, `wasserstein_dist`)
  - Globally optimal matching via Earth Mover's Distance
  - Entropy-regularized Sinkhorn for efficiency

- **Bayesian Uncertainty** (`bayesian_match`, `credible_interval`)
  - Posterior inference on matched values
  - CIA (Conditional Independence Assumption) testing

- **Embedding Distance** (`learn_embeddings`, `embedding_dist`)
  - Target encoding and SVD for high-cardinality categoricals
  - Better handling of complex categorical relationships

- **Survey Weights** (`calibrate_weights`, `design_effect`, `replicate_variance`)
  - Complex survey design support
  - Weight calibration via iterative proportional fitting

- **Diagnostics Dashboard** (`match_diagnostics`, `love_plot`)
  - Balance tables, SMD calculation
  - HTML report generation

## Installation

```bash
pip install py-statmatch
```

For development:
```bash
git clone https://github.com/PolicyEngine/py-statmatch.git
cd py-statmatch
pip install -e ".[dev]"
```

## Quick Start

### Basic Matching

```python
import pandas as pd
from statmatch import nnd_hotdeck, create_fused

# Donor dataset (has X and Y variables)
donors = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [30000, 45000, 55000, 65000, 80000],
    'satisfaction': [7, 8, 6, 9, 8]  # Variable to donate
})

# Recipient dataset (has X but missing Y)
recipients = pd.DataFrame({
    'age': [28, 33, 42],
    'income': [35000, 50000, 70000]
})

# Perform matching
result = nnd_hotdeck(
    data_rec=recipients,
    data_don=donors,
    match_vars=['age', 'income']
)

# Create fused dataset
fused = create_fused(
    data_rec=recipients,
    data_don=donors,
    mtc_ids=result['mtc.ids'],
    z_vars=['satisfaction']
)
```

### Multiple Imputation (Proper Uncertainty)

```python
from statmatch import mi_nnd_hotdeck, mi_create_fused, mi_summary

# Generate 5 imputed datasets
mi_results = mi_nnd_hotdeck(
    data_rec=recipients,
    data_don=donors,
    match_vars=['age', 'income'],
    m=5
)

# Create fused datasets
fused_datasets = mi_create_fused(
    data_rec=recipients,
    data_don=donors,
    mi_results=mi_results,
    z_vars=['satisfaction']
)

# Get summary with confidence intervals
summary = mi_summary(fused_datasets, 'satisfaction')
print(summary)  # estimate, std_error, ci_lower, ci_upper
```

### ML Propensity Matching

```python
from statmatch import propensity_hotdeck

result = propensity_hotdeck(
    data_rec=recipients,
    data_don=donors,
    match_vars=['age', 'income'],
    estimator='gbm',  # or 'random_forest', 'neural_net', 'logistic'
    caliper=0.1       # optional: max propensity score distance
)
```

### Match Quality Diagnostics

```python
from statmatch import match_diagnostics

diag = match_diagnostics(
    result=result,
    data_rec=recipients,
    data_don=donors,
    match_vars=['age', 'income']
)

# View balance table
print(diag.balance_table())

# Generate HTML report
diag.to_html('match_report.html')

# Love plot visualization
diag.love_plot()
```

## Documentation

Full documentation: https://policyengine.github.io/py-statmatch/

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=statmatch

# Run R comparison tests (requires R + rpy2 + StatMatch)
pytest -k "against_r" -v

# Format code
black . -l 79
```

## License

MIT License

## Citation

```bibtex
@software{pystatmatch2024,
  title = {py-statmatch: Statistical matching in Python with advanced methods},
  author = {PolicyEngine},
  year = {2024},
  url = {https://github.com/PolicyEngine/py-statmatch}
}
```

## Acknowledgments

Core matching functions are a Python port of the R StatMatch package by Marcello D'Orazio. Advanced methods (MI, propensity, OT, Bayesian, embeddings, diagnostics) are original contributions.
