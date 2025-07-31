# py-statmatch

Python implementation of R's StatMatch package for statistical matching and data fusion.

## Overview

`py-statmatch` provides tools for statistical matching (also known as data fusion or synthetic data matching) between different datasets. This package is a Python port of the popular R package [StatMatch](https://cran.r-project.org/web/packages/StatMatch/), implementing various methods to match records from different data sources that share some common variables.

## Features

- **NND.hotdeck**: Nearest Neighbor Distance Hot Deck matching
  - Multiple distance metrics (Euclidean, Manhattan, Mahalanobis, etc.)
  - Donation classes (match within groups/strata)
  - Constrained matching using Hungarian algorithm
  - Handle missing values appropriately
- **Coming soon**:
  - RANDwNND.hotdeck: Random distance hot deck
  - rankNND.hotdeck: Rank distance hot deck
  - create.fused: Create fused datasets from matching results

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

```python
import pandas as pd
from statmatch import nnd_hotdeck

# Create donor dataset (has X and Y variables)
donor_data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [30000, 45000, 55000, 65000, 80000],
    'education': ['HS', 'BA', 'BA', 'MA', 'PhD'],
    'job_satisfaction': [7, 8, 6, 9, 8]  # Variable to donate
})

# Create recipient dataset (has X variables but missing Y)
recipient_data = pd.DataFrame({
    'age': [28, 33, 42],
    'income': [35000, 50000, 70000],
    'education': ['BA', 'BA', 'MA']
})

# Perform matching
result = nnd_hotdeck(
    data_rec=recipient_data,
    data_don=donor_data,
    match_vars=['age', 'income'],
    dist_fun='euclidean'
)

# Get matched donor indices
print(result['noad.index'])  # [0, 1, 3] for example

# Create fused dataset
fused_data = recipient_data.copy()
fused_data['job_satisfaction'] = donor_data.iloc[result['noad.index']]['job_satisfaction'].values
```

## API Reference

### nnd_hotdeck

```python
nnd_hotdeck(
    data_rec,
    data_don,
    match_vars,
    don_class=None,
    dist_fun="euclidean",
    cut_don=None,
    k=None,
    w_don=None,
    w_rec=None,
    constr_alg=None
)
```

**Parameters:**
- `data_rec` (pd.DataFrame): The recipient dataset
- `data_don` (pd.DataFrame): The donor dataset
- `match_vars` (List[str]): List of variable names to use for matching
- `don_class` (str, optional): Variable name defining donation classes
- `dist_fun` (str): Distance function - "euclidean", "manhattan", "mahalanobis", etc.
- `k` (int, optional): Maximum number of times each donor can be used
- `constr_alg` (str, optional): Algorithm for constrained matching - "lpsolve" or "hungarian"

**Returns:**
- Dictionary containing:
  - `mtc.ids`: DataFrame with recipient and donor IDs
  - `noad.index`: Array of donor indices for each recipient (0-based)
  - `dist.rd`: Array of distances between matched recipients and donors

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=statmatch

# Run specific test
pytest tests/test_nnd_hotdeck.py::TestNNDHotdeck::test_euclidean_distance_matching
```

### Code Style

This project uses Black for code formatting:

```bash
black . -l 79
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this package in your research, please cite both this package and the original R package:

```bibtex
@software{pystatmatch2024,
  title = {py-statmatch: Python implementation of R's StatMatch package},
  author = {PolicyEngine},
  year = {2024},
  url = {https://github.com/PolicyEngine/py-statmatch}
}

@Manual{rstatmatch,
  title = {StatMatch: Statistical Matching or Data Fusion},
  author = {Marcello D'Orazio},
  year = {2023},
  note = {R package version 1.4.2},
  url = {https://CRAN.R-project.org/package=StatMatch},
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

This is a Python port of the R StatMatch package by Marcello D'Orazio. We are grateful for the original implementation which has been invaluable to the statistical matching community.