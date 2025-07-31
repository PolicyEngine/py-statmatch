# Getting Started

## Installation

### From PyPI

The easiest way to install py-statmatch is via pip:

```bash
pip install py-statmatch
```

### From Source

For development or to get the latest features:

```bash
git clone https://github.com/PolicyEngine/py-statmatch.git
cd py-statmatch
pip install -e ".[dev]"
```

## Basic Usage

The main function in py-statmatch is `nnd_hotdeck`, which performs Nearest Neighbor Distance Hot Deck matching:

```python
from statmatch import nnd_hotdeck
import pandas as pd

# Load your data
donor_df = pd.read_csv('donors.csv')
recipient_df = pd.read_csv('recipients.csv')

# Perform matching
result = nnd_hotdeck(
    data_rec=recipient_df,
    data_don=donor_df,
    match_vars=['var1', 'var2', 'var3']
)
```

## Understanding the Results

The `nnd_hotdeck` function returns a dictionary with three components:

1. **`mtc.ids`**: A DataFrame mapping recipient IDs to donor IDs
2. **`noad.index`**: An array of donor indices (0-based) for each recipient
3. **`dist.rd`**: An array of distances between matched recipients and donors

### Example: Creating a Fused Dataset

```python
# Perform matching
result = nnd_hotdeck(
    data_rec=recipients,
    data_don=donors,
    match_vars=['age', 'income', 'education']
)

# Create fused dataset
fused_data = recipients.copy()

# Add variables from donors
for col in ['job_satisfaction', 'health_score']:
    fused_data[col] = donors.iloc[result['noad.index']][col].values

# Add matching quality information
fused_data['match_distance'] = result['dist.rd']
fused_data['donor_id'] = result['noad.index']
```

## Common Parameters

### Distance Functions

py-statmatch supports various distance metrics:

- `"euclidean"` (default): Standard Euclidean distance
- `"manhattan"`: City-block distance
- `"mahalanobis"`: Accounts for variable correlations
- `"minimax"`: Maximum absolute difference
- `"cosine"`: Cosine similarity

### Donation Classes

Match within specific groups using the `don_class` parameter:

```python
result = nnd_hotdeck(
    data_rec=recipients,
    data_don=donors,
    match_vars=['age', 'income'],
    don_class='region'  # Match only within same region
)
```

### Constrained Matching

Limit how many times each donor can be used:

```python
result = nnd_hotdeck(
    data_rec=recipients,
    data_don=donors,
    match_vars=['age', 'income'],
    k=3,  # Each donor used at most 3 times
    constr_alg='hungarian'  # or 'lpsolve'
)
```

## Handling Missing Values

py-statmatch automatically handles missing values by imputing them with column means during distance calculations. This ensures matching can proceed even with incomplete data:

```python
# Data with missing values
donors_with_na = pd.DataFrame({
    'x1': [1.0, 2.0, np.nan, 4.0],
    'x2': [2.0, np.nan, 6.0, 8.0],
    'y': [10, 20, 30, 40]
})

# Matching will still work
result = nnd_hotdeck(
    data_rec=recipients,
    data_don=donors_with_na,
    match_vars=['x1', 'x2']
)
```

## Next Steps

- See {doc}`examples/index` for more detailed examples
- Check the {doc}`api-reference` for all available options
- Read about the {doc}`methodology` to understand the algorithms