# API Reference

## Main Functions

### nnd_hotdeck

```{eval-rst}
.. autofunction:: statmatch.nnd_hotdeck
```

## Parameters

### data_rec : pd.DataFrame
The recipient dataset. This is the dataset that needs variables to be imputed.

### data_don : pd.DataFrame  
The donor dataset. This dataset contains the variables that will be donated to recipients.

### match_vars : List[str]
List of variable names to use for matching. These variables must exist in both datasets.

### don_class : Optional[str] = None
Variable name defining donation classes. When specified, matching is constrained to occur only within the same class. This is useful for ensuring matches respect important grouping variables like geographic region or demographic categories.

### dist_fun : str = "euclidean"
Distance function to use. Options include:
- `"euclidean"`: Standard Euclidean distance
- `"manhattan"`: Manhattan/city-block distance (sum of absolute differences)
- `"mahalanobis"`: Mahalanobis distance (accounts for variable correlations)
- `"minimax"` or `"chebyshev"`: Maximum absolute difference
- `"cosine"`: Cosine distance

### k : Optional[int] = None
Maximum number of times each donor can be used. When specified with `constr_alg`, enables constrained matching to ensure donors aren't over-used.

### constr_alg : Optional[str] = None
Algorithm for constrained matching:
- `"hungarian"`: Uses the Hungarian algorithm for optimal assignment
- `"lpsolve"`: Uses linear programming via the lpSolve library

Both require `k` to be specified.

### cut_don : Optional[str] = None
*(Not yet implemented)* Variable name to use for cutting the donor pool.

### w_don : Optional[Union[np.ndarray, List[float]]] = None
*(Not yet implemented)* Weights for donor units.

### w_rec : Optional[Union[np.ndarray, List[float]]] = None
*(Not yet implemented)* Weights for recipient units.

## Return Value

Returns a dictionary containing:

### mtc.ids : pd.DataFrame
DataFrame with two columns:
- `rec.id`: Recipient identifiers (index from recipient dataset)
- `don.id`: Matched donor identifiers (index from donor dataset)

### noad.index : np.ndarray
Array of donor indices (0-based) for each recipient. This provides direct indexing into the donor dataset.

### dist.rd : np.ndarray  
Array of distances between each recipient and their matched donor. Lower values indicate better matches.

## Example Usage

```python
import pandas as pd
import numpy as np
from statmatch import nnd_hotdeck

# Create example datasets
donors = pd.DataFrame({
    'age': np.random.normal(40, 10, 100),
    'income': np.random.normal(50000, 15000, 100),
    'satisfaction': np.random.randint(1, 11, 100)
})

recipients = pd.DataFrame({
    'age': np.random.normal(38, 12, 50),
    'income': np.random.normal(48000, 16000, 50)
})

# Basic matching
result = nnd_hotdeck(
    data_rec=recipients,
    data_don=donors,
    match_vars=['age', 'income']
)

# Matching with constraints
result_constrained = nnd_hotdeck(
    data_rec=recipients,
    data_don=donors,
    match_vars=['age', 'income'],
    k=2,  # Each donor used at most twice
    constr_alg='hungarian'
)

# Matching within classes
donors['region'] = np.random.choice(['A', 'B', 'C'], 100)
recipients['region'] = np.random.choice(['A', 'B', 'C'], 50)

result_classed = nnd_hotdeck(
    data_rec=recipients,
    data_don=donors,
    match_vars=['age', 'income'],
    don_class='region'
)
```

## Notes

- The function automatically handles missing values by imputing with column means
- When using donation classes, ensure the class variable exists in both datasets
- For large datasets, consider using constrained matching to improve donor diversity
- Distance calculations are performed after standardizing variables internally