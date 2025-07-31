# py-statmatch

```{include} ../README.md
:start-after: # py-statmatch
:end-before: ## Installation
```

```{toctree}
:hidden:
:maxdepth: 2

getting-started
api-reference
examples/index
methodology
contributing
changelog
```

## What is Statistical Matching?

Statistical matching, also known as data fusion or synthetic data matching, is a technique used to integrate information from different data sources that share some common variables but have no or few units in common. This is particularly useful when:

- You have two datasets with different but complementary variables
- Direct linkage is not possible due to lack of common identifiers
- You want to enrich one dataset with information from another

## Key Features

```{include} ../README.md
:start-after: ## Features
:end-before: ## Installation
```

## Quick Example

Here's a simple example of using py-statmatch for nearest neighbor matching:

```python
import pandas as pd
from statmatch import nnd_hotdeck

# Donor data has variables X and Y
donor_data = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'income': [30000, 45000, 55000, 65000],
    'satisfaction': [7, 8, 6, 9]  # This will be donated
})

# Recipient data has only X variables
recipient_data = pd.DataFrame({
    'age': [28, 33, 42],
    'income': [35000, 50000, 70000]
})

# Perform matching
result = nnd_hotdeck(
    data_rec=recipient_data,
    data_don=donor_data,
    match_vars=['age', 'income']
)

# Create fused dataset
fused = recipient_data.copy()
fused['satisfaction'] = donor_data.iloc[result['noad.index']]['satisfaction'].values
```

## Next Steps

- Read the {doc}`getting-started` guide for installation and basic usage
- Check out the {doc}`api-reference` for detailed function documentation
- Explore {doc}`examples/index` for more complex use cases
- Learn about the {doc}`methodology` behind statistical matching