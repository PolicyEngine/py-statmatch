# py-statmatch

Python implementation of R's StatMatch package for statistical matching and data fusion.

## Installation

```bash
pip install py-statmatch
```

## Features

- **NND.hotdeck**: Distance hot deck method for finding nearest neighbor donors
- **RANDwNND.hotdeck**: Random distance hot deck (randomly select from k nearest neighbors)
- **rankNND.hotdeck**: Rank distance hot deck using Mahalanobis distance
- **create.fused**: Create fused dataset from matching results

## Usage

```python
from statmatch import nnd_hotdeck

# Perform nearest neighbor distance hot deck imputation
matched_data, donor_indices = nnd_hotdeck(
    receiver=receiver_df,
    donor=donor_df,
    matching_variables=['age', 'income'],
    z_variables=['spending']
)
```

## Development

This package is under active development. Contributions are welcome!

## License

MIT License - See LICENSE file for details.