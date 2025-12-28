# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

py-statmatch is a Python implementation of R's StatMatch package for statistical matching and data fusion. It enables matching records from different data sources that share common variables, allowing imputation of missing variables from a donor dataset to a recipient dataset.

## Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests with coverage
pytest --cov=statmatch

# Run specific test
pytest tests/test_nnd_hotdeck.py::TestNNDHotdeck::test_euclidean_distance_matching

# Run tests against R implementation (requires R + rpy2 + StatMatch)
pytest tests/test_nnd_hotdeck.py::TestNNDHotdeck::test_basic_matching_against_r

# Format code (line length 79)
black . -l 79

# Check formatting without modifying
black . -l 79 --check

# Build documentation
cd docs && jupyter-book build .

# Build package
python -m build
```

## CI Workflows

- **Push to main**: Runs lint, tests (Python 3.13), and deploys docs to GitHub Pages
- **Pull requests**: Runs lint, tests, and validates changelog entry

PRs require a `changelog_entry.yaml` file describing the change.

## Architecture

The package follows a simple structure:

- `statmatch/nnd_hotdeck.py` - Core implementation of Nearest Neighbor Distance Hot Deck matching
  - `nnd_hotdeck()` - Main public API function
  - `_find_matches()` - Internal function for computing distances and finding nearest neighbors
  - `_constrained_matching()` - Internal function for constrained matching (Hungarian algorithm)

### Key Concepts

**Statistical Matching**: Records from a "donor" dataset (containing variables X and Y) are matched to records in a "recipient" dataset (containing only X) based on similarity in the matching variables X. The Y values are then "donated" to the recipients.

**Distance Functions**: euclidean, manhattan, mahalanobis, minimax (chebyshev), cosine

**Donation Classes**: Optional stratification variable that restricts matches to occur only within the same class

**Constrained Matching**: Limits how many times each donor can be used via the Hungarian algorithm

## Testing Against R

The test suite includes comparisons against the original R StatMatch package. These tests verify that distances match exactly between implementations.

**Requirements:**
1. R installed
2. rpy2 Python package
3. StatMatch R package (`install.packages("StatMatch")` in R)

Tests with `@pytest.mark.skipif(not R_AVAILABLE, ...)` are skipped if R is not available.

**Known differences from R:**
- R returns results grouped by donation class; Python returns results in original recipient order
- When donors are equidistant (ties), Python picks the lower index; R may pick differently. Both are valid nearest neighbors.
