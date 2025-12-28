# Methodology

This page documents the statistical methods implemented in py-statmatch and how we validate our implementation against R's StatMatch package.

## R Validation Approach

py-statmatch is designed to produce **identical results** to R's StatMatch package. Our validation approach ensures numerical equivalence:

### Test Suite

The test suite includes approximately 15+ comparison tests that verify:

1. **Distance calculations** match to 1e-10 tolerance
2. **Matched donor indices** are identical (accounting for ties)
3. **Frechet bounds** match within floating-point precision

### Running R Comparison Tests

To run the full R comparison test suite:

```bash
# Prerequisites:
# 1. Install R
# 2. Install rpy2: pip install rpy2
# 3. Install StatMatch in R: install.packages("StatMatch")

# Run comparison tests
pytest tests/ -k "against_r" -v
```

### Handling Ties

When multiple donors have the same distance to a recipient, both implementations pick a valid nearest neighbor, but may pick different ones. Our tests account for this by:

1. Verifying the selected donor is indeed at the minimum distance
2. Accepting either choice when ties exist

## Statistical Matching Theory

### Overview

Statistical matching (also known as data fusion) combines information from multiple datasets that share common variables (X) but have different unique variables (Y and Z). The goal is to create a synthetic dataset containing all variables.

```
Dataset A: [X, Y]     →     Fused: [X, Y, Z]
Dataset B: [X, Z]     →
```

### Conditional Independence Assumption (CIA)

The fundamental assumption in statistical matching is:

> **Y and Z are independent given X**

This means that once we control for the common variables X, knowing Y provides no additional information about Z (and vice versa).

### Frechet Bounds

When CIA doesn't hold, we can compute bounds on the joint distribution P(Y, Z | X) using marginal information:

- **Lower bound**: Maximum possible negative dependence
- **Upper bound**: Maximum possible positive dependence

The `frechet_bounds_cat` function computes these bounds for categorical variables.

## Hot Deck Matching Methods

### Nearest Neighbor Distance (NND)

The `nnd_hotdeck` function finds the nearest donor for each recipient based on distance in the X space:

1. Compute pairwise distances between all recipients and donors
2. For each recipient, select the donor with minimum distance
3. Optionally apply constraints (max uses per donor)

**Distance metrics:**
- Euclidean (default)
- Manhattan (L1)
- Mahalanobis (covariance-adjusted)
- Chebyshev/Minimax (L∞)
- Cosine similarity

### Random from k-Nearest (RANDwNND)

The `rand_hotdeck` function adds randomness by:

1. Finding k nearest donors for each recipient
2. Randomly selecting one from this set

This reduces bias from always picking the single nearest neighbor.

### Rank-based NND

The `rank_nnd_hotdeck` function uses ECDF (empirical cumulative distribution function) transformation:

1. Transform each variable to its rank percentile
2. Perform matching in the rank space

This provides better matching when variables have different scales or non-linear relationships.

## Constrained Matching

When donors are scarce or you want to limit reuse:

### Hungarian Algorithm

Optimal one-to-one assignment that minimizes total distance. Available via:

```python
nnd_hotdeck(..., k=1, constr_alg='hungarian')
```

### LP Relaxation

Linear programming approach for when k > 1:

```python
nnd_hotdeck(..., k=3, constr_alg='lpsolve')
```

## Distance Functions

### Gower Distance

For mixed-type data (numeric, categorical, ordinal):

```python
from statmatch import gower_dist

dist = gower_dist(data_x, data_y, kr_corr=True)
```

The Kaufman-Rousseeuw correction (`kr_corr=True`) improves handling of ordinal variables.

### Mahalanobis Distance

Accounts for variable correlations and scales:

```python
from statmatch import mahalanobis_dist

dist = mahalanobis_dist(data_x, data_y, vc=None)
```

When `vc=None`, the covariance matrix is estimated from the combined data.

## References

1. D'Orazio, M., Di Zio, M., & Scanu, M. (2006). *Statistical Matching: Theory and Practice*. John Wiley & Sons.

2. Rässler, S. (2002). *Statistical Matching: A Frequentist Theory, Practical Applications, and Alternative Bayesian Approaches*. Springer.

3. StatMatch R Package: [CRAN](https://cran.r-project.org/package=StatMatch)
