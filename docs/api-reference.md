# API Reference

This page documents all 21 functions in py-statmatch, organized by category.

## Hot Deck Matching

### nnd_hotdeck

Nearest Neighbor Distance Hot Deck matching.

```python
from statmatch import nnd_hotdeck

result = nnd_hotdeck(
    data_rec=recipients,
    data_don=donors,
    match_vars=['age', 'income'],
    don_class=None,      # Optional: stratification variable
    dist_fun='euclidean', # Options: euclidean, manhattan, mahalanobis, minimax, cosine
    k=None,              # Max times each donor can be used
    constr_alg=None      # 'hungarian' or 'lpsolve' for constrained matching
)
```

**Returns**: Dict with `mtc.ids`, `noad.index`, `dist.rd`

### rand_hotdeck

Random selection from k-nearest donors (RANDwNND.hotdeck).

```python
from statmatch import rand_hotdeck

result = rand_hotdeck(
    data_rec=recipients,
    data_don=donors,
    match_vars=['age', 'income'],
    don_class=None,
    dist_fun='euclidean',
    cut_don='rot'  # Options: 'rot', 'span', 'exact', 'min', 'k.dist'
)
```

**Returns**: Dict with `mtc.ids`, `sum.dist`, `noad`

### rank_nnd_hotdeck

ECDF-based rank matching.

```python
from statmatch import rank_nnd_hotdeck

result = rank_nnd_hotdeck(
    data_rec=recipients,
    data_don=donors,
    match_vars=['age', 'income'],
    don_class=None,
    dist_fun='euclidean'
)
```

**Returns**: Dict with `mtc.ids`, `dist.rd`, `noad`

---

## Data Fusion

### create_fused

Create a synthetic fused dataset from matching results.

```python
from statmatch import create_fused

fused = create_fused(
    data_rec=recipients,
    data_don=donors,
    mtc_ids=result['mtc.ids'],
    z_vars=['satisfaction', 'rating'],  # Variables to donate
    dup_x=False  # Include matching vars with '.don' suffix
)
```

**Returns**: DataFrame with recipient data plus donated z variables

---

## Distance Functions

All distance functions return a distance matrix of shape `(n_x, n_y)`.

### gower_dist

Gower's distance for mixed-type data (numeric, categorical, ordered).

```python
from statmatch import gower_dist

dist_matrix = gower_dist(
    data_x=df1,
    data_y=df2,          # Optional, defaults to data_x
    rngs=None,           # Custom ranges for normalization
    kr_corr=True,        # Kaufman-Rousseeuw correction for ordinals
    var_weights=None     # Variable weights
)
```

### mahalanobis_dist

Mahalanobis distance (covariance-adjusted).

```python
from statmatch import mahalanobis_dist

dist_matrix = mahalanobis_dist(
    data_x=df1,
    data_y=df2,  # Optional
    vc=None      # Custom covariance matrix
)
```

### maximum_dist

Maximum (L-infinity/Chebyshev) distance.

```python
from statmatch import maximum_dist

dist_matrix = maximum_dist(
    data_x=df1,
    data_y=df2,   # Optional
    rank=False    # Use rank transformation for scale invariance
)
```

---

## Comparison Functions

### comp_cont

Compare continuous variable distributions between datasets.

```python
from statmatch import comp_cont

result = comp_cont(
    data_A=df_a['income'],
    data_B=df_b['income'],
    w_A=weights_a,       # Optional weights
    w_B=weights_b
)
```

**Returns**: Dict with KS statistic, TVD, Hellinger distance, etc.

### comp_prop

Compare categorical proportions between datasets.

```python
from statmatch import comp_prop

result = comp_prop(
    data_A=df_a['category'],
    data_B=df_b['category'],
    w_A=None,
    w_B=None
)
```

**Returns**: Dict with TVD, chi-square statistic, overlap coefficient

### pw_assoc

Compute pairwise association measures.

```python
from statmatch import pw_assoc

result = pw_assoc(
    data=df[['var1', 'var2', 'var3']],
    weights=None
)
```

**Returns**: Dict with Cramer's V, uncertainty coefficient, etc.

---

## Plotting Functions

### plot_bounds

Visualize Frechet bounds for cell probabilities.

```python
from statmatch import plot_bounds

plot_bounds(bounds_result)
```

### plot_cont

Compare continuous distributions visually.

```python
from statmatch import plot_cont

plot_cont(data_A=df_a['income'], data_B=df_b['income'])
```

### plot_tab

Compare contingency tables visually.

```python
from statmatch import plot_tab

plot_tab(tab_A=table_a, tab_B=table_b)
```

---

## Sample Utilities

### comb_samples

Combine survey samples with different variables.

```python
from statmatch import comb_samples

combined = comb_samples(
    svy_A=survey_a,
    svy_B=survey_b,
    svy_C=survey_c    # Optional third survey
)
```

### harmonize_x

Harmonize matching variables across datasets.

```python
from statmatch import harmonize_x

harmonized_A, harmonized_B = harmonize_x(
    data_A=df_a,
    data_B=df_b,
    match_vars=['age', 'income']
)
```

### fact2dummy

Convert factor variables to dummy variables.

```python
from statmatch import fact2dummy

dummies = fact2dummy(data=df[['category1', 'category2']])
```

---

## Mixed Matching

### mixed_mtc

Mixed matching with multiple constraints.

```python
from statmatch import mixed_mtc

result = mixed_mtc(
    data_rec=recipients,
    data_don=donors,
    match_vars=['age', 'income'],
    y_var='satisfaction',
    z_var='rating'
)
```

### sel_mtc_by_unc

Select matches by minimizing uncertainty.

```python
from statmatch import sel_mtc_by_unc

selected = sel_mtc_by_unc(
    mtc_results=results_list,
    criterion='min_unc'
)
```

---

## Frechet Bounds

### frechet_bounds_cat

Compute Frechet bounds for categorical data.

```python
from statmatch import frechet_bounds_cat

result = frechet_bounds_cat(
    tab_x=table_x,     # X marginal distribution
    tab_xy=table_xy,   # X vs Y contingency table
    tab_xz=table_xz,   # X vs Z contingency table
    print_f='tables'   # 'tables' or 'data.frame'
)
```

**Returns**: Dict with `low.u`, `up.u`, `low.cx`, `up.cx`, `CIA`, `uncertainty`

### fbwidths_by_x

Compute Frechet bounds for all X variable subsets.

```python
from statmatch import fbwidths_by_x

result = fbwidths_by_x(
    tab_x=table_x,
    tab_xy=table_xy,
    tab_xz=table_xz
)
```

**Returns**: Dict with `sum.unc` DataFrame summarizing all subsets

### p_bayes

Pseudo-Bayes estimation for sparse contingency tables.

```python
from statmatch import p_bayes

result = p_bayes(
    x=table,
    method='Jeffreys'  # Options: Jeffreys, minimax, invcat, user, m.ind, h.assoc
)
```

**Returns**: Dict with `info`, `prior`, `pseudoB`

---

## R Compatibility

All functions produce results that match R's StatMatch package. The test suite includes comparison tests that verify:

- Distances match to 1e-10 tolerance
- Matched donor indices are identical (or tied with equal distances)
- Frechet bounds match within floating-point precision

Run R comparison tests with:
```bash
pytest tests/ -k "against_r" -v
```

Requires `rpy2` and R's `StatMatch` package installed.
