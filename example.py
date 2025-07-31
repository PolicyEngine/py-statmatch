"""Example usage of py-statmatch package."""

import pandas as pd
import numpy as np
from statmatch import nnd_hotdeck

# Create example donor dataset
np.random.seed(42)
n_donors = 100
donor_data = pd.DataFrame(
    {
        "age": np.random.normal(40, 10, n_donors),
        "income": np.random.normal(50000, 20000, n_donors),
        "education_years": np.random.normal(14, 3, n_donors),
        "region": np.random.choice(
            ["North", "South", "East", "West"], n_donors
        ),
        "job_satisfaction": np.random.randint(
            1, 11, n_donors
        ),  # Variable to donate
        "health_score": np.random.randint(
            1, 101, n_donors
        ),  # Variable to donate
    }
)

# Create example recipient dataset (missing job_satisfaction and health_score)
n_recipients = 50
recipient_data = pd.DataFrame(
    {
        "age": np.random.normal(38, 12, n_recipients),
        "income": np.random.normal(48000, 18000, n_recipients),
        "education_years": np.random.normal(13, 3, n_recipients),
        "region": np.random.choice(
            ["North", "South", "East", "West"], n_recipients
        ),
    }
)

print("Donor data shape:", donor_data.shape)
print("Recipient data shape:", recipient_data.shape)
print("\nDonor data head:")
print(donor_data.head())
print("\nRecipient data head:")
print(recipient_data.head())

# Perform matching within regions using Euclidean distance
print("\n" + "=" * 60)
print("Performing statistical matching...")
result = nnd_hotdeck(
    data_rec=recipient_data,
    data_don=donor_data,
    match_vars=["age", "income", "education_years"],
    don_class="region",
    dist_fun="euclidean",
)

print("\nMatching complete!")
print(f"Average distance: {result['dist.rd'].mean():.2f}")
print(
    f"Distance range: [{result['dist.rd'].min():.2f}, {result['dist.rd'].max():.2f}]"
)

# Create fused dataset
fused_data = recipient_data.copy()
fused_data["job_satisfaction"] = donor_data.iloc[result["noad.index"]][
    "job_satisfaction"
].values
fused_data["health_score"] = donor_data.iloc[result["noad.index"]][
    "health_score"
].values
fused_data["donor_id"] = result["noad.index"]
fused_data["match_distance"] = result["dist.rd"]

print("\nFused data head:")
print(fused_data.head())

# Analyze results by region
print("\n" + "=" * 60)
print("Results by region:")
for region in ["North", "South", "East", "West"]:
    region_data = fused_data[fused_data["region"] == region]
    if len(region_data) > 0:
        print(f"\n{region}:")
        print(f"  Recipients: {len(region_data)}")
        print(
            f"  Avg match distance: {region_data['match_distance'].mean():.2f}"
        )
        print(
            f"  Avg job satisfaction: {region_data['job_satisfaction'].mean():.1f}"
        )
        print(f"  Avg health score: {region_data['health_score'].mean():.1f}")

# Example with constrained matching
print("\n" + "=" * 60)
print("Performing constrained matching (each donor used at most 2 times)...")
result_constrained = nnd_hotdeck(
    data_rec=recipient_data,
    data_don=donor_data,
    match_vars=["age", "income", "education_years"],
    dist_fun="euclidean",
    constr_alg="lpsolve",
    k=2,
)

# Check constraint is satisfied
donor_usage = pd.Series(result_constrained["noad.index"]).value_counts()
print(f"\nMax donor usage: {donor_usage.max()}")
print(f"Number of unique donors used: {len(donor_usage)}")
print(
    f"Average distance (constrained): {result_constrained['dist.rd'].mean():.2f}"
)
