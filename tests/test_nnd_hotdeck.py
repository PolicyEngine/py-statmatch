"""Test NND.hotdeck against R's StatMatch implementation."""

import numpy as np
import pandas as pd
import pytest
from statmatch.nnd_hotdeck import nnd_hotdeck

# Only import rpy2 for test generation
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, conversion
    from rpy2.robjects.packages import importr

    # Use the new conversion context method
    from rpy2.robjects.conversion import localconverter

    # Import StatMatch
    statmatch_r = importr("StatMatch")
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False


class TestNNDHotdeck:
    """Test suite for NND.hotdeck function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample donor and recipient datasets."""
        np.random.seed(42)

        # Donor dataset with matching variables X and donation variable Y
        n_donors = 50
        donor_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n_donors),
                "x2": np.random.normal(0, 2, n_donors),
                "x3": np.random.uniform(0, 10, n_donors),
                "y": np.random.normal(100, 20, n_donors),  # Variable to donate
                "donation_class": np.random.choice(["A", "B", "C"], n_donors),
            }
        )

        # Recipient dataset with matching variables X but no Y
        n_recipients = 30
        recipient_data = pd.DataFrame(
            {
                "x1": np.random.normal(0.5, 1, n_recipients),
                "x2": np.random.normal(0.5, 2, n_recipients),
                "x3": np.random.uniform(1, 11, n_recipients),
                "donation_class": np.random.choice(
                    ["A", "B", "C"], n_recipients
                ),
            }
        )

        return donor_data, recipient_data

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_basic_matching_against_r(self, sample_data):
        """Test basic matching functionality against R implementation."""
        donor_data, recipient_data = sample_data

        # Python implementation
        result_py = nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            don_class="donation_class",
            dist_fun="euclidean",
        )

        # R implementation
        # Convert data to R using the new conversion context
        with localconverter(ro.default_converter + pandas2ri.converter):
            donor_r = ro.conversion.py2rpy(donor_data)
            recipient_r = ro.conversion.py2rpy(recipient_data)

            # Call R function
            result_r = statmatch_r.NND_hotdeck(
                data_rec=recipient_r,
                data_don=donor_r,
                match_vars=ro.StrVector(["x1", "x2", "x3"]),
                don_class="donation_class",
                dist_fun="Euclidean",
            )

        # Extract results from R NamedList
        dist_rd_r = np.array(result_r.getbyname("dist.rd"))
        mtc_ids_flat = np.array(result_r.getbyname("mtc.ids"))

        # R returns results grouped by donation class, not in recipient order
        # Parse mtc.ids (column-major: first n elements are rec.id, next n are don.id)
        n = len(dist_rd_r)
        rec_ids_r = np.array([int(x) for x in mtc_ids_flat[:n]])
        don_ids_r = np.array([int(x) for x in mtc_ids_flat[n:]])

        # Create mapping from recipient ID to (donor_id, distance) for R results
        r_matches = {
            rec_ids_r[i]: (don_ids_r[i], dist_rd_r[i]) for i in range(n)
        }

        # Compare results by recipient ID (handles R's grouping by donation class)
        for rec_id in range(len(recipient_data)):
            py_dist = result_py["dist.rd"][rec_id]
            r_don_id, r_dist = r_matches[rec_id]

            # Distances should match exactly
            np.testing.assert_allclose(
                py_dist,
                r_dist,
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Distance mismatch for recipient {rec_id}",
            )

            # Donor IDs may differ due to ties (same distance, different donor)
            # Both are valid if the distance is the same
            py_don_id = result_py["noad.index"][rec_id]
            if py_don_id != r_don_id:
                # Verify this is a tie (same distance to both donors)
                rec = recipient_data.iloc[rec_id][["x1", "x2", "x3"]].values
                don_py = donor_data.iloc[py_don_id][["x1", "x2", "x3"]].values
                don_r = donor_data.iloc[r_don_id][["x1", "x2", "x3"]].values
                dist_to_py = np.sqrt(np.sum((rec - don_py) ** 2))
                dist_to_r = np.sqrt(np.sum((rec - don_r) ** 2))
                np.testing.assert_allclose(
                    dist_to_py,
                    dist_to_r,
                    rtol=1e-5,
                    atol=1e-8,
                    err_msg=f"Donor mismatch for recipient {rec_id} is not a tie",
                )

    def test_euclidean_distance_matching(self, sample_data):
        """Test matching with Euclidean distance."""
        donor_data, recipient_data = sample_data

        result = nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="euclidean",
        )

        # Basic checks
        assert "dist.rd" in result
        assert "noad.index" in result
        assert "mtc.ids" in result

        # Check dimensions
        assert len(result["dist.rd"]) == len(recipient_data)
        assert len(result["noad.index"]) == len(recipient_data)

        # Check that all distances are non-negative
        assert np.all(result["dist.rd"] >= 0)

        # Check that donor indices are valid
        assert np.all(result["noad.index"] >= 0)
        assert np.all(result["noad.index"] < len(donor_data))

    def test_manhattan_distance_matching(self, sample_data):
        """Test matching with Manhattan distance."""
        donor_data, recipient_data = sample_data

        result = nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="manhattan",
        )

        # Check that Manhattan distances are computed correctly
        # Manhattan distance should be >= Euclidean distance
        result_euclidean = nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="euclidean",
        )

        # For same points, Manhattan >= Euclidean (with equality only for 1D)
        assert np.all(result["dist.rd"] >= result_euclidean["dist.rd"] - 1e-10)

    def test_donation_classes(self, sample_data):
        """Test matching within donation classes."""
        donor_data, recipient_data = sample_data

        result = nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            don_class="donation_class",
            dist_fun="euclidean",
        )

        # Check that matches respect donation classes
        for i, rec_class in enumerate(recipient_data["donation_class"]):
            donor_idx = result["noad.index"][i]
            donor_class = donor_data.iloc[donor_idx]["donation_class"]
            assert rec_class == donor_class

    def test_constrained_matching(self, sample_data):
        """Test constrained matching (each donor used at most k times)."""
        donor_data, recipient_data = sample_data

        # Constrain each donor to be used at most 2 times
        result = nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="euclidean",
            constr_alg="lpsolve",
            k=2,
        )

        # Count how many times each donor is used
        donor_counts = np.bincount(
            result["noad.index"], minlength=len(donor_data)
        )

        # Check constraint is satisfied
        assert np.all(donor_counts <= 2)

    def test_missing_values_handling(self):
        """Test handling of missing values."""
        # Create data with missing values
        donor_data = pd.DataFrame(
            {
                "x1": [1.0, 2.0, np.nan, 4.0, 5.0],
                "x2": [2.0, np.nan, 6.0, 8.0, 10.0],
                "y": [10, 20, 30, 40, 50],
            }
        )

        recipient_data = pd.DataFrame(
            {"x1": [1.5, np.nan, 3.5], "x2": [3.0, 5.0, np.nan]}
        )

        result = nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            dist_fun="euclidean",
        )

        # Check that result has correct shape despite missing values
        assert len(result["dist.rd"]) == len(recipient_data)
        assert len(result["noad.index"]) == len(recipient_data)

    def test_single_matching_variable(self):
        """Test with single matching variable."""
        donor_data = pd.DataFrame(
            {
                "x": np.array([1, 2, 3, 4, 5]),
                "y": np.array([10, 20, 30, 40, 50]),
            }
        )

        recipient_data = pd.DataFrame({"x": np.array([1.1, 2.9, 4.5])})

        result = nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x"],
            dist_fun="euclidean",
        )

        # Check expected matches based on nearest neighbor
        # 1.1 -> 1 (index 0), 2.9 -> 3 (index 2), 4.5 -> 5 (index 4) or 4 (index 3)
        expected_indices = [0, 2, 3]  # or [0, 2, 4] depending on tie-breaking
        assert result["noad.index"][0] == 0
        assert result["noad.index"][1] == 2
        assert result["noad.index"][2] in [3, 4]

    def test_create_fused_dataset(self, sample_data):
        """Test creating fused dataset from matching results."""
        donor_data, recipient_data = sample_data

        # First perform matching
        result = nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="euclidean",
        )

        # Create fused dataset
        fused_data = result["mtc.ids"].copy()

        # Add donated variable
        fused_data["y_donated"] = donor_data.iloc[result["noad.index"]][
            "y"
        ].values

        # Check that fused dataset has correct shape
        assert len(fused_data) == len(recipient_data)
        assert "y_donated" in fused_data.columns

        # Check that all values were donated
        assert not fused_data["y_donated"].isna().any()
