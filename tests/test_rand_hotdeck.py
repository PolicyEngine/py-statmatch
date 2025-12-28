"""Test RANDwNND.hotdeck against R's StatMatch implementation."""

import numpy as np
import pandas as pd
import pytest
from statmatch.rand_hotdeck import rand_hotdeck

# Only import rpy2 for test generation
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter

    # Import StatMatch
    statmatch_r = importr("StatMatch")
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False


class TestRANDHotdeck:
    """Test suite for RANDwNND.hotdeck function."""

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
                "y": np.random.normal(100, 20, n_donors),
                "donation_class": np.random.choice(["A", "B", "C"], n_donors),
                "weight": np.random.uniform(0.5, 2.0, n_donors),
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

    @pytest.fixture
    def simple_data(self):
        """Create simple deterministic data for basic tests."""
        donor_data = pd.DataFrame(
            {
                "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "x2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y": [10, 20, 30, 40, 50],
            }
        )
        recipient_data = pd.DataFrame(
            {
                "x1": [1.1, 2.9, 4.5],
                "x2": [1.1, 2.9, 4.5],
            }
        )
        return donor_data, recipient_data

    def test_basic_matching(self, simple_data):
        """Test basic matching functionality."""
        donor_data, recipient_data = simple_data

        result = rand_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            dist_fun="euclidean",
            cut_don="exact",
            k=1,
        )

        # Basic checks
        assert "mtc.ids" in result
        assert "sum.dist" in result
        assert "noad" in result

        # Check dimensions
        assert len(result["noad"]) == len(recipient_data)

        # With k=1, all noad should be 1
        assert np.all(result["noad"] == 1)

        # Check that donor indices are valid
        don_ids = result["mtc.ids"]["don.id"].values
        assert np.all(don_ids >= 0)
        assert np.all(don_ids < len(donor_data))

    def test_cut_don_rot(self, sample_data):
        """Test cut_don='rot' (square root of donors)."""
        donor_data, recipient_data = sample_data

        result = rand_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="euclidean",
            cut_don="rot",
        )

        # Expected k = ceil(sqrt(50)) = 8
        expected_k = int(np.ceil(np.sqrt(len(donor_data))))

        # All noad should be <= expected_k
        assert np.all(result["noad"] <= expected_k)
        assert np.all(result["noad"] >= 1)

    def test_cut_don_exact(self, sample_data):
        """Test cut_don='exact' with specified k."""
        donor_data, recipient_data = sample_data

        k = 5
        result = rand_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="euclidean",
            cut_don="exact",
            k=k,
        )

        # All noad should be exactly k
        assert np.all(result["noad"] == k)

    def test_cut_don_min(self, sample_data):
        """Test cut_don='min' (only minimum distance donors)."""
        donor_data, recipient_data = sample_data

        result = rand_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="euclidean",
            cut_don="min",
        )

        # All noad should be >= 1
        assert np.all(result["noad"] >= 1)

        # Distances should be minimum distances
        sum_dist = result["sum.dist"]
        np.testing.assert_allclose(
            sum_dist["dist.rd"].values,
            sum_dist["min"].values,
            rtol=1e-10,
        )

    def test_cut_don_span(self, sample_data):
        """Test cut_don='span' (proportion of donors)."""
        donor_data, recipient_data = sample_data

        k = 0.2  # 20% of donors
        result = rand_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="euclidean",
            cut_don="span",
            k=k,
        )

        # Expected noad = ceil(50 * 0.2) = 10
        expected_noad = int(np.ceil(len(donor_data) * k))
        assert np.all(result["noad"] == expected_noad)

    def test_cut_don_k_dist(self, sample_data):
        """Test cut_don='k.dist' (distance threshold)."""
        donor_data, recipient_data = sample_data

        # Use a large distance threshold to get multiple donors
        k = 5.0
        result = rand_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="euclidean",
            cut_don="k.dist",
            k=k,
        )

        # All selected donors should have distance <= k
        sum_dist = result["sum.dist"]
        assert np.all(sum_dist["dist.rd"].values <= k)
        assert np.all(sum_dist["cut"].values == k)

    def test_manhattan_distance(self, sample_data):
        """Test matching with Manhattan distance."""
        donor_data, recipient_data = sample_data

        result = rand_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="manhattan",
            cut_don="exact",
            k=3,
        )

        assert len(result["noad"]) == len(recipient_data)
        assert np.all(result["noad"] == 3)

    def test_donation_classes(self, sample_data):
        """Test matching within donation classes."""
        donor_data, recipient_data = sample_data

        result = rand_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            don_class="donation_class",
            dist_fun="euclidean",
            cut_don="exact",
            k=2,
        )

        # Check that matches respect donation classes
        mtc_ids = result["mtc.ids"]
        for i in range(len(recipient_data)):
            rec_id = mtc_ids.iloc[i]["rec.id"]
            don_id = mtc_ids.iloc[i]["don.id"]
            rec_class = recipient_data.iloc[rec_id]["donation_class"]
            donor_class = donor_data.iloc[don_id]["donation_class"]
            assert rec_class == donor_class

    def test_weighted_selection(self, sample_data):
        """Test weighted donor selection."""
        donor_data, recipient_data = sample_data

        # Run many times and check that weighted donors are selected
        # more often (statistical test)
        np.random.seed(123)

        result = rand_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="euclidean",
            cut_don="exact",
            k=5,
            weight_don="weight",
        )

        # Just check that it runs without error and returns valid results
        assert len(result["noad"]) == len(recipient_data)
        assert "mtc.ids" in result

    def test_no_match_vars(self, sample_data):
        """Test matching without matching variables (random selection)."""
        donor_data, recipient_data = sample_data

        np.random.seed(42)
        result = rand_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=None,
        )

        # All noad should equal number of donors
        assert np.all(result["noad"] == len(donor_data))

        # Should have valid matches
        assert len(result["mtc.ids"]) == len(recipient_data)

    def test_sum_dist_columns(self, sample_data):
        """Test that sum.dist has correct columns."""
        donor_data, recipient_data = sample_data

        result = rand_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="euclidean",
            cut_don="exact",
            k=3,
        )

        sum_dist = result["sum.dist"]
        expected_cols = ["min", "max", "sd", "cut", "dist.rd"]
        for col in expected_cols:
            assert col in sum_dist.columns

        # min should be <= dist.rd
        assert np.all(sum_dist["min"].values <= sum_dist["dist.rd"].values)

        # max should be >= dist.rd
        assert np.all(sum_dist["max"].values >= sum_dist["dist.rd"].values)

    def test_reproducibility_with_seed(self, sample_data):
        """Test that results are reproducible with same seed."""
        donor_data, recipient_data = sample_data

        np.random.seed(42)
        result1 = rand_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="euclidean",
            cut_don="exact",
            k=5,
        )

        np.random.seed(42)
        result2 = rand_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="euclidean",
            cut_don="exact",
            k=5,
        )

        # Results should be identical
        np.testing.assert_array_equal(
            result1["mtc.ids"]["don.id"].values,
            result2["mtc.ids"]["don.id"].values,
        )
        np.testing.assert_array_equal(result1["noad"], result2["noad"])

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_basic_matching_against_r(self, sample_data):
        """Test basic matching output structure against R implementation."""
        donor_data, recipient_data = sample_data

        # Python implementation (with fixed seed for comparison)
        np.random.seed(42)
        result_py = rand_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="euclidean",
            cut_don="exact",
            k=3,
        )

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            donor_r = ro.conversion.py2rpy(donor_data)
            recipient_r = ro.conversion.py2rpy(recipient_data)

            # Set R seed for reproducibility
            ro.r("set.seed(42)")

            result_r = statmatch_r.RANDwNND_hotdeck(
                data_rec=recipient_r,
                data_don=donor_r,
                match_vars=ro.StrVector(["x1", "x2", "x3"]),
                dist_fun="Euclidean",
                cut_don="exact",
                k=3,
            )

        # Extract R results
        noad_r = np.array(result_r.rx2("noad"))
        sum_dist_r = np.array(result_r.rx2("sum.dist"))

        # Check noad matches exactly (deterministic based on k)
        np.testing.assert_array_equal(
            result_py["noad"],
            noad_r,
            err_msg="noad mismatch",
        )

        # Check sum.dist columns match (min, max, sd, cut are deterministic)
        # dist.rd may differ due to random selection
        np.testing.assert_allclose(
            result_py["sum.dist"]["min"].values,
            sum_dist_r[:, 0],
            rtol=1e-5,
            err_msg="min distance mismatch",
        )

        np.testing.assert_allclose(
            result_py["sum.dist"]["max"].values,
            sum_dist_r[:, 1],
            rtol=1e-5,
            err_msg="max distance mismatch",
        )

        np.testing.assert_allclose(
            result_py["sum.dist"]["cut"].values,
            sum_dist_r[:, 3],
            rtol=1e-5,
            err_msg="cut distance mismatch",
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_donation_class_against_r(self, sample_data):
        """Test matching with donation classes against R."""
        donor_data, recipient_data = sample_data

        np.random.seed(42)
        result_py = rand_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            don_class="donation_class",
            dist_fun="euclidean",
            cut_don="rot",
        )

        with localconverter(ro.default_converter + pandas2ri.converter):
            donor_r = ro.conversion.py2rpy(donor_data)
            recipient_r = ro.conversion.py2rpy(recipient_data)

            ro.r("set.seed(42)")

            result_r = statmatch_r.RANDwNND_hotdeck(
                data_rec=recipient_r,
                data_don=donor_r,
                match_vars=ro.StrVector(["x1", "x2", "x3"]),
                don_class="donation_class",
                dist_fun="Euclidean",
                cut_don="rot",
            )

        # Check that results have same length
        noad_r = np.array(result_r.rx2("noad"))
        assert len(result_py["noad"]) == len(noad_r)

        # Check that all matches respect donation classes
        # (Python returns in original order, R groups by class)
        mtc_ids = result_py["mtc.ids"]
        for i in range(len(recipient_data)):
            rec_id = mtc_ids.iloc[i]["rec.id"]
            don_id = mtc_ids.iloc[i]["don.id"]
            rec_class = recipient_data.iloc[rec_id]["donation_class"]
            donor_class = donor_data.iloc[don_id]["donation_class"]
            assert rec_class == donor_class

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_cut_don_min_against_r(self, sample_data):
        """Test cut_don='min' against R implementation."""
        donor_data, recipient_data = sample_data

        result_py = rand_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="euclidean",
            cut_don="min",
        )

        with localconverter(ro.default_converter + pandas2ri.converter):
            donor_r = ro.conversion.py2rpy(donor_data)
            recipient_r = ro.conversion.py2rpy(recipient_data)

            result_r = statmatch_r.RANDwNND_hotdeck(
                data_rec=recipient_r,
                data_don=donor_r,
                match_vars=ro.StrVector(["x1", "x2", "x3"]),
                dist_fun="Euclidean",
                cut_don="min",
            )

        sum_dist_r = np.array(result_r.rx2("sum.dist"))

        # With cut_don='min', distances should match exactly
        # (though tie-breaking may differ)
        np.testing.assert_allclose(
            result_py["sum.dist"]["min"].values,
            sum_dist_r[:, 0],
            rtol=1e-5,
            err_msg="min distance mismatch with cut_don='min'",
        )

        # For cut_don='min', dist.rd should equal min
        np.testing.assert_allclose(
            result_py["sum.dist"]["dist.rd"].values,
            result_py["sum.dist"]["min"].values,
            rtol=1e-10,
            err_msg="dist.rd should equal min with cut_don='min'",
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_manhattan_against_r(self, sample_data):
        """Test Manhattan distance against R."""
        donor_data, recipient_data = sample_data

        result_py = rand_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="manhattan",
            cut_don="exact",
            k=3,
        )

        with localconverter(ro.default_converter + pandas2ri.converter):
            donor_r = ro.conversion.py2rpy(donor_data)
            recipient_r = ro.conversion.py2rpy(recipient_data)

            result_r = statmatch_r.RANDwNND_hotdeck(
                data_rec=recipient_r,
                data_don=donor_r,
                match_vars=ro.StrVector(["x1", "x2", "x3"]),
                dist_fun="Manhattan",
                cut_don="exact",
                k=3,
            )

        sum_dist_r = np.array(result_r.rx2("sum.dist"))

        # Check that min/max/cut distances match
        np.testing.assert_allclose(
            result_py["sum.dist"]["min"].values,
            sum_dist_r[:, 0],
            rtol=1e-5,
            err_msg="Manhattan min distance mismatch",
        )

        np.testing.assert_allclose(
            result_py["sum.dist"]["max"].values,
            sum_dist_r[:, 1],
            rtol=1e-5,
            err_msg="Manhattan max distance mismatch",
        )

    def test_invalid_cut_don(self, sample_data):
        """Test that invalid cut_don raises error."""
        donor_data, recipient_data = sample_data

        with pytest.raises(ValueError, match="Invalid cut_don"):
            rand_hotdeck(
                data_rec=recipient_data,
                data_don=donor_data,
                match_vars=["x1", "x2", "x3"],
                cut_don="invalid",
            )

    def test_invalid_match_vars(self, sample_data):
        """Test that invalid match_vars raises error."""
        donor_data, recipient_data = sample_data

        with pytest.raises(ValueError, match="not found"):
            rand_hotdeck(
                data_rec=recipient_data,
                data_don=donor_data,
                match_vars=["nonexistent"],
            )

    def test_k_required_for_exact(self, sample_data):
        """Test that k is required for cut_don='exact'."""
        donor_data, recipient_data = sample_data

        with pytest.raises(ValueError, match="k is required"):
            rand_hotdeck(
                data_rec=recipient_data,
                data_don=donor_data,
                match_vars=["x1", "x2", "x3"],
                cut_don="exact",
                k=None,
            )

    def test_k_required_for_span(self, sample_data):
        """Test that k is required for cut_don='span'."""
        donor_data, recipient_data = sample_data

        with pytest.raises(ValueError, match="k is required"):
            rand_hotdeck(
                data_rec=recipient_data,
                data_don=donor_data,
                match_vars=["x1", "x2", "x3"],
                cut_don="span",
                k=None,
            )
