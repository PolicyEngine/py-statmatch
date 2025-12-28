"""Test rankNND.hotdeck against R's StatMatch implementation."""

import numpy as np
import pandas as pd
import pytest

# Only import rpy2 for test generation
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    # Import StatMatch
    statmatch_r = importr("StatMatch")
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False


class TestRankNNDHotdeck:
    """Test suite for rankNND.hotdeck function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample donor and recipient datasets."""
        np.random.seed(42)

        # Donor dataset with matching variable and donation variable
        n_donors = 50
        donor_data = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, n_donors),
                "income": np.random.uniform(20000, 150000, n_donors),
                "y": np.random.normal(100, 20, n_donors),
                "region": np.random.choice(["A", "B", "C"], n_donors),
            }
        )

        # Recipient dataset with matching variable but no Y
        n_recipients = 30
        recipient_data = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, n_recipients),
                "income": np.random.uniform(20000, 150000, n_recipients),
                "region": np.random.choice(["A", "B", "C"], n_recipients),
            }
        )

        return donor_data, recipient_data

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_basic_matching_against_r(self, sample_data):
        """Test basic matching functionality against R implementation."""
        donor_data, recipient_data = sample_data

        # Import Python implementation
        from statmatch.rank_hotdeck import rank_nnd_hotdeck

        # Python implementation
        result_py = rank_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            var_rec="age",
            var_don="age",
        )

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            donor_r = ro.conversion.py2rpy(donor_data)
            recipient_r = ro.conversion.py2rpy(recipient_data)

            result_r = statmatch_r.rankNND_hotdeck(
                data_rec=recipient_r,
                data_don=donor_r,
                var_rec="age",
                var_don="age",
            )

        # Extract R results
        dist_rd_r = np.array(result_r.rx2("dist.rd"))

        # Compare distances - should match exactly
        np.testing.assert_allclose(
            result_py["dist.rd"],
            dist_rd_r,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Distance mismatch between Python and R",
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_weighted_matching_against_r(self, sample_data):
        """Test weighted matching against R implementation."""
        donor_data, recipient_data = sample_data

        # Add weights
        np.random.seed(123)
        donor_data = donor_data.copy()
        recipient_data = recipient_data.copy()
        donor_data["weight"] = np.random.uniform(0.5, 2.0, len(donor_data))
        recipient_data["weight"] = np.random.uniform(
            0.5, 2.0, len(recipient_data)
        )

        from statmatch.rank_hotdeck import rank_nnd_hotdeck

        # Python implementation
        result_py = rank_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            var_rec="age",
            var_don="age",
            weight_rec="weight",
            weight_don="weight",
        )

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            donor_r = ro.conversion.py2rpy(donor_data)
            recipient_r = ro.conversion.py2rpy(recipient_data)

            result_r = statmatch_r.rankNND_hotdeck(
                data_rec=recipient_r,
                data_don=donor_r,
                var_rec="age",
                var_don="age",
                weight_rec="weight",
                weight_don="weight",
            )

        dist_rd_r = np.array(result_r.rx2("dist.rd"))

        np.testing.assert_allclose(
            result_py["dist.rd"],
            dist_rd_r,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Weighted distance mismatch",
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_donation_classes_against_r(self, sample_data):
        """Test matching within donation classes against R."""
        donor_data, recipient_data = sample_data

        from statmatch.rank_hotdeck import rank_nnd_hotdeck

        # Python implementation
        result_py = rank_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            var_rec="age",
            var_don="age",
            don_class="region",
        )

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            donor_r = ro.conversion.py2rpy(donor_data)
            recipient_r = ro.conversion.py2rpy(recipient_data)

            result_r = statmatch_r.rankNND_hotdeck(
                data_rec=recipient_r,
                data_don=donor_r,
                var_rec="age",
                var_don="age",
                don_class="region",
            )

        dist_rd_r = np.array(result_r.rx2("dist.rd"))

        np.testing.assert_allclose(
            result_py["dist.rd"],
            dist_rd_r,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Donation class distance mismatch",
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_constrained_matching_against_r(self, sample_data):
        """Test constrained matching against R implementation."""
        donor_data, recipient_data = sample_data

        from statmatch.rank_hotdeck import rank_nnd_hotdeck

        # Python implementation - constrained
        result_py = rank_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            var_rec="age",
            var_don="age",
            constrained=True,
            constr_alg="hungarian",
        )

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            donor_r = ro.conversion.py2rpy(donor_data)
            recipient_r = ro.conversion.py2rpy(recipient_data)

            result_r = statmatch_r.rankNND_hotdeck(
                data_rec=recipient_r,
                data_don=donor_r,
                var_rec="age",
                var_don="age",
                constrained=True,
                constr_alg="Hungarian",
            )

        dist_rd_r = np.array(result_r.rx2("dist.rd"))

        # For constrained matching, total distance should be similar
        # (exact match may differ due to different tie-breaking)
        assert (
            abs(np.sum(result_py["dist.rd"]) - np.sum(dist_rd_r)) < 1e-5
        ), "Constrained matching total distance mismatch"

    def test_basic_matching_output_structure(self, sample_data):
        """Test that output has correct structure."""
        donor_data, recipient_data = sample_data

        from statmatch.rank_hotdeck import rank_nnd_hotdeck

        result = rank_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            var_rec="age",
            var_don="age",
        )

        # Check required keys
        assert "mtc.ids" in result
        assert "dist.rd" in result
        assert "noad" in result

        # Check dimensions
        assert len(result["dist.rd"]) == len(recipient_data)
        assert len(result["noad"]) == len(recipient_data)
        assert len(result["mtc.ids"]) == len(recipient_data)

        # Check that all distances are non-negative
        assert np.all(result["dist.rd"] >= 0)
        assert np.all(result["dist.rd"] <= 1)  # ECDF distances are in [0, 1]

        # Check that noad counts are positive integers
        assert np.all(result["noad"] >= 1)

    def test_different_var_names(self, sample_data):
        """Test matching with different variable names in rec and don."""
        donor_data, recipient_data = sample_data

        # Rename variable in donor
        donor_data = donor_data.rename(columns={"age": "donor_age"})

        from statmatch.rank_hotdeck import rank_nnd_hotdeck

        result = rank_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            var_rec="age",
            var_don="donor_age",
        )

        assert len(result["dist.rd"]) == len(recipient_data)

    def test_donation_classes_respect(self, sample_data):
        """Test that matches respect donation class boundaries."""
        donor_data, recipient_data = sample_data

        from statmatch.rank_hotdeck import rank_nnd_hotdeck

        result = rank_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            var_rec="age",
            var_don="age",
            don_class="region",
        )

        # Check that each recipient was matched to donor in same class
        mtc_ids = result["mtc.ids"]
        for i in range(len(recipient_data)):
            rec_class = recipient_data.iloc[i]["region"]
            donor_idx = mtc_ids.iloc[i]["don.id"]
            don_class = donor_data.iloc[donor_idx]["region"]
            assert rec_class == don_class, (
                f"Recipient {i} in class {rec_class} matched to "
                f"donor in class {don_class}"
            )

    def test_constrained_matching_unique_donors(self, sample_data):
        """Test that constrained matching uses each donor at most once."""
        donor_data, recipient_data = sample_data

        # Ensure enough donors
        if len(donor_data) < len(recipient_data):
            pytest.skip("Need more donors than recipients for this test")

        from statmatch.rank_hotdeck import rank_nnd_hotdeck

        result = rank_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            var_rec="age",
            var_don="age",
            constrained=True,
        )

        # Check that each donor is used at most once
        donor_ids = result["mtc.ids"]["don.id"].values
        unique_donors = len(np.unique(donor_ids))
        assert unique_donors == len(recipient_data), (
            f"Expected {len(recipient_data)} unique donors, "
            f"got {unique_donors}"
        )

    def test_single_variable_matching(self):
        """Test with a simple single-variable case."""
        donor_data = pd.DataFrame({"x": [10, 20, 30, 40, 50]})
        recipient_data = pd.DataFrame({"x": [15, 35, 45]})

        from statmatch.rank_hotdeck import rank_nnd_hotdeck

        result = rank_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            var_rec="x",
            var_don="x",
        )

        # Check dimensions
        assert len(result["dist.rd"]) == 3
        assert len(result["noad"]) == 3

        # Distances should be based on ECDF ranks
        # All distances should be in [0, 1]
        assert np.all(result["dist.rd"] >= 0)
        assert np.all(result["dist.rd"] <= 1)

    def test_error_on_missing_variable(self, sample_data):
        """Test that appropriate error is raised for missing variable."""
        donor_data, recipient_data = sample_data

        from statmatch.rank_hotdeck import rank_nnd_hotdeck

        with pytest.raises(ValueError, match="not found"):
            rank_nnd_hotdeck(
                data_rec=recipient_data,
                data_don=donor_data,
                var_rec="nonexistent",
                var_don="age",
            )

        with pytest.raises(ValueError, match="not found"):
            rank_nnd_hotdeck(
                data_rec=recipient_data,
                data_don=donor_data,
                var_rec="age",
                var_don="nonexistent",
            )

    def test_error_on_missing_donation_class(self, sample_data):
        """Test error when donation class variable is missing."""
        donor_data, recipient_data = sample_data

        from statmatch.rank_hotdeck import rank_nnd_hotdeck

        with pytest.raises(ValueError, match="not found"):
            rank_nnd_hotdeck(
                data_rec=recipient_data,
                data_don=donor_data,
                var_rec="age",
                var_don="age",
                don_class="nonexistent",
            )

    def test_mtc_ids_structure(self, sample_data):
        """Test that mtc.ids has correct structure."""
        donor_data, recipient_data = sample_data

        from statmatch.rank_hotdeck import rank_nnd_hotdeck

        result = rank_nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            var_rec="age",
            var_don="age",
        )

        mtc_ids = result["mtc.ids"]

        # Should be a DataFrame with rec.id and don.id columns
        assert isinstance(mtc_ids, pd.DataFrame)
        assert "rec.id" in mtc_ids.columns
        assert "don.id" in mtc_ids.columns

        # Check all donor indices are valid
        assert np.all(mtc_ids["don.id"] >= 0)
        assert np.all(mtc_ids["don.id"] < len(donor_data))
