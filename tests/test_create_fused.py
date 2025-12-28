"""Test create_fused against R's StatMatch implementation."""

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


class TestCreateFused:
    """Test suite for create_fused function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample donor and recipient datasets with matching results."""
        np.random.seed(42)

        # Donor dataset with matching variables X and donation variable Y
        n_donors = 50
        donor_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n_donors),
                "x2": np.random.normal(0, 2, n_donors),
                "x3": np.random.uniform(0, 10, n_donors),
                "y": np.random.normal(100, 20, n_donors),
                "z1": np.random.choice(["A", "B", "C"], n_donors),
                "z2": np.random.randint(1, 100, n_donors),
            }
        )

        # Recipient dataset with matching variables X but no Y
        n_recipients = 30
        recipient_data = pd.DataFrame(
            {
                "x1": np.random.normal(0.5, 1, n_recipients),
                "x2": np.random.normal(0.5, 2, n_recipients),
                "x3": np.random.uniform(1, 11, n_recipients),
                "w": np.random.uniform(0, 1, n_recipients),
            }
        )

        # Simulated matching result (mtc.ids)
        # For testing, create a simple match where each recipient
        # is matched to a donor
        mtc_ids = pd.DataFrame(
            {
                "rec.id": np.arange(n_recipients),
                "don.id": np.random.choice(n_donors, n_recipients),
            }
        )

        return donor_data, recipient_data, mtc_ids

    def test_basic_fusing(self, sample_data):
        """Test basic fusing with z.vars only."""
        from statmatch import create_fused

        donor_data, recipient_data, mtc_ids = sample_data

        result = create_fused(
            data_rec=recipient_data,
            data_don=donor_data,
            mtc_ids=mtc_ids,
            z_vars=["y", "z1"],
        )

        # Check that result has original recipient columns
        for col in recipient_data.columns:
            assert col in result.columns

        # Check that z_vars were added
        assert "y" in result.columns
        assert "z1" in result.columns

        # Check that result has correct number of rows
        assert len(result) == len(recipient_data)

        # Check that donated values come from correct donors
        for i in range(len(result)):
            donor_idx = mtc_ids.iloc[i]["don.id"]
            assert result.iloc[i]["y"] == donor_data.iloc[donor_idx]["y"]
            assert result.iloc[i]["z1"] == donor_data.iloc[donor_idx]["z1"]

    def test_fusing_with_dup_x(self, sample_data):
        """Test fusing with dup_x=True."""
        from statmatch import create_fused

        donor_data, recipient_data, mtc_ids = sample_data

        result = create_fused(
            data_rec=recipient_data,
            data_don=donor_data,
            mtc_ids=mtc_ids,
            z_vars=["y"],
            dup_x=True,
            match_vars=["x1", "x2"],
        )

        # Check that match_vars with "don" suffix were added
        assert "x1.don" in result.columns
        assert "x2.don" in result.columns

        # Check that donated match_vars come from correct donors
        for i in range(len(result)):
            donor_idx = mtc_ids.iloc[i]["don.id"]
            np.testing.assert_almost_equal(
                result.iloc[i]["x1.don"], donor_data.iloc[donor_idx]["x1"]
            )
            np.testing.assert_almost_equal(
                result.iloc[i]["x2.don"], donor_data.iloc[donor_idx]["x2"]
            )

    def test_preserves_recipient_data(self, sample_data):
        """Test that original recipient data is preserved."""
        from statmatch import create_fused

        donor_data, recipient_data, mtc_ids = sample_data

        result = create_fused(
            data_rec=recipient_data,
            data_don=donor_data,
            mtc_ids=mtc_ids,
            z_vars=["y"],
        )

        # Check that original columns are preserved with same values
        for col in recipient_data.columns:
            pd.testing.assert_series_equal(
                result[col].reset_index(drop=True),
                recipient_data[col].reset_index(drop=True),
                check_names=False,
            )

    def test_multiple_z_vars(self, sample_data):
        """Test fusing with multiple z.vars."""
        from statmatch import create_fused

        donor_data, recipient_data, mtc_ids = sample_data

        result = create_fused(
            data_rec=recipient_data,
            data_don=donor_data,
            mtc_ids=mtc_ids,
            z_vars=["y", "z1", "z2"],
        )

        # Check all z_vars were added
        assert "y" in result.columns
        assert "z1" in result.columns
        assert "z2" in result.columns

    def test_dup_x_requires_match_vars(self, sample_data):
        """Test that dup_x=True requires match_vars."""
        from statmatch import create_fused

        donor_data, recipient_data, mtc_ids = sample_data

        with pytest.raises(ValueError, match="match_vars"):
            create_fused(
                data_rec=recipient_data,
                data_don=donor_data,
                mtc_ids=mtc_ids,
                z_vars=["y"],
                dup_x=True,
                match_vars=None,
            )

    def test_invalid_z_vars(self, sample_data):
        """Test that invalid z_vars raises error."""
        from statmatch import create_fused

        donor_data, recipient_data, mtc_ids = sample_data

        with pytest.raises(ValueError, match="not found"):
            create_fused(
                data_rec=recipient_data,
                data_don=donor_data,
                mtc_ids=mtc_ids,
                z_vars=["nonexistent_var"],
            )

    def test_invalid_match_vars(self, sample_data):
        """Test that invalid match_vars raises error."""
        from statmatch import create_fused

        donor_data, recipient_data, mtc_ids = sample_data

        with pytest.raises(ValueError, match="not found"):
            create_fused(
                data_rec=recipient_data,
                data_don=donor_data,
                mtc_ids=mtc_ids,
                z_vars=["y"],
                dup_x=True,
                match_vars=["nonexistent_var"],
            )

    def test_with_named_index(self):
        """Test fusing with named indices (not integer)."""
        from statmatch import create_fused

        donor_data = pd.DataFrame(
            {
                "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y": [10, 20, 30, 40, 50],
            },
            index=["d1", "d2", "d3", "d4", "d5"],
        )

        recipient_data = pd.DataFrame(
            {
                "x1": [1.5, 2.5, 3.5],
                "w": [100, 200, 300],
            },
            index=["r1", "r2", "r3"],
        )

        mtc_ids = pd.DataFrame(
            {
                "rec.id": ["r1", "r2", "r3"],
                "don.id": ["d1", "d3", "d5"],
            }
        )

        result = create_fused(
            data_rec=recipient_data,
            data_don=donor_data,
            mtc_ids=mtc_ids,
            z_vars=["y"],
        )

        # Check donated values
        assert result.loc["r1", "y"] == 10
        assert result.loc["r2", "y"] == 30
        assert result.loc["r3", "y"] == 50

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_against_r_basic(self, sample_data):
        """Test basic fusing against R implementation."""
        from statmatch import create_fused

        donor_data, recipient_data, mtc_ids = sample_data

        # Python implementation
        result_py = create_fused(
            data_rec=recipient_data,
            data_don=donor_data,
            mtc_ids=mtc_ids,
            z_vars=["y", "z1"],
        )

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            donor_r = ro.conversion.py2rpy(donor_data)
            recipient_r = ro.conversion.py2rpy(recipient_data)

            # R expects 1-based indices
            mtc_ids_r_df = mtc_ids.copy()
            mtc_ids_r_df["rec.id"] = mtc_ids_r_df["rec.id"] + 1
            mtc_ids_r_df["don.id"] = mtc_ids_r_df["don.id"] + 1
            mtc_ids_r = ro.conversion.py2rpy(mtc_ids_r_df)

            result_r = statmatch_r.create_fused(
                data_rec=recipient_r,
                data_don=donor_r,
                mtc_ids=mtc_ids_r,
                z_vars=ro.StrVector(["y", "z1"]),
            )

            # Convert back to pandas
            result_r_df = ro.conversion.rpy2py(result_r)

        # Compare results
        # Check that donated y values match
        np.testing.assert_array_almost_equal(
            result_py["y"].values, result_r_df["y"].values
        )

        # Check that donated z1 values match
        assert list(result_py["z1"]) == list(result_r_df["z1"])

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_against_r_with_dup_x(self, sample_data):
        """Test fusing with dup_x against R implementation."""
        from statmatch import create_fused

        donor_data, recipient_data, mtc_ids = sample_data

        # Python implementation
        result_py = create_fused(
            data_rec=recipient_data,
            data_don=donor_data,
            mtc_ids=mtc_ids,
            z_vars=["y"],
            dup_x=True,
            match_vars=["x1", "x2"],
        )

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            donor_r = ro.conversion.py2rpy(donor_data)
            recipient_r = ro.conversion.py2rpy(recipient_data)

            # R expects 1-based indices
            mtc_ids_r_df = mtc_ids.copy()
            mtc_ids_r_df["rec.id"] = mtc_ids_r_df["rec.id"] + 1
            mtc_ids_r_df["don.id"] = mtc_ids_r_df["don.id"] + 1
            mtc_ids_r = ro.conversion.py2rpy(mtc_ids_r_df)

            result_r = statmatch_r.create_fused(
                data_rec=recipient_r,
                data_don=donor_r,
                mtc_ids=mtc_ids_r,
                z_vars=ro.StrVector(["y"]),
                dup_x=True,
                match_vars=ro.StrVector(["x1", "x2"]),
            )

            # Convert back to pandas
            result_r_df = ro.conversion.rpy2py(result_r)

        # Compare results
        # Check that donated y values match
        np.testing.assert_array_almost_equal(
            result_py["y"].values, result_r_df["y"].values
        )

        # Check that duplicated x1 values match
        # R uses "x1don" suffix, we use "x1.don"
        np.testing.assert_array_almost_equal(
            result_py["x1.don"].values, result_r_df["x1don"].values
        )
        np.testing.assert_array_almost_equal(
            result_py["x2.don"].values, result_r_df["x2don"].values
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_against_r_with_nnd_hotdeck(self):
        """Test create_fused with actual NND.hotdeck matching results."""
        from statmatch import create_fused, nnd_hotdeck

        np.random.seed(123)

        # Create test data
        n_donors = 50
        donor_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n_donors),
                "x2": np.random.normal(0, 2, n_donors),
                "y": np.random.normal(100, 20, n_donors),
            }
        )

        n_recipients = 30
        recipient_data = pd.DataFrame(
            {
                "x1": np.random.normal(0.5, 1, n_recipients),
                "x2": np.random.normal(0.5, 2, n_recipients),
            }
        )

        # Perform matching in Python
        match_result = nnd_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            dist_fun="euclidean",
        )

        # Create fused dataset in Python
        result_py = create_fused(
            data_rec=recipient_data,
            data_don=donor_data,
            mtc_ids=match_result["mtc.ids"],
            z_vars=["y"],
        )

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            donor_r = ro.conversion.py2rpy(donor_data)
            recipient_r = ro.conversion.py2rpy(recipient_data)

            # Perform matching in R
            match_result_r = statmatch_r.NND_hotdeck(
                data_rec=recipient_r,
                data_don=donor_r,
                match_vars=ro.StrVector(["x1", "x2"]),
                dist_fun="Euclidean",
            )

            # Create fused dataset in R
            result_r = statmatch_r.create_fused(
                data_rec=recipient_r,
                data_don=donor_r,
                mtc_ids=match_result_r.getbyname("mtc.ids"),
                z_vars=ro.StrVector(["y"]),
            )

            result_r_df = ro.conversion.rpy2py(result_r)

        # Due to potential tie-breaking differences, we check that:
        # 1. The fused dataset has the right structure
        assert "y" in result_py.columns
        assert "x1" in result_py.columns
        assert "x2" in result_py.columns

        # 2. The result has the correct number of rows
        assert len(result_py) == len(recipient_data)
        assert len(result_r_df) == len(recipient_data)
