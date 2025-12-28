"""Test Optimal Transport matching implementation."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from statmatch.ot_matching import ot_hotdeck, wasserstein_dist


class TestOTHotdeck:
    """Test suite for OT hotdeck matching function."""

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
            }
        )

        # Recipient dataset with matching variables X but no Y
        n_recipients = 30
        recipient_data = pd.DataFrame(
            {
                "x1": np.random.normal(0.5, 1, n_recipients),
                "x2": np.random.normal(0.5, 2, n_recipients),
                "x3": np.random.uniform(1, 11, n_recipients),
            }
        )

        return donor_data, recipient_data

    @pytest.fixture
    def small_data(self):
        """Create small balanced datasets for exact OT testing."""
        # Balanced case: same number of donors and recipients
        donor_data = pd.DataFrame(
            {
                "x1": [0.0, 1.0, 2.0],
                "x2": [0.0, 1.0, 2.0],
            }
        )

        recipient_data = pd.DataFrame(
            {
                "x1": [0.1, 1.1, 2.1],
                "x2": [0.1, 1.1, 2.1],
            }
        )

        return donor_data, recipient_data

    @pytest.fixture
    def unbalanced_data(self):
        """Create unbalanced datasets (more donors than recipients)."""
        donor_data = pd.DataFrame(
            {
                "x1": [0.0, 1.0, 2.0, 3.0],
                "x2": [0.0, 1.0, 2.0, 3.0],
            }
        )

        recipient_data = pd.DataFrame(
            {
                "x1": [0.1, 1.1, 2.1],
                "x2": [0.1, 1.1, 2.1],
            }
        )

        return donor_data, recipient_data

    def test_ot_matching_returns_valid_structure(self, sample_data):
        """Test that OT matching returns expected dictionary structure."""
        donor_data, recipient_data = sample_data

        result = ot_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
        )

        # Check required keys exist
        assert "mtc.ids" in result
        assert "noad.index" in result
        assert "dist.rd" in result
        assert "transport_plan" in result
        assert "total_cost" in result

        # Check dimensions
        assert len(result["noad.index"]) == len(recipient_data)
        assert len(result["dist.rd"]) == len(recipient_data)

    def test_ot_matching_valid_assignments(self, sample_data):
        """Test that OT matching produces valid assignments."""
        donor_data, recipient_data = sample_data

        result = ot_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
        )

        # All donor indices should be valid
        assert np.all(result["noad.index"] >= 0)
        assert np.all(result["noad.index"] < len(donor_data))

        # All distances should be non-negative
        assert np.all(result["dist.rd"] >= 0)

    def test_ot_matching_euclidean_distance(self, sample_data):
        """Test OT matching with Euclidean distance."""
        donor_data, recipient_data = sample_data

        result = ot_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="euclidean",
        )

        # Verify distances match computed Euclidean distances
        rec_data = recipient_data[["x1", "x2", "x3"]].values
        don_data = donor_data[["x1", "x2", "x3"]].values

        for i, donor_idx in enumerate(result["noad.index"]):
            expected_dist = np.sqrt(
                np.sum((rec_data[i] - don_data[donor_idx]) ** 2)
            )
            np.testing.assert_allclose(
                result["dist.rd"][i], expected_dist, rtol=1e-10
            )

    def test_ot_matching_manhattan_distance(self, sample_data):
        """Test OT matching with Manhattan distance."""
        donor_data, recipient_data = sample_data

        result = ot_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            dist_fun="manhattan",
        )

        # Verify distances match computed Manhattan distances
        rec_data = recipient_data[["x1", "x2", "x3"]].values
        don_data = donor_data[["x1", "x2", "x3"]].values

        for i, donor_idx in enumerate(result["noad.index"]):
            expected_dist = np.sum(np.abs(rec_data[i] - don_data[donor_idx]))
            np.testing.assert_allclose(
                result["dist.rd"][i], expected_dist, rtol=1e-10
            )

    def test_emd_method_exact_solution(self, small_data):
        """Test that EMD method gives exact optimal transport solution."""
        donor_data, recipient_data = small_data

        result = ot_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            method="emd",
        )

        # With balanced data [0,1,2] recipients and [0,1,2] donors,
        # the optimal matching should map each to their nearest:
        # recipient 0 (at 0.1) -> donor 0 (at 0)
        # recipient 1 (at 1.1) -> donor 1 (at 1)
        # recipient 2 (at 2.1) -> donor 2 (at 2)
        expected_matches = [0, 1, 2]
        np.testing.assert_array_equal(result["noad.index"], expected_matches)

    def test_sinkhorn_method_approximate_solution(self, small_data):
        """Test Sinkhorn method gives approximate optimal transport."""
        donor_data, recipient_data = small_data

        result = ot_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            method="sinkhorn",
            reg=0.01,
        )

        # Sinkhorn should give similar results to EMD for well-separated data
        # With balanced data, each recipient should match to corresponding donor
        expected_matches = [0, 1, 2]
        np.testing.assert_array_equal(result["noad.index"], expected_matches)

    def test_regularization_affects_transport_plan(self, sample_data):
        """Test that regularization affects the transport plan."""
        donor_data, recipient_data = sample_data

        # Low regularization (closer to exact OT)
        result_low_reg = ot_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            method="sinkhorn",
            reg=0.001,
        )

        # High regularization (more smoothed)
        result_high_reg = ot_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            method="sinkhorn",
            reg=1.0,
        )

        # Transport plans should be different
        # High reg should have higher entropy (more spread out)
        plan_low = result_low_reg["transport_plan"]
        plan_high = result_high_reg["transport_plan"]

        # Compute entropy of normalized plans
        def entropy(p):
            p_flat = p.flatten()
            p_flat = p_flat[p_flat > 1e-10]
            p_flat = p_flat / p_flat.sum()
            return -np.sum(p_flat * np.log(p_flat))

        entropy_low = entropy(plan_low)
        entropy_high = entropy(plan_high)

        # High regularization should give higher entropy
        assert entropy_high > entropy_low

    def test_total_cost_is_minimized(self, small_data):
        """Test that OT achieves minimum total transport cost."""
        donor_data, recipient_data = small_data

        result = ot_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            method="emd",
        )

        # Compute cost matrix
        rec_data = recipient_data[["x1", "x2"]].values
        don_data = donor_data[["x1", "x2"]].values
        cost_matrix = cdist(rec_data, don_data, metric="euclidean")

        # For balanced OT with uniform marginals, the transport plan has
        # 1/n mass on each optimal assignment.
        # The optimal matching [0->0, 1->1, 2->2] gives total cost of:
        n = len(rec_data)
        optimal_cost = sum(cost_matrix[i, i] for i in range(n)) / n

        # Compare with returned total cost
        # OT cost = sum of (transport_plan * cost_matrix)
        np.testing.assert_allclose(
            result["total_cost"], optimal_cost, rtol=1e-10
        )

    def test_transport_plan_marginals(self, sample_data):
        """Test that transport plan marginals are correct."""
        donor_data, recipient_data = sample_data

        result = ot_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2", "x3"],
            method="emd",
        )

        plan = result["transport_plan"]
        n_rec = len(recipient_data)
        n_don = len(donor_data)

        # Row sums should be 1/n_rec (uniform recipient weights)
        row_sums = plan.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(n_rec) / n_rec, rtol=1e-5)

        # Column sums should be 1/n_don (uniform donor weights)
        col_sums = plan.sum(axis=0)
        np.testing.assert_allclose(col_sums, np.ones(n_don) / n_don, rtol=1e-5)

    def test_invalid_match_vars_raises_error(self, sample_data):
        """Test that invalid match variables raise an error."""
        donor_data, recipient_data = sample_data

        with pytest.raises(ValueError, match="not found in recipient"):
            ot_hotdeck(
                data_rec=recipient_data,
                data_don=donor_data,
                match_vars=["x1", "nonexistent"],
            )

        with pytest.raises(ValueError, match="not found in donor"):
            # Add column to recipient only
            recipient_data["extra"] = 1
            ot_hotdeck(
                data_rec=recipient_data,
                data_don=donor_data,
                match_vars=["x1", "extra"],
            )

    def test_invalid_method_raises_error(self, sample_data):
        """Test that invalid method raises an error."""
        donor_data, recipient_data = sample_data

        with pytest.raises(ValueError, match="Unknown method"):
            ot_hotdeck(
                data_rec=recipient_data,
                data_don=donor_data,
                match_vars=["x1", "x2"],
                method="invalid_method",
            )

    def test_emd_vs_sinkhorn_convergence(self, small_data):
        """Test that Sinkhorn converges to EMD as reg -> 0."""
        donor_data, recipient_data = small_data

        result_emd = ot_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            method="emd",
        )

        result_sinkhorn = ot_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            method="sinkhorn",
            reg=0.0001,
        )

        # Results should be very similar
        np.testing.assert_array_equal(
            result_emd["noad.index"], result_sinkhorn["noad.index"]
        )

        # Total costs should be close
        np.testing.assert_allclose(
            result_emd["total_cost"], result_sinkhorn["total_cost"], rtol=0.01
        )

    def test_unbalanced_matching(self, unbalanced_data):
        """Test OT matching when n_rec != n_don."""
        donor_data, recipient_data = unbalanced_data

        result = ot_hotdeck(
            data_rec=recipient_data,
            data_don=donor_data,
            match_vars=["x1", "x2"],
            method="emd",
        )

        # Verify transport plan has correct marginals
        plan = result["transport_plan"]
        n_rec = len(recipient_data)
        n_don = len(donor_data)

        # Row sums should be 1/n_rec (uniform recipient weights)
        row_sums = plan.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(n_rec) / n_rec, rtol=1e-5)

        # Column sums should be 1/n_don (uniform donor weights)
        col_sums = plan.sum(axis=0)
        np.testing.assert_allclose(col_sums, np.ones(n_don) / n_don, rtol=1e-5)

        # All assignments should be valid
        assert np.all(result["noad.index"] >= 0)
        assert np.all(result["noad.index"] < n_don)


class TestWassersteinDist:
    """Test suite for Wasserstein distance function."""

    def test_wasserstein_identical_distributions(self):
        """Test that identical distributions have zero Wasserstein distance."""
        data = np.random.normal(0, 1, (100, 2))
        dist = wasserstein_dist(data, data, p=1)
        np.testing.assert_allclose(dist, 0, atol=1e-10)

    def test_wasserstein_1d_simple(self):
        """Test 1D Wasserstein distance."""
        # Simple case: two point masses
        x = np.array([[0.0]])
        y = np.array([[1.0]])
        dist = wasserstein_dist(x, y, p=1)
        np.testing.assert_allclose(dist, 1.0, rtol=1e-5)

    def test_wasserstein_2d(self):
        """Test 2D Wasserstein distance."""
        np.random.seed(42)
        x = np.random.normal(0, 1, (50, 2))
        y = np.random.normal(1, 1, (50, 2))

        # Distance should be positive for different distributions
        dist = wasserstein_dist(x, y, p=1)
        assert dist > 0

    def test_wasserstein_symmetry(self):
        """Test that Wasserstein distance is symmetric."""
        np.random.seed(42)
        x = np.random.normal(0, 1, (30, 2))
        y = np.random.normal(1, 2, (30, 2))

        dist_xy = wasserstein_dist(x, y, p=1)
        dist_yx = wasserstein_dist(y, x, p=1)

        np.testing.assert_allclose(dist_xy, dist_yx, rtol=1e-5)

    def test_wasserstein_1_vs_2(self):
        """Test that W1 and W2 give different results."""
        np.random.seed(42)
        x = np.random.normal(0, 1, (50, 2))
        y = np.random.normal(2, 1, (50, 2))

        w1 = wasserstein_dist(x, y, p=1)
        w2 = wasserstein_dist(x, y, p=2)

        # Generally W2 >= W1 for the same distributions
        # but they should be different
        assert not np.isclose(w1, w2)

    def test_wasserstein_triangle_inequality(self):
        """Test triangle inequality: W(x,z) <= W(x,y) + W(y,z)."""
        np.random.seed(42)
        x = np.random.normal(0, 1, (30, 2))
        y = np.random.normal(1, 1, (30, 2))
        z = np.random.normal(2, 1, (30, 2))

        d_xz = wasserstein_dist(x, z, p=1)
        d_xy = wasserstein_dist(x, y, p=1)
        d_yz = wasserstein_dist(y, z, p=1)

        assert d_xz <= d_xy + d_yz + 1e-10

    def test_wasserstein_scaled_data(self):
        """Test Wasserstein distance scales with data scaling."""
        np.random.seed(42)
        x = np.random.normal(0, 1, (30, 1))
        y = np.random.normal(1, 1, (30, 1))

        dist_original = wasserstein_dist(x, y, p=1)
        dist_scaled = wasserstein_dist(2 * x, 2 * y, p=1)

        # Scaling by factor k should scale W1 by k
        np.testing.assert_allclose(dist_scaled, 2 * dist_original, rtol=0.1)

    def test_wasserstein_with_pandas(self):
        """Test Wasserstein distance works with pandas DataFrames."""
        df_x = pd.DataFrame({"a": [0, 1, 2], "b": [0, 1, 2]})
        df_y = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})

        dist = wasserstein_dist(df_x.values, df_y.values, p=1)
        assert dist > 0
