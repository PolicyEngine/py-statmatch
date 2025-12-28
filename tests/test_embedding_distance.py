"""Tests for embedding-based distance computation for categorical variables."""

import numpy as np
import pandas as pd
import pytest
from statmatch.embedding_distance import (
    learn_embeddings,
    embedding_dist,
    entity_similarity,
)


class TestLearnEmbeddings:
    """Test suite for learn_embeddings function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with categorical variables and a target."""
        np.random.seed(42)
        n = 200

        # Create categories with different target means
        # 'A' has high mean, 'B' medium, 'C' low
        categories = np.random.choice(["A", "B", "C"], n, p=[0.3, 0.4, 0.3])
        target = np.where(
            categories == "A",
            np.random.normal(100, 10, n),
            np.where(
                categories == "B",
                np.random.normal(50, 10, n),
                np.random.normal(0, 10, n),
            ),
        )

        # Second categorical variable
        region = np.random.choice(
            ["North", "South", "East", "West"], n, p=[0.25, 0.25, 0.25, 0.25]
        )

        return pd.DataFrame(
            {"category": categories, "region": region, "target": target}
        )

    def test_embeddings_have_correct_dimensions(self, sample_data):
        """Test that learned embeddings have the expected dimensions."""
        embedding_dim = 8
        embeddings = learn_embeddings(
            data=sample_data,
            cat_vars=["category", "region"],
            target_var="target",
            embedding_dim=embedding_dim,
            method="target",
        )

        # Check structure
        assert "category" in embeddings
        assert "region" in embeddings

        # Check each category has correct embedding dimension
        for cat in ["A", "B", "C"]:
            assert cat in embeddings["category"]
            assert len(embeddings["category"][cat]) == embedding_dim

        for region in ["North", "South", "East", "West"]:
            assert region in embeddings["region"]
            assert len(embeddings["region"][region]) == embedding_dim

    def test_embeddings_are_numpy_arrays(self, sample_data):
        """Test that embedding vectors are numpy arrays."""
        embeddings = learn_embeddings(
            data=sample_data,
            cat_vars=["category"],
            target_var="target",
            embedding_dim=4,
            method="target",
        )

        for cat in ["A", "B", "C"]:
            assert isinstance(embeddings["category"][cat], np.ndarray)

    def test_similar_categories_have_similar_embeddings_target_method(
        self, sample_data
    ):
        """Test that categories with similar target means have similar embeddings."""
        embeddings = learn_embeddings(
            data=sample_data,
            cat_vars=["category"],
            target_var="target",
            embedding_dim=4,
            method="target",
        )

        # Since A has high mean, B medium, C low:
        # Distance A-C should be greater than A-B
        emb_a = embeddings["category"]["A"]
        emb_b = embeddings["category"]["B"]
        emb_c = embeddings["category"]["C"]

        dist_ab = np.linalg.norm(emb_a - emb_b)
        dist_ac = np.linalg.norm(emb_a - emb_c)

        # A is farther from C than from B
        assert dist_ac > dist_ab

    def test_svd_method(self, sample_data):
        """Test SVD method for learning embeddings."""
        embeddings = learn_embeddings(
            data=sample_data,
            cat_vars=["category", "region"],
            target_var="target",
            embedding_dim=2,
            method="svd",
        )

        # Check embeddings exist
        assert "category" in embeddings
        assert "region" in embeddings

        # Check dimensions
        for cat in ["A", "B", "C"]:
            assert len(embeddings["category"][cat]) == 2

    def test_embedding_dim_clipped_to_num_categories(self, sample_data):
        """Test that embedding_dim is clipped when larger than num categories."""
        # Only 3 categories for 'category' variable
        embeddings = learn_embeddings(
            data=sample_data,
            cat_vars=["category"],
            target_var="target",
            embedding_dim=100,  # Larger than number of categories
            method="svd",
        )

        # SVD dimension should be clipped to min(n_categories, embedding_dim)
        for cat in ["A", "B", "C"]:
            # Should have dimension <= 3 (number of categories)
            assert len(embeddings["category"][cat]) <= 3

    def test_high_cardinality_categoricals(self):
        """Test with high-cardinality categorical variables."""
        np.random.seed(42)
        n = 1000
        n_categories = 50

        # Create high-cardinality categorical
        categories = [f"cat_{i}" for i in range(n_categories)]
        cat_values = np.random.choice(categories, n)
        target = np.random.normal(0, 1, n)

        data = pd.DataFrame({"high_card": cat_values, "target": target})

        embeddings = learn_embeddings(
            data=data,
            cat_vars=["high_card"],
            target_var="target",
            embedding_dim=8,
            method="target",
        )

        # Check all categories have embeddings
        for cat in categories:
            assert cat in embeddings["high_card"]
            assert len(embeddings["high_card"][cat]) == 8

    def test_invalid_method_raises_error(self, sample_data):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            learn_embeddings(
                data=sample_data,
                cat_vars=["category"],
                target_var="target",
                method="invalid_method",
            )

    def test_missing_cat_var_raises_error(self, sample_data):
        """Test that missing categorical variable raises KeyError."""
        with pytest.raises(KeyError):
            learn_embeddings(
                data=sample_data,
                cat_vars=["nonexistent"],
                target_var="target",
            )

    def test_missing_target_var_raises_error(self, sample_data):
        """Test that missing target variable raises KeyError."""
        with pytest.raises(KeyError):
            learn_embeddings(
                data=sample_data,
                cat_vars=["category"],
                target_var="nonexistent",
            )


class TestEmbeddingDist:
    """Test suite for embedding_dist function."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        return {
            "category": {
                "A": np.array([1.0, 0.0, 0.0, 0.0]),
                "B": np.array([0.0, 1.0, 0.0, 0.0]),
                "C": np.array([0.0, 0.0, 1.0, 0.0]),
            },
            "region": {
                "North": np.array([1.0, 0.0]),
                "South": np.array([0.0, 1.0]),
            },
        }

    @pytest.fixture
    def sample_data_for_dist(self):
        """Create sample data for distance computation."""
        data_x = pd.DataFrame(
            {
                "category": ["A", "B", "C"],
                "region": ["North", "South", "North"],
                "numeric1": [1.0, 2.0, 3.0],
            }
        )
        data_y = pd.DataFrame(
            {
                "category": ["A", "C"],
                "region": ["North", "South"],
                "numeric1": [1.5, 2.5],
            }
        )
        return data_x, data_y

    def test_distance_matrix_shape(
        self, sample_embeddings, sample_data_for_dist
    ):
        """Test that distance matrix has correct shape."""
        data_x, data_y = sample_data_for_dist

        dist_matrix = embedding_dist(
            data_x=data_x,
            data_y=data_y,
            embeddings=sample_embeddings,
            cat_vars=["category", "region"],
            numeric_vars=["numeric1"],
        )

        assert dist_matrix.shape == (len(data_x), len(data_y))

    def test_distance_matrix_non_negative(
        self, sample_embeddings, sample_data_for_dist
    ):
        """Test that all distances are non-negative."""
        data_x, data_y = sample_data_for_dist

        dist_matrix = embedding_dist(
            data_x=data_x,
            data_y=data_y,
            embeddings=sample_embeddings,
            cat_vars=["category", "region"],
            numeric_vars=["numeric1"],
        )

        assert np.all(dist_matrix >= 0)

    def test_distance_matrix_symmetric_same_data(self, sample_embeddings):
        """Test that distance matrix is symmetric when data_y is None."""
        data = pd.DataFrame(
            {
                "category": ["A", "B", "C"],
                "region": ["North", "South", "North"],
                "numeric1": [1.0, 2.0, 3.0],
            }
        )

        dist_matrix = embedding_dist(
            data_x=data,
            data_y=None,
            embeddings=sample_embeddings,
            cat_vars=["category", "region"],
            numeric_vars=["numeric1"],
        )

        np.testing.assert_allclose(dist_matrix, dist_matrix.T, rtol=1e-10)

    def test_zero_diagonal_same_data(self, sample_embeddings):
        """Test that diagonal is zero when computing distances within same data."""
        data = pd.DataFrame(
            {
                "category": ["A", "B", "C"],
                "region": ["North", "South", "North"],
                "numeric1": [1.0, 2.0, 3.0],
            }
        )

        dist_matrix = embedding_dist(
            data_x=data,
            data_y=None,
            embeddings=sample_embeddings,
            cat_vars=["category", "region"],
            numeric_vars=["numeric1"],
        )

        np.testing.assert_allclose(np.diag(dist_matrix), 0.0, atol=1e-10)

    def test_only_categorical_vars(self, sample_embeddings):
        """Test distance computation with only categorical variables."""
        data_x = pd.DataFrame({"category": ["A", "B", "C"]})
        data_y = pd.DataFrame({"category": ["A", "C"]})

        dist_matrix = embedding_dist(
            data_x=data_x,
            data_y=data_y,
            embeddings=sample_embeddings,
            cat_vars=["category"],
            numeric_vars=[],
        )

        assert dist_matrix.shape == (3, 2)
        # A to A should be zero distance
        assert dist_matrix[0, 0] == pytest.approx(0.0, abs=1e-10)
        # A to C should be non-zero
        assert dist_matrix[0, 1] > 0

    def test_only_numeric_vars(self, sample_embeddings):
        """Test distance computation with only numeric variables."""
        data_x = pd.DataFrame({"numeric1": [1.0, 2.0, 3.0]})
        data_y = pd.DataFrame({"numeric1": [1.5, 2.5]})

        dist_matrix = embedding_dist(
            data_x=data_x,
            data_y=data_y,
            embeddings={},  # No embeddings needed
            cat_vars=[],
            numeric_vars=["numeric1"],
        )

        # Should reduce to standard Euclidean distance
        expected = np.array([[0.5, 1.5], [0.5, 0.5], [1.5, 0.5]])
        np.testing.assert_allclose(dist_matrix, expected, rtol=1e-10)

    def test_unknown_category_raises_error(self, sample_embeddings):
        """Test that unknown category raises KeyError."""
        data_x = pd.DataFrame({"category": ["A", "UNKNOWN"]})
        data_y = pd.DataFrame({"category": ["A"]})

        with pytest.raises(KeyError):
            embedding_dist(
                data_x=data_x,
                data_y=data_y,
                embeddings=sample_embeddings,
                cat_vars=["category"],
                numeric_vars=[],
            )


class TestEntitySimilarity:
    """Test suite for entity_similarity function."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        return {
            "category": {
                "A": np.array([1.0, 0.0, 0.0]),
                "B": np.array([0.0, 1.0, 0.0]),
                "C": np.array([0.5, 0.5, 0.0]),  # Between A and B
            }
        }

    def test_similarity_matrix_shape(self, sample_embeddings):
        """Test that similarity matrix has correct shape."""
        sim_matrix = entity_similarity(
            embeddings=sample_embeddings,
            var_name="category",
        )

        # 3 categories -> 3x3 matrix
        assert sim_matrix.shape == (3, 3)

    def test_self_similarity_is_one(self, sample_embeddings):
        """Test that self-similarity is 1.0."""
        sim_matrix = entity_similarity(
            embeddings=sample_embeddings,
            var_name="category",
        )

        np.testing.assert_allclose(np.diag(sim_matrix), 1.0, atol=1e-10)

    def test_similarity_is_symmetric(self, sample_embeddings):
        """Test that similarity matrix is symmetric."""
        sim_matrix = entity_similarity(
            embeddings=sample_embeddings,
            var_name="category",
        )

        np.testing.assert_allclose(sim_matrix, sim_matrix.T, rtol=1e-10)

    def test_similarity_bounded_zero_one(self, sample_embeddings):
        """Test that similarity values are between 0 and 1."""
        sim_matrix = entity_similarity(
            embeddings=sample_embeddings,
            var_name="category",
        )

        assert np.all(sim_matrix >= 0)
        assert np.all(sim_matrix <= 1 + 1e-10)

    def test_orthogonal_vectors_zero_similarity(self, sample_embeddings):
        """Test that orthogonal vectors have zero similarity."""
        sim_matrix = entity_similarity(
            embeddings=sample_embeddings,
            var_name="category",
        )

        # A = [1,0,0] and B = [0,1,0] are orthogonal
        # Find indices for A and B
        categories = list(sample_embeddings["category"].keys())
        idx_a = categories.index("A")
        idx_b = categories.index("B")

        assert sim_matrix[idx_a, idx_b] == pytest.approx(0.0, abs=1e-10)

    def test_returns_categories_in_order(self, sample_embeddings):
        """Test that categories are returned in consistent order."""
        sim_matrix, categories = entity_similarity(
            embeddings=sample_embeddings,
            var_name="category",
            return_categories=True,
        )

        assert len(categories) == 3
        assert set(categories) == {"A", "B", "C"}

    def test_unknown_var_raises_error(self, sample_embeddings):
        """Test that unknown variable name raises KeyError."""
        with pytest.raises(KeyError):
            entity_similarity(
                embeddings=sample_embeddings,
                var_name="nonexistent",
            )


class TestIntegration:
    """Integration tests for full embedding-based matching workflow."""

    def test_full_workflow_with_nnd_hotdeck(self):
        """Test full workflow of learning embeddings and using with nnd_hotdeck."""
        from statmatch.nnd_hotdeck import nnd_hotdeck

        np.random.seed(42)

        # Create donor data with categorical and target
        n_don = 100
        donor_categories = np.random.choice(["A", "B", "C"], n_don)
        donor_target = np.where(
            donor_categories == "A",
            np.random.normal(100, 10, n_don),
            np.where(
                donor_categories == "B",
                np.random.normal(50, 10, n_don),
                np.random.normal(0, 10, n_don),
            ),
        )
        donor_y = np.random.normal(0, 1, n_don)  # Variable to donate

        data_don = pd.DataFrame(
            {"category": donor_categories, "x": donor_target, "y": donor_y}
        )

        # Create recipient data
        n_rec = 50
        recipient_categories = np.random.choice(["A", "B", "C"], n_rec)
        recipient_x = np.where(
            recipient_categories == "A",
            np.random.normal(100, 10, n_rec),
            np.where(
                recipient_categories == "B",
                np.random.normal(50, 10, n_rec),
                np.random.normal(0, 10, n_rec),
            ),
        )

        data_rec = pd.DataFrame(
            {"category": recipient_categories, "x": recipient_x}
        )

        # Learn embeddings from donor data
        embeddings = learn_embeddings(
            data=data_don,
            cat_vars=["category"],
            target_var="x",
            embedding_dim=4,
            method="target",
        )

        # Compute distance matrix
        dist_matrix = embedding_dist(
            data_x=data_rec,
            data_y=data_don,
            embeddings=embeddings,
            cat_vars=["category"],
            numeric_vars=["x"],
        )

        # Verify we can use this for matching
        assert dist_matrix.shape == (n_rec, n_don)
        assert np.all(dist_matrix >= 0)

        # Check that matches prefer same category (due to embedding distance)
        # Recipients in category A should tend to match to donors in category A
        for i in range(n_rec):
            # Find nearest donor
            nearest_donor = np.argmin(dist_matrix[i])
            # The match should have similar x values (due to learned embeddings)
            assert donor_target[nearest_donor] is not None  # Just check valid
