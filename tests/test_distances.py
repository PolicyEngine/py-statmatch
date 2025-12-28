"""Tests for distance functions comparing against R's StatMatch implementation."""

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


class TestGowerDist:
    """Test suite for gower_dist function."""

    @pytest.fixture
    def numeric_data(self):
        """Create sample numeric data for testing."""
        np.random.seed(42)
        data_x = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, 10),
                "x2": np.random.uniform(0, 10, 10),
                "x3": np.random.normal(5, 2, 10),
            }
        )
        data_y = pd.DataFrame(
            {
                "x1": np.random.normal(0.5, 1, 8),
                "x2": np.random.uniform(1, 11, 8),
                "x3": np.random.normal(5.5, 2, 8),
            }
        )
        return data_x, data_y

    @pytest.fixture
    def mixed_data(self):
        """Create sample mixed data (numeric, categorical, ordered) for testing."""
        np.random.seed(42)
        data_x = pd.DataFrame(
            {
                "num1": np.random.normal(0, 1, 10),
                "num2": np.random.uniform(0, 10, 10),
                "cat1": pd.Categorical(
                    np.random.choice(["A", "B", "C"], 10)
                ),
                "ord1": pd.Categorical(
                    np.random.choice(["low", "medium", "high"], 10),
                    categories=["low", "medium", "high"],
                    ordered=True,
                ),
            }
        )
        data_y = pd.DataFrame(
            {
                "num1": np.random.normal(0.5, 1, 8),
                "num2": np.random.uniform(1, 11, 8),
                "cat1": pd.Categorical(np.random.choice(["A", "B", "C"], 8)),
                "ord1": pd.Categorical(
                    np.random.choice(["low", "medium", "high"], 8),
                    categories=["low", "medium", "high"],
                    ordered=True,
                ),
            }
        )
        return data_x, data_y

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_gower_numeric_against_r(self, numeric_data):
        """Test gower_dist with numeric data against R implementation."""
        from statmatch.distances import gower_dist

        data_x, data_y = numeric_data

        # Python implementation
        result_py = gower_dist(data_x, data_y)

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            data_x_r = ro.conversion.py2rpy(data_x)
            data_y_r = ro.conversion.py2rpy(data_y)
            result_r = np.array(statmatch_r.gower_dist(data_x_r, data_y_r))

        np.testing.assert_allclose(
            result_py,
            result_r,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Gower distances do not match R implementation",
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_gower_within_dataset_against_r(self, numeric_data):
        """Test gower_dist within same dataset against R implementation."""
        from statmatch.distances import gower_dist

        data_x, _ = numeric_data

        # Python implementation (distance within data_x)
        result_py = gower_dist(data_x)

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            data_x_r = ro.conversion.py2rpy(data_x)
            result_r = np.array(statmatch_r.gower_dist(data_x_r))

        np.testing.assert_allclose(
            result_py,
            result_r,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Gower distances (within dataset) do not match R",
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_gower_with_weights_against_r(self, numeric_data):
        """Test gower_dist with variable weights against R implementation."""
        from statmatch.distances import gower_dist

        data_x, data_y = numeric_data
        weights = [1.0, 2.0, 0.5]

        # Python implementation
        result_py = gower_dist(data_x, data_y, var_weights=weights)

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            data_x_r = ro.conversion.py2rpy(data_x)
            data_y_r = ro.conversion.py2rpy(data_y)
            result_r = np.array(
                statmatch_r.gower_dist(
                    data_x_r, data_y_r, var_weights=ro.FloatVector(weights)
                )
            )

        np.testing.assert_allclose(
            result_py,
            result_r,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Gower distances with weights do not match R",
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_gower_with_ranges_against_r(self, numeric_data):
        """Test gower_dist with custom ranges against R implementation."""
        from statmatch.distances import gower_dist

        data_x, data_y = numeric_data
        rngs = [2.0, 10.0, 4.0]

        # Python implementation
        result_py = gower_dist(data_x, data_y, rngs=rngs)

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            data_x_r = ro.conversion.py2rpy(data_x)
            data_y_r = ro.conversion.py2rpy(data_y)
            result_r = np.array(
                statmatch_r.gower_dist(
                    data_x_r, data_y_r, rngs=ro.FloatVector(rngs)
                )
            )

        np.testing.assert_allclose(
            result_py,
            result_r,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Gower distances with ranges do not match R",
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_gower_kr_corr_false_against_r(self, numeric_data):
        """Test gower_dist with KR.corr=False against R implementation."""
        from statmatch.distances import gower_dist

        data_x, data_y = numeric_data

        # Python implementation
        result_py = gower_dist(data_x, data_y, kr_corr=False)

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            data_x_r = ro.conversion.py2rpy(data_x)
            data_y_r = ro.conversion.py2rpy(data_y)
            result_r = np.array(
                statmatch_r.gower_dist(data_x_r, data_y_r, KR_corr=False)
            )

        np.testing.assert_allclose(
            result_py,
            result_r,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Gower distances with KR.corr=False do not match R",
        )

    def test_gower_basic_functionality(self, numeric_data):
        """Test basic gower_dist functionality without R comparison."""
        from statmatch.distances import gower_dist

        data_x, data_y = numeric_data

        result = gower_dist(data_x, data_y)

        # Basic checks
        assert result.shape == (len(data_x), len(data_y))
        assert np.all(result >= 0)
        assert np.all(result <= 1)  # Gower distances are in [0, 1]

    def test_gower_symmetric(self, numeric_data):
        """Test that gower_dist is symmetric within a dataset."""
        from statmatch.distances import gower_dist

        data_x, _ = numeric_data

        result = gower_dist(data_x)

        # Check symmetry
        np.testing.assert_allclose(
            result, result.T, rtol=1e-10, atol=1e-10
        )

        # Check diagonal is zero
        np.testing.assert_allclose(
            np.diag(result), np.zeros(len(data_x)), atol=1e-10
        )


class TestMahalanobisDist:
    """Test suite for mahalanobis_dist function."""

    @pytest.fixture
    def numeric_data(self):
        """Create sample numeric data for testing."""
        np.random.seed(42)
        data_x = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, 15),
                "x2": np.random.uniform(0, 10, 15),
                "x3": np.random.normal(5, 2, 15),
            }
        )
        data_y = pd.DataFrame(
            {
                "x1": np.random.normal(0.5, 1, 10),
                "x2": np.random.uniform(1, 11, 10),
                "x3": np.random.normal(5.5, 2, 10),
            }
        )
        return data_x, data_y

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_mahalanobis_against_r(self, numeric_data):
        """Test mahalanobis_dist against R implementation."""
        from statmatch.distances import mahalanobis_dist

        data_x, data_y = numeric_data

        # Python implementation
        result_py = mahalanobis_dist(data_x, data_y)

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            data_x_r = ro.conversion.py2rpy(data_x)
            data_y_r = ro.conversion.py2rpy(data_y)
            result_r = np.array(
                statmatch_r.mahalanobis_dist(data_x_r, data_y_r)
            )

        np.testing.assert_allclose(
            result_py,
            result_r,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Mahalanobis distances do not match R implementation",
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_mahalanobis_within_dataset_against_r(self, numeric_data):
        """Test mahalanobis_dist within same dataset against R."""
        from statmatch.distances import mahalanobis_dist

        data_x, _ = numeric_data

        # Python implementation
        result_py = mahalanobis_dist(data_x)

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            data_x_r = ro.conversion.py2rpy(data_x)
            result_r = np.array(statmatch_r.mahalanobis_dist(data_x_r))

        np.testing.assert_allclose(
            result_py,
            result_r,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Mahalanobis distances (within dataset) do not match R",
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_mahalanobis_with_custom_cov_against_r(self, numeric_data):
        """Test mahalanobis_dist with custom covariance matrix against R."""
        from statmatch.distances import mahalanobis_dist

        data_x, data_y = numeric_data

        # Create a custom covariance matrix
        combined = pd.concat([data_x, data_y], ignore_index=True)
        vc = np.cov(combined.values.T)

        # Python implementation
        result_py = mahalanobis_dist(data_x, data_y, vc=vc)

        # R implementation - need to convert covariance matrix properly
        with localconverter(ro.default_converter + pandas2ri.converter):
            data_x_r = ro.conversion.py2rpy(data_x)
            data_y_r = ro.conversion.py2rpy(data_y)

        # Create R matrix outside the converter context
        # R stores matrices column-major, so we need to transpose
        vc_r = ro.r.matrix(
            ro.FloatVector(vc.T.flatten()),
            nrow=vc.shape[0],
            ncol=vc.shape[1],
        )
        result_r = np.array(
            statmatch_r.mahalanobis_dist(data_x_r, data_y_r, vc=vc_r)
        )

        np.testing.assert_allclose(
            result_py,
            result_r,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Mahalanobis distances with custom cov do not match R",
        )

    def test_mahalanobis_basic_functionality(self, numeric_data):
        """Test basic mahalanobis_dist functionality without R comparison."""
        from statmatch.distances import mahalanobis_dist

        data_x, data_y = numeric_data

        result = mahalanobis_dist(data_x, data_y)

        # Basic checks
        assert result.shape == (len(data_x), len(data_y))
        assert np.all(result >= 0)

    def test_mahalanobis_diagonal_zero(self, numeric_data):
        """Test that mahalanobis_dist has zero diagonal within same dataset."""
        from statmatch.distances import mahalanobis_dist

        data_x, _ = numeric_data

        result = mahalanobis_dist(data_x)

        # Check diagonal is zero
        np.testing.assert_allclose(
            np.diag(result), np.zeros(len(data_x)), atol=1e-10
        )


class TestMaximumDist:
    """Test suite for maximum_dist function."""

    @pytest.fixture
    def numeric_data(self):
        """Create sample numeric data for testing."""
        np.random.seed(42)
        data_x = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, 10),
                "x2": np.random.uniform(0, 10, 10),
                "x3": np.random.normal(5, 2, 10),
            }
        )
        data_y = pd.DataFrame(
            {
                "x1": np.random.normal(0.5, 1, 8),
                "x2": np.random.uniform(1, 11, 8),
                "x3": np.random.normal(5.5, 2, 8),
            }
        )
        return data_x, data_y

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_maximum_against_r(self, numeric_data):
        """Test maximum_dist against R implementation."""
        from statmatch.distances import maximum_dist

        data_x, data_y = numeric_data

        # Python implementation
        result_py = maximum_dist(data_x, data_y)

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            data_x_r = ro.conversion.py2rpy(data_x)
            data_y_r = ro.conversion.py2rpy(data_y)
            result_r = np.array(
                statmatch_r.maximum_dist(data_x_r, data_y_r)
            )

        np.testing.assert_allclose(
            result_py,
            result_r,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Maximum distances do not match R implementation",
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_maximum_within_dataset_against_r(self, numeric_data):
        """Test maximum_dist within same dataset against R implementation."""
        from statmatch.distances import maximum_dist

        data_x, _ = numeric_data

        # Python implementation
        result_py = maximum_dist(data_x)

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            data_x_r = ro.conversion.py2rpy(data_x)
            result_r = np.array(statmatch_r.maximum_dist(data_x_r))

        np.testing.assert_allclose(
            result_py,
            result_r,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Maximum distances (within dataset) do not match R",
        )

    @pytest.mark.skipif(not R_AVAILABLE, reason="R/rpy2 not available")
    def test_maximum_with_rank_against_r(self, numeric_data):
        """Test maximum_dist with rank=True against R implementation."""
        from statmatch.distances import maximum_dist

        data_x, data_y = numeric_data

        # Python implementation
        result_py = maximum_dist(data_x, data_y, rank=True)

        # R implementation
        with localconverter(ro.default_converter + pandas2ri.converter):
            data_x_r = ro.conversion.py2rpy(data_x)
            data_y_r = ro.conversion.py2rpy(data_y)
            result_r = np.array(
                statmatch_r.maximum_dist(data_x_r, data_y_r, rank=True)
            )

        np.testing.assert_allclose(
            result_py,
            result_r,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Maximum distances with rank do not match R",
        )

    def test_maximum_basic_functionality(self, numeric_data):
        """Test basic maximum_dist functionality without R comparison."""
        from statmatch.distances import maximum_dist

        data_x, data_y = numeric_data

        result = maximum_dist(data_x, data_y)

        # Basic checks
        assert result.shape == (len(data_x), len(data_y))
        assert np.all(result >= 0)

    def test_maximum_symmetric(self, numeric_data):
        """Test that maximum_dist is symmetric within a dataset."""
        from statmatch.distances import maximum_dist

        data_x, _ = numeric_data

        result = maximum_dist(data_x)

        # Check symmetry
        np.testing.assert_allclose(
            result, result.T, rtol=1e-10, atol=1e-10
        )

        # Check diagonal is zero
        np.testing.assert_allclose(
            np.diag(result), np.zeros(len(data_x)), atol=1e-10
        )

    def test_maximum_known_values(self):
        """Test maximum_dist with known values."""
        from statmatch.distances import maximum_dist

        # Simple test case
        data_x = pd.DataFrame({"a": [0, 1, 2], "b": [0, 0, 0]})
        data_y = pd.DataFrame({"a": [0, 0], "b": [0, 1]})

        result = maximum_dist(data_x, data_y)

        # Expected: max of |x_a - y_a|, |x_b - y_b| for each pair
        # (0,0) vs (0,0): max(0, 0) = 0
        # (0,0) vs (0,1): max(0, 1) = 1
        # (1,0) vs (0,0): max(1, 0) = 1
        # (1,0) vs (0,1): max(1, 1) = 1
        # (2,0) vs (0,0): max(2, 0) = 2
        # (2,0) vs (0,1): max(2, 1) = 2
        expected = np.array([[0, 1], [1, 1], [2, 2]])

        np.testing.assert_allclose(result, expected, atol=1e-10)
