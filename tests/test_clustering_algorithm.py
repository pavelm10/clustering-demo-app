from unittest.mock import Mock

import numpy as np
import pytest
from app.clustering.base import ClusteringAlgorithm
from app.data_loader.base import BaseDataLoader
from sklearn.cluster import KMeans


@pytest.fixture
def mock_data_loader():
    mock_data_loader = Mock(spec=BaseDataLoader)
    mock_data_loader.load_data.return_value = np.array([[1, 2], [3, 4], [5, 6]])
    return mock_data_loader


@pytest.fixture
def mock_algorithm():
    mock_algorithm = Mock(spec=KMeans)
    mock_algorithm.fit_predict.return_value = np.array([0, 1, 0])
    return mock_algorithm


@pytest.fixture
def clustering(mock_data_loader, mock_algorithm):
    return ClusteringAlgorithm(data_loader=mock_data_loader, algorithm=mock_algorithm)


class TestClusteringAlgorithm:
    """Test cases for ClusteringAlgorithm class."""

    def test_load_data(self, clustering, mock_data_loader):
        """Test load_data method calls data_loader.load_data()."""
        result = clustering.load_data()

        mock_data_loader.load_data.assert_called_once()
        assert np.array_equal(result, np.array([[1, 2], [3, 4], [5, 6]]))

    def test_fit_predict_with_valid_data(self, clustering, mock_algorithm):
        """Test fit_predict method with valid data."""
        test_data = np.array([[1, 2], [3, 4], [5, 6]])
        result = clustering.fit_predict(test_data)

        mock_algorithm.fit_predict.assert_called_once()
        assert np.array_equal(result, np.array([0, 1, 0]))

    def test_fit_predict_with_1d_data(self, clustering, mock_algorithm):
        """Test fit_predict method with 1D data that gets reshaped."""
        test_data = np.array([1, 2, 3, 4, 5, 6])
        expected_reshaped = test_data.reshape(-1, 1)

        result = clustering.fit_predict(test_data)

        # Verify the algorithm was called with reshaped data
        mock_algorithm.fit_predict.assert_called_once()
        call_args = mock_algorithm.fit_predict.call_args[0][0]
        assert np.array_equal(call_args, expected_reshaped)
        assert np.array_equal(result, np.array([0, 1, 0]))

    def test_validate_data_with_valid_2d_data(self, clustering):
        """Test _validate_data method with valid 2D data."""
        test_data = np.array([[1, 2], [3, 4]])
        result = clustering._validate_data(test_data)

        assert np.array_equal(result, test_data)

    def test_validate_data_with_valid_1d_data(self, clustering):
        """Test _validate_data method with valid 1D data."""
        test_data = np.array([1, 2, 3, 4])
        expected = test_data.reshape(-1, 1)
        result = clustering._validate_data(test_data)

        assert np.array_equal(result, expected)

    def test_validate_data_with_empty_data_raises_error(self, clustering):
        """Test _validate_data method raises ValueError for empty data."""
        empty_data = np.array([])

        with pytest.raises(ValueError, match="Data cannot be empty"):
            clustering._validate_data(empty_data)

    def test_validate_data_with_empty_2d_data_raises_error(self, clustering):
        """Test _validate_data method raises ValueError for empty 2D data."""
        empty_2d_data = np.array([[]])

        with pytest.raises(ValueError, match="Data cannot be empty"):
            clustering._validate_data(empty_2d_data)
