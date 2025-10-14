import numpy as np
from app.data_loader.base import BaseDataLoader
from sklearn.base import ClusterMixin


class ClusteringAlgorithm:
    """
    Class for clustering algorithms with data loader and algorithm.
    """

    def __init__(self, data_loader: BaseDataLoader, algorithm: ClusterMixin) -> None:
        self.data_loader = data_loader
        self.algorithm = algorithm

    def load_data(self) -> np.ndarray:
        """
        Load data from the file.

        Returns:
            Data points.
        """
        return self.data_loader.load_data()

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """
        Fits and predict cluster labels for new data points.

        Args:
            data: Data points to assign to clusters.

        Returns:
            Array of cluster labels.

        Raises:
            ValueError: If model is not fitted or data is invalid.
        """
        data = self._validate_data(data)
        return self.algorithm.fit_predict(data)

    def _validate_data(self, data: np.ndarray) -> np.ndarray:
        """
        Validate and convert input data to NumPy array.

        Args:
            data: Input data to validate.

        Returns:
            Validated NumPy array.

        Raises:
            ValueError: If data is invalid or empty.
        """
        if data.size == 0:
            raise ValueError("Data cannot be empty")

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        return data
