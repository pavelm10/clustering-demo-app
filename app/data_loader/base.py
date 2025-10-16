from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    """

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        if not self.file_path.exists():
            raise FileNotFoundError(f"File {self.file_path} not found")

    @abstractmethod
    def load_data(self) -> np.ndarray:
        pass
