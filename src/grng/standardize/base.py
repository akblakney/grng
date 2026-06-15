"""Base class for standardizers."""

from abc import ABC, abstractmethod
from typing import Any, List


class Standardizer(ABC):
    """Abstract base class that converts raw source data into a standard format.
    """

    @abstractmethod
    def standardize(self, raw: Any) -> List[int]:
        """Convert raw source data into a flat list of integers."""
        raise NotImplementedError
