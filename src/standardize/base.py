"""Base class for standardizers."""

from abc import ABC, abstractmethod
from typing import Any, List


class Standardizer(ABC):
    """Abstract base class that converts raw source data into a standard format.

    The standard format is a flat `List[int]`, where each element is a
    single integer value parsed from the source's native raw data
    (e.g. one audio sample, or one pixel channel value).
    """

    @abstractmethod
    def standardize(self, raw: Any) -> List[int]:
        """Convert raw source data into a flat list of integers."""
        raise NotImplementedError
