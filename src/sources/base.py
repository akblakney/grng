"""Base class for entropy sources."""

from abc import ABC, abstractmethod
from typing import Any


class EntropySource(ABC):
    """Abstract base class for a device/source that produces raw entropy data.

    Subclasses wrap a specific input device (microphone, camera, etc.) and
    expose a uniform interface for opening/closing the device and reading
    raw data in whatever native format that device produces.
    """

    @abstractmethod
    def open(self) -> None:
        """Open/initialize the underlying device or resource."""
        raise NotImplementedError

    @abstractmethod
    def read_raw(self, *args: Any, **kwargs: Any) -> Any:
        """Read raw data from the device in its native format.

        The shape and type of the return value is source-specific
        (e.g. bytes for audio, an image array for a camera). It is the
        job of a corresponding Standardizer to convert this into a
        standard format.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close/release the underlying device or resource."""
        raise NotImplementedError

    def __enter__(self) -> "EntropySource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
