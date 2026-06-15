"""Bit extraction strategies: convert a list of standardized ints into a bitstream."""

from abc import ABC, abstractmethod
from typing import List

from bitarray import bitarray


class BitExtractor(ABC):
    """Abstract base class for strategies that extract bits from integer values."""

    @abstractmethod
    def extract(self, values: List[int]) -> bitarray:
        """Convert a list of integer values into a bitarray of extracted bits."""
        raise NotImplementedError


class LSBExtractor(BitExtractor):
    """Extracts the lowest `n` bits of each integer value.

    For each value, the n least-significant bits are taken (using the
    value's two's-complement bit pattern for negative numbers) and
    appended to the output bitarray, most-significant of the n bits first.
    """

    def __init__(self, n: int = 1):
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n

    def extract(self, values: List[int]) -> bitarray:
        bits = bitarray()
        mask = (1 << self.n) - 1 # mask with n lowest bits

        for value in values:
            low_bits = value & mask
            for i in range(self.n - 1, -1, -1):
                bits.append((low_bits >> i) & 1)

        return bits
