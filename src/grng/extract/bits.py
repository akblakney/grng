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
    """Extracts the lowest `lsb_bits` bits of each integer value.

    For each value, the lsb_bits least-significant bits are taken (using the
    value's two's-complement bit pattern for negative numbers) and
    appended to the output bitarray, most-significant of the lsb_bits bits first.
    """

    def __init__(self, lsb_bits: int = 1, interval: int = 1):
        if lsb_bits < 1:
            raise ValueError(f"lsb_bits must be >= 1, got {lsb_bits}")
        if interval < 1:
            raise ValueError(f"interval must be >= 1, got {interval}")
        self.lsb_bits = lsb_bits
        self.interval = interval

    def extract(self, values: List[int]) -> bitarray:
        bits = bitarray()
        mask = (1 << self.lsb_bits) - 1 # mask with lsb_bits lowest bits

        for i in range(0, len(values), self.interval):
            low_bits = values[i] & mask
            for j in range(self.lsb_bits - 1, -1, -1):
                bits.append((low_bits >> j) & 1)

        return bits
