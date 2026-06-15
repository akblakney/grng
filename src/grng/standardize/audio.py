"""Standardizer for raw 16-bit PCM audio data."""

import struct
from typing import List

from .base import Standardizer


class AudioStandardizer(Standardizer):
    """Converts raw 16-bit signed little-endian PCM bytes into a list of ints.
    """

    def standardize(self, raw: bytes) -> List[int]:
        if len(raw) % 2 != 0:
            raise ValueError(
                f"Raw audio data length must be a multiple of 2 bytes, got {len(raw)}"
            )

        num_samples = len(raw) // 2
        return list(struct.unpack(f"<{num_samples}h", raw))
