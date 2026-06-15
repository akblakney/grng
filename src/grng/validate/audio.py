"""Validator for audio entropy source."""

from collections import Counter
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from scipy.stats import chi2

from .base import Validator


class AudioValidator(Validator):
    """Validation checks for audio entropy data.

    `check_waveform_plot` displays the standardized values as a waveform,
    so the user can visually confirm that meaningful audio was captured.

    `check_low_bits_uniformity` extracts the lowest `n_bits` of each
    standardized value and runs a chi-square goodness-of-fit test
    against a uniform distribution over the resulting range.
    """

    def __init__(self, sample_rate: int = 44100, n_bits: int = 4):
        self.sample_rate = sample_rate
        self.n_bits = n_bits

    def check_waveform_plot(self, raw: bytes, values: List[int]) -> None:
        times = [i / self.sample_rate for i in range(len(values))]

        plt.figure(figsize=(12, 4))
        plt.plot(times, values, linewidth=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Sample value")
        plt.title("Audio waveform (standardized values)")
        plt.tight_layout()
        plt.show()

    def check_low_bits_uniformity(self, raw: bytes, values: List[int]) -> Dict[str, Any]:
        num_bins = 1 << self.n_bits
        mask = num_bins - 1

        low_values = [value & mask for value in values]
        counts = Counter(low_values)

        expected = len(values) / num_bins
        chi_square = sum(
            (counts.get(i, 0) - expected) ** 2 / expected for i in range(num_bins)
        )

        degrees_of_freedom = num_bins - 1
        p_value = chi2.sf(chi_square, degrees_of_freedom)

        return {
            "n_bits": self.n_bits,
            "num_bins": num_bins,
            "counts": dict(sorted(counts.items())),
            "expected_per_bin": expected,
            "chi_square": chi_square,
            "degrees_of_freedom": degrees_of_freedom,
            "p_value": p_value,
        }
