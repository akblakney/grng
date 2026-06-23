"""Validator for audio entropy source."""
import json
import sys
from collections import Counter
from typing import Any, Dict, List
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import chi2

from .base import Validator


class AudioValidator(Validator):
    """Validation checks for audio entropy data.
    """

    def __init__(self, sample_rate: int = 44100, n_bits: int = 4, thresh: int = 1.0, plot=False):
        self.sample_rate = sample_rate
        self.n_bits = n_bits
        self.has_plotted = False
        self._counts = Counter()
        self._total_values = 0
        self.thresh = thresh
        self.plot = plot

    def check_waveform_plot(self, raw: bytes, values: List[int]) -> None:
        if self.has_plotted or not self.plot:
            return
        times = [i / self.sample_rate for i in range(len(values))]
        plt.figure(figsize=(12, 4))
        plt.plot(times, values, linewidth=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Sample value")
        plt.title("Audio waveform (standardized values)")
        plt.tight_layout()
        plt.show()
        self.has_plotted = True

    def accumulate(self, raw: bytes, values: List[int]) -> None:
        mask = (1 << self.n_bits) - 1
        self._counts.update(value & mask for value in values)
        self._total_values += len(values)

    def finalize(self) -> None:
        num_bins = 1 << self.n_bits
        expected = self._total_values / num_bins
        chi_square = sum(
            (self._counts.get(i, 0) - expected) ** 2 / expected
            for i in range(num_bins)
        )
        degrees_of_freedom = num_bins - 1
        p_value = chi2.sf(chi_square, degrees_of_freedom)
        result = {
            "n_bits": self.n_bits,
            "num_bins": num_bins,
            "total_values": self._total_values,
            "counts": dict(sorted(self._counts.items())),
            "expected_per_bin": round(expected, 2),
            "chi_square": round(chi_square, 4),
            "degrees_of_freedom": degrees_of_freedom,
            "p_value": round(p_value, 4),
        }
        print("\n===== LOW BITS UNIFORMITY (cumulative) =====", file=sys.stderr)
        print(json.dumps(result, indent=2), file=sys.stderr)
        print("=============================================\n", file=sys.stderr)
    
    def check_lsb_autocorrelation(self, raw: bytes, values: List[int]) -> Dict[str, Any]:
        # Extract LSB stream as +1/-1 (zero-mean) for standard autocorrelation
        bits = np.array([(v & 1) * 2 - 1 for v in values], dtype=float)

        n = len(bits)
        variance = np.var(bits)

        lags = [2**i for i in range(11)]  # 1, 2, 4, ..., 1024
        autocorr = {
            lag: round(float(np.mean(bits[:n - lag] * bits[lag:]) / variance), 4)
            for lag in lags
            if lag < n
        }

        return {"lsb_autocorrelation": autocorr}

    def reset(self) -> None:
        self._counts = Counter()
        self._total_values = 0
        self.has_plotted = False

    def to_dict(self) -> dict:
        if self._total_values == 0:
            return {}
        num_bins = 1 << self.n_bits
        expected = self._total_values / num_bins
        chi_square = sum(
            (self._counts.get(i, 0) - expected) ** 2 / expected
            for i in range(num_bins)
        )
        degrees_of_freedom = num_bins - 1
        p_value = float(chi2.sf(chi_square, degrees_of_freedom))
        return {
            "n_bits": self.n_bits,
            "num_bins": num_bins,
            "total_values": self._total_values,
            "counts": dict(sorted(self._counts.items())),
            "expected_per_bin": round(expected, 2),
            "chi_square": round(chi_square, 4),
            "degrees_of_freedom": degrees_of_freedom,
            "p_value": round(p_value, 4),
        }