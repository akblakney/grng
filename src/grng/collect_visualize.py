"""Visualize grng-collect binary output files.

Usage:
    grng-collect-visualize <data-dir> <period> [options]

Where <period> is one of:
    2026-06-22        # use data from the full day
    2026-06-22T14     # use data from a single hour
"""
import argparse
import os
import sys

import numpy as np

from .visualize.random_walk import plot_1d, plot_2d
from .visualize.scatter import plot_scatter
from .visualize.monte_carlo import plot_monte_carlo
from .visualize.hilbert import plot_hilbert


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def bin_path(data_dir: str, date_str: str, hour: int) -> str:
    return os.path.join(data_dir, date_str, f"{hour:02d}.bin")


def load_hour(data_dir: str, date_str: str, hour: int) -> np.ndarray:
    path = bin_path(data_dir, date_str, hour)
    if not os.path.exists(path):
        print(f"Error: no .bin file found at {path}", file=sys.stderr)
        sys.exit(1)
    return np.fromfile(path, dtype=np.uint8)


def load_day(data_dir: str, date_str: str, max_bytes: int) -> np.ndarray:
    """Load up to max_bytes from the day's .bin files in hour order."""
    chunks = []
    total = 0
    for hour in range(24):
        path = bin_path(data_dir, date_str, hour)
        if not os.path.exists(path):
            continue
        chunk = np.fromfile(path, dtype=np.uint8)
        remaining = max_bytes - total
        if len(chunk) >= remaining:
            chunks.append(chunk[:remaining])
            total += remaining
            break
        chunks.append(chunk)
        total += len(chunk)
    if not chunks:
        print(f"Error: no .bin files found for {date_str}", file=sys.stderr)
        sys.exit(1)
    return np.concatenate(chunks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_period(period: str) -> tuple[str, int | None]:
    from datetime import datetime
    if "T" in period:
        parts = period.split("T")
        date_str = parts[0]
        try:
            hour = int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid hour: {parts[1]!r}")
        if not 0 <= hour <= 23:
            raise ValueError(f"Hour must be 0-23, got {hour}")
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str, hour
    else:
        from datetime import datetime
        datetime.strptime(period, "%Y-%m-%d")
        return period, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grng-collect-visualize",
        description="Visualize grng-collect binary output files.",
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Root data directory.",
    )
    parser.add_argument(
        "period",
        type=str,
        help="Period to visualize: 2026-06-22 or 2026-06-22T14.",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=["walk1d", "walk2d", "scatter", "montecarlo", "hilbert"],
        default=["walk1d", "walk2d", "scatter", "montecarlo", "hilbert"],
        metavar="PLOT",
        help=(
            "Which plots to show. Choices: walk1d walk2d scatter montecarlo hilbert. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--hilbert-order",
        type=int,
        default=7,
        metavar="N",
        help="Hilbert curve order (default: 7 → 128×128, needs 16384 bytes).",
    )
    return parser


def main(argv=None) -> int:
    import matplotlib.pyplot as plt

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        date_str, hour = parse_period(args.period)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Determine how many bytes we need at most
    hilbert_bytes = 4 ** args.hilbert_order  # e.g. 16384 for order 7
    max_bytes = max(
        100_000 * 8,    # walk1d: 100k bits = 12.5k bytes
        50_000 * 2,     # walk2d: 50k steps = 100k bytes
        50_000 * 2,     # scatter: 50k pairs = 100k bytes
        10_000 * 2,     # montecarlo: 10k points = 20k bytes
        hilbert_bytes,
    )

    if hour is not None:
        data = load_hour(data_dir=args.data_dir, date_str=date_str, hour=hour)
    else:
        data = load_day(data_dir=args.data_dir, date_str=date_str, max_bytes=max_bytes)

    plots = set(args.plots)
    figures = []

    if "walk1d" in plots:
        figures.append(plot_1d(data))
    if "walk2d" in plots:
        figures.append(plot_2d(data))
    if "scatter" in plots:
        figures.append(plot_scatter(data))
    if "montecarlo" in plots:
        figures.append(plot_monte_carlo(data))
    if "hilbert" in plots:
        try:
            figures.append(plot_hilbert(data, order=args.hilbert_order))
        except ValueError as e:
            print(f"Warning: hilbert plot skipped — {e}", file=sys.stderr)

    if not figures:
        print("No plots to show.", file=sys.stderr)
        return 1

    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())