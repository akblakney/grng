"""Distill grng-collect binary output into a whitened daily file.

For each consecutive pair of hourly .bin files, XORs them together
and concatenates the results into a single daily output file.

This provides a provable quadratic improvement in statistical distance
from uniform for each pair, with no external randomness required.

Usage:
    grng-distill <data-dir> <date> [--output FILE]
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def date_dir(data_dir: str, date_str: str) -> str:
    return os.path.join(data_dir, date_str)


def bin_path(data_dir: str, date_str: str, hour: int) -> str:
    return os.path.join(date_dir(data_dir, date_str), f"{hour:02d}.bin")


def default_output_path(data_dir: str, date_str: str) -> str:
    return os.path.join(date_dir(data_dir, date_str), "daily.bin")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bin(path: str) -> np.ndarray | None:
    """Load a .bin file as a uint8 array. Returns None if missing or empty."""
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    return data if len(data) > 0 else None


# ---------------------------------------------------------------------------
# Core distillation
# ---------------------------------------------------------------------------

import numpy as np

def xor_fold(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Recursively XOR-fold a uint8 array until its length is == n.
    If the final length is greater than n, truncate to n.

    Parameters
    ----------
    arr : np.ndarray
        1-D uint8 array.
    n : int
        Desired maximum output length.

    Returns
    -------
    np.ndarray
        Folded uint8 array of length n or None if
        initially shorter than n).
    """
    if arr.dtype != np.uint8:
        raise TypeError("arr must have dtype=np.uint8")
    if arr.ndim != 1:
        raise ValueError("arr must be 1-dimensional")
    if n <= 0:
        raise ValueError("n must be positive")

    # return None if it cannot be done
    if len(arr) < n:
        return None

    half = len(arr) // 2

    # no fold needed just truncate
    if half < n:
        return arr[:n]

    # Drop the middle element if the length is odd.
    a = arr[:half]
    b = arr[-half:]

    folded = np.bitwise_xor(a, b)

    return xor_fold(folded, n)

def xor_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """XOR two uint8 arrays, truncating to the shorter length."""
    n = min(len(a), len(b))
    if len(a) < len(b):
        discarded_bytes = b[n:]
    elif len(b) < len(a):
        discarded_bytes = a[n:]
    else:
        discarded_bytes = np.ndarray(0, dtype=np.uint8)
    return np.bitwise_xor(a[:n], b[:n]), discarded_bytes


def distill_day(
    data_dir: str,
    date_str: str,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Load all hourly .bin files for a day, XOR consecutive pairs,
    and concatenate the results.

    Returns (distilled_data, report, discarded_bytes) where report contains statistics
    about which pairs were processed, skipped, and how many bytes were
    discarded due to length mismatches.
    """
    # Load all present hours
    hours_data = {}
    for hour in range(24):
        path = bin_path(data_dir, date_str, hour)
        data = load_bin(path)
        if data is not None:
            hours_data[hour] = data
        elif verbose:
            print(f"  Hour {hour:02d}: missing or empty, skipping", file=sys.stderr)

    present_hours = sorted(hours_data.keys())

    if len(present_hours) < 2:
        print(
            f"Error: need at least 2 hours of data, found {len(present_hours)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Form consecutive pairs from present hours
    # e.g. if hours 00,01,02,03,05,06 are present:
    # pairs: (00,01), (02,03), (05,06)
    pairs = []
    i = 0
    while i < len(present_hours) - 1:
        h0 = present_hours[i]
        h1 = present_hours[i + 1]
        pairs.append((h0, h1))
        i += 2

    # If odd number of present hours, last one is unpaired — skip it
    if len(present_hours) % 2 != 0:
        skipped_hour = present_hours[-1]
        if verbose:
            print(
                f"  Hour {skipped_hour:02d}: unpaired (odd number of present hours), "
                f"skipping",
                file=sys.stderr,
            )
    else:
        skipped_hour = None

    # XOR each pair and collect chunks
    chunks = []
    discarded_chunks = []
    pair_reports = []

    for h0, h1 in pairs:
        a = hours_data[h0]
        b = hours_data[h1]
        result, discarded_bytes = xor_pair(a, b)
        kept = result.size
        discarded = discarded_bytes.size
        chunks.append(result)
        discarded_chunks.append(discarded_bytes)
        pair_reports.append({
            "hours": f"{h0:02d}+{h1:02d}",
            "bytes_a": len(a),
            "bytes_b": len(b),
            "bytes_kept": kept,
            "bytes_discarded": discarded,
        })
        if verbose:
            print(
                f"  Pair {h0:02d}+{h1:02d}: {len(a):,} + {len(b):,} bytes "
                f"→ {kept:,} bytes "
                + (f"({discarded:,} discarded)" if discarded > 0 else ""),
                file=sys.stderr,
            )

    distilled = np.concatenate(chunks)
    diamond = xor_fold(np.concatenate(discarded_chunks), 32)

    report = {
        "date": date_str,
        "hours_present": present_hours,
        "hours_skipped": (
            [skipped_hour] if skipped_hour is not None else []
        ),
        "num_pairs": len(pairs),
        "pairs": pair_reports,
        "total_input_bytes": sum(len(hours_data[h]) for h in present_hours),
        "total_output_bytes": len(distilled),
    }

    return distilled, report, diamond


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grng-distill",
        description=(
            "Distill grng-collect hourly output into a whitened daily file "
            "by XORing consecutive pairs of hourly files."
        ),
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Root data directory (same as --output-dir passed to grng-collect).",
    )
    parser.add_argument(
        "date",
        type=str,
        help="Date to distill (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "Output file path. Defaults to <data-dir>/<date>/daily.bin."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-pair progress to stderr.",
    )
    return parser


def main(argv=None) -> int:
    import json

    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate date format
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"Error: invalid date format {args.date!r}, expected YYYY-MM-DD",
              file=sys.stderr)
        return 1

    # Check data directory exists
    d = date_dir(args.data_dir, args.date)
    if not os.path.exists(d):
        print(f"Error: no data directory found at {d}", file=sys.stderr)
        return 1

    output_path = args.output or default_output_path(args.data_dir, args.date)

    if args.verbose:
        print(f"Distilling {args.date}...", file=sys.stderr)

    distilled, report, _ = distill_day(args.data_dir, args.date, verbose=args.verbose)

    # Write distilled output
    distilled.tofile(output_path)

    # Print report as JSON to stdout
    print(json.dumps(report, indent=2))

    if args.verbose:
        print(
            f"\nWrote {len(distilled):,} bytes to {output_path}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())