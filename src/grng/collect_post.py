"""Post-processing script for grng-collect raw output.

Scans raw/ for date directories and for each date:
  1. Computes statistics on the raw hourly .bin files
     -> raw/<date>/daily_stats.json  (if not already present)
  2. Distills hourly files into a whitened daily.bin by XORing pairs
     -> post/<date>/daily.bin        (if not already present)
  3. Computes statistics on the distilled daily.bin
     -> post/<date>/daily_stats.json (if not already present)

Usage:
    grng-collect-post <base-dir>

Where <base-dir> contains raw/ and post/ subdirectories.
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

from .collect_stats import (
    load_day_bytes,
    test_byte_chi_square,
    test_byte_entropy,
    test_byte_mean_std,
    test_serial_correlation,
    test_byte_pair_chi_square,
    test_byte_longest_run,
    test_monobit,
    test_bit_runs,
    test_block_frequency,
    test_bit_longest_run,
    unpack_bits,
)
from .post.distill import distill_day
from .post.bitmap import bitmap256


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def raw_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "raw")


def post_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "post")


def raw_date_dir(base_dir: str, date_str: str) -> str:
    return os.path.join(raw_dir(base_dir), date_str)


def post_date_dir(base_dir: str, date_str: str) -> str:
    return os.path.join(post_dir(base_dir), date_str)


def raw_daily_stats_path(base_dir: str, date_str: str) -> str:
    return os.path.join(raw_date_dir(base_dir, date_str), "daily_stats.json")


def post_daily_bin_path(base_dir: str, date_str: str) -> str:
    return os.path.join(post_date_dir(base_dir, date_str), "daily.bin")

def post_diamond_bin_path(base_dir: str, date_str: str) -> str:
    return os.path.join(post_date_dir(base_dir, date_str), "diamond.bin")

def get_post_diamond_bitmap_path(base_dir: str, date_str: str) -> str:
    return os.path.join(post_date_dir(base_dir, date_str), "diamond.png")

def post_daily_stats_path(base_dir: str, date_str: str) -> str:
    return os.path.join(post_date_dir(base_dir, date_str), "daily_stats.json")


# ---------------------------------------------------------------------------
# Stats computation (shared for raw and post)
# ---------------------------------------------------------------------------

def compute_stats(data: np.ndarray, label: str) -> dict:
    """Run all statistical tests on a uint8 array and return as a dict."""
    bits = unpack_bits(data)
    return {
        "label": label,
        "total_bytes": len(data),
        "total_bits": len(bits),
        "byte_level": {
            "chi_square": test_byte_chi_square(data),
            "entropy": test_byte_entropy(data),
            "mean_std": test_byte_mean_std(data),
            "serial_correlation": test_serial_correlation(data),
            "byte_pair_chi_square": test_byte_pair_chi_square(data),
            "longest_run": test_byte_longest_run(data),
        },
        "bit_level": {
            "monobit": test_monobit(bits),
            "runs": test_bit_runs(bits),
            "block_frequency": test_block_frequency(bits),
            "longest_run": test_bit_longest_run(bits),
        },
    }


def write_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Per-date processing
# ---------------------------------------------------------------------------

def is_valid_date_dir(name: str) -> bool:
    """Check if a directory name is a valid YYYY-MM-DD date."""
    try:
        datetime.strptime(name, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def process_date(base_dir: str, date_str: str, verbose: bool) -> None:
    """Run all post-processing steps for a single date."""

    def log(msg: str) -> None:
        print(f"[{date_str}] {msg}", file=sys.stderr)

    # Ensure post date directory exists
    os.makedirs(post_date_dir(base_dir, date_str), exist_ok=True)

    raw_stats_path = raw_daily_stats_path(base_dir, date_str)
    post_bin_path = post_daily_bin_path(base_dir, date_str)
    post_diamond_path = post_diamond_bin_path(base_dir, date_str)
    post_diamond_bitmap_path = get_post_diamond_bitmap_path(base_dir, date_str)
    post_stats_path = post_daily_stats_path(base_dir, date_str)

    all_present = (
        os.path.exists(raw_stats_path)
        and os.path.exists(post_bin_path)
        and os.path.exists(post_stats_path)
    )
    if all_present:
        log("all output files already exist, skipping")
        return

    # ------------------------------------------------------------------
    # Step 1 — raw daily_stats
    # ------------------------------------------------------------------
    if os.path.exists(raw_stats_path):
        log(f"raw/daily_stats.json already exists, skipping")
    else:
        log("computing raw daily stats...")
        try:
            raw_data, missing_hours = load_day_bytes(
                raw_dir(base_dir), date_str
            )
            stats = compute_stats(raw_data, label=f"raw/{date_str}")
            stats["hours_missing"] = missing_hours
            write_json(raw_stats_path, stats)
            log(f"wrote raw/daily_stats.json ({len(raw_data):,} bytes)")
        except SystemExit:
            log("no raw .bin files found, skipping date entirely")
            return

    # ------------------------------------------------------------------
    # Step 2 — post daily.bin
    # ------------------------------------------------------------------
    if os.path.exists(post_bin_path):
        log("post/daily.bin already exists, skipping distillation")
        distilled = np.fromfile(post_bin_path, dtype=np.uint8)
    else:
        log("distilling hourly files...")
        try:
            distilled, distill_report, diamond = distill_day(
                raw_dir(base_dir), date_str, verbose=verbose
            )
        except SystemExit:
            log("distillation failed (not enough hours), skipping")
            return
        distilled.tofile(post_bin_path)
        if diamond is not None:
            diamond.tofile(post_diamond_path)
            bitmap256(diamond, post_diamond_bitmap_path)

        else:
            log("not enough discarded bytes to write diamond, skipping")
        log(
            f"wrote post/daily.bin "
            f"({len(distilled):,} bytes from {distill_report['num_pairs']} pairs)"
        )

        # Write distill report alongside the bin
        distill_report_path = os.path.join(
            post_date_dir(base_dir, date_str), "distill_report.json"
        )
        write_json(distill_report_path, distill_report)

    # ------------------------------------------------------------------
    # Step 3 — post daily_stats
    # ------------------------------------------------------------------
    if os.path.exists(post_stats_path):
        log("post/daily_stats.json already exists, skipping")
    else:
        log("computing post daily stats...")
        post_stats = compute_stats(distilled, label=f"post/{date_str}")
        write_json(post_stats_path, post_stats)
        log("wrote post/daily_stats.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_dates(base_dir: str) -> list[str]:
    """Find all valid date directories in raw/."""
    raw = raw_dir(base_dir)
    if not os.path.exists(raw):
        print(f"Error: raw/ directory not found at {raw}", file=sys.stderr)
        sys.exit(1)
    dates = sorted(
        name for name in os.listdir(raw)
        if os.path.isdir(os.path.join(raw, name)) and is_valid_date_dir(name)
    )
    return dates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grng-collect-post",
        description=(
            "Post-process grng-collect raw output. "
            "Computes statistics on raw hourly data, distills into a "
            "whitened daily.bin, and computes statistics on the distilled data."
        ),
    )
    parser.add_argument(
        "base_dir",
        type=str,
        help="Base directory containing raw/ and post/ subdirectories.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-pair distillation progress.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dates = find_dates(args.base_dir)

    if not dates:
        print("No date directories found in raw/", file=sys.stderr)
        return 1

    print(f"Found {len(dates)} date(s) to process: {', '.join(dates)}", file=sys.stderr)

    for date_str in dates:
        process_date(args.base_dir, date_str, verbose=args.verbose)

    print("Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())