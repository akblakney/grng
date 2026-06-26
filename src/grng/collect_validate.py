"""Summarize grng-collect validation and meta files.

Usage:
    grng-collect-validate <data-dir> <period>

Where <period> is one of:
    2026-06-22        # full day summary
    2026-06-22T14     # single hour summary
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def date_dir(data_dir: str, date_str: str) -> str:
    return os.path.join(data_dir, date_str)


def hour_paths(data_dir: str, date_str: str, hour: int) -> dict:
    d = date_dir(data_dir, date_str)
    stem = f"{hour:02d}"
    return {
        "bin":        os.path.join(d, f"{stem}.bin"),
        "meta":       os.path.join(d, f"{stem}.meta.json"),
        "validation": os.path.join(d, f"{stem}.validation.json"),
        "log":        os.path.join(d, f"{stem}.log"),
    }


def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Summarization helpers
# ---------------------------------------------------------------------------

def summarize_autocorrelation(blocks: list[dict]) -> dict | None:
    """Average autocorrelation per lag across all blocks."""
    if not blocks:
        return None
    all_lags = set()
    for block in blocks:
        all_lags.update(int(k) for k in block.keys())
    averaged = {}
    for lag in sorted(all_lags):
        values = [block[str(lag)] for block in blocks if str(lag) in block]
        averaged[lag] = round(sum(values) / len(values), 4) if values else None
    return {
        "num_blocks": len(blocks),
        "averaged_per_lag": averaged,
    }


def merge_chi_square(chi_results: list[dict]) -> dict | None:
    """Merge chi-square results by summing counts and recomputing statistics."""
    if not chi_results:
        return None

    from scipy.stats import chi2 as scipy_chi2

    # All results must share the same n_bits
    n_bits = chi_results[0]["n_bits"]
    num_bins = chi_results[0]["num_bins"]

    merged_counts: Counter = Counter()
    total_values = 0
    for result in chi_results:
        merged_counts.update({int(k): v for k, v in result["counts"].items()})
        total_values += result["total_values"]

    expected = total_values / num_bins
    chi_square = sum(
        (merged_counts.get(i, 0) - expected) ** 2 / expected
        for i in range(num_bins)
    )
    degrees_of_freedom = num_bins - 1
    p_value = float(scipy_chi2.sf(chi_square, degrees_of_freedom))

    return {
        "n_bits": n_bits,
        "num_bins": num_bins,
        "total_values": total_values,
        "counts": dict(sorted({str(k): v for k, v in merged_counts.items()}.items())),
        "expected_per_bin": round(expected, 2),
        "chi_square": round(chi_square, 4),
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": round(p_value, 4),
    }


def summarize_values(all_values: list[int]) -> dict | None:
    """Basic summary statistics over the collected values."""
    if not all_values:
        return None
    import numpy as np
    arr = np.array(all_values, dtype=float)
    return {
        "count": len(all_values),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
    }


def summarize_meta(meta: dict | None, hour: int) -> dict:
    """Extract relevant fields from a meta file, or return a missing entry."""
    if meta is None:
        return {
            "hour": f"{hour:02d}",
            "present": False,
            "bytes_written": 0,
            "complete": False,
            "hour_start_utc": None,
            "hour_end_utc": None,
            "resume_count": None,
        }
    return {
        "hour": f"{hour:02d}",
        "present": True,
        "bytes_written": meta.get("bytes_written", 0),
        "complete": meta.get("complete", False),
        "hour_start_utc": meta.get("hour_start_utc"),
        "hour_end_utc": meta.get("hour_end_utc"),
        "resume_count": meta.get("resume_count", 0),
    }


# ---------------------------------------------------------------------------
# Single hour summary
# ---------------------------------------------------------------------------

def summarize_hour(data_dir: str, date_str: str, hour: int) -> dict:
    paths = hour_paths(data_dir, date_str, hour)
    meta = load_json(paths["meta"])
    validation = load_json(paths["validation"])

    result: dict[str, Any] = {
        "period": f"{date_str}T{hour:02d}",
        "meta": summarize_meta(meta, hour),
    }

    print(paths)
    if validation is None:
        result["validation"] = None
        return result

    chi_results = []
    if "chi_square_results" in validation:
        chi_results.append(validation["chi_square_results"])

    autocorr_blocks = validation.get("autocorrelation_results", [])
    values = validation.get("values_results", [])

    result["validation"] = {
        "chi_square": merge_chi_square(chi_results),
        "autocorrelation": summarize_autocorrelation(autocorr_blocks),
        "values": summarize_values(values),
    }

    return result


# ---------------------------------------------------------------------------
# Full day summary
# ---------------------------------------------------------------------------

def summarize_day(data_dir: str, date_str: str) -> dict:
    d = date_dir(data_dir, date_str)
    if not os.path.exists(d):
        print(f"Error: no data directory found for {date_str} at {d}", file=sys.stderr)
        sys.exit(1)

    per_hour_meta = []
    all_chi = []
    all_autocorr_blocks = []
    all_values = []
    total_bytes = 0
    hours_present = 0
    hours_complete = 0

    for hour in range(24):
        paths = hour_paths(data_dir, date_str, hour)
        meta = load_json(paths["meta"])
        validation = load_json(paths["validation"])

        hour_meta = summarize_meta(meta, hour)
        per_hour_meta.append(hour_meta)
        total_bytes += hour_meta["bytes_written"]
        if hour_meta["present"]:
            hours_present += 1
        if hour_meta["complete"]:
            hours_complete += 1

        if validation is not None:
            if "chi_square_results" in validation:
                all_chi.append(validation["chi_square_results"])
            all_autocorr_blocks.extend(validation.get("autocorrelation_results", []))
            all_values.extend(validation.get("values_results", []))

    missing_hours = [h["hour"] for h in per_hour_meta if not h["present"]]

    return {
        "period": date_str,
        "meta_summary": {
            "total_bytes": total_bytes,
            "hours_present": hours_present,
            "hours_complete": hours_complete,
            "hours_missing": missing_hours,
            "per_hour": per_hour_meta,
        },
        "validation": {
            "chi_square": merge_chi_square(all_chi),
            "autocorrelation": summarize_autocorrelation(all_autocorr_blocks),
            "values": summarize_values(all_values),
        },
    }

def plot_values(values: list[int], period: str) -> None:
    """Plot raw 16-bit signed int values as a waveform and histogram."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"Values — {period}", fontsize=14)

    # Waveform
    ax1.plot(values, linewidth=0.5)
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Sample value")
    ax1.set_ylim(-32768, 32767)
    ax1.set_title("Waveform")

    # Histogram
    ax2.hist(values, bins=1024, range=(-32768, 32767), edgecolor="none")
    ax2.axhline(
        y=len(values) / 1024,
        color="r",
        linestyle="--",
        linewidth=1,
        label="Expected (uniform)",
    )
    ax2.set_xlabel("Sample value (-32768 to 32767)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution")
    ax2.legend()

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_period(period: str) -> tuple[str, int | None]:
    """Parse a period string into (date_str, hour_or_None).

    Accepts:
        2026-06-22        -> ('2026-06-22', None)
        2026-06-22T14     -> ('2026-06-22', 14)
    """
    if "T" in period:
        parts = period.split("T")
        if len(parts) != 2:
            raise ValueError(f"Invalid period format: {period!r}")
        date_str = parts[0]
        try:
            hour = int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid hour in period: {parts[1]!r}")
        if not 0 <= hour <= 23:
            raise ValueError(f"Hour must be 0-23, got {hour}")
        # Validate date format
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str, hour
    else:
        datetime.strptime(period, "%Y-%m-%d")
        return period, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grng-collect-validate",
        description="Summarize grng-collect validation and meta files.",
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Root data directory (same as --output-dir passed to grng-collect).",
    )
    parser.add_argument(
        "period",
        type=str,
        help=(
            "Period to summarize. Either a date (2026-06-22) for a full day "
            "summary, or a date and hour (2026-06-22T14) for a single hour."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the values from the validation file (hourly mode only).",
    )

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        date_str, hour = parse_period(args.period)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if hour is not None:
        result = summarize_hour(args.data_dir, date_str, hour)
        print(json.dumps(result, indent=2))
        if args.plot:
            validation = load_json(
                hour_paths(args.data_dir, date_str, hour)["validation"]
            )
            if validation is None or not validation.get("values_results"):
                print("No values data available to plot.", file=sys.stderr)
            else:
                plot_values(validation["values_results"], result["period"])
    else:
        if args.plot:
            print("Warning: --plot is only supported for hourly mode. Ignoring.", file=sys.stderr)
        result = summarize_day(args.data_dir, date_str)
        print(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())