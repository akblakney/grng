"""Summarize a single statistical test across all processed dates.

Reads raw/<date>/daily_stats.json and post/<date>/daily_stats.json
for all available dates and prints a side-by-side table for the
requested test key.

Usage:
    grng-collect-summarize-stats <base-dir> <test-key>

Example:
    grng-collect-summarize-stats ~/grng-data monobit
    grng-collect-summarize-stats ~/grng-data entropy
    grng-collect-summarize-stats ~/grng-data serial_correlation
"""
import argparse
import json
import os
import sys
from datetime import datetime


# ---------------------------------------------------------------------------
# Test-specific field definitions
# ---------------------------------------------------------------------------

# Maps test key -> list of (field_name, display_label) tuples to show.
# p_value is handled separately and shown for all tests that have it.
TEST_FIELDS: dict[str, list[tuple[str, str]]] = {
    "monobit": [
        ("proportion_ones", "prop_ones"),
    ],
    "runs": [
        ("num_runs", "num_runs"),
        ("expected_runs", "exp_runs"),
    ],
    "block_frequency": [
        ("chi_square", "chi_sq"),
    ],
    "chi_square": [
        ("statistic", "statistic"),
    ],
    "entropy": [
        ("bits_per_byte", "bits/byte"),
    ],
    "mean_std": [
        ("mean", "mean"),
        ("std", "std"),
    ],
    "serial_correlation": [
        ("coefficient", "coefficient"),
    ],
    "byte_pair_chi_square": [
        ("statistic", "statistic"),
    ],
    "longest_run": [
        ("observed_length", "observed"),
        ("expected_length", "expected"),
    ],
}

# Tests that have no p_value
NO_P_VALUE = {"entropy", "mean_std", "serial_correlation", "longest_run"}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def raw_stats_path(base_dir: str, date_str: str) -> str:
    return os.path.join(base_dir, "raw", date_str, "daily_stats.json")


def post_stats_path(base_dir: str, date_str: str) -> str:
    return os.path.join(base_dir, "post", date_str, "daily_stats.json")


def find_dates(base_dir: str) -> list[str]:
    raw = os.path.join(base_dir, "raw")
    if not os.path.exists(raw):
        print(f"Error: raw/ directory not found at {raw}", file=sys.stderr)
        sys.exit(1)
    return sorted(
        name for name in os.listdir(raw)
        if os.path.isdir(os.path.join(raw, name))
        and is_valid_date(name)
    )


def is_valid_date(name: str) -> bool:
    try:
        datetime.strptime(name, "%Y-%m-%d")
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Stats extraction
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def find_test_result(stats: dict, key: str) -> dict | None:
    """Search both byte_level and bit_level for the requested key."""
    if stats is None:
        return None
    for level in ("byte_level", "bit_level"):
        if level in stats and key in stats[level]:
            return stats[level][key]
    return None


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def format_value(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.6f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def build_columns(key: str) -> list[tuple[str, str]]:
    """Return list of (field_name, display_label) for the given test key.
    Prepends p_value for tests that have it.
    """
    extra = TEST_FIELDS.get(key, [])
    if key in NO_P_VALUE:
        return extra
    return [("p_value", "p_value")] + extra


def print_table(
    dates: list[str],
    raw_results: dict[str, dict | None],
    post_results: dict[str, dict | None],
    raw_stats: dict[str, dict | None],
    post_stats: dict[str, dict | None],
    key: str,
) -> None:
    columns = build_columns(key)
    col_labels = [label for _, label in columns]

    date_w = 12
    col_w = 14
    bytes_w = 14

    def pad(s: str, w: int) -> str:
        return s[:w].ljust(w)

    def header_row() -> str:
        parts = [pad("date", date_w)]
        parts.append(pad("raw:bytes", bytes_w))
        for label in col_labels:
            parts.append(pad(f"raw:{label}", col_w))
        parts.append(pad("post:bytes", bytes_w))
        for label in col_labels:
            parts.append(pad(f"post:{label}", col_w))
        return "  ".join(parts)

    def separator_row() -> str:
        total_width = (
            date_w + 2 +
            (bytes_w + 2) +
            (col_w + 2) * len(col_labels) +
            (bytes_w + 2) +
            (col_w + 2) * len(col_labels)
        )
        return "-" * total_width

    def data_row(date_str: str) -> str:
        raw = raw_results.get(date_str)
        post = post_results.get(date_str)
        raw_s = raw_stats.get(date_str)
        post_s = post_stats.get(date_str)
        parts = [pad(date_str, date_w)]
        raw_bytes = raw_s.get("total_bytes") if raw_s else None
        parts.append(pad(format_value(raw_bytes), bytes_w))
        for field, _ in columns:
            v = raw.get(field) if raw else None
            parts.append(pad(format_value(v), col_w))
        post_bytes = post_s.get("total_bytes") if post_s else None
        parts.append(pad(format_value(post_bytes), bytes_w))
        for field, _ in columns:
            v = post.get(field) if post else None
            parts.append(pad(format_value(v), col_w))
        return "  ".join(parts)

    print(f"\nTest: {key}")
    print(header_row())
    print(separator_row())
    for date_str in dates:
        print(data_row(date_str))
    print()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grng-collect-summarize-stats",
        description=(
            "Summarize a single statistical test across all processed dates, "
            "showing raw and post results side by side."
        ),
    )
    parser.add_argument(
        "base_dir",
        type=str,
        help="Base directory containing raw/ and post/ subdirectories.",
    )
    parser.add_argument(
        "test_key",
        type=str,
        help=(
            "Name of the test to summarize. "
            f"Available: {', '.join(sorted(TEST_FIELDS.keys()))}."
        ),
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.test_key not in TEST_FIELDS:
        print(
            f"Error: unknown test key {args.test_key!r}. "
            f"Available: {', '.join(sorted(TEST_FIELDS.keys()))}",
            file=sys.stderr,
        )
        return 1

    dates = find_dates(args.base_dir)
    if not dates:
        print("No date directories found in raw/", file=sys.stderr)
        return 1

    raw_results: dict[str, dict | None] = {}
    post_results: dict[str, dict | None] = {}
    raw_stats_all: dict[str, dict | None] = {}
    post_stats_all: dict[str, dict | None] = {}
    dates_with_data = []

    for date_str in dates:
        raw_stats = load_json(raw_stats_path(args.base_dir, date_str))
        post_stats = load_json(post_stats_path(args.base_dir, date_str))

        raw_result = find_test_result(raw_stats, args.test_key)
        post_result = find_test_result(post_stats, args.test_key)

        if raw_result is None and post_result is None:
            print(
                f"[{date_str}] no stats files found, skipping",
                file=sys.stderr,
            )
            continue

        raw_results[date_str] = raw_result
        post_results[date_str] = post_result
        raw_stats_all[date_str] = raw_stats
        post_stats_all[date_str] = post_stats
        dates_with_data.append(date_str)

    if not dates_with_data:
        print(
            f"No stats files found for test {args.test_key!r}.",
            file=sys.stderr,
        )
        return 1

    print_table(
        dates_with_data,
        raw_results,
        post_results,
        raw_stats_all,
        post_stats_all,
        args.test_key,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())