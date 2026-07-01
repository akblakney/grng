"""Stream stored random data to stdout.

Reads raw hourly .bin files or post daily.bin files and writes them
to stdout in chronological order, optionally filtered by date range.

Usage:
    grng-collect-stream <base-dir> raw [--from YYYY-MM-DD] [--to YYYY-MM-DD]
    grng-collect-stream <base-dir> post [--from YYYY-MM-DD] [--to YYYY-MM-DD]

Examples:
    grng-collect-stream ~/grng-data raw | xxd | head
    grng-collect-stream ~/grng-data post --from 2026-06-01 --to 2026-06-30 | wc -c
    grng-collect-stream ~/grng-data post | dieharder -a -g 200
"""
import argparse
import os
import sys
from datetime import datetime


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def find_dates(base_dir: str, source: str) -> list[str]:
    """Find all valid date directories in raw/ or post/."""
    d = os.path.join(base_dir, source)
    if not os.path.exists(d):
        print(f"Error: {source}/ directory not found at {d}", file=sys.stderr)
        sys.exit(1)
    return sorted(
        name for name in os.listdir(d)
        if os.path.isdir(os.path.join(d, name)) and is_valid_date(name)
    )


def is_valid_date(name: str) -> bool:
    try:
        datetime.strptime(name, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def filter_dates(
    dates: list[str],
    from_date: str | None,
    to_date: str | None,
) -> list[str]:
    """Filter dates to the given inclusive range."""
    result = dates
    if from_date is not None:
        result = [d for d in result if d >= from_date]
    if to_date is not None:
        result = [d for d in result if d <= to_date]
    return result


def raw_files_for_date(base_dir: str, date_str: str) -> list[str]:
    """Return all hourly .bin file paths for a date in hour order."""
    d = os.path.join(base_dir, "raw", date_str)
    paths = []
    for hour in range(24):
        path = os.path.join(d, f"{hour:02d}.bin")
        paths.append(path)
    return paths


def post_file_for_date(base_dir: str, date_str: str) -> str:
    return os.path.join(base_dir, "post", date_str, "daily.bin")


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

CHUNK_SIZE = 65536  # 64KB read buffer


def stream_file(path: str) -> bool:
    """Stream a single file to stdout. Returns True if successful."""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) == 0:
        return False
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
        return True
    except BrokenPipeError:
        # Consumer closed the pipe (e.g. head, xxd) — exit cleanly
        sys.exit(0)


def stream_raw(base_dir: str, dates: list[str]) -> None:
    """Stream all hourly .bin files for each date in order."""
    for date_str in dates:
        paths = raw_files_for_date(base_dir, date_str)
        any_found = False
        for path in paths:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                continue
            ok = stream_file(path)
            if ok:
                any_found = True
                print(
                    f"[{date_str}] streamed {os.path.basename(path)} "
                    f"({os.path.getsize(path):,} bytes)",
                    file=sys.stderr,
                )
        if not any_found:
            print(
                f"[{date_str}] warning: no hourly .bin files found, skipping",
                file=sys.stderr,
            )


def stream_post(base_dir: str, dates: list[str]) -> None:
    """Stream the daily.bin file for each date in order."""
    for date_str in dates:
        path = post_file_for_date(base_dir, date_str)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(
                f"[{date_str}] warning: no post/daily.bin found, skipping",
                file=sys.stderr,
            )
            continue
        ok = stream_file(path)
        if ok:
            print(
                f"[{date_str}] streamed daily.bin "
                f"({os.path.getsize(path):,} bytes)",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grng-collect-stream",
        description=(
            "Stream stored random data to stdout. "
            "Use 'raw' to stream hourly files or 'post' to stream distilled daily files."
        ),
    )
    parser.add_argument(
        "base_dir",
        type=str,
        help="Base directory containing raw/ and post/ subdirectories.",
    )
    parser.add_argument(
        "source",
        choices=["raw", "post"],
        help="Which data to stream: raw hourly files or post distilled daily files.",
    )
    parser.add_argument(
        "--from",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        dest="from_date",
        help="Start date (inclusive). Defaults to earliest available.",
    )
    parser.add_argument(
        "--to",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        dest="to_date",
        help="End date (inclusive). Defaults to latest available.",
    )
    return parser


def validate_date_arg(value: str, name: str) -> None:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        print(
            f"Error: invalid {name} date {value!r}, expected YYYY-MM-DD",
            file=sys.stderr,
        )
        sys.exit(1)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.from_date is not None:
        validate_date_arg(args.from_date, "--from")
    if args.to_date is not None:
        validate_date_arg(args.to_date, "--to")

    dates = find_dates(args.base_dir, args.source)
    if not dates:
        print(
            f"No date directories found in {args.source}/",
            file=sys.stderr,
        )
        return 1

    dates = filter_dates(dates, args.from_date, args.to_date)
    if not dates:
        print(
            f"No dates found in range "
            f"{args.from_date or 'start'} — {args.to_date or 'end'}",
            file=sys.stderr,
        )
        return 1

    print(
        f"Streaming {len(dates)} date(s) from {args.source}/ "
        f"({dates[0]} to {dates[-1]})",
        file=sys.stderr,
    )

    if args.source == "raw":
        stream_raw(args.base_dir, dates)
    else:
        stream_post(args.base_dir, dates)

    return 0


if __name__ == "__main__":
    sys.exit(main())