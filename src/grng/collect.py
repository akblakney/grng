"""Continuous random data collector for grng.

Runs indefinitely, writing one binary file of random data per UTC hour,
alongside a meta file, validation file, and log file for each hour.

Directory structure:
    <output-dir>/
    └── YYYY-MM-DD/
        ├── HH.bin              # raw random bytes
        ├── HH.meta.json        # hash, byte count, timestamps, params
        ├── HH.validation.json  # chi-square and autocorrelation results
        └── HH.log              # chronological log for this hour
"""
import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone

from .extract.bits import LSBExtractor
from .extract.von_neumann import VonNeumannExtractor
from .pipeline import Pipeline
from .sources.microphone import MicrophoneSource
from .validate.audio import AudioValidator
from .constants.audio import SAMPLE_RATE

SOURCES = {
    "audio": (MicrophoneSource, AudioValidator),
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def hour_dir(output_dir: str, dt: datetime) -> str:
    """Return the date directory for a given UTC datetime, creating if needed."""
    path = os.path.join(output_dir, dt.strftime("%Y-%m-%d"))
    os.makedirs(path, exist_ok=True)
    return path


def hour_stem(dt: datetime) -> str:
    """Return the hour stem (e.g. '14') for a given UTC datetime."""
    return dt.strftime("%H")


def hour_paths(output_dir: str, dt: datetime) -> dict:
    """Return all file paths for a given UTC hour."""
    d = hour_dir(output_dir, dt)
    stem = hour_stem(dt)
    return {
        "bin":        os.path.join(d, f"{stem}.bin"),
        "meta":       os.path.join(d, f"{stem}.meta.json"),
        "validation": os.path.join(d, f"{stem}.validation.json"),
        "log":        os.path.join(d, f"{stem}.log"),
    }


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()

def write_preliminary_meta(
    paths: dict,
    hour_start: datetime,
    existing_bytes: int,
    args: argparse.Namespace,
    resume_count: int,
) -> None:
    meta = {
        "source": args.source,
        "hour_start_utc": hour_start.isoformat(),
        "hour_end_utc": None,
        "bytes_written": existing_bytes,  # reflects .bin size at start of this run
        "sha256": None,
        "resume_count": resume_count,
        "complete": False,
        "partial_start": resume_count == 0 and hour_start.minute != 0,
        "params": {
            "lsb_bits": args.lsb_bits,
            "interval": args.interval,
            "validate": args.validate,
        },
    }
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)

def write_meta(
    paths: dict,
    hour_start: datetime,
    hour_end: datetime,
    bytes_written: int,
    args: argparse.Namespace,
    resume_count: int,
) -> None:
    sha256 = compute_sha256(paths["bin"])
    meta = {
        "source": args.source,
        "hour_start_utc": hour_start.isoformat(),
        "hour_end_utc": hour_end.isoformat(),
        "bytes_written": bytes_written,
        "sha256": sha256,
        "resume_count": resume_count,
        "complete": True,
        "partial_start": resume_count == 0 and hour_start.minute != 0,
        "params": {
            "lsb_bits": args.lsb_bits,
            "interval": args.interval,
            "validate": args.validate,
        },
    }
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)


def write_validation(paths: dict, validator: AudioValidator) -> None:
    data = validator.to_dict()
    with open(paths["validation"], "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(paths: dict, message: str) -> None:
    """Append a timestamped message to the hour's log file and stderr."""
    now = datetime.now(timezone.utc).isoformat()
    line = f"[{now}] {message}\n"
    with open(paths["log"], "a") as f:
        f.write(line)
    #print(line, end="", file=sys.stderr)


# ---------------------------------------------------------------------------
# Resume handling
# ---------------------------------------------------------------------------

def load_resume_state(paths: dict) -> tuple[int, int]:
    """Return (existing_byte_count, resume_count) by inspecting the .bin file.

    The .bin file size is the ground truth — it reflects exactly how many
    bytes were written regardless of when the process was interrupted.
    The resume_count is read from meta if available.
    """
    existing_bytes = 0
    resume_count = 0

    if os.path.exists(paths["bin"]):
        existing_bytes = os.path.getsize(paths["bin"])

    if existing_bytes > 0 and os.path.exists(paths["meta"]):
        try:
            with open(paths["meta"]) as f:
                meta = json.load(f)
            resume_count = meta.get("resume_count", 0) + 1
        except (json.JSONDecodeError, KeyError):
            resume_count = 1  # meta corrupted but .bin exists — still a resume

    return existing_bytes, resume_count


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(args: argparse.Namespace) -> Pipeline:
    bit_extractor = LSBExtractor(lsb_bits=args.lsb_bits, interval=args.interval)
    von_neumann = VonNeumannExtractor()
    validator = None
    if args.validate is not None:
        if args.source == "audio":
            validator = AudioValidator(args.sample_rate, thresh=args.validate)
        else:
            raise ValueError(f"Unknown source: {args.source}")
    if args.source == "audio":
        source = MicrophoneSource(rate=args.sample_rate)
        return Pipeline(source, bit_extractor, von_neumann, validator=validator)
    raise ValueError(f"Unknown source: {args.source}")


# ---------------------------------------------------------------------------
# Core collection loop
# ---------------------------------------------------------------------------

def collect_hour(
    pipeline: Pipeline,
    paths: dict,
    hour_start: datetime,
    deadline: float,
    args: argparse.Namespace,
) -> None:
    existing_bytes, resume_count = load_resume_state(paths)

    if existing_bytes > 0:
        log(paths, f"Resuming hour {hour_start.strftime('%H')} "
                   f"(resume #{resume_count}, {existing_bytes} bytes already written)")
    else:
        log(paths, f"Starting hour {hour_start.strftime('%H')} UTC")

    # Write preliminary meta before we start so interrupts are detectable
    write_preliminary_meta(paths, hour_start, existing_bytes, args, resume_count)

    if pipeline.validator is not None:
        pipeline.validator.reset()

    # Append to existing .bin file on resume, otherwise overwrite
    bin_mode = "ab"
    bytes_written_this_run = pipeline.run_to_file(
        paths["bin"],
        deadline=deadline,
        fmt="binary",
        file_mode=bin_mode,
    )

    total_bytes = existing_bytes + bytes_written_this_run
    hour_end = datetime.now(timezone.utc)

    write_meta(paths, hour_start, hour_end, total_bytes, args, resume_count)
    log(paths, f"Wrote meta: {total_bytes} bytes total, sha256 recomputed over full file")

    if pipeline.validator is not None:
        write_validation(paths, pipeline.validator)
        log(paths, "Wrote validation file")

    log(paths, f"Hour complete. {bytes_written_this_run} bytes written this run, "
               f"{total_bytes} bytes total.")


def run_collector(args: argparse.Namespace) -> None:
    pipeline = build_pipeline(args)
    first = True

    while True:
        now = datetime.now(timezone.utc)

        # First iteration may be a partial hour to align to UTC boundary
        if first:
            seconds = Pipeline.seconds_until_next_utc_hour()
            hour_start = now
            first = False
        else:
            seconds = 3600.0
            hour_start = now

        deadline = Pipeline.make_deadline(seconds)
        paths = hour_paths(args.output_dir, hour_start)

        try:
            collect_hour(pipeline, paths, hour_start, deadline, args)
        except KeyboardInterrupt:
            log(paths, "Interrupted by user. Exiting.")
            sys.exit(0)
        except Exception:
            log(paths, f"Unexpected error:\n{traceback.format_exc()}")
            log(paths, "Continuing to next hour.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grng-collect",
        description="Continuously generate random bytes and store them hourly.",
    )
    parser.add_argument(
        "source",
        choices=sorted(SOURCES.keys()),
        help="Entropy source to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory to write output files to.",
    )
    parser.add_argument(
        "--lsb-bits",
        type=int,
        default=1,
        metavar="N",
        help="Number of least-significant bits to extract per sample (default: 1).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        metavar="N",
        help="Sample rate for audio sampling."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        metavar="N",
        help="Stride between samples used for bit extraction (default: 1).",
    )
    parser.add_argument(
        "--validate",
        nargs="?",
        const=1.0,
        default=None,
        type=float,
        metavar="RATE",
        help="Enable validation. Optionally specify a sampling rate (default: 1.0).",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        run_collector(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())