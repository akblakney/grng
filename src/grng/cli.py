"""Command-line interface for grng."""

import argparse
import sys

from .extract.bits import LSBExtractor
from .extract.von_neumann import VonNeumannExtractor
from .pipeline import Pipeline
from .sources.microphone import MicrophoneSource
from .validate.audio import AudioValidator

from .constants.audio import SAMPLE_RATE


SOURCES = {
    "audio": (MicrophoneSource, AudioValidator),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grng",
        description="Generate random bytes from various entropy sources.",
    )
    parser.add_argument(
        "source",
        choices=sorted(SOURCES.keys()),
        help="Entropy source to use.",
    )
    parser.add_argument(
        "--bytes",
        type=int,
        default=None,
        metavar="N",
        dest="n_bytes",
        help="Number of random bytes to generate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="File to write the output bytes to. If none provided, write to stdout.",
    )
    parser.add_argument(
        "--lsb-bits",
        type=int,
        default=1,
        metavar="N",
        help="Number of least-significant bits to extract from each "
             "standardized value (default: 1).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="interval between each value to be extracted"
    )
    parser.add_argument(
        "--validate",
        nargs="?",
        const=1.0,
        default=None,
        type=float,
        metavar="RATE",
        help="Run validation checks on the raw/standardized data. Optionally "
            "specify a sampling rate between 0 and 1 (e.g. 0.1 to run 10%% "
            "of the time). If omitted entirely, validation is disabled. "
            "If passed without a value, validation runs every iteration.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["binary", "hex"],
        default="binary",
        dest="fmt",
        help="Output format: binary (default) or hex.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress updates to stderr during generation.",
    )
    return parser


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    bit_extractor = LSBExtractor(lsb_bits=args.lsb_bits, interval=args.interval)
    von_neumann = VonNeumannExtractor()
    validator = build_validator(args) if args.validate else None
    if args.source == "audio":
        source = MicrophoneSource()
        return Pipeline(source, bit_extractor, von_neumann, validator=validator)
    raise ValueError(f"Unknown source: {args.source}")


def build_validator(args: argparse.Namespace):
    if args.source == "audio":
        return AudioValidator(SAMPLE_RATE, thresh=args.validate)
    raise ValueError(f"Unknown source: {args.source}")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pipeline = build_pipeline(args)
    pipeline.run_to_file(args.output, args.n_bytes, verbose=args.verbose, fmt=args.fmt)
    return 0


if __name__ == "__main__":
    sys.exit(main())