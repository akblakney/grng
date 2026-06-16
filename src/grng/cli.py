"""Command-line interface for grng."""

import argparse
import sys

from .extract.bits import LSBExtractor
from .extract.von_neumann import VonNeumannExtractor
from .pipeline import Pipeline
from .sources.microphone import MicrophoneSource
from .validate.audio import AudioValidator

from .constants.audio import SAMPLE_RATE

# Registry mapping source names to (EntropySource factory, Standardizer factory,
# Validator factory). Each factory is a zero-arg callable that constructs a
# fresh instance.
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
        "--lsb-bits",
        type=int,
        default=1,
        metavar="N",
        help="Number of least-significant bits to extract from each "
        "standardized value (default: 1).",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Write output bytes to FILE. If omitted, output is printed "
        "to stdout as a hex string.",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks on the raw/standardized data before "
        "generating output (e.g. waveform plot, low-bits uniformity test).",
    )

    # Audio-specific options
    audio_group = parser.add_argument_group("audio source options")
    audio_group.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Number of samples per chunk read from the audio stream "
        "(default: 1024).",
    )
    audio_group.add_argument(
        "--num-chunks",
        type=int,
        default=10,
        help="Number of chunks to read from the audio stream (default: 10).",
    )

    return parser


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    bit_extractor = LSBExtractor(n=args.lsb_bits)
    von_neumann = VonNeumannExtractor()
    validator = build_validator(args) if args.validate else None

    if args.source == "audio":
        source = MicrophoneSource(
            chunk_size=args.chunk_size,
        )
        return Pipeline(source, bit_extractor, von_neumann, validator=validator)

    raise ValueError(f"Unknown source: {args.source}")


def build_validator(args: argparse.Namespace):
    if args.source == "audio":
        return AudioValidator(SAMPLE_RATE)

    raise ValueError(f"Unknown source: {args.source}")

def run_source(args: argparse.Namespace, pipeline: Pipeline) -> bytearray:
    if args.source == "audio":
        return pipeline.run(num_chunks=args.num_chunks)

    raise ValueError(f"Unknown source: {args.source}")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    pipeline = build_pipeline(args)

    output_bytes = run_source(args, pipeline)

    if not output_bytes:
        print(
            "Warning: no output bytes were produced (Von Neumann extraction "
            "may have discarded all bits). Try reading more data.",
            file=sys.stderr,
        )

    if args.output:
        with open(args.output, "wb") as f:
            f.write(output_bytes)
        print(f"Wrote {len(output_bytes)} bytes to {args.output}", file=sys.stderr)
    else:
        print(output_bytes.hex())

    return 0


if __name__ == "__main__":
    sys.exit(main())
