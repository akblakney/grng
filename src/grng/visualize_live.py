"""Live visualization of grng random data.

Generates a fixed number of bytes from a live entropy source and
displays the same visualizations as grng-collect-visualize.

Usage:
    grng-visualize <source> [options]
"""
import argparse
import sys

import numpy as np

from .extract.bits import LSBExtractor
from .extract.von_neumann import VonNeumannExtractor
from .pipeline import Pipeline
from .sources.microphone import MicrophoneSource
from .constants.audio import SAMPLE_RATE
from .visualize.random_walk import plot_1d, plot_2d
from .visualize.scatter import plot_scatter
from .visualize.monte_carlo import plot_monte_carlo
from .visualize.hilbert import plot_hilbert


# Hardcoded pipeline defaults
_LSB_BITS = 1
_INTERVAL = 1

SOURCES = {
    "audio": MicrophoneSource,
}


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(args: argparse.Namespace) -> Pipeline:
    bit_extractor = LSBExtractor(lsb_bits=_LSB_BITS)
    von_neumann = VonNeumannExtractor()
    if args.source == "audio":
        source = MicrophoneSource()
        return Pipeline(source, bit_extractor, von_neumann)
    raise ValueError(f"Unknown source: {args.source}")


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_bytes(pipeline: Pipeline, n_bytes: int) -> np.ndarray:
    """Generate n_bytes from the pipeline and return as a uint8 array."""
    import io
    buf = io.BytesIO()
    print(f"Generating {n_bytes:,} bytes from live source...", file=sys.stderr)
    pipeline.run_to_file(
        path=None,
        n_bytes=n_bytes,
        verbose=True,
        fmt="binary",
    )
    # run_to_file writes to stdout when path=None — instead capture to buffer
    # by temporarily redirecting stdout.buffer
    return _generate_to_buffer(pipeline, n_bytes)


def _generate_to_buffer(pipeline: Pipeline, n_bytes: int) -> np.ndarray:
    """Run the pipeline and collect output into a numpy uint8 array."""
    import contextlib
    import io

    chunks = []
    written = 0
    while written < n_bytes:
        batch = pipeline.run()
        if not batch:
            continue
        remaining = n_bytes - written
        chunk = batch[:remaining]
        chunks.append(np.frombuffer(bytes(chunk), dtype=np.uint8))
        written += len(chunk)
        print(
            f"\r{written:,} / {n_bytes:,} bytes generated",
            end="",
            file=sys.stderr,
        )
    print(file=sys.stderr)
    return np.concatenate(chunks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grng-visualize",
        description="Generate random bytes from a live source and visualize them.",
    )
    parser.add_argument(
        "source",
        choices=sorted(SOURCES.keys()),
        help="Entropy source to use.",
    )
    parser.add_argument(
        "--bytes",
        type=int,
        default=100_000,
        metavar="N",
        dest="n_bytes",
        help="Number of bytes to generate before plotting (default: 100000).",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=["walk1d", "walk2d", "scatter", "montecarlo", "hilbert"],
        default=["walk1d", "walk2d", "scatter", "montecarlo", "hilbert"],
        metavar="PLOT",
        help=(
            "Which plots to show. "
            "Choices: walk1d walk2d scatter montecarlo hilbert. "
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

    # Validate that enough bytes are requested for the hilbert plot
    hilbert_bytes = 4 ** args.hilbert_order
    if "hilbert" in args.plots and args.n_bytes < hilbert_bytes:
        print(
            f"Warning: --bytes {args.n_bytes} is less than the {hilbert_bytes:,} "
            f"bytes needed for hilbert order {args.hilbert_order}. "
            f"Hilbert plot will be skipped.",
            file=sys.stderr,
        )
        args.plots = [p for p in args.plots if p != "hilbert"]

    pipeline = build_pipeline(args)
    data = _generate_to_buffer(pipeline, args.n_bytes)

    plots = set(args.plots)

    if "walk1d" in plots:
        plot_1d(data)
    if "walk2d" in plots:
        plot_2d(data)
    if "scatter" in plots:
        plot_scatter(data)
    if "montecarlo" in plots:
        plot_monte_carlo(data)
    if "hilbert" in plots:
        plot_hilbert(data, order=args.hilbert_order)

    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())