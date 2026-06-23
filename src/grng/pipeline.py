"""Pipeline that composes an entropy source through to final random bytes."""
import sys
import time
import contextlib
import random
from datetime import datetime, timezone
from .extract.bits import BitExtractor
from .extract.von_neumann import VonNeumannExtractor
from .sources.base import EntropySource
from .validate.base import Validator


class Pipeline:
    """Composes a full entropy-to-random-bytes pipeline.

    The pipeline runs the following stages in order:
        1. EntropySource.read_raw()      -> raw source data
        2. .standardize()                -> List[int]
        3. BitExtractor.extract()        -> bitarray
        4. VonNeumannExtractor.extract() -> bytearray
    """

    def __init__(
        self,
        source: EntropySource,
        bit_extractor: BitExtractor,
        von_neumann_extractor: VonNeumannExtractor,
        validator: Validator = None,
    ):
        self.source = source
        self.bit_extractor = bit_extractor
        self.von_neumann_extractor = von_neumann_extractor
        self.validator = validator

    def run_validation(self) -> bool:
        if self.validator is None:
            return False
        return random.random() < self.validator.thresh

    def run(self, *source_args, **source_kwargs) -> bytearray:
        """Run one pass through the pipeline and return the resulting bytes.

        Any positional/keyword arguments are forwarded to
        `source.read_raw()` (e.g. number of chunks to read).
        """
        with self.source:
            raw = self.source.read_raw(*source_args, **source_kwargs)
        values = self.source.standardize(raw)
        if self.run_validation():
            results = self.validator.run_all(raw, values)
            self.validator.accumulate(raw, values)
            #self.validator.print_results(results)
        bits = self.bit_extractor.extract(values)
        return self.von_neumann_extractor.extract(bits)

    def run_to_file(
        self,
        path: str | None,
        n_bytes: int | None = None,
        deadline: float | None = None,
        verbose: bool = False,
        fmt: str = "binary",
        file_mode: str = "wb",
        *source_args,
        **source_kwargs,
    ) -> int:
        """Loop the pipeline, writing to path, until a stop condition is met.

        Exactly one of n_bytes or deadline must be provided:
            n_bytes:  stop after exactly this many bytes are written.
            deadline: stop after this monotonic time (from time.monotonic())
                      is reached. Use make_deadline() to construct one.

        Returns the number of bytes written.
        """
        assert (n_bytes is None) != (deadline is None), \
            "Exactly one of n_bytes or deadline must be provided"

        written = 0
        start_time = time.time()
        with self._open_output(path, fmt, file_mode) as f:
            while self._should_continue(n_bytes, deadline, written):
                batch = self.run(*source_args, **source_kwargs)
                if not batch:
                    continue
                chunk = batch
                if n_bytes is not None:
                    remaining = n_bytes - written
                    chunk = batch[:remaining]
                f.write(self._encode(chunk, fmt))
                written += len(chunk)
                elapsed = time.time() - start_time
                if verbose:
                    if n_bytes is not None:
                        msg = f"{written} / {n_bytes} bytes written in {elapsed:.2f}s"
                    else:
                        msg = f"{written} bytes written in {elapsed:.2f}s"
                    print(f"\r{msg:<60}", end="", file=sys.stderr)

        if self.validator is not None:
            self.validator.finalize()
        if verbose:
            elapsed = time.time() - start_time
            print(file=sys.stderr)
            print(f"Complete. {written} bytes written to {path} in {elapsed:.2f}s", file=sys.stderr)
            vn = self.von_neumann_extractor
            print(
                f"Processed {vn.pairs_processed} total pairs. "
                f"Kept {vn.pairs_output}, discarded {vn.pairs_discarded}.",
                file=sys.stderr,
            )
        return written

    @staticmethod
    def make_deadline(seconds: float) -> float:
        """Return a monotonic deadline that expires in `seconds` from now."""
        return time.monotonic() + seconds

    @staticmethod
    def seconds_until_next_utc_hour() -> float:
        """Return the number of seconds until the next UTC hour boundary."""
        now = datetime.now(timezone.utc)
        seconds_into_hour = now.minute * 60 + now.second + now.microsecond / 1e6
        return 3600.0 - seconds_into_hour

    @staticmethod
    def _should_continue(
        n_bytes: int | None,
        deadline: float | None,
        written: int,
    ) -> bool:
        if n_bytes is not None:
            return written < n_bytes
        return time.monotonic() < deadline

    @contextlib.contextmanager
    def _open_output(self, path: str | None, fmt: str = "binary", file_mode: str = "wb"):
        if fmt == "hex":
            file_mode = "w"
        if path is None:
            yield sys.stdout.buffer if fmt == "binary" else sys.stdout
        else:
            f = open(path, file_mode)
            try:
                yield f
            finally:
                f.close()

    @staticmethod
    def _encode(chunk: bytearray, fmt: str) -> bytes:
        if fmt == "binary":
            return bytes(chunk)
        elif fmt == "hex":
            return chunk.hex()
        raise ValueError(f"Unknown format: {fmt}")