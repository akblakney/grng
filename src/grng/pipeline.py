"""Pipeline that composes an entropy source through to final random bytes."""
import sys
from .extract.bits import BitExtractor
from .extract.von_neumann import VonNeumannExtractor
from .sources.base import EntropySource
from .validate.base import Validator


class Pipeline:
    """Composes a full entropy-to-random-bytes pipeline.

    The pipeline runs the following stages in order:
        1. EntropySource.read_raw()      -> raw source data
        2. Standardizer.standardize()    -> List[int]
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

    def run(self, *source_args, **source_kwargs) -> bytearray:
        """Run one pass through the pipeline and return the resulting bytes.

        Any positional/keyword arguments are forwarded to
        `source.read_raw()` (e.g. number of chunks to read).
        """
        with self.source:
            raw = self.source.read_raw(*source_args, **source_kwargs)

        values = self.source.standardize(raw)

        if self.validator is not None:
            results = self.validator.run_all(raw, values)
            self.validator.print_results(results)

        bits = self.bit_extractor.extract(values)
        return self.von_neumann_extractor.extract(bits)

    def run_to_file(
        self,
        path: str,
        n_bytes: int,
        verbose: bool = False,
        *source_args,
        **source_kwargs,
    ) -> None:
        """Loop the pipeline until exactly n_bytes have been written to path.

        Each iteration performs one source read, extracts bytes, and flushes
        them to the file — keeping memory usage bounded to a single batch at
        a time regardless of how large n_bytes is.
        """
        written = 0
        with open(path, "wb") as f:
            while written < n_bytes:
                batch = self.run(*source_args, **source_kwargs)
                if not batch:
                    continue
                remaining = n_bytes - written
                chunk = batch[:remaining]  # truncate final batch if needed
                f.write(chunk)
                written += len(chunk)
                if verbose:
                    print(f"{written} / {n_bytes} bytes written", file=sys.stderr)
        if verbose:
            print(f"Done. {written} bytes written to {path}", file=sys.stderr)