"""Pipeline that composes an entropy source through to final random bytes."""
import sys
import time

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
            self.validator.accumulate(raw, values)
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
        """
        written = 0
        start_time = time.time()
        with open(path, "wb") as f:
            while written < n_bytes:
                batch = self.run(*source_args, **source_kwargs)
                if not batch:
                    continue
                remaining = n_bytes - written
                chunk = batch[:remaining]
                f.write(chunk)
                written += len(chunk)
                elapsed_time = time.time() - start_time
                if verbose:
                    msg = f"{written} / {n_bytes} bytes written in {elapsed_time:.2f} seconds"
                    print(f"\r{msg:<50}", end="", file=sys.stderr)

        if self.validator is not None:
            self.validator.finalize()
        if verbose:
            print(f"Complete. {written} bytes written to {path} in {elapsed_time:.2f} seconds", file=sys.stderr)
            vn = self.von_neumann_extractor
            print('Processed {} total pairs of bits. Kept {} and discarded {}'.format(vn.pairs_processed, vn.pairs_output, vn.pairs_discarded))