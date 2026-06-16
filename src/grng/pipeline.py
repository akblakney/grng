"""Pipeline that composes an entropy source through to final random bytes."""

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
        validator: Validator = None
    ):
        self.source = source
        self.bit_extractor = bit_extractor
        self.von_neumann_extractor = von_neumann_extractor
        self.validator = validator

    def run(self, *source_args, **source_kwargs) -> bytearray:
        """Run the full pipeline and return the final random bytes.

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
