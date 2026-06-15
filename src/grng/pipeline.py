"""Pipeline that composes an entropy source through to final random bytes."""

from .extract.bits import BitExtractor
from .extract.von_neumann import VonNeumannExtractor
from .sources.base import EntropySource
from .standardize.base import Standardizer


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
        standardizer: Standardizer,
        bit_extractor: BitExtractor,
        von_neumann_extractor: VonNeumannExtractor,
    ):
        self.source = source
        self.standardizer = standardizer
        self.bit_extractor = bit_extractor
        self.von_neumann_extractor = von_neumann_extractor

    def run(self, *source_args, **source_kwargs) -> bytearray:
        """Run the full pipeline and return the final random bytes.

        Any positional/keyword arguments are forwarded to
        `source.read_raw()` (e.g. number of chunks to read).
        """
        with self.source:
            raw = self.source.read_raw(*source_args, **source_kwargs)

        values = self.standardizer.standardize(raw)
        bits = self.bit_extractor.extract(values)
        return self.von_neumann_extractor.extract(bits)
