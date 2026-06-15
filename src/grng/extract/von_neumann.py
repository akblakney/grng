"""Von Neumann randomness extractor for debiasing a bitstream."""

from bitarray import bitarray


class VonNeumannExtractor:
    """Applies the Von Neumann debiasing algorithm to a bitstream.

    The input bits are grouped into non-overlapping pairs. Each pair
    "01" produces an output bit 0, each pair "10" produces an output
    bit 1, and pairs "00" or "11" are discarded. Any leftover single
    bit at the end (if the input has odd length) is discarded.

    The output bits are packed into a `bytearray`. If the number of
    output bits is not a multiple of 8, the final partial byte is
    either dropped or zero-padded depending on `pad_final_byte`.
    """

    def __init__(self, pad_final_byte: bool = False):
        self.pad_final_byte = pad_final_byte

    def extract(self, bits: bitarray) -> bytearray:
        output_bits = bitarray()

        # Process bits in pairs; ignore a trailing odd bit if present.
        num_pairs = len(bits) // 2
        for i in range(num_pairs):
            first = bits[2 * i]
            second = bits[2 * i + 1]

            if first == second:
                continue  # discard 00 or 11
            elif first == 0 and second == 1:
                output_bits.append(0)
            else:  # first == 1 and second == 0
                output_bits.append(1)

        return self._pack_to_bytearray(output_bits)

    def _pack_to_bytearray(self, bits: bitarray) -> bytearray:
        remainder = len(bits) % 8

        if remainder == 0:
            return bytearray(bits.tobytes())

        if self.pad_final_byte:
            padded = bits.copy()
            padded.extend([0] * (8 - remainder))
            return bytearray(padded.tobytes())
        else:
            # Drop the trailing partial byte's worth of bits.
            trimmed = bits[: len(bits) - remainder]
            return bytearray(trimmed.tobytes())
