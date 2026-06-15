"""Quick manual test for BitExtractor (LSB) and VonNeumannExtractor."""

from bitarray import bitarray
from src.extract import LSBExtractor, VonNeumannExtractor


def test_lsb_extractor():
    print("=== LSBExtractor ===")

    # Values chosen so low bits are easy to verify by hand.
    # 0b...0101 = 5  -> low 1 bit = 1, low 4 bits = 0101
    # 0b...0010 = 2  -> low 1 bit = 0, low 4 bits = 0010
    # -1 in two's complement = ...1111 -> low bits all 1
    values = [5, 2, -1, 8]  # 8 = 0b1000

    extractor_1 = LSBExtractor(n=1)
    bits_1 = extractor_1.extract(values)
    print(f"values={values}")
    print(f"n=1 -> bits: {bits_1.to01()} (expected 1 0 1 0)")

    extractor_4 = LSBExtractor(n=4)
    bits_4 = extractor_4.extract(values)
    print(f"n=4 -> bits: {bits_4.to01()} (expected 0101 0010 1111 1000)")

    print()


def test_von_neumann():
    print("=== VonNeumannExtractor ===")

    # Construct a known bitstream:
    # pairs: 01 -> 0, 10 -> 1, 00 -> discard, 11 -> discard, 01 -> 0
    bits = bitarray("01" "10" "00" "11" "01" "10" "00" "10" "01" "01" "01" "10" "11" "01" "01")
    print(f"input bits: {bits.to01()}")

    vn = VonNeumannExtractor()
    output = vn.extract(bits)
    output2 = list(output)
    print('mine {}'.format(output2))
    print(f"output bytearray: {output!r}")
    print(f"output length: {len(output)} bytes")

    # Expected: output bits are 0, 1, 0 -> only 3 bits, not enough for a byte
    # with pad_final_byte=False, this should be dropped -> empty bytearray
    print(f"expected output bits would be '010' (3 bits) -> dropped, empty bytearray")

    print()

    # Test with padding enabled
    vn_pad = VonNeumannExtractor(pad_final_byte=True)
    output_pad = vn_pad.extract(bits)
    print(f"with pad_final_byte=True: output={output_pad!r}, len={len(output_pad)}")
    print(f"expected: '010' padded to '01000000' = 0x40")

    print()


def test_full_pipeline_synthetic():
    print("=== Full pipeline on larger synthetic data ===")

    # Generate enough values to produce a meaningful number of output bits.
    import random
    random.seed(42)
    values = [random.randint(-32768, 32767) for _ in range(1000)]

    lsb = LSBExtractor(n=1)
    bits = lsb.extract(values)
    print(f"extracted {len(bits)} bits from {len(values)} values")
    print(f"bit balance: {bits.count(1)} ones, {bits.count(0)} zeros")

    vn = VonNeumannExtractor()
    output = vn.extract(bits)
    print(f"Von Neumann output: {len(output)} bytes ({len(output) * 8} bits)")
    print(f"output hex: {output.hex()}")


if __name__ == "__main__":
    test_lsb_extractor()
    test_von_neumann()
    test_full_pipeline_synthetic()
