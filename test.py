"""Quick manual test: read from microphone and standardize the data."""

import sys
import matplotlib.pyplot as plt
from src.sources import MicrophoneSource
from src.standardize import AudioStandardizer


def main():
    rate = 44100
    chunk_size = 1024
    duration_seconds = 2

    # Number of chunks needed to cover the desired duration
    num_chunks = int((rate * duration_seconds) / chunk_size)

    source = MicrophoneSource(rate=rate, channels=1, chunk_size=chunk_size)
    standardizer = AudioStandardizer()

    with source:
        raw = source.read_raw(num_chunks=num_chunks)

    print(f"Raw bytes read: {len(raw)}")

    values = standardizer.standardize(raw)
    print(f"Standardized values: {len(values)} ints")
    print(f"First 20 values: {values[:20]}")
    print(f"Min: {min(values)}, Max: {max(values)}")

    # Print the raw bytes corresponding to the first 20 values (2 bytes each)
    # for manual sanity-checking of the byte -> int conversion.
    print("\nFirst 20 values with corresponding raw bytes:")
    for i, value in enumerate(values[:20]):
        sample_bytes = raw[i * 2 : i * 2 + 2]
        print(f"  value={value:>7}  bytes={sample_bytes.hex()}")

    # Plot the waveform
    times = [i / rate for i in range(len(values))]
    plt.figure(figsize=(12, 4))
    plt.plot(times, values, linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Sample value")
    plt.title(f"Microphone waveform ({duration_seconds}s, {rate} Hz)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
