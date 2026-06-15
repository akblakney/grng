"""Entropy source backed by a microphone via PyAudio."""

import pyaudio

from .base import EntropySource


class MicrophoneSource(EntropySource):
    """Reads raw 16-bit PCM audio samples from a microphone.

    `read_raw` returns a `bytes` object containing the raw byte stream
    as produced by PyAudio, where each sample is a 16-bit signed
    little-endian integer (paInt16).
    """

    def __init__(
        self,
        rate: int = 44100,
        channels: int = 1,
        chunk_size: int = 1024,
        input_device_index: int | None = None,
    ) -> None:
        self.rate = rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.input_device_index = input_device_index
        self.format = pyaudio.paInt16

        self._pa: pyaudio.PyAudio | None = None
        self._stream = None

    def open(self) -> None:
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.chunk_size,
        )

    def read_raw(self, num_chunks: int = 1) -> bytes:
        """Read `num_chunks` chunks of audio and return the raw bytes.

        Each chunk contains `chunk_size` samples; each sample is 2 bytes
        (16-bit signed integer).
        """
        if self._stream is None:
            raise RuntimeError("MicrophoneSource is not open. Call open() first.")

        data = bytearray()
        for _ in range(num_chunks):
            data.extend(self._stream.read(self.chunk_size, exception_on_overflow=False))
        return bytes(data)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
