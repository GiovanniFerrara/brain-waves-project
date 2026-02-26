"""AudioOutput — sounddevice OutputStream wrapper."""

from __future__ import annotations

from collections import deque

import numpy as np
import sounddevice as sd


class AudioOutput:
    """Wraps a sounddevice OutputStream with a simple write() interface.

    Uses a callback-based stream fed from a deque buffer.
    This is the single interception point for future network streaming.

    Usage::

        out = AudioOutput(sample_rate=44100, block_size=2205)
        out.start()
        out.write(audio_block)  # numpy float64 array
        ...
        out.stop()
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        block_size: int = 2205,
        channels: int = 1,
        buffer_blocks: int = 8,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_blocks)
        self._stream: sd.OutputStream | None = None

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if self._buffer:
            block = self._buffer.popleft()
            n = min(len(block), frames)
            outdata[:n, 0] = block[:n]
            if n < frames:
                outdata[n:, 0] = 0.0
        else:
            outdata[:, 0] = 0.0

    def start(self) -> None:
        """Open and start the audio stream."""
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def write(self, audio: np.ndarray) -> None:
        """Queue an audio block for playback."""
        self._buffer.append(audio.astype(np.float32))

    def stop(self) -> None:
        """Stop and close the audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._buffer.clear()
