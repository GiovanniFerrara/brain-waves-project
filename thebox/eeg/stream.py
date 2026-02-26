"""EEGStream — fixed-size ring buffer per EEG channel."""

from __future__ import annotations

import numpy as np

from ..ble.protocol import CHANNEL_NAMES, SAMPLE_RATE


class EEGStream:
    """Ring buffer that stores the last ``duration`` seconds per channel.

    Usage::

        stream = EEGStream(duration=10.0)
        stream.append("AF7", [1.0, 2.0, 3.0])
        window = stream.get_window("AF7", seconds=2.0)
    """

    def __init__(self, duration: float = 10.0):
        self.duration = duration
        self.capacity = int(duration * SAMPLE_RATE)
        self._buffers: dict[str, np.ndarray] = {
            name: np.zeros(self.capacity, dtype=np.float64)
            for name in CHANNEL_NAMES
        }
        self._write_pos: dict[str, int] = {name: 0 for name in CHANNEL_NAMES}
        self._count: dict[str, int] = {name: 0 for name in CHANNEL_NAMES}

    def append(self, channel: str, samples: list[float] | np.ndarray) -> None:
        """Append samples to a channel's ring buffer."""
        buf = self._buffers[channel]
        pos = self._write_pos[channel]
        n = len(samples)

        if n >= self.capacity:
            # More samples than buffer — keep only the last capacity
            buf[:] = np.asarray(samples[-self.capacity:], dtype=np.float64)
            self._write_pos[channel] = 0
            self._count[channel] = self.capacity
            return

        end = pos + n
        if end <= self.capacity:
            buf[pos:end] = samples
        else:
            first = self.capacity - pos
            buf[pos:] = samples[:first]
            buf[:n - first] = samples[first:]

        self._write_pos[channel] = end % self.capacity
        self._count[channel] = min(self._count[channel] + n, self.capacity)

    def get_window(self, channel: str, seconds: float | None = None) -> np.ndarray:
        """Return the last ``seconds`` of data (or all available data).

        Returns a contiguous copy in chronological order.
        """
        count = self._count[channel]
        if count == 0:
            return np.array([], dtype=np.float64)

        if seconds is not None:
            n = min(int(seconds * SAMPLE_RATE), count)
        else:
            n = count

        buf = self._buffers[channel]
        pos = self._write_pos[channel]

        # Samples are at positions [pos-n, pos) in the ring buffer (mod capacity)
        start = (pos - n) % self.capacity
        if start < pos:
            return buf[start:pos].copy()
        else:
            return np.concatenate([buf[start:], buf[:pos]])

    def sample_count(self, channel: str) -> int:
        """Number of samples received on a channel (capped at capacity)."""
        return self._count[channel]

    def total_samples(self) -> int:
        """Total samples across all channels."""
        return sum(self._count.values())
