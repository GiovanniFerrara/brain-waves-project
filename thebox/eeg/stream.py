"""EEGStream — fixed-size ring buffer per EEG channel."""

from __future__ import annotations

import numpy as np

from ..ble.protocol import CHANNEL_NAMES, SAMPLE_RATE


class EEGStream:
    """Ring buffer that stores the last ``duration`` seconds per channel.

    Alongside the samples it keeps a validity mask: False marks samples
    interpolated over lost BLE packets.

    Usage::

        stream = EEGStream(duration=10.0)
        stream.append("AF7", samples, valid)
        window = stream.get_window("AF7", seconds=2.0)
        quality = stream.valid_fraction("AF7", seconds=2.0)
    """

    def __init__(self, duration: float = 10.0):
        self.duration = duration
        self.capacity = int(duration * SAMPLE_RATE)
        self._buffers = {
            name: np.zeros(self.capacity, dtype=np.float64) for name in CHANNEL_NAMES
        }
        self._valid = {name: np.zeros(self.capacity, dtype=bool) for name in CHANNEL_NAMES}
        self._write_pos = {name: 0 for name in CHANNEL_NAMES}
        self._count = {name: 0 for name in CHANNEL_NAMES}

    def append(
        self,
        channel: str,
        samples: list[float] | np.ndarray,
        valid: np.ndarray | None = None,
    ) -> None:
        """Append samples (and their validity; default all valid) to a channel."""
        samples = np.asarray(samples, dtype=np.float64)
        if valid is None:
            valid = np.ones(len(samples), dtype=bool)
        self._write(self._buffers[channel], self._write_pos[channel], samples)
        self._write(self._valid[channel], self._write_pos[channel], valid)

        n = len(samples)
        self._write_pos[channel] = (self._write_pos[channel] + n) % self.capacity
        self._count[channel] = min(self._count[channel] + n, self.capacity)

    def _write(self, buf: np.ndarray, pos: int, data: np.ndarray) -> None:
        n = len(data)
        if n >= self.capacity:
            # Keep only the newest `capacity` samples, ending at pos + n
            end = (pos + n) % self.capacity
            tail = data[-self.capacity:]
            buf[end:] = tail[: self.capacity - end]
            buf[:end] = tail[self.capacity - end:]
            return
        end = pos + n
        if end <= self.capacity:
            buf[pos:end] = data
        else:
            first = self.capacity - pos
            buf[pos:] = data[:first]
            buf[: n - first] = data[first:]

    def _read(self, buf: np.ndarray, channel: str, seconds: float | None) -> np.ndarray:
        count = self._count[channel]
        n = count if seconds is None else min(int(seconds * SAMPLE_RATE), count)
        if n == 0:
            return buf[:0].copy()
        pos = self._write_pos[channel]
        start = (pos - n) % self.capacity
        if start < pos:
            return buf[start:pos].copy()
        return np.concatenate([buf[start:], buf[:pos]])

    def get_window(self, channel: str, seconds: float | None = None) -> np.ndarray:
        """Return the last ``seconds`` of data (or all), oldest first."""
        return self._read(self._buffers[channel], channel, seconds)

    def get_valid(self, channel: str, seconds: float | None = None) -> np.ndarray:
        """Validity mask matching ``get_window``."""
        return self._read(self._valid[channel], channel, seconds)

    def valid_fraction(self, channel: str, seconds: float | None = None) -> float:
        """Share of real (not interpolated) samples in the window."""
        mask = self.get_valid(channel, seconds)
        return float(mask.mean()) if len(mask) else 0.0

    def sample_count(self, channel: str) -> int:
        """Number of samples held for a channel (capped at capacity)."""
        return self._count[channel]

    def total_samples(self) -> int:
        """Total samples across all channels."""
        return sum(self._count.values())
