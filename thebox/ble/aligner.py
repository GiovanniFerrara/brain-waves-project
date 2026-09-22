"""PacketAligner — rebuild a gap-free 256 Hz timeline from lossy BLE packets."""

from __future__ import annotations

import numpy as np

from .protocol import CHANNEL_NAMES, SAMPLE_RATE, SAMPLES_PER_PACKET, SEQ_MODULO

_EMPTY = np.array([], dtype=np.float64)
_EMPTY_MASK = np.array([], dtype=bool)


class PacketAligner:
    """Place each packet at its true position in time using its sequence number.

    BLE drops notifications. If samples are simply concatenated, a recording
    looks shorter than it was and every frequency is shifted up. Instead,
    missing packets are filled by linear interpolation and flagged invalid,
    so each channel stays on a 256 Hz timeline and sample ``i`` is the same
    instant on all four channels.

    Usage::

        aligner = PacketAligner()
        samples, valid = aligner.push("AF7", seq, decoded_samples)
    """

    def __init__(
        self,
        channels: list[str] = CHANNEL_NAMES,
        max_fill_seconds: float = 10.0,
    ):
        self.channels = list(channels)
        self.max_fill_packets = int(max_fill_seconds * SAMPLE_RATE / SAMPLES_PER_PACKET)
        self.reset()

    def reset(self) -> None:
        """Forget all sequence state (e.g. after a reconnect)."""
        self._origin: int | None = None
        self._next_seq: dict[str, int | None] = {ch: None for ch in self.channels}
        self._last_value: dict[str, float | None] = {ch: None for ch in self.channels}
        self.received = {ch: 0 for ch in self.channels}
        self.lost = {ch: 0 for ch in self.channels}
        self.resyncs = 0

    def loss_fraction(self, channel: str | None = None) -> float:
        """Fraction of packets lost, for one channel or all combined."""
        chans = [channel] if channel else self.channels
        received = sum(self.received[c] for c in chans)
        lost = sum(self.lost[c] for c in chans)
        total = received + lost
        return lost / total if total else 0.0

    def push(
        self, channel: str, seq: int, samples: list[float] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Accept one packet; return the samples to append and their validity mask.

        The returned arrays include interpolated fill for any packets lost
        since the previous one on this channel. Duplicate or out-of-order
        packets return empty arrays.
        """
        samples = np.asarray(samples, dtype=np.float64)
        if self._origin is None:
            # All channels share one counter, so the first packet seen on any
            # channel anchors t=0 for every channel.
            self._origin = seq

        expected = self._next_seq[channel]
        if expected is None:
            expected = self._origin
        missing = (seq - expected) % SEQ_MODULO

        if missing >= SEQ_MODULO // 2:
            return _EMPTY, _EMPTY_MASK  # already placed: duplicate or reordered

        self._next_seq[channel] = (seq + 1) % SEQ_MODULO
        self.received[channel] += 1
        prev = self._last_value[channel]
        self._last_value[channel] = float(samples[-1])

        if missing == 0:
            return samples, np.ones(len(samples), dtype=bool)

        if missing > self.max_fill_packets:
            # Counter jumped too far to be packet loss (headband restarted
            # its counter). Resume from here rather than inventing data.
            self.resyncs += 1
            return samples, np.ones(len(samples), dtype=bool)

        self.lost[channel] += missing
        n_fill = missing * len(samples)
        start = samples[0] if prev is None else prev
        fill = np.linspace(start, samples[0], n_fill + 2)[1:-1]
        out = np.concatenate([fill, samples])
        valid = np.concatenate([np.zeros(n_fill, dtype=bool), np.ones(len(samples), dtype=bool)])
        return out, valid
