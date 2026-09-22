"""EEG filters — offline zero-phase filters and a real-time streaming filter."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, sosfilt, sosfilt_zi, tf2sos

from ..ble.protocol import SAMPLE_RATE

MAINS_FREQUENCY = 50.0  # Hz — Europe; 60 in the Americas


def _band_sos(low: float, high: float, order: int, sample_rate: float) -> np.ndarray:
    return butter(order, [low, high], btype="band", fs=sample_rate, output="sos")


def bandpass(
    data: np.ndarray,
    low: float,
    high: float,
    order: int = 4,
    sample_rate: float = SAMPLE_RATE,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass (offline: needs the whole window).

    Second-order sections keep narrow low bands (e.g. delta at 256 Hz)
    numerically stable, where the b/a form can blow up.
    """
    sos = _band_sos(low, high, order, sample_rate)
    return sosfiltfilt(sos, data, axis=-1)


def notch(
    data: np.ndarray,
    freq: float = MAINS_FREQUENCY,
    quality: float = 30.0,
    sample_rate: float = SAMPLE_RATE,
) -> np.ndarray:
    """Zero-phase notch removing mains hum at ``freq``."""
    b, a = iirnotch(freq, quality, fs=sample_rate)
    return sosfiltfilt(tf2sos(b, a), data, axis=-1)


def clean(
    data: np.ndarray,
    low: float = 1.0,
    high: float = 45.0,
    mains: float | None = MAINS_FREQUENCY,
    sample_rate: float = SAMPLE_RATE,
) -> np.ndarray:
    """Standard EEG cleanup: remove offset, mains hum, and out-of-band content."""
    out = np.asarray(data, dtype=np.float64)
    out = out - np.mean(out, axis=-1, keepdims=True)
    if mains is not None and mains < sample_rate / 2:
        out = notch(out, mains, sample_rate=sample_rate)
    return bandpass(out, low, high, sample_rate=sample_rate)


class StreamingBandpass:
    """Causal IIR bandpass that processes consecutive chunks incrementally.

    Feed it each *new* chunk exactly once — state carries across calls, so
    re-feeding overlapping windows corrupts the output.

    Usage::

        filt = StreamingBandpass(8.0, 13.0)  # alpha band
        for chunk in chunks:
            filtered = filt.process(chunk)
    """

    def __init__(
        self,
        low: float,
        high: float,
        order: int = 4,
        sample_rate: float = SAMPLE_RATE,
    ):
        self.sos = _band_sos(low, high, order, sample_rate)
        self.reset()

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Filter a chunk of samples, maintaining state across calls."""
        filtered, self._zi = sosfilt(self.sos, chunk, zi=self._zi)
        return filtered

    def reset(self) -> None:
        """Reset filter state (zero initial conditions)."""
        self._zi = np.zeros_like(sosfilt_zi(self.sos))
