"""EEG filters — offline bandpass and real-time streaming filter."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, sosfilt, sosfilt_zi

from ..ble.protocol import SAMPLE_RATE


def bandpass(
    data: np.ndarray,
    low: float,
    high: float,
    order: int = 4,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter (offline use).

    Uses filtfilt for zero phase distortion — requires the full signal.
    """
    nyq = sample_rate / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)


class StreamingBandpass:
    """Causal IIR bandpass filter that processes chunks incrementally.

    Uses second-order sections (sos) with sosfilt for numerical stability.
    Maintains filter state across calls so it can process real-time chunks.

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
        sample_rate: int = SAMPLE_RATE,
    ):
        nyq = sample_rate / 2
        self.sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
        self._zi = sosfilt_zi(self.sos)
        # Scale initial conditions to zero (no assumed DC offset)
        self._zi = self._zi * 0.0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Filter a chunk of samples, maintaining state across calls."""
        filtered, self._zi = sosfilt(self.sos, chunk, zi=self._zi)
        return filtered

    def reset(self) -> None:
        """Reset filter state."""
        self._zi = sosfilt_zi(self.sos) * 0.0
