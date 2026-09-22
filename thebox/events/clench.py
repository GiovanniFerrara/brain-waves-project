"""ClenchDetector — detect jaw clenches from TP9+TP10 high-frequency bursts."""

from __future__ import annotations

import numpy as np

from ..eeg.filters import bandpass
from ..eeg.stream import EEGStream
from .base import Event, EventDetector, EventType


class ClenchDetector(EventDetector):
    """Detects jaw clenches via high-frequency (20-45 Hz) RMS on temporal channels.

    TP9 and TP10 sit behind the ears, close to the jaw muscles. Clenching
    produces a burst of high-frequency (EMG) activity.
    """

    def __init__(
        self,
        threshold: float = 30.0,
        window: float = 0.5,
        debounce: float = 0.5,
    ):
        self.threshold = threshold
        self.window = window
        self.debounce = debounce
        self._last_clench: float = 0.0

    def detect(self, stream: EEGStream, now: float) -> list[Event]:
        if now - self._last_clench < self.debounce:
            return []

        tp9 = stream.get_window("TP9", self.window)
        tp10 = stream.get_window("TP10", self.window)

        if len(tp9) < 64 or len(tp10) < 64:
            return []

        # Filter each window from scratch to 20-45 Hz (EMG, below 50 Hz mains).
        # Windows overlap between calls, so a stateful filter would re-see samples.
        filt9 = bandpass(tp9, 20.0, 45.0)
        filt10 = bandpass(tp10, 20.0, 45.0)

        rms9 = float(np.sqrt(np.mean(filt9 ** 2)))
        rms10 = float(np.sqrt(np.mean(filt10 ** 2)))
        rms = max(rms9, rms10)

        if rms > self.threshold:
            self._last_clench = now
            return [Event(EventType.CLENCH, timestamp=now, value=rms)]

        return []
