"""BlinkDetector — detect eye blinks from AF7+AF8 spike amplitude."""

from __future__ import annotations

import numpy as np

from ..eeg.stream import EEGStream
from .base import Event, EventDetector, EventType


class BlinkDetector(EventDetector):
    """Detects blinks as large peak-to-peak spikes on frontal channels.

    AF7 and AF8 (forehead electrodes) show a characteristic high-amplitude
    deflection during eye blinks. We detect when peak-to-peak amplitude
    in a short window exceeds a threshold.
    """

    def __init__(
        self,
        threshold: float = 200.0,
        window: float = 0.2,
        debounce: float = 0.3,
    ):
        self.threshold = threshold
        self.window = window
        self.debounce = debounce
        self._last_blink: float = 0.0

    def detect(self, stream: EEGStream, now: float) -> list[Event]:
        if now - self._last_blink < self.debounce:
            return []

        af7 = stream.get_window("AF7", self.window)
        af8 = stream.get_window("AF8", self.window)

        if len(af7) < 10 or len(af8) < 10:
            return []

        pp7 = float(np.ptp(af7))
        pp8 = float(np.ptp(af8))
        peak = max(pp7, pp8)

        if peak > self.threshold:
            self._last_blink = now
            return [Event(EventType.BLINK, timestamp=now, value=peak)]

        return []
