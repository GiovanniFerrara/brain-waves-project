"""ClenchDetector — detect jaw clenches from TP9+TP10 high-frequency bursts."""

from __future__ import annotations

import numpy as np

from ..eeg.filters import StreamingBandpass
from ..eeg.stream import EEGStream
from .base import Event, EventDetector, EventType


class ClenchDetector(EventDetector):
    """Detects jaw clenches via high-frequency (20-50 Hz) RMS on temporal channels.

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
        self._filt_tp9 = StreamingBandpass(20.0, 50.0)
        self._filt_tp10 = StreamingBandpass(20.0, 50.0)

    def detect(self, stream: EEGStream, now: float) -> list[Event]:
        if now - self._last_clench < self.debounce:
            return []

        tp9 = stream.get_window("TP9", self.window)
        tp10 = stream.get_window("TP10", self.window)

        if len(tp9) < 10 or len(tp10) < 10:
            return []

        # Filter to 20-50 Hz and compute RMS
        filt9 = self._filt_tp9.process(tp9)
        filt10 = self._filt_tp10.process(tp10)

        rms9 = float(np.sqrt(np.mean(filt9 ** 2)))
        rms10 = float(np.sqrt(np.mean(filt10 ** 2)))
        rms = max(rms9, rms10)

        if rms > self.threshold:
            self._last_clench = now
            return [Event(EventType.CLENCH, timestamp=now, value=rms)]

        return []
