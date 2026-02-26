"""AlphaBurstDetector — detect sustained alpha power increases."""

from __future__ import annotations

from collections import deque

import numpy as np

from ..eeg.bands import ALPHA, compute_band_powers
from ..eeg.stream import EEGStream
from .base import Event, EventDetector, EventType


class AlphaBurstDetector(EventDetector):
    """Detects alpha bursts — sustained periods where alpha power exceeds a rolling baseline.

    Alpha (8-13 Hz) power increases when the subject relaxes or closes their eyes.
    This detector tracks a rolling baseline and fires on/off events when
    alpha power crosses a threshold relative to that baseline.
    """

    def __init__(
        self,
        ratio_threshold: float = 1.5,
        baseline_seconds: float = 10.0,
        analysis_window: float = 1.0,
        update_interval: float = 0.5,
    ):
        self.ratio_threshold = ratio_threshold
        self.baseline_seconds = baseline_seconds
        self.analysis_window = analysis_window
        self.update_interval = update_interval

        self._baseline_values: deque[float] = deque(
            maxlen=int(baseline_seconds / update_interval)
        )
        self._in_burst = False
        self._last_update: float = 0.0

    def detect(self, stream: EEGStream, now: float) -> list[Event]:
        if now - self._last_update < self.update_interval:
            return []
        self._last_update = now

        # Use AF7+AF8 average for alpha detection (frontal alpha is most reliable)
        af7 = stream.get_window("AF7", self.analysis_window)
        af8 = stream.get_window("AF8", self.analysis_window)

        if len(af7) < 128 or len(af8) < 128:
            return []

        # Average frontal alpha power
        p7 = compute_band_powers(af7, bands=[ALPHA])
        p8 = compute_band_powers(af8, bands=[ALPHA])
        alpha_power = (p7["Alpha"] + p8["Alpha"]) / 2

        self._baseline_values.append(alpha_power)

        if len(self._baseline_values) < 4:
            return []

        baseline = float(np.median(list(self._baseline_values)))
        if baseline <= 0:
            return []

        ratio = alpha_power / baseline
        events: list[Event] = []

        if not self._in_burst and ratio > self.ratio_threshold:
            self._in_burst = True
            events.append(
                Event(
                    EventType.ALPHA_BURST_START,
                    timestamp=now,
                    value=ratio,
                    metadata={"alpha_power": alpha_power, "baseline": baseline},
                )
            )
        elif self._in_burst and ratio < 1.0:
            self._in_burst = False
            events.append(
                Event(
                    EventType.ALPHA_BURST_END,
                    timestamp=now,
                    value=ratio,
                    metadata={"alpha_power": alpha_power, "baseline": baseline},
                )
            )

        return events
