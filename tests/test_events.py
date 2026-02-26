"""Unit tests for event detectors with synthetic EEG data."""

import numpy as np
import pytest

from thebox.ble.protocol import SAMPLE_RATE
from thebox.eeg.stream import EEGStream
from thebox.events.alpha import AlphaBurstDetector
from thebox.events.base import EventType
from thebox.events.blink import BlinkDetector
from thebox.events.bus import EventBus
from thebox.events.clench import ClenchDetector


def _fill_stream(stream: EEGStream, duration: float = 2.0, amplitude: float = 10.0):
    """Fill all channels with low-amplitude noise (baseline)."""
    n = int(duration * SAMPLE_RATE)
    rng = np.random.default_rng(42)
    for ch in ["TP9", "AF7", "AF8", "TP10"]:
        stream.append(ch, (rng.standard_normal(n) * amplitude).tolist())


class TestBlinkDetector:
    def test_no_blink_on_calm_signal(self):
        stream = EEGStream(duration=5.0)
        _fill_stream(stream, amplitude=10.0)
        detector = BlinkDetector(threshold=200.0)
        events = detector.detect(stream, now=1.0)
        assert len(events) == 0

    def test_blink_on_large_spike(self):
        stream = EEGStream(duration=5.0)
        _fill_stream(stream, amplitude=10.0)
        # Inject a large spike on AF7
        spike = np.zeros(int(0.2 * SAMPLE_RATE))
        spike[10] = 300.0
        spike[20] = -300.0
        stream.append("AF7", spike.tolist())

        detector = BlinkDetector(threshold=200.0)
        events = detector.detect(stream, now=2.0)
        assert len(events) == 1
        assert events[0].type == EventType.BLINK
        assert events[0].value >= 200.0

    def test_debounce(self):
        stream = EEGStream(duration=5.0)
        _fill_stream(stream, amplitude=10.0)
        spike = np.zeros(int(0.2 * SAMPLE_RATE))
        spike[10] = 300.0
        spike[20] = -300.0
        stream.append("AF7", spike.tolist())

        detector = BlinkDetector(threshold=200.0, debounce=0.3)
        events1 = detector.detect(stream, now=1.0)
        assert len(events1) == 1

        # Too soon — should be debounced
        events2 = detector.detect(stream, now=1.1)
        assert len(events2) == 0

        # After debounce period
        stream.append("AF7", spike.tolist())
        events3 = detector.detect(stream, now=1.4)
        assert len(events3) == 1


class TestClenchDetector:
    def test_no_clench_on_calm_signal(self):
        stream = EEGStream(duration=5.0)
        _fill_stream(stream, amplitude=5.0)
        detector = ClenchDetector(threshold=30.0)
        events = detector.detect(stream, now=1.0)
        assert len(events) == 0

    def test_clench_on_hf_burst(self):
        stream = EEGStream(duration=5.0)
        _fill_stream(stream, amplitude=5.0)
        # Inject high-frequency burst on TP9
        t = np.arange(int(0.5 * SAMPLE_RATE)) / SAMPLE_RATE
        burst = 100.0 * np.sin(2 * np.pi * 35 * t)  # 35 Hz, large amplitude
        stream.append("TP9", burst.tolist())

        detector = ClenchDetector(threshold=30.0)
        events = detector.detect(stream, now=2.0)
        assert len(events) == 1
        assert events[0].type == EventType.CLENCH


class TestAlphaBurstDetector:
    def test_no_burst_on_noise(self):
        stream = EEGStream(duration=15.0)
        _fill_stream(stream, duration=12.0, amplitude=10.0)
        detector = AlphaBurstDetector(ratio_threshold=1.5)
        # Run several cycles to build baseline
        events = []
        for i in range(20):
            events.extend(detector.detect(stream, now=float(i) * 0.5))
        bursts = [e for e in events if e.type == EventType.ALPHA_BURST_START]
        assert len(bursts) == 0

    def test_burst_on_strong_alpha(self):
        stream = EEGStream(duration=15.0)
        # Build baseline with low-amplitude noise
        _fill_stream(stream, duration=10.0, amplitude=10.0)
        detector = AlphaBurstDetector(ratio_threshold=1.5, update_interval=0.1)

        # Build baseline
        for i in range(30):
            detector.detect(stream, now=float(i) * 0.1)

        # Inject strong 10 Hz alpha on AF7+AF8
        t = np.arange(int(2.0 * SAMPLE_RATE)) / SAMPLE_RATE
        alpha = 100.0 * np.sin(2 * np.pi * 10 * t)
        stream.append("AF7", alpha.tolist())
        stream.append("AF8", alpha.tolist())

        events = detector.detect(stream, now=5.0)
        bursts = [e for e in events if e.type == EventType.ALPHA_BURST_START]
        assert len(bursts) == 1


class TestEventBus:
    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventType.BLINK, lambda e: received.append(e))

        from thebox.events.base import Event
        event = Event(EventType.BLINK, timestamp=1.0, value=250.0)
        bus.publish(event)

        assert len(received) == 1
        assert received[0].type == EventType.BLINK

    def test_wildcard_subscriber(self):
        bus = EventBus()
        received = []
        bus.subscribe(None, lambda e: received.append(e))

        from thebox.events.base import Event
        bus.publish(Event(EventType.BLINK, timestamp=1.0))
        bus.publish(Event(EventType.CLENCH, timestamp=2.0))

        assert len(received) == 2

    def test_no_cross_delivery(self):
        bus = EventBus()
        blinks = []
        bus.subscribe(EventType.BLINK, lambda e: blinks.append(e))

        from thebox.events.base import Event
        bus.publish(Event(EventType.CLENCH, timestamp=1.0))

        assert len(blinks) == 0
