"""Stream, recording, filters, quality and report on synthetic EEG."""

import numpy as np
import pytest

from thebox.ble.protocol import CHANNEL_NAMES, SAMPLE_RATE
from thebox.eeg.bands import ALL_BANDS, epoch_band_powers
from thebox.eeg.filters import bandpass, clean, notch
from thebox.eeg.quality import ARTIFACT, FLAT, LOST, assess
from thebox.eeg.recording import Recording
from thebox.eeg.stream import EEGStream
from thebox.report import SessionReport

FS = SAMPLE_RATE


def _sine(freq: float, seconds: float, amp: float = 10.0) -> np.ndarray:
    t = np.arange(int(seconds * FS)) / FS
    return amp * np.sin(2 * np.pi * freq * t)


def _power_at(x: np.ndarray, freq: float) -> float:
    spec = np.abs(np.fft.rfft(x)) / len(x)
    freqs = np.fft.rfftfreq(len(x), 1 / FS)
    return float(spec[np.argmin(np.abs(freqs - freq))])


class TestEEGStream:
    def test_valid_mask_follows_samples(self):
        s = EEGStream(duration=1.0)
        s.append("AF7", np.ones(100), np.r_[np.ones(50, bool), np.zeros(50, bool)])
        assert s.valid_fraction("AF7") == 0.5
        assert len(s.get_valid("AF7")) == len(s.get_window("AF7")) == 100

    def test_wraparound_keeps_newest(self):
        s = EEGStream(duration=1.0)  # 256 samples
        s.append("AF7", np.arange(200.0))
        s.append("AF7", np.arange(200.0, 300.0))
        np.testing.assert_array_equal(s.get_window("AF7"), np.arange(44.0, 300.0))

    def test_oversized_append_keeps_newest(self):
        s = EEGStream(duration=1.0)
        s.append("AF7", np.arange(10.0))
        s.append("AF7", np.arange(1000.0))
        np.testing.assert_array_equal(s.get_window("AF7"), np.arange(744.0, 1000.0))
        s.append("AF7", [1000.0])
        assert s.get_window("AF7", seconds=2 / FS).tolist() == [999.0, 1000.0]


class TestRecording:
    def test_save_load_roundtrip(self, tmp_path):
        rec = Recording(meta={"device": "Muse-TEST"})
        for ch in CHANNEL_NAMES:
            rec.append(ch, np.arange(24.0), np.r_[np.zeros(12, bool), np.ones(12, bool)])
        rec.append("AF7", np.arange(12.0), np.ones(12, bool))  # one channel ahead
        path = rec.save(tmp_path / "r.npz")

        back = Recording.load(path)
        assert back.channels == CHANNEL_NAMES
        assert back.data.shape == (4, 24)  # trimmed to the shortest channel
        assert back.valid[:, :12].sum() == 0
        assert back.meta == {"device": "Muse-TEST"}


class TestFilters:
    def test_notch_removes_mains(self):
        x = _sine(10, 8) + _sine(50, 8, amp=50)
        y = notch(x)[2 * FS:-2 * FS]  # skip edge transients
        mid = x[2 * FS:-2 * FS]
        assert _power_at(y, 50) < 0.01 * _power_at(mid, 50)
        assert _power_at(y, 10) > 0.95 * _power_at(mid, 10)

    def test_delta_bandpass_is_stable(self):
        """b/a-form filtfilt goes unstable here; SOS must not."""
        x = _sine(2, 10) + _sine(20, 10)
        y = bandpass(x, 0.5, 4.0)
        assert np.all(np.isfinite(y))
        assert _power_at(y, 2) > 0.9 * _power_at(x, 2)
        assert _power_at(y, 20) < 0.01 * _power_at(x, 20)

    def test_clean_removes_offset_and_drift(self):
        x = 800 + np.linspace(0, 200, 4 * FS) + _sine(10, 4)
        y = clean(x)
        assert abs(y.mean()) < 1
        assert np.ptp(y[FS:-FS]) < 25  # just the 10 Hz sine left (±10 µV)

    def test_clean_works_on_all_channels_at_once(self):
        x = np.stack([_sine(10, 4), _sine(20, 4)])
        assert clean(x).shape == x.shape


class TestBands:
    def test_alpha_sine_lands_in_alpha(self):
        epochs = _sine(10, 4).reshape(2, -1)
        bp = epoch_band_powers(epochs)
        names = [b.name for b in ALL_BANDS]
        assert np.all(bp.argmax(axis=-1) == names.index("Alpha"))


class TestQuality:
    def test_flags(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 8 * FS)) * 10
        valid = np.ones_like(x, dtype=bool)
        x[0, 2 * FS + 100] = 400          # epoch 1: blink-sized spike
        x[0, 4 * FS:6 * FS] = 0            # epoch 2: flat
        valid[0, 6 * FS:6 * FS + FS] = False  # epoch 3: 50% lost
        q = assess(x, valid)
        assert q.flags[0].tolist() == [0, ARTIFACT, FLAT, LOST]
        assert q.good_fraction()[0] == 0.25


class TestReport:
    def test_finds_alpha_peak_and_plots(self, tmp_path):
        rng = np.random.default_rng(1)
        rec = Recording()
        n = 20 * FS
        for ch in CHANNEL_NAMES:
            x = rng.standard_normal(n) * 5 + _sine(10, 20, amp=15)
            rec.append(ch, x, np.ones(n, bool))
        report = SessionReport(rec)
        assert "alpha peak 10.0 Hz" in report.summary()
        paths = report.save_plots(tmp_path / "r")
        assert all(p.stat().st_size > 10_000 for p in paths)
