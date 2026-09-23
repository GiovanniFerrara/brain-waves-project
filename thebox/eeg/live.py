"""LiveFeatures — turn the incoming EEG into a calibrated brain state, in real time.

Feed samples with ``push`` (fits ``MuseConnection.on_eeg``) and call
``update`` every ``hop`` seconds. Time is counted in samples received, so the
same code runs live or replaying a recording faster than real time.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos, welch

from ..ble.protocol import CHANNEL_NAMES, SAMPLE_RATE
from .bands import ALL_BANDS, epoch_band_powers
from .features import ALPHA_SEARCH, above_background_db
from .filters import MAINS_FREQUENCY
from .quality import robust_sigma
from .stream import EEGStream

BAND_NAMES = [b.name for b in ALL_BANDS]
LEFT, RIGHT = ("TP9", "AF7"), ("TP10", "AF8")


@dataclass
class BrainState:
    """What the music listens to. Levels are 0-1 with 0.5 = your baseline."""

    time: float = 0.0
    calibrated: bool = False
    calibration_progress: float = 0.0          # 0-1 while calibrating
    levels: dict[str, float] = field(default_factory=lambda: {b: 0.5 for b in BAND_NAMES})
    quality: float = 0.0                       # share of channels currently clean
    asymmetry: float = 0.0                     # alpha, right minus left, -1…1
    alpha_weights: dict[str, float] = field(default_factory=dict)  # who alpha listens to
    channel_good: dict[str, bool] = field(default_factory=dict)


class _CausalFilter:
    """Stateful SOS filter per channel: every sample is filtered exactly once."""

    def __init__(self, sos: np.ndarray, channels: list[str]):
        self.sos = sos
        self.zi: dict[str, np.ndarray | None] = {ch: None for ch in channels}

    def __call__(self, ch: str, x: np.ndarray) -> np.ndarray:
        if self.zi[ch] is None:
            # Start in steady state for the first value: no step transient from
            # the electrode's DC offset
            self.zi[ch] = sosfilt_zi(self.sos) * x[0]
        y, self.zi[ch] = sosfilt(self.sos, x, zi=self.zi[ch])
        return y


class LiveFeatures:
    def __init__(
        self,
        channels: list[str] = CHANNEL_NAMES,
        fs: float = SAMPLE_RATE,
        segment: float = 1.0,          # s of signal per spectrum
        tau: float = 2.5,              # s, smoothing of band power
        warmup: float = 3.0,           # s ignored while filters and contact settle
        calibration: float = 20.0,     # s of baseline
        baseline_tau: float = 120.0,   # s, slow re-centring after calibration
        spike_sigma: float = 6.0,
        spike_limits: tuple[float, float] = (40.0, 75.0),
    ):
        self.channels = list(channels)
        self.fs = fs
        self.segment = segment
        self.tau = tau
        self.warmup = warmup
        self.calibration = calibration
        self.baseline_tau = baseline_tau
        self.spike_sigma = spike_sigma
        self.spike_limits = spike_limits

        notch = tf2sos(*iirnotch(MAINS_FREQUENCY, 30, fs=fs))
        band = butter(4, [1.0, 45.0], btype="band", fs=fs, output="sos")
        self._clean_filter = _CausalFilter(np.vstack([notch, band]), self.channels)
        # Notch here too: a causal 45 Hz edge lets mains hum through, and poorly
        # seated electrodes pick up far more hum than muscle
        emg = butter(4, [20.0, 45.0], btype="band", fs=fs, output="sos")
        self._emg_filter = _CausalFilter(np.vstack([notch, emg]), self.channels)
        # Blinks are slow (<10 Hz) deflections; muscle is fast. Detecting them in
        # separate bands stops a jaw clench from registering as a blink.
        self._slow_filter = _CausalFilter(
            butter(2, [1.0, 10.0], btype="band", fs=fs, output="sos"), self.channels)
        self.clean = EEGStream(duration=12.0)
        self.emg = EEGStream(duration=4.0)
        self.slow = EEGStream(duration=4.0)
        self._received = {ch: 0 for ch in self.channels}

        n_ch, n_b = len(self.channels), len(BAND_NAMES)
        self._power = np.full((n_ch, n_b), np.nan)     # smoothed, linear µV²
        self._psd: np.ndarray | None = None             # (ch, freq), slow average
        self._psd_freqs: np.ndarray | None = None
        self._good_ema = np.zeros(n_ch)
        self._sigma = np.full(n_ch, np.nan)
        self._calib_db: list[np.ndarray] = []           # (ch, band) snapshots
        self._calib_sigma: list[np.ndarray] = []
        self._calib_emg: list[np.ndarray] = []
        self.baseline: np.ndarray | None = None         # (ch, band) dB
        self.spread: np.ndarray | None = None
        self._emg_baseline: np.ndarray | None = None
        self._last_time = 0.0
        self._last_event = {"blink": -1e9, "clench": -1e9}
        self._clench_hold = 0
        self.events: deque[str] = deque()
        self.state = BrainState()

    @property
    def time(self) -> float:
        return min(self._received.values()) / self.fs

    def push(self, channel: str, samples: np.ndarray, valid: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float64)
        self.clean.append(channel, self._clean_filter(channel, samples), valid)
        self.emg.append(channel, self._emg_filter(channel, samples), valid)
        self.slow.append(channel, self._slow_filter(channel, samples), valid)
        self._received[channel] += len(samples)

    # --- per-update analysis ---

    def _limit(self, i: int) -> float:
        sigma = self._sigma[i] if np.isfinite(self._sigma[i]) else self.spike_limits[1] / 6
        return float(np.clip(self.spike_sigma * sigma, *self.spike_limits))

    def update(self) -> BrainState:
        t = self.time
        dt = max(t - self._last_time, 1e-3)
        self._last_time = t
        n_seg = int(self.segment * self.fs)
        a = 1 - np.exp(-dt / self.tau)
        good = np.zeros(len(self.channels), bool)

        for i, ch in enumerate(self.channels):
            seg = self.clean.get_window(ch, self.segment)
            if len(seg) < n_seg or t < self.warmup:
                continue
            if self.baseline is None:
                # Track each channel's noise level while calibrating
                recent = self.clean.get_window(ch, 8.0)
                self._sigma[i] = float(robust_sigma(recent[None])[0])
            ok = (self.clean.valid_fraction(ch, self.segment) >= 0.8
                  and np.abs(seg).max() <= self._limit(i))
            good[i] = ok
            if ok:
                p = epoch_band_powers(seg[None], self.fs)[0]
                self._power[i] = p if np.isnan(self._power[i]).any() else self._power[i] + a * (p - self._power[i])
                self._update_psd(i, seg, dt)
        self._good_ema += (1 - np.exp(-dt / 2.0)) * (good - self._good_ema)

        with np.errstate(divide="ignore", invalid="ignore"):
            db = 10 * np.log10(self._power)
        self._calibrate(t, dt, db, good)
        self._detect_events(t)
        self.state = self._make_state(t, db)
        return self.state

    def _update_psd(self, i: int, seg: np.ndarray, dt: float) -> None:
        """Slow (~15 s) average spectrum per channel, to see whose alpha is clearest."""
        freqs, p = welch(seg, fs=self.fs, nperseg=len(seg))
        if self._psd is None:
            self._psd_freqs = freqs
            self._psd = np.full((len(self.channels), len(freqs)), np.nan)
        if np.isnan(self._psd[i]).any():
            self._psd[i] = p
        else:
            self._psd[i] += (1 - np.exp(-dt / 15.0)) * (p - self._psd[i])

    def alpha_weights(self) -> np.ndarray:
        """How much each channel should count for alpha.

        The eyes-closed alpha rise can be strong on one electrode and absent on
        the others (fit, hair, head shape). Weighting by how far each channel's
        alpha peak stands above its 1/f background lets the music follow the
        channel that actually sees it, instead of averaging it away.
        """
        w = np.ones(len(self.channels))
        if self._psd is None:
            return w
        m = (self._psd_freqs >= ALPHA_SEARCH[0]) & (self._psd_freqs <= ALPHA_SEARCH[1])
        for i, p in enumerate(self._psd):
            if np.isfinite(p).all() and (p > 0).all():
                prominence = above_background_db(self._psd_freqs[1:], p[1:])[m[1:]].max()
                w[i] = (np.clip(prominence, 0, 12) + 0.5) ** 2
        return w

    def _calibrate(self, t: float, dt: float, db: np.ndarray, good: np.ndarray) -> None:
        if self.baseline is None:
            if t < self.warmup:
                return
            if good.any():
                self._calib_db.append(np.where(good[:, None], db, np.nan))
                self._calib_sigma.append(self._sigma.copy())
                self._calib_emg.append(self._emg_rms(0.5))
            if t >= self.warmup + self.calibration and len(self._calib_db) >= 10:
                hist = np.array(self._calib_db)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # never-clean channels
                    self.baseline = np.nanmedian(hist, axis=0)
                    mad = np.nanmedian(np.abs(hist - self.baseline), axis=0)
                self.spread = np.maximum(1.4826 * np.nan_to_num(mad, nan=2.0), 1.5)
                # Channels that never came clean get a neutral baseline
                self.baseline = np.where(np.isfinite(self.baseline), self.baseline,
                                         np.nanmedian(self.baseline, axis=0, keepdims=True))
                self._sigma = np.nanmedian(np.array(self._calib_sigma), axis=0)
                self._emg_baseline = np.median(np.array(self._calib_emg), axis=0)
            return
        # After calibration: follow slow drifts (sweat, headband shifting)
        b = 1 - np.exp(-dt / self.baseline_tau)
        upd = good[:, None] & np.isfinite(db)
        self.baseline = np.where(upd, self.baseline + b * (db - self.baseline), self.baseline)

    def _emg_rms(self, seconds: float) -> np.ndarray:
        return np.array([
            np.sqrt(np.mean(w ** 2)) if len(w := self.emg.get_window(ch, seconds)) else 0.0
            for ch in self.channels
        ])

    def _detect_events(self, t: float) -> None:
        if self.baseline is None:
            return
        idx = {ch: i for i, ch in enumerate(self.channels)}
        window = 0.35
        # Blink: a big deflection on BOTH frontal channels at once
        if t - self._last_event["blink"] > 0.6 and {"AF7", "AF8"} <= idx.keys():
            peaks = [np.abs(self.slow.get_window(ch, window)).max(initial=0) for ch in ("AF7", "AF8")]
            limits = [max(self._limit(idx[ch]), 60.0) for ch in ("AF7", "AF8")]
            if all(p > lim for p, lim in zip(peaks, limits)):
                self._fire("blink", t)
        # Jaw clench: a sustained, two-sided muscle (20-45 Hz) burst. Measured on
        # a Muse 2: real clenches raise EMG >2x on BOTH temporal channels about
        # equally (and on the forehead too); swallows and electrode rubs hit one
        # side; blinks are dominated by the forehead.
        if {"TP9", "TP10"} <= idx.keys():
            rms = self._emg_rms(window)
            ratio = {ch: rms[i] / max(self._emg_baseline[i], 1.0) for ch, i in idx.items()}
            tp = np.array([ratio["TP9"], ratio["TP10"]])
            frontal = np.mean([ratio.get("AF7", 1.0), ratio.get("AF8", 1.0)])
            burst = (tp.min() > 2.0 and tp.min() / tp.max() >= 0.5
                     and frontal < 1.5 * tp.mean()
                     and t - self._last_event["blink"] > 1.0)
            self._clench_hold = self._clench_hold + 1 if burst else 0
            if self._clench_hold >= 2 and t - self._last_event["clench"] > 1.5:
                self._fire("clench", t)

    def _fire(self, name: str, t: float) -> None:
        self._last_event[name] = t
        self.events.append(name)

    def _make_state(self, t: float, db: np.ndarray) -> BrainState:
        progress = float(np.clip((t - self.warmup) / self.calibration, 0, 1))
        good = self._good_ema > 0.5
        state = BrainState(
            time=t,
            calibrated=self.baseline is not None,
            calibration_progress=1.0 if self.baseline is not None else progress,
            quality=float(np.mean(self._good_ema)),
            channel_good={ch: bool(g) for ch, g in zip(self.channels, good)},
        )
        if self.baseline is None or not good.any():
            return state
        z = (db - self.baseline) / self.spread               # (ch, band)
        z_good = np.where(good[:, None] & np.isfinite(z), z, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN bands
            mean_z = np.nanmean(z_good, axis=0)
        alpha = BAND_NAMES.index("Alpha")
        weights = np.where(np.isfinite(z_good[:, alpha]), self.alpha_weights(), 0.0)
        if weights.sum() > 0:
            mean_z[alpha] = float(np.nansum(z_good[:, alpha] * weights) / weights.sum())
            state.alpha_weights = {ch: round(float(w / weights.sum()), 2)
                                   for ch, w in zip(self.channels, weights)}
        state.levels = {
            name: float(np.clip(0.5 + mz / 4, 0, 1)) if np.isfinite(mz) else 0.5
            for name, mz in zip(BAND_NAMES, mean_z)
        }
        side = lambda chans: np.nanmean([z_good[self.channels.index(c), alpha]
                                         for c in chans if c in self.channels] or [np.nan])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            diff = side(RIGHT) - side(LEFT)
        state.asymmetry = float(np.tanh(diff / 2)) if np.isfinite(diff) else 0.0
        return state
