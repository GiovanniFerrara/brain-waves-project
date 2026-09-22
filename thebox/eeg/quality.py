"""Signal quality — flag the parts of a recording that shouldn't be analysed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..ble.protocol import SAMPLE_RATE


@dataclass(frozen=True)
class QualityThresholds:
    max_lost_fraction: float = 0.2    # >20% interpolated samples → unreliable spectrum
    spike_sigma: float = 6.0          # |x| beyond this many robust SDs is an artifact
    spike_floor: float = 40.0         # µV; never call anything smaller a spike
    max_peak_to_peak: float = 150.0   # µV; hard ceiling whatever the channel's noise
    spike_margin: float = 0.25        # s masked either side of a spike
    min_std: float = 1.0              # µV; flatter than this = electrode not touching


# Reasons a window is rejected (bit flags so several can apply)
LOST = 1
ARTIFACT = 2
FLAT = 4


def robust_sigma(x: np.ndarray) -> np.ndarray:
    """Per-channel SD estimated from the median absolute deviation.

    Unlike the plain SD it is barely moved by the artifacts we're looking for.
    """
    med = np.median(x, axis=-1, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=-1)


def artifact_mask(
    cleaned: np.ndarray,
    sample_rate: float = SAMPLE_RATE,
    thresholds: QualityThresholds = QualityThresholds(),
    sigma: np.ndarray | None = None,
) -> np.ndarray:
    """True around samples too large to be EEG: blinks, swallows, jaw, movement.

    The limit adapts to each channel's own noise (``spike_sigma`` robust SDs,
    at least ``spike_floor``, at most half of ``max_peak_to_peak``). Pass
    ``sigma`` to use a baseline estimated elsewhere (e.g. during calibration).
    """
    x = np.atleast_2d(cleaned)
    if sigma is None:
        sigma = robust_sigma(x)
    limit = np.clip(
        thresholds.spike_sigma * np.asarray(sigma, dtype=float),
        thresholds.spike_floor,
        thresholds.max_peak_to_peak / 2,
    )
    hits = np.abs(x) > limit[:, None]
    margin = int(thresholds.spike_margin * sample_rate)
    if margin and hits.any():
        # Dilate each hit to ±margin samples
        kernel = np.ones(2 * margin + 1)
        hits = np.stack([np.convolve(h, kernel, mode="same") > 0 for h in hits])
    return hits


def windows(x: np.ndarray, win: int, step: int) -> np.ndarray:
    """(..., n) → (..., n_windows, win) view of sliding windows."""
    return sliding_window_view(x, win, axis=-1)[..., ::step, :]


@dataclass
class WindowQuality:
    """Rejection flags per channel and window. ``flags`` is (channels, windows)."""

    win_seconds: float
    step_seconds: float
    flags: np.ndarray

    @property
    def good(self) -> np.ndarray:
        return self.flags == 0

    @property
    def centers(self) -> np.ndarray:
        """Window centre times (s)."""
        return np.arange(self.flags.shape[-1]) * self.step_seconds + self.win_seconds / 2

    def good_fraction(self) -> np.ndarray:
        """Share of good windows per channel."""
        return self.good.mean(axis=-1) if self.flags.size else np.zeros(len(self.flags))


def assess(
    cleaned: np.ndarray,
    valid: np.ndarray,
    win_seconds: float = 2.0,
    step_seconds: float | None = None,
    sample_rate: float = SAMPLE_RATE,
    thresholds: QualityThresholds = QualityThresholds(),
    artifacts: np.ndarray | None = None,
) -> WindowQuality:
    """Flag windows of cleaned (band-passed) EEG. Inputs are (channels, samples).

    ``step_seconds`` defaults to ``win_seconds`` (non-overlapping epochs).
    """
    step_seconds = step_seconds or win_seconds
    win, step = int(win_seconds * sample_rate), int(step_seconds * sample_rate)
    x = np.atleast_2d(cleaned)
    if artifacts is None:
        artifacts = artifact_mask(x, sample_rate, thresholds)

    w_x = windows(x, win, step)
    w_lost = 1.0 - windows(np.atleast_2d(valid), win, step).mean(axis=-1)
    w_art = windows(artifacts, win, step).any(axis=-1)

    flags = np.zeros(w_x.shape[:2], dtype=np.uint8)
    flags[w_lost > thresholds.max_lost_fraction] |= LOST
    flags[w_art | (np.ptp(w_x, axis=-1) > thresholds.max_peak_to_peak)] |= ARTIFACT
    flags[w_x.std(axis=-1) < thresholds.min_std] |= FLAT
    return WindowQuality(win_seconds, step_seconds, flags)
