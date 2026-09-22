"""Spectral features that stay meaningful despite EEG's 1/f background."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

ALPHA_SEARCH = (7.0, 14.0)
FIT_RANGE = (2.0, 40.0)


def aperiodic_fit(
    freqs: np.ndarray,
    psd: np.ndarray,
    fit_range: tuple[float, float] = FIT_RANGE,
    exclude: tuple[float, float] = ALPHA_SEARCH,
) -> tuple[float, float]:
    """Straight-line fit of log10(power) vs log10(freq), ignoring the alpha range.

    EEG power falls roughly as 1/f^k, so on log-log axes the background is a
    line and rhythms (alpha, beta) are bumps above it. Returns (offset, slope).
    """
    m = (freqs >= fit_range[0]) & (freqs <= fit_range[1])
    m &= ~((freqs >= exclude[0]) & (freqs <= exclude[1]))
    m &= psd > 0
    slope, offset = np.polyfit(np.log10(freqs[m]), np.log10(psd[m]), 1)
    return float(offset), float(slope)


def above_background_db(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Power relative to the fitted 1/f background, in dB (0 = no rhythm)."""
    offset, slope = aperiodic_fit(freqs, psd)
    with np.errstate(divide="ignore"):
        background = offset + slope * np.log10(freqs)
        return 10 * (np.log10(psd) - background)


@dataclass(frozen=True)
class Peak:
    freq: float           # Hz
    prominence_db: float  # height above the 1/f background


def alpha_peak(
    freqs: np.ndarray,
    psd: np.ndarray,
    search: tuple[float, float] = ALPHA_SEARCH,
    min_prominence_db: float = 1.5,
) -> Peak | None:
    """The alpha peak, if there is a real one above the 1/f background.

    Taking the plain maximum of the 7-14 Hz range just returns its lower
    edge, because the background falls with frequency.
    """
    excess = above_background_db(freqs, psd)
    m = np.flatnonzero((freqs >= search[0]) & (freqs <= search[1]))
    i = m[np.argmax(excess[m])]
    is_local_max = excess[i] >= excess[i - 1] and excess[i] >= excess[i + 1]
    if not is_local_max or excess[i] < min_prominence_db:
        return None
    return Peak(float(freqs[i]), float(excess[i]))


def db_change(powers: np.ndarray, axis: int = -2) -> np.ndarray:
    """Band power in dB relative to its median over time (NaNs ignored).

    ``powers`` is (..., windows, bands). 0 dB = typical for this session,
    +3 dB = double.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10 * np.log10(powers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN channels
        return db - np.nanmedian(db, axis=axis, keepdims=True)


def smooth_powers(
    powers: np.ndarray, good: np.ndarray, n: int, min_good: int = 3
) -> np.ndarray:
    """Trailing mean of band power over the last ``n`` clean segments.

    ``powers`` is (..., segments, bands) in linear units, ``good`` is
    (..., segments). Averaging only clean segments means one blink costs a
    second of data instead of a whole window. Windows with fewer than
    ``min_good`` clean segments are NaN.
    """
    p = np.where(good[..., None], powers, 0.0)
    g = good[..., None].astype(float)
    cp = np.cumsum(p, axis=-2)
    cg = np.cumsum(g, axis=-2)
    # Rolling sums: s[i] - s[i-n]
    cp[..., n:, :] = cp[..., n:, :] - cp[..., :-n, :]
    cg[..., n:, :] = cg[..., n:, :] - cg[..., :-n, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cg >= min_good, cp / cg, np.nan)
