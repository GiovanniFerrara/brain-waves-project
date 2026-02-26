"""EEG frequency band definitions and power computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from ..ble.protocol import SAMPLE_RATE


@dataclass(frozen=True)
class FrequencyBand:
    name: str
    low: float
    high: float
    color: str


# Standard EEG frequency bands
DELTA = FrequencyBand("Delta", 0.5, 4.0, "#9467bd")
THETA = FrequencyBand("Theta", 4.0, 8.0, "#8c564b")
ALPHA = FrequencyBand("Alpha", 8.0, 13.0, "#e377c2")
BETA = FrequencyBand("Beta", 13.0, 30.0, "#17becf")
GAMMA = FrequencyBand("Gamma", 30.0, 50.0, "#bcbd22")

ALL_BANDS = [DELTA, THETA, ALPHA, BETA, GAMMA]

# Legacy dict format for backward compatibility with plotting
BANDS_DICT = {
    f"{b.name} ({b.low}-{int(b.high)} Hz)": (b.low, b.high)
    for b in ALL_BANDS
}
BAND_COLORS = [b.color for b in ALL_BANDS]


def compute_band_powers(
    data: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    bands: list[FrequencyBand] | None = None,
) -> dict[str, float]:
    """Compute power in each frequency band using Welch's method.

    Args:
        data: 1D array of EEG samples (should be at least 1s of data).
        sample_rate: Sampling rate in Hz.
        bands: Frequency bands to compute. Defaults to ALL_BANDS.

    Returns:
        Dict mapping band name to absolute power (µV²/Hz).
    """
    if bands is None:
        bands = ALL_BANDS

    if len(data) < sample_rate:
        return {b.name: 0.0 for b in bands}

    nperseg = min(len(data), sample_rate * 2)
    freqs, psd = welch(data, fs=sample_rate, nperseg=nperseg)

    powers = {}
    for band in bands:
        mask = (freqs >= band.low) & (freqs <= band.high)
        powers[band.name] = float(np.trapezoid(psd[mask], freqs[mask]))

    return powers


def normalize_band_powers(powers: dict[str, float]) -> dict[str, float]:
    """Normalize band powers to relative values summing to 1.0."""
    total = sum(powers.values())
    if total == 0:
        return {k: 0.0 for k in powers}
    return {k: v / total for k, v in powers.items()}
