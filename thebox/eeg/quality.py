"""Signal quality — split a recording into epochs and flag the unusable ones."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ble.protocol import SAMPLE_RATE


@dataclass(frozen=True)
class QualityThresholds:
    max_lost_fraction: float = 0.2   # >20% interpolated samples → unreliable spectrum
    max_peak_to_peak: float = 150.0  # µV after cleaning; blinks, movement, clenches
    min_std: float = 1.0             # µV; flatter than this = electrode not touching


# Reasons an epoch is rejected (bit flags so several can apply)
LOST = 1
ARTIFACT = 2
FLAT = 4


@dataclass
class EpochQuality:
    """Per-channel, per-epoch rejection flags. ``flags`` is (channels, epochs)."""

    epoch_seconds: float
    flags: np.ndarray

    @property
    def good(self) -> np.ndarray:
        return self.flags == 0

    def good_fraction(self) -> np.ndarray:
        """Share of good epochs per channel."""
        return self.good.mean(axis=1) if self.flags.size else np.zeros(len(self.flags))


def epoch_view(x: np.ndarray, epoch_samples: int) -> np.ndarray:
    """Reshape (..., n) into (..., epochs, epoch_samples), dropping the remainder."""
    n_epochs = x.shape[-1] // epoch_samples
    trimmed = x[..., : n_epochs * epoch_samples]
    return trimmed.reshape(*x.shape[:-1], n_epochs, epoch_samples)


def assess(
    cleaned: np.ndarray,
    valid: np.ndarray,
    epoch_seconds: float = 2.0,
    sample_rate: float = SAMPLE_RATE,
    thresholds: QualityThresholds = QualityThresholds(),
) -> EpochQuality:
    """Flag epochs of cleaned (band-passed) EEG. Inputs are (channels, samples)."""
    epoch_samples = int(epoch_seconds * sample_rate)
    ep = epoch_view(np.atleast_2d(cleaned), epoch_samples)
    ep_valid = epoch_view(np.atleast_2d(valid), epoch_samples)

    flags = np.zeros(ep.shape[:2], dtype=np.uint8)
    flags[1.0 - ep_valid.mean(axis=-1) > thresholds.max_lost_fraction] |= LOST
    flags[np.ptp(ep, axis=-1) > thresholds.max_peak_to_peak] |= ARTIFACT
    flags[ep.std(axis=-1) < thresholds.min_std] |= FLAT
    return EpochQuality(epoch_seconds, flags)
