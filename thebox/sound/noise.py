"""NoiseSource — filtered noise shaped by EEG parameters."""

from __future__ import annotations

import numpy as np

from .base import SoundParameters, SoundSource

AUDIO_SAMPLE_RATE = 44100


class NoiseSource(SoundSource):
    """Generates noise whose gain is driven by gamma power.

    Also produces a burst on jaw clench events.
    """

    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE):
        self.sample_rate = sample_rate

    def generate(self, params: SoundParameters, n_frames: int) -> np.ndarray:
        noise = np.random.default_rng().standard_normal(n_frames)

        # Base noise level from gamma
        level = params.noise_gain * 0.3

        # Clench burst overlay
        if params.clench_trigger > 0.01:
            burst_len = min(n_frames, int(0.02 * self.sample_rate))
            t = np.arange(burst_len) / self.sample_rate
            envelope = params.clench_trigger * np.exp(-t * 30)
            noise[:burst_len] = noise[:burst_len] * envelope + noise[:burst_len]
            level = max(level, params.clench_trigger * 0.5)

        return noise * level
