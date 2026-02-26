"""Mixer — combines sound sources and applies modulators."""

from __future__ import annotations

import numpy as np

from .base import SoundModulator, SoundParameters, SoundSource


class Mixer:
    """Sums multiple SoundSources with individual gains, applies modulators, normalizes.

    Usage::

        mixer = Mixer()
        mixer.add_source(OscillatorSource(), gain=0.7)
        mixer.add_source(NoiseSource(), gain=0.3)
        audio = mixer.generate(params, n_frames=2205)
    """

    def __init__(self, master_volume: float = 0.5):
        self.master_volume = master_volume
        self._sources: list[tuple[SoundSource, float]] = []
        self._modulators: list[SoundModulator] = []

    def add_source(self, source: SoundSource, gain: float = 1.0) -> None:
        self._sources.append((source, gain))

    def add_modulator(self, modulator: SoundModulator) -> None:
        self._modulators.append(modulator)

    def generate(self, params: SoundParameters, n_frames: int) -> np.ndarray:
        """Generate a mixed audio block."""
        mixed = np.zeros(n_frames, dtype=np.float64)

        for source, gain in self._sources:
            mixed += source.generate(params, n_frames) * gain

        for modulator in self._modulators:
            mixed = modulator.process(mixed, params)

        # Soft clip to [-1, 1]
        mixed = np.tanh(mixed * self.master_volume)

        return mixed
