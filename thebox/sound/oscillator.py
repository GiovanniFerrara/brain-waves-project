"""OscillatorSource — sine/saw tone generation from EEG parameters."""

from __future__ import annotations

import numpy as np

from .base import SoundParameters, SoundSource

AUDIO_SAMPLE_RATE = 44100


class OscillatorSource(SoundSource):
    """Generates a tone whose frequency, amplitude, and timbre are driven by EEG.

    Uses phase accumulation for glitch-free frequency changes between blocks.
    Brightness controls the mix between sine (warm) and sawtooth (bright).
    """

    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._phase = 0.0

    def generate(self, params: SoundParameters, n_frames: int) -> np.ndarray:
        freq = params.base_frequency
        amp = params.amplitude

        # Phase accumulation
        phase_inc = 2.0 * np.pi * freq / self.sample_rate
        phases = self._phase + phase_inc * np.arange(n_frames)
        self._phase = float(phases[-1] % (2.0 * np.pi)) if n_frames > 0 else self._phase

        # Sine component
        sine = np.sin(phases)

        # Sawtooth component (from phase)
        saw = 2.0 * (phases / (2.0 * np.pi) % 1.0) - 1.0

        # Brightness blends sine → sawtooth
        brightness = np.clip(params.brightness, 0.0, 1.0)
        tone = (1.0 - brightness) * sine + brightness * saw

        # Blink trigger: percussive click
        if params.blink_trigger > 0.01:
            click_len = min(n_frames, int(0.01 * self.sample_rate))
            t = np.arange(click_len) / self.sample_rate
            click = params.blink_trigger * np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 50)
            tone[:click_len] += click

        return tone * amp
