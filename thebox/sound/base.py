"""Sound engine ABCs and SoundParameters bridge dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SoundParameters:
    """Bridge between EEG processing and the sound engine.

    Updated each pipeline cycle with normalized EEG-derived values.
    """
    # Normalized band powers (0-1)
    alpha: float = 0.0
    beta: float = 0.0
    theta: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0

    # Derived parameters
    amplitude: float = 0.3       # driven by alpha power
    base_frequency: float = 220.0  # driven by beta/alpha ratio (110-880 Hz)
    brightness: float = 0.1      # driven by theta (lower = warmer, less saw)
    noise_gain: float = 0.05     # driven by gamma

    # Event triggers (decay over time)
    blink_trigger: float = 0.0   # 0-1, decays ~200ms
    clench_trigger: float = 0.0  # 0-1, decays ~300ms
    alpha_state: bool = False    # sustained during alpha burst


class SoundSource(ABC):
    """Generates audio samples from SoundParameters."""

    @abstractmethod
    def generate(self, params: SoundParameters, n_frames: int) -> np.ndarray:
        """Generate ``n_frames`` of mono audio as float64 in [-1, 1]."""
        ...


class SoundModulator(ABC):
    """Modifies an audio buffer based on SoundParameters."""

    @abstractmethod
    def process(
        self, audio: np.ndarray, params: SoundParameters
    ) -> np.ndarray:
        """Process audio buffer in-place or return modified copy."""
        ...
