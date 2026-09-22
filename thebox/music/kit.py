"""SampleKit — load the WAVs in samples/, generating any that are missing.

To use your own sounds, drop a WAV with the same name into samples/. Pitched
samples must be tuned to the note in ``KIT`` (all Ds), or edit that entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from .synth import Synth, _resample
from .theory import ratio

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "samples"


@dataclass(frozen=True)
class SampleSpec:
    root: int | None   # MIDI note the sample is tuned to; None = unpitched
    make: str          # Synth method that generates the default
    description: str


KIT = {
    "kick":    SampleSpec(None, "kick",    "beta: the pulse"),
    "hat":     SampleSpec(None, "hat",     "beta: hi-hat density"),
    "clap":    SampleSpec(None, "clap",    "high beta: backbeat"),
    "tick":    SampleSpec(None, "tick",    "gamma/EMG: ghost clicks"),
    "bell":    SampleSpec(74,   "bell",    "alpha: melody (calm)"),
    "pluck":   SampleSpec(62,   "pluck",   "alpha: melody (alert)"),
    "pad":     SampleSpec(62,   "pad",     "theta: chords"),
    "sub":     SampleSpec(38,   "sub",     "delta: bass drone"),
    "texture": SampleSpec(None, "texture", "gamma/EMG: wind layer (looped)"),
    "chime":   SampleSpec(None, "chime",   "blink gesture"),
    "boom":    SampleSpec(None, "boom",    "jaw clench gesture"),
}


def _read_wav(path: Path, sr: int) -> np.ndarray:
    file_sr, data = wavfile.read(path)
    if data.dtype.kind == "i":
        data = data / np.iinfo(data.dtype).max
    elif data.dtype.kind == "u":
        data = (data - 128) / 128.0
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if file_sr != sr:
        g = gcd(sr, file_sr)
        data = resample_poly(data, sr // g, file_sr // g)
    return data.astype(np.float32)


class SampleKit:
    def __init__(self, sr: int = 48000, directory: Path = DEFAULT_DIR):
        self.sr = sr
        self.directory = Path(directory)
        self.samples: dict[str, np.ndarray] = {}
        self._pitched: dict[tuple[str, int], np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        synth = None
        for name, spec in KIT.items():
            path = self.directory / f"{name}.wav"
            if not path.exists():
                synth = synth or Synth(self.sr)
                audio = getattr(synth, spec.make)()
                wavfile.write(path, self.sr, (audio * 32767).astype(np.int16))
            self.samples[name] = _read_wav(path, self.sr)

    def get(self, name: str, midi: int | None = None) -> np.ndarray:
        """The sample, transposed to ``midi`` if it is pitched."""
        spec = KIT[name]
        if midi is None or spec.root is None or midi == spec.root:
            return self.samples[name]
        key = (name, midi)
        if key not in self._pitched:
            self._pitched[key] = _resample(self.samples[name], ratio(midi - spec.root))
        return self._pitched[key]
