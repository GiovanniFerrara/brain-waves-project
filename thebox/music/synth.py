"""Synthesise the default sample kit (replaceable with your own WAVs)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from .theory import PENTATONIC, ratio


def midi_hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


class Synth:
    """Generators for each kit sample, all at ``sr`` Hz, mono float32."""

    def __init__(self, sr: int = 48000, seed: int = 7):
        self.sr = sr
        self.rng = np.random.default_rng(seed)

    # --- building blocks ---

    def t(self, seconds: float) -> np.ndarray:
        return np.arange(int(seconds * self.sr)) / self.sr

    def noise(self, seconds: float) -> np.ndarray:
        return self.rng.standard_normal(int(seconds * self.sr))

    def filt(self, x: np.ndarray, kind: str, freq, order: int = 2) -> np.ndarray:
        return sosfilt(butter(order, freq, btype=kind, fs=self.sr, output="sos"), x)

    def reverb(self, x: np.ndarray, seconds: float = 2.5, mix: float = 0.3) -> np.ndarray:
        """Convolve with a synthetic hall: decaying, darkening noise."""
        t = self.t(seconds)
        ir = self.rng.standard_normal(len(t)) * np.exp(-6.9 * t / seconds)
        ir = self.filt(ir, "low", 5000)
        ir /= np.sqrt(np.sum(ir**2))
        wet = fftconvolve(np.concatenate([x, np.zeros(len(ir))]), ir)[: len(x) + len(ir)]
        dry = np.concatenate([x, np.zeros(len(ir))])
        return (1 - mix) * dry + mix * wet * (np.abs(x).max() / (np.abs(wet).max() + 1e-12))

    @staticmethod
    def fade(x: np.ndarray, sr: int, fade_in: float = 0.002, fade_out: float = 0.02) -> np.ndarray:
        x = x.copy()
        a, b = int(fade_in * sr), int(fade_out * sr)
        if a:
            x[:a] *= np.linspace(0, 1, a)
        if b:
            x[-b:] *= np.linspace(1, 0, b)
        return x

    @staticmethod
    def normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
        return (x * peak / (np.abs(x).max() + 1e-12)).astype(np.float32)

    def _finish(self, x: np.ndarray, fade_out: float = 0.02) -> np.ndarray:
        return self.normalize(self.fade(x, self.sr, fade_out=fade_out))

    # --- drums ---

    def kick(self) -> np.ndarray:
        t = self.t(0.6)
        freq = 42 + 110 * np.exp(-t / 0.035)
        body = np.sin(2 * np.pi * np.cumsum(freq) / self.sr) * np.exp(-t / 0.22)
        click = self.filt(self.noise(0.6), "band", [1500, 6000]) * np.exp(-t / 0.004) * 0.3
        return self._finish(np.tanh(1.8 * (body + click)))

    def hat(self) -> np.ndarray:
        t = self.t(0.12)
        x = self.filt(self.noise(0.12), "high", 7500, 4) * np.exp(-t / 0.025)
        return self._finish(x, 0.005) * 0.8

    def clap(self) -> np.ndarray:
        t = self.t(0.5)
        env = np.zeros_like(t)
        for delay in (0.0, 0.011, 0.023):  # three hands, slightly apart
            env += (t >= delay) * np.exp(-np.clip(t - delay, 0, None) / 0.006)
        env += (t >= 0.03) * np.exp(-np.clip(t - 0.03, 0, None) / 0.12) * 0.5
        x = self.filt(self.noise(0.5), "band", [900, 4500]) * env
        return self._finish(self.reverb(x, 0.8, 0.15))

    def tick(self) -> np.ndarray:
        t = self.t(0.08)
        x = np.sin(2 * np.pi * 1900 * t) * np.exp(-t / 0.01)
        x += self.filt(self.noise(0.08), "band", [3000, 8000]) * np.exp(-t / 0.006) * 0.5
        return self._finish(x, 0.005) * 0.7

    # --- pitched (tuned to D) ---

    def bell(self, midi: int = 74) -> np.ndarray:
        """FM bell: inharmonic modulator ratio, brightness decaying faster than volume."""
        f = midi_hz(midi)
        t = self.t(4.0)
        index = 3.0 * np.exp(-t / 0.6)
        x = np.sin(2 * np.pi * f * t + index * np.sin(2 * np.pi * 1.4 * f * t))
        x += 0.25 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t / 0.4)
        x *= np.exp(-t / 1.1)
        return self._finish(self.reverb(self.fade(x, self.sr), 3.0, 0.35), 0.3)

    def pluck(self, midi: int = 62) -> np.ndarray:
        """Karplus-Strong string, computed one period at a time."""
        n = int(round(self.sr / midi_hz(midi)))
        total = int(2.5 * self.sr)
        y = np.zeros(total + n + 1)
        y[: n + 1] = self.filt(self.rng.uniform(-1, 1, n + 1), "low", 4000)
        for start in range(n + 1, total + n + 1, n):
            end = min(start + n, total + n + 1)
            k = end - start
            y[start:end] = 0.996 * 0.5 * (y[start - n:start - n + k] + y[start - n - 1:start - n - 1 + k])
        x = self.filt(y[n + 1:], "low", 3500)
        return self._finish(self.reverb(x, 2.0, 0.25), 0.3)

    def pad(self, midi: int = 62, seconds: float = 12.0) -> np.ndarray:
        """Seven detuned saws, low-passed, slow attack and release."""
        t = self.t(seconds)
        f = midi_hz(midi)
        x = np.zeros_like(t)
        for cents in (-14, -8, -3, 0, 3, 8, 14):
            fi = f * 2 ** (cents / 1200)
            phase = self.rng.uniform(0, 1)
            x += 2 * ((fi * t + phase) % 1.0) - 1
        x = self.filt(x, "low", 1400, 2)
        env = np.minimum(t / 2.5, 1.0) * np.minimum((seconds - t) / 3.0, 1.0)
        return self._finish(self.reverb(x * env, 3.0, 0.3), 0.5)

    def sub(self, midi: int = 38, seconds: float = 12.0) -> np.ndarray:
        t = self.t(seconds)
        f = midi_hz(midi)
        x = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)
        env = np.minimum(t / 0.8, 1.0) * (0.7 + 0.3 * np.exp(-t / 2)) * np.minimum((seconds - t) / 3.0, 1.0)
        return self._finish(np.tanh(1.5 * x) * env)

    def texture(self, seconds: float = 8.0) -> np.ndarray:
        """Noise shaped into a 'wind choir' on D-minor partials. Loops seamlessly
        because it is built in the frequency domain (circular by construction)."""
        n = int(seconds * self.sr)
        freqs = np.fft.rfftfreq(n, 1 / self.sr)
        shape = np.zeros_like(freqs)
        for midi in (50, 57, 62, 65, 69, 74, 77):
            fc = midi_hz(midi)
            shape += np.exp(-0.5 * ((freqs - fc) / (fc * 0.004)) ** 2) / np.sqrt(fc)
        shape += 0.02 * np.exp(-freqs / 3000)  # a little air
        spec = shape * np.exp(2j * np.pi * self.rng.uniform(size=len(freqs)))
        x = np.fft.irfft(spec, n)
        return self.normalize(x)

    # --- gestures ---

    def chime(self) -> np.ndarray:
        """Blink: a quick upward run of bell notes."""
        base = self.bell(86)
        out = np.zeros(len(base) + int(0.5 * self.sr))
        for i, degree in enumerate(PENTATONIC + [12]):
            note = _resample(base, ratio(degree))
            start = int(i * 0.055 * self.sr)
            seg = note[: len(out) - start] * (0.9 ** i)
            out[start:start + len(seg)] += seg
        return self._finish(out, 0.3)

    def boom(self) -> np.ndarray:
        """Jaw clench: a deep impact with a long tail."""
        t = self.t(1.6)
        freq = 28 + 70 * np.exp(-t / 0.08)
        body = np.sin(2 * np.pi * np.cumsum(freq) / self.sr) * np.exp(-t / 0.6)
        hit = self.filt(self.noise(1.6), "low", 900) * np.exp(-t / 0.05)
        return self._finish(self.reverb(np.tanh(2 * (body + 0.5 * hit)), 3.5, 0.35), 0.5)


def _resample(x: np.ndarray, rate: float) -> np.ndarray:
    """Play ``x`` at ``rate`` times speed (linear interpolation; rate>1 = higher)."""
    idx = np.arange(0, len(x) - 1, rate)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)
