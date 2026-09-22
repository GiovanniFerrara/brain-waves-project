"""MusicEngine — sample-accurate sequencer, voices, buses and effects.

The engine knows nothing about brains: a ``Composer`` gets called on every
16th-note step to trigger sounds, and once per block to set bus levels.
``render`` works the same live (from the audio callback) or offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.signal import butter, sosfilt

from .kit import SampleKit

STEPS_PER_BEAT = 4  # 16th notes


class Composer(Protocol):
    def on_block(self, engine: MusicEngine, frames: int) -> None: ...
    def on_step(self, engine: MusicEngine, step: int, offset: int) -> None: ...


@dataclass
class Voice:
    audio: np.ndarray
    bus: str
    gain: float
    pan: float                 # -1 left … +1 right
    pos: int                   # read position; negative = starts later in this block
    loop: bool = False
    tag: str | None = None
    release_total: int = 0
    release_left: int | None = None

    def pan_gains(self) -> tuple[float, float]:
        angle = (self.pan + 1) * np.pi / 4  # equal-power
        return np.cos(angle) * self.gain, np.sin(angle) * self.gain


@dataclass
class Bus:
    gain: float = 1.0          # target; ramped per block to avoid clicks
    cutoff: float | None = None  # low-pass Hz, None = open
    send: float = 0.0          # to the delay
    _gain_now: float = 0.0
    _cutoff_now: float | None = None
    _sos: np.ndarray | None = None
    _zi: np.ndarray | None = None
    buffer: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))


class StereoDelay:
    """Ping-pong feedback delay; vectorised because delay ≥ block size."""

    def __init__(self, sr: int, max_seconds: float = 2.0, feedback: float = 0.45):
        self.buf = np.zeros((int(max_seconds * sr), 2), np.float32)
        self.w = 0
        self.feedback = feedback
        self.tone = butter(1, 3000, fs=sr, output="sos")
        self.zi = np.zeros((1, 2, 2))

    def process(self, x: np.ndarray, delay: int) -> np.ndarray:
        n, size = len(x), len(self.buf)
        delay = max(delay, n)
        read_idx = (self.w - delay + np.arange(n)) % size
        wet = self.buf[read_idx]
        # Feed back crossed over (ping-pong) and darkened a little each repeat
        fb, self.zi = sosfilt(self.tone, wet[:, ::-1] * self.feedback, axis=0, zi=self.zi)
        self.buf[(self.w + np.arange(n)) % size] = x + fb
        self.w = (self.w + n) % size
        return wet


class MusicEngine:
    def __init__(self, kit: SampleKit, composer: Composer, bpm: float = 84.0, seed: int = 1):
        self.kit = kit
        self.sr = kit.sr
        self.composer = composer
        self.bpm = bpm
        self.rng = np.random.default_rng(seed)
        self.voices: list[Voice] = []
        self.buses: dict[str, Bus] = {}
        self.delay = StereoDelay(self.sr)
        self.master = 0.85
        self.step = 0
        self._to_next_step = 0.0  # samples until the next 16th
        self.max_voices = 64

    # --- API for composers ---

    def bus(self, name: str) -> Bus:
        return self.buses.setdefault(name, Bus())

    @property
    def step_samples(self) -> float:
        return self.sr * 60.0 / self.bpm / STEPS_PER_BEAT

    def play(
        self,
        name: str,
        bus: str,
        *,
        midi: int | None = None,
        gain: float = 1.0,
        pan: float = 0.0,
        offset: int = 0,
        loop: bool = False,
        tag: str | None = None,
    ) -> None:
        """Start a sample ``offset`` samples into the current block."""
        if len(self.voices) >= self.max_voices:
            self.voices.sort(key=lambda v: v.gain)
            self.voices.pop(0)  # steal the quietest
        self.bus(bus)
        self.voices.append(Voice(self.kit.get(name, midi), bus, gain,
                                 float(np.clip(pan, -1, 1)), -offset, loop, tag))

    def release(self, tag: str, seconds: float = 1.0) -> None:
        """Fade out every voice with ``tag``."""
        n = max(1, int(seconds * self.sr))
        for v in self.voices:
            if v.tag == tag and v.release_left is None:
                v.release_total = v.release_left = n

    def has(self, tag: str) -> bool:
        return any(v.tag == tag and v.release_left is None for v in self.voices)

    # --- rendering ---

    def render(self, frames: int) -> np.ndarray:
        """Next ``frames`` of stereo audio, float32 in [-1, 1]."""
        self.composer.on_block(self, frames)

        # Fire every step that falls inside this block, at its exact offset
        pos = self._to_next_step
        while pos < frames:
            self.composer.on_step(self, self.step, int(pos))
            self.step += 1
            pos += self.step_samples
        self._to_next_step = pos - frames

        for bus in self.buses.values():
            if len(bus.buffer) != frames:
                bus.buffer = np.zeros((frames, 2), np.float32)
            else:
                bus.buffer.fill(0)
        self._mix_voices(frames)

        out = np.zeros((frames, 2), np.float32)
        send = np.zeros((frames, 2), np.float32)
        ramp = np.linspace(0, 1, frames, endpoint=False, dtype=np.float32)[:, None]
        for bus in self.buses.values():
            x = self._lowpass(bus, bus.buffer)
            g = bus._gain_now + (bus.gain - bus._gain_now) * ramp
            bus._gain_now = bus.gain
            x = x * g
            out += x
            if bus.send:
                send += x * bus.send
        delay = int(self.step_samples * 3)  # dotted eighth
        out += self.delay.process(send, delay)
        return np.tanh(out * self.master).astype(np.float32)

    def _mix_voices(self, frames: int) -> None:
        alive = []
        for v in self.voices:
            start = max(0, -v.pos)
            n = frames - start
            if n <= 0:
                v.pos += frames
                alive.append(v)
                continue
            src_from = max(0, v.pos)
            if v.loop:
                idx = (src_from + np.arange(n)) % len(v.audio)
                chunk = v.audio[idx]
            else:
                chunk = v.audio[src_from:src_from + n]
            if v.release_left is not None:
                left = v.release_left - np.arange(len(chunk))
                env = np.clip(left / v.release_total, 0, 1).astype(np.float32)
                chunk = chunk * env
                v.release_left -= len(chunk)
            gl, gr = v.pan_gains()
            buf = self.buses[v.bus].buffer
            buf[start:start + len(chunk), 0] += chunk * gl
            buf[start:start + len(chunk), 1] += chunk * gr
            v.pos += frames
            done = (not v.loop and v.pos >= len(v.audio)) or (
                v.release_left is not None and v.release_left <= 0)
            if not done:
                alive.append(v)
        self.voices = alive

    def _lowpass(self, bus: Bus, x: np.ndarray) -> np.ndarray:
        if bus.cutoff is None:
            return x
        cutoff = float(np.clip(bus.cutoff, 60, self.sr * 0.45))
        if bus._cutoff_now is None or abs(cutoff - bus._cutoff_now) > 0.02 * bus._cutoff_now:
            bus._sos = butter(2, cutoff, fs=self.sr, output="sos")
            bus._cutoff_now = cutoff
            if bus._zi is None:
                bus._zi = np.zeros((bus._sos.shape[0], 2, 2))
        y, bus._zi = sosfilt(bus._sos, x, axis=0, zi=bus._zi)
        return y.astype(np.float32)
