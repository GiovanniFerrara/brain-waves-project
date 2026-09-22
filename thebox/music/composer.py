"""BrainComposer — the mapping from brain state to music.

    delta  → sub-bass drone level
    theta  → chord pad level, and slower chord changes when high (dreamy)
    alpha  → bell/pluck melody: more notes, higher and brighter as it rises
    beta   → drums: from silence to kick, then hats, backbeat, busier hats
    gamma  → (mostly muscle tension on a Muse) wind texture and ghost clicks
    asymmetry (right-left alpha) → where the melody sits in the stereo field
    blink  → chime run          jaw clench → boom and a drum fill

Levels are 0-1 around a personal baseline of 0.5, so the music follows
changes in *your* state rather than absolute numbers. Everything stays in
D minor on a 16th-note grid, so whatever the brain does, it sounds musical.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..eeg.live import BrainState
from .engine import MusicEngine
from .theory import PENTATONIC, PROGRESSION, ROOT, scale_note

MELODY_ROOT = ROOT + 12       # D5
DEGREE_RANGE = (-3, 8)        # pentatonic degrees around D5


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 below lo, 1 above hi, linear between."""
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


class BrainComposer:
    def __init__(self, seed: int = 3):
        self.brain = BrainState()
        self.events: deque[str] = deque()
        self.rng = np.random.default_rng(seed)
        self.smooth: dict[str, float] = dict(self.brain.levels)
        self.quality = 0.0
        self.chord_index = -1
        self.bars_per_chord = 2
        self.degree = 2
        self.fill_steps = 0
        self.last_event: str | None = None

    # --- state ---

    def level(self, band: str) -> float:
        """Smoothed band level, pulled to neutral when the signal is poor."""
        return 0.5 + (self.smooth[band] - 0.5) * self.quality

    @property
    def live(self) -> bool:
        return self.brain.calibrated

    # --- engine callbacks ---

    def on_block(self, engine: MusicEngine, frames: int) -> None:
        dt = frames / engine.sr
        k = 1 - np.exp(-dt / 1.0)  # ~1 s glide on top of the EEG smoothing
        for band, value in self.brain.levels.items():
            self.smooth[band] += k * (value - self.smooth[band])
        target_q = self.brain.quality if self.live else 0.0
        self.quality += k * (target_q - self.quality)

        delta, theta, alpha = self.level("Delta"), self.level("Theta"), self.level("Alpha")
        gamma = self.level("Gamma")

        engine.bus("sub").gain = 0.35 + 0.4 * delta
        engine.bus("pad").gain = 0.22 + 0.45 * theta
        engine.bus("pad").cutoff = 900 + 5000 * alpha
        engine.bus("pad").send = 0.1
        melody = engine.bus("melody")
        melody.gain = 0.55
        melody.cutoff = 1800 + 9000 * alpha
        melody.send = 0.25 + 0.35 * alpha
        engine.bus("drums").gain = 0.6
        engine.bus("drums").send = 0.05
        engine.bus("texture").gain = 0.08 + 0.4 * _ramp(gamma, 0.5, 0.9) * self.quality
        engine.bus("texture").cutoff = 2500 + 6000 * gamma
        engine.bus("fx").gain = 0.7
        engine.bus("fx").send = 0.3

        while self.events:
            self._gesture(engine, self.events.popleft())

    def on_step(self, engine: MusicEngine, step: int, offset: int) -> None:
        s16, bar = step % 16, step // 16
        if step == 0:
            engine.play("texture", "texture", loop=True, tag="texture", gain=0.8)
        if s16 == 0:
            self._on_bar(engine, bar, offset)
        self._melody(engine, s16, offset)
        self._drums(engine, s16, offset)

    # --- musical decisions ---

    def _on_bar(self, engine: MusicEngine, bar: int, offset: int) -> None:
        # Tempo breathes with arousal (beta), at most 1 BPM per bar
        target = 76 + 22 * self.level("Beta")
        engine.bpm += float(np.clip(target - engine.bpm, -1.0, 1.0))

        if self.chord_index >= 0 and bar % self.bars_per_chord:
            return
        # Dreamy (high theta) → linger on each chord
        self.bars_per_chord = 4 if self.level("Theta") > 0.65 else 2
        self.chord_index = (self.chord_index + 1) % len(PROGRESSION)
        chord = PROGRESSION[self.chord_index]
        engine.release("pad", 2.5)
        engine.release("sub", 1.5)
        for i, tone in enumerate(chord.tones):
            pan = (i / max(len(chord.tones) - 1, 1) - 0.5) * 1.2
            engine.play("pad", "pad", midi=ROOT + tone, gain=0.3, pan=pan, offset=offset, tag="pad")
        engine.play("sub", "sub", midi=38 + chord.root, gain=0.9, offset=offset, tag="sub")

    def _melody(self, engine: MusicEngine, s16: int, offset: int) -> None:
        if s16 % 2:
            return  # melody lives on 8ths
        alpha, beta = self.level("Alpha"), self.level("Beta")
        p = 0.04 + (0.65 * _ramp(alpha, 0.35, 0.9) if self.live else 0.0)
        if s16 == 0:
            p += 0.15  # lean on the downbeat
        if self.rng.random() >= p:
            return

        step = self.rng.choice([-2, -1, -1, 1, 1, 2])
        lo, hi = DEGREE_RANGE
        lift = 3 if alpha > 0.75 else 0  # blooms upward as alpha rises
        self.degree = int(np.clip(self.degree + step, lo + lift, hi + lift))
        semis = scale_note(self.degree, PENTATONIC)
        if s16 == 0:
            semis = self._nearest_chord_tone(semis)  # downbeats land on the harmony

        instrument = "bell" if alpha >= beta else "pluck"
        accent = 1.0 if s16 in (0, 8) else 0.75
        gain = accent * self.rng.uniform(0.45, 0.8)
        pan = 0.7 * self.brain.asymmetry + self.rng.uniform(-0.25, 0.25)
        engine.play(instrument, "melody", midi=MELODY_ROOT + semis, gain=gain, pan=pan, offset=offset)

    def _nearest_chord_tone(self, semis: int) -> int:
        chord = PROGRESSION[max(self.chord_index, 0)]
        pcs = {t % 12 for t in chord.tones}
        for d in (0, -1, 1, -2, 2):
            if (semis + d) % 12 in pcs:
                return semis + d
        return semis

    def _drums(self, engine: MusicEngine, s16: int, offset: int) -> None:
        fill = self.fill_steps > 0
        if fill:
            self.fill_steps -= 1
        drive = _ramp(self.level("Beta"), 0.4, 0.85) if self.live else 0.0
        gamma = self.level("Gamma") if self.live else 0.0
        r = self.rng.random

        if (s16 == 0 and drive > 0.05) or (s16 == 8 and drive > 0.35) \
                or (s16 == 10 and drive > 0.75 and r() < 0.5) or (fill and s16 % 4 == 0):
            engine.play("kick", "drums", gain=0.9, offset=offset)
        if s16 in (4, 12) and (drive > 0.55 or fill):
            engine.play("clap", "drums", gain=0.5, pan=0.1, offset=offset)
        hat = (s16 % 4 == 2 and drive > 0.2) or (s16 % 2 == 0 and drive > 0.45 and r() < drive) \
            or (drive > 0.65 and r() < (drive - 0.6)) or fill
        if hat:
            engine.play("hat", "drums", gain=self.rng.uniform(0.25, 0.5) * (1.3 if fill else 1),
                        pan=self.rng.uniform(-0.4, 0.4), offset=offset)
        if gamma > 0.6 and r() < 0.25 * _ramp(gamma, 0.6, 1.0):
            engine.play("tick", "drums", gain=self.rng.uniform(0.2, 0.4),
                        pan=self.rng.uniform(-0.8, 0.8), offset=offset)

    def _gesture(self, engine: MusicEngine, event: str) -> None:
        self.last_event = event
        if event == "blink":
            engine.play("chime", "fx", gain=0.55, pan=self.rng.uniform(-0.6, 0.6))
        elif event == "clench":
            engine.play("boom", "fx", gain=0.9)
            self.fill_steps = 8
