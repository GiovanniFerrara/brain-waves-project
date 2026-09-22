"""BrainMusic — glue between EEG features, the composer and the audio engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from ..eeg.live import BAND_NAMES, LiveFeatures
from ..eeg.recording import Recording
from .composer import BrainComposer
from .engine import MusicEngine
from .kit import SampleKit
from .theory import PROGRESSION

HOP = 0.25  # s between brain-state updates
AUDIO_SR = 48000


class BrainMusic:
    def __init__(self, sr: int = AUDIO_SR, seed: int = 1, **feature_kwargs):
        self.features = LiveFeatures(**feature_kwargs)
        self.composer = BrainComposer(seed=seed + 2)
        self.engine = MusicEngine(SampleKit(sr), self.composer, seed=seed)
        self.recent_event: tuple[str, float] | None = None

    def push(self, channel: str, samples: np.ndarray, valid: np.ndarray) -> None:
        self.features.push(channel, samples, valid)

    def tick(self) -> None:
        """Refresh the brain state the music follows; call every ``HOP`` s."""
        self.composer.brain = self.features.update()
        while self.features.events:
            event = self.features.events.popleft()
            self.composer.events.append(event)
            self.recent_event = (event, self.features.time)

    def status(self) -> str:
        s = self.composer.brain
        if not s.calibrated:
            n = int(20 * s.calibration_progress)
            return (f"calibrating [{'█' * n}{'░' * (20 - n)}] {100 * s.calibration_progress:3.0f}%"
                    f"  — sit still, eyes open   quality {100 * s.quality:3.0f}%")
        bars = " ".join(f"{b[0]}{_bar(self.composer.level(b))}" for b in BAND_NAMES)
        chord = PROGRESSION[max(self.composer.chord_index, 0)].name
        pan = "◀" if s.asymmetry < -0.2 else "▶" if s.asymmetry > 0.2 else "·"
        event = ""
        if self.recent_event and s.time - self.recent_event[1] < 1.5:
            event = f"  ✦ {self.recent_event[0]}"
        chans = "".join("●" if s.channel_good.get(c) else "○" for c in self.features.channels)
        return (f"{bars}  |  signal {chans} {100 * s.quality:3.0f}%  |  {pan}  "
                f"{self.engine.bpm:3.0f} BPM  {chord:7s}{event}")


def _bar(level: float) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    return blocks[int(np.clip(level, 0, 0.999) * len(blocks))]


def render_recording(rec: Recording, out: Path, seed: int = 1) -> Path:
    """Play a saved EEG session through the music offline and write a WAV."""
    music = BrainMusic(seed=seed)
    eeg_hop = int(HOP * rec.sample_rate)
    audio_hop = int(HOP * music.engine.sr)
    blocks = []
    for start in range(0, rec.n_samples - eeg_hop + 1, eeg_hop):
        for i, ch in enumerate(rec.channels):
            music.push(ch, rec.data[i, start:start + eeg_hop], rec.valid[i, start:start + eeg_hop])
        music.tick()
        for _ in range(audio_hop // 1000):
            blocks.append(music.engine.render(1000))
    audio = np.concatenate(blocks)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(out, music.engine.sr, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    return out
