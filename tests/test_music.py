"""Music engine, composer and live features on synthetic data."""

import numpy as np
import pytest

from thebox.ble.protocol import CHANNEL_NAMES
from thebox.eeg.live import LiveFeatures
from thebox.eeg.recording import Recording
from thebox.music.engine import MusicEngine
from thebox.music.kit import KIT, SampleKit
from thebox.music.session import render_recording

FS = 256


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    return SampleKit(48000, tmp_path_factory.mktemp("samples"))


class RecordingComposer:
    """Logs what the engine asks of it; plays a kick on every beat."""

    def __init__(self):
        self.steps = []

    def on_block(self, engine, frames):
        engine.bus("drums").gain = 1.0

    def on_step(self, engine, step, offset):
        self.steps.append((step, offset))
        if step % 4 == 0:
            engine.play("kick", "drums", offset=offset)


class TestKit:
    def test_generates_every_sample(self, kit):
        assert set(kit.samples) == set(KIT)
        assert all(np.abs(s).max() > 0.1 for s in kit.samples.values())

    def test_pitch_shift_changes_length(self, kit):
        up = kit.get("bell", 74 + 12)
        assert abs(len(up) - len(kit.samples["bell"]) / 2) < 2


class TestEngine:
    def test_steps_land_on_the_grid(self, kit):
        """At 120 BPM a 16th is 6000 samples; steps must hit it exactly across blocks."""
        comp = RecordingComposer()
        eng = MusicEngine(kit, comp, bpm=120)
        times = []
        for block in range(20):
            before = len(comp.steps)
            eng.render(1000)
            times += [block * 1000 + off for _, off in comp.steps[before:]]
        assert times == [0, 6000, 12000, 18000]

    def test_output_is_stereo_bounded_and_not_silent(self, kit):
        eng = MusicEngine(kit, RecordingComposer(), bpm=120)
        out = np.concatenate([eng.render(1024) for _ in range(40)])
        assert out.shape == (40 * 1024, 2)
        assert np.abs(out).max() <= 1.0
        assert np.abs(out).max() > 0.1

    def test_release_fades_voice_out(self, kit):
        eng = MusicEngine(kit, RecordingComposer())
        eng.play("pad", "pad", tag="pad")
        eng.render(1000)
        eng.release("pad", seconds=0.05)
        for _ in range(4):
            eng.render(1000)
        assert not eng.has("pad") and not any(v.tag == "pad" for v in eng.voices)


def _eeg(seconds: float, alpha: float = 5.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    return rng.standard_normal((4, n)) * 8 + alpha * np.sin(2 * np.pi * 10 * t)


def _feed(features: LiveFeatures, data: np.ndarray, hop: int = 64):
    states = []
    for start in range(0, data.shape[1] - hop + 1, hop):
        for i, ch in enumerate(CHANNEL_NAMES):
            features.push(ch, data[i, start:start + hop], np.ones(hop, bool))
        states.append(features.update())
    return states


class TestLiveFeatures:
    def test_calibrates_then_tracks_alpha(self):
        f = LiveFeatures(calibration=10)
        states = _feed(f, _eeg(15, alpha=5))
        assert states[-1].calibrated
        assert abs(states[-1].levels["Alpha"] - 0.5) < 0.15
        states = _feed(f, _eeg(8, alpha=25, seed=1))  # eyes closed
        assert states[-1].levels["Alpha"] > 0.8

    def test_blink_on_both_frontal_channels(self):
        f = LiveFeatures(calibration=10)
        _feed(f, _eeg(15))
        blink = _eeg(1, seed=2)
        bump = 150 * np.exp(-0.5 * ((np.arange(FS) - 128) / 20) ** 2)
        blink[1] += bump
        blink[2] += bump
        _feed(f, blink)
        assert list(f.events) == ["blink"]

    def test_artifacts_do_not_move_levels(self):
        f = LiveFeatures(calibration=10)
        _feed(f, _eeg(15))
        noisy = _eeg(5, seed=3)
        noisy[:, ::64] += 400  # spikes everywhere
        state = _feed(f, noisy)[-1]
        assert state.quality < 0.2


def test_render_recording_end_to_end(tmp_path):
    rec = Recording()
    data = _eeg(30)
    for i, ch in enumerate(CHANNEL_NAMES):
        rec.append(ch, data[i], np.ones(data.shape[1], bool))
    out = render_recording(rec, tmp_path / "music.wav")
    from scipy.io import wavfile
    sr, audio = wavfile.read(out)
    assert sr == 48000 and audio.shape[1] == 2
    assert abs(len(audio) / sr - 30) < 0.5
    assert np.abs(audio).max() > 1000
