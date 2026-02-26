"""Unit tests for audio generation."""

import numpy as np
import pytest

from thebox.sound.base import SoundParameters
from thebox.sound.mixer import Mixer
from thebox.sound.noise import NoiseSource
from thebox.sound.oscillator import OscillatorSource


class TestOscillatorSource:
    def test_generates_correct_length(self):
        osc = OscillatorSource(sample_rate=44100)
        params = SoundParameters(base_frequency=440.0, amplitude=0.5)
        audio = osc.generate(params, 2205)
        assert len(audio) == 2205

    def test_amplitude_scaling(self):
        osc = OscillatorSource(sample_rate=44100)
        params = SoundParameters(base_frequency=440.0, amplitude=0.5, brightness=0.0)
        audio = osc.generate(params, 44100)
        assert np.max(np.abs(audio)) <= 0.55  # sine * 0.5 with small tolerance

    def test_zero_amplitude(self):
        osc = OscillatorSource(sample_rate=44100)
        params = SoundParameters(base_frequency=440.0, amplitude=0.0, brightness=0.0)
        audio = osc.generate(params, 2205)
        assert np.max(np.abs(audio)) < 0.001

    def test_phase_continuity(self):
        """Consecutive blocks should not have discontinuities."""
        osc = OscillatorSource(sample_rate=44100)
        params = SoundParameters(base_frequency=440.0, amplitude=0.5, brightness=0.0)
        block1 = osc.generate(params, 1000)
        block2 = osc.generate(params, 1000)
        # Check the junction between blocks
        diff = abs(block2[0] - block1[-1])
        # Should be small — consecutive samples of a sine at 440 Hz
        max_step = 0.5 * 2 * np.pi * 440 / 44100  # max derivative * dt
        assert diff < max_step + 0.01

    def test_blink_trigger_adds_click(self):
        osc = OscillatorSource(sample_rate=44100)
        params_no_blink = SoundParameters(base_frequency=440.0, amplitude=0.3, blink_trigger=0.0)
        params_blink = SoundParameters(base_frequency=440.0, amplitude=0.3, blink_trigger=1.0)

        osc_clean = OscillatorSource(sample_rate=44100)
        audio_no = osc_clean.generate(params_no_blink, 2205)

        osc_click = OscillatorSource(sample_rate=44100)
        audio_yes = osc_click.generate(params_blink, 2205)

        # Blink should make the start of the buffer louder
        assert np.max(np.abs(audio_yes[:100])) > np.max(np.abs(audio_no[:100]))


class TestNoiseSource:
    def test_generates_correct_length(self):
        noise = NoiseSource(sample_rate=44100)
        params = SoundParameters(noise_gain=0.5)
        audio = noise.generate(params, 2205)
        assert len(audio) == 2205

    def test_low_gain_is_quiet(self):
        noise = NoiseSource(sample_rate=44100)
        params = SoundParameters(noise_gain=0.01)
        audio = noise.generate(params, 44100)
        assert np.std(audio) < 0.05

    def test_clench_trigger_increases_level(self):
        noise1 = NoiseSource(sample_rate=44100)
        noise2 = NoiseSource(sample_rate=44100)
        params_no = SoundParameters(noise_gain=0.1, clench_trigger=0.0)
        params_yes = SoundParameters(noise_gain=0.1, clench_trigger=1.0)
        audio_no = noise1.generate(params_no, 2205)
        audio_yes = noise2.generate(params_yes, 2205)
        assert np.std(audio_yes) > np.std(audio_no)


class TestMixer:
    def test_generates_correct_length(self):
        mixer = Mixer(master_volume=0.5)
        mixer.add_source(OscillatorSource(), gain=1.0)
        params = SoundParameters(base_frequency=440.0, amplitude=0.5)
        audio = mixer.generate(params, 2205)
        assert len(audio) == 2205

    def test_output_bounded(self):
        """Mixer output should be in [-1, 1] due to tanh soft clipping."""
        mixer = Mixer(master_volume=1.0)
        mixer.add_source(OscillatorSource(), gain=5.0)
        mixer.add_source(NoiseSource(), gain=5.0)
        params = SoundParameters(
            base_frequency=440.0, amplitude=1.0, noise_gain=1.0
        )
        audio = mixer.generate(params, 44100)
        assert np.all(audio >= -1.0)
        assert np.all(audio <= 1.0)

    def test_empty_mixer(self):
        mixer = Mixer()
        params = SoundParameters()
        audio = mixer.generate(params, 2205)
        assert np.all(audio == 0.0)

    def test_multiple_sources_sum(self):
        mixer = Mixer(master_volume=0.1)
        mixer.add_source(OscillatorSource(), gain=0.5)
        mixer.add_source(OscillatorSource(), gain=0.5)
        params = SoundParameters(base_frequency=440.0, amplitude=0.5, brightness=0.0)
        audio = mixer.generate(params, 2205)
        # Should be non-silent
        assert np.max(np.abs(audio)) > 0.01


class TestEEGStream:
    """Test the ring buffer separately since sound tests depend on it."""

    def test_ring_buffer_wrap(self):
        from thebox.eeg.stream import EEGStream
        stream = EEGStream(duration=1.0)  # 256 samples capacity
        # Write more than capacity
        for i in range(30):
            stream.append("AF7", [float(i)] * 12)

        window = stream.get_window("AF7", seconds=0.5)
        assert len(window) == 128
        # Should have recent values, not early ones
        assert window[-1] == 29.0
