#!/usr/bin/env python3
"""Test tone to verify sounddevice works on the Pi."""

import time

import numpy as np

from thebox.sound.base import SoundParameters
from thebox.sound.mixer import Mixer
from thebox.sound.noise import NoiseSource
from thebox.sound.oscillator import OscillatorSource
from thebox.sound.output import AudioOutput

SAMPLE_RATE = 44100
BLOCK_SIZE = 2205  # ~50ms


def main():
    print("Testing audio output...")
    print("You should hear a tone sweep with noise texture.\n")

    output = AudioOutput(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    mixer = Mixer(master_volume=0.5)
    mixer.add_source(OscillatorSource(SAMPLE_RATE), gain=0.7)
    mixer.add_source(NoiseSource(SAMPLE_RATE), gain=0.3)

    params = SoundParameters()
    output.start()

    try:
        # Sweep frequency 220 → 440 → 220 over 4 seconds
        for i in range(80):  # 80 blocks × 50ms = 4s
            t = i / 80.0
            params.base_frequency = 220 + 220 * np.sin(t * 2 * np.pi)
            params.amplitude = 0.4
            params.brightness = t  # sine → saw over time
            params.noise_gain = 0.1 + 0.3 * t

            audio = mixer.generate(params, BLOCK_SIZE)
            output.write(audio)
            time.sleep(0.05)

        # Simulate blink
        print("  Blink click...")
        params.blink_trigger = 1.0
        for i in range(10):
            params.blink_trigger *= 0.7
            audio = mixer.generate(params, BLOCK_SIZE)
            output.write(audio)
            time.sleep(0.05)

        # Simulate clench
        print("  Clench burst...")
        params.clench_trigger = 1.0
        for i in range(10):
            params.clench_trigger *= 0.7
            audio = mixer.generate(params, BLOCK_SIZE)
            output.write(audio)
            time.sleep(0.05)

        # Let buffer drain
        time.sleep(0.5)

    finally:
        output.stop()

    print("\nDone! If you heard sound, audio is working.")


if __name__ == "__main__":
    main()
