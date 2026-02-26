"""Pipeline — wires BLE → EEG → Events → Sound → Output."""

from __future__ import annotations

import asyncio
import time

import numpy as np

from .ble.connection import MuseConnection
from .config import TheBoxConfig
from .eeg.bands import ALL_BANDS, compute_band_powers, normalize_band_powers
from .eeg.stream import EEGStream
from .events.alpha import AlphaBurstDetector
from .events.base import Event, EventType
from .events.blink import BlinkDetector
from .events.bus import EventBus
from .events.clench import ClenchDetector
from .sound.base import SoundParameters
from .sound.mixer import Mixer
from .sound.noise import NoiseSource
from .sound.oscillator import OscillatorSource
from .sound.output import AudioOutput


class Pipeline:
    """Orchestrates the full EEG-to-sound pipeline.

    Usage::

        config = TheBoxConfig()
        pipeline = Pipeline(config)
        await pipeline.start()  # blocks until Ctrl+C
    """

    def __init__(self, config: TheBoxConfig | None = None):
        self.config = config or TheBoxConfig()
        c = self.config

        # BLE + EEG
        self.connection = MuseConnection(
            c.device_name,
            scan_timeout=c.scan_timeout,
            connect_timeout=c.connect_timeout,
            max_retries=c.max_retries,
            retry_delay=c.retry_delay,
        )
        self.stream = EEGStream(duration=c.eeg_buffer_seconds)

        # Events
        self.bus = EventBus()
        self.detectors = [
            BlinkDetector(c.blink_threshold, c.blink_window, c.blink_debounce),
            ClenchDetector(c.clench_threshold, c.clench_window, c.clench_debounce),
            AlphaBurstDetector(c.alpha_burst_ratio, c.alpha_baseline_seconds),
        ]

        # Sound
        self.params = SoundParameters()
        self.mixer = Mixer(master_volume=c.master_volume)
        self.mixer.add_source(OscillatorSource(c.audio_sample_rate), gain=0.7)
        self.mixer.add_source(NoiseSource(c.audio_sample_rate), gain=0.3)
        self.output = AudioOutput(
            sample_rate=c.audio_sample_rate,
            block_size=c.audio_block_size,
            channels=c.audio_channels,
        )

        self._running = False

    def _eeg_callback(
        self, channel: str, samples: list[float], timestamp: float
    ) -> None:
        """Called by MuseConnection for each decoded EEG packet."""
        self.stream.append(channel, samples)

    def _handle_event(self, event: Event) -> None:
        """Update SoundParameters in response to detected events."""
        if event.type == EventType.BLINK:
            self.params.blink_trigger = 1.0
        elif event.type == EventType.CLENCH:
            self.params.clench_trigger = 1.0
        elif event.type == EventType.ALPHA_BURST_START:
            self.params.alpha_state = True
        elif event.type == EventType.ALPHA_BURST_END:
            self.params.alpha_state = False

    def _update_sound_params(self) -> None:
        """Map EEG band powers to sound parameters."""
        c = self.config

        # Use AF7 as representative channel (frontal)
        window = self.stream.get_window("AF7", seconds=2.0)
        if len(window) < c.sample_rate:
            return

        powers = compute_band_powers(window)
        norm = normalize_band_powers(powers)

        self.params.alpha = norm.get("Alpha", 0.0)
        self.params.beta = norm.get("Beta", 0.0)
        self.params.theta = norm.get("Theta", 0.0)
        self.params.delta = norm.get("Delta", 0.0)
        self.params.gamma = norm.get("Gamma", 0.0)

        # Map to sound parameters per the mapping table
        self.params.amplitude = np.clip(0.1 + self.params.alpha * 0.8, 0.05, 0.9)

        beta_alpha = self.params.beta / max(self.params.alpha, 0.01)
        freq_lo, freq_hi = c.base_frequency_range
        self.params.base_frequency = freq_lo + np.clip(
            beta_alpha / 3.0, 0.0, 1.0
        ) * (freq_hi - freq_lo)

        self.params.brightness = np.clip(1.0 - self.params.theta * 2.0, 0.0, 1.0)
        self.params.noise_gain = np.clip(self.params.gamma * 3.0, 0.0, 1.0)

    def _decay_triggers(self, dt: float) -> None:
        """Decay event triggers over time."""
        c = self.config
        if self.params.blink_trigger > 0:
            self.params.blink_trigger *= max(0, 1.0 - dt / c.blink_decay)
            if self.params.blink_trigger < 0.01:
                self.params.blink_trigger = 0.0

        if self.params.clench_trigger > 0:
            self.params.clench_trigger *= max(0, 1.0 - dt / c.clench_decay)
            if self.params.clench_trigger < 0.01:
                self.params.clench_trigger = 0.0

    async def start(self) -> None:
        """Connect to Muse and run the processing loop until cancelled."""
        self.connection.on_eeg(self._eeg_callback)
        self.bus.subscribe(None, self._handle_event)

        await self.connection.connect()
        self.output.start()
        self._running = True

        print("Pipeline running. Press Ctrl+C to stop.")

        try:
            last_time = time.monotonic()
            while self._running:
                now = time.monotonic()
                dt = now - last_time
                last_time = now

                # 1. Run event detectors
                for detector in self.detectors:
                    for event in detector.detect(self.stream, now):
                        self.bus.publish(event)

                # 2. Update sound parameters from EEG
                self._update_sound_params()

                # 3. Decay event triggers
                self._decay_triggers(dt)

                # 4. Generate and output audio
                audio = self.mixer.generate(
                    self.params, self.config.audio_block_size
                )
                self.output.write(audio)

                await asyncio.sleep(self.config.process_interval)

        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Gracefully shut down all components."""
        self._running = False
        self.output.stop()
        await self.connection.disconnect()
        print("Pipeline stopped.")
