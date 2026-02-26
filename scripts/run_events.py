#!/usr/bin/env python3
"""Connect to Muse and print detected events in real-time (for tuning thresholds)."""

import asyncio
import signal
import time

from thebox.ble.connection import MuseConnection
from thebox.config import TheBoxConfig
from thebox.eeg.stream import EEGStream
from thebox.events.alpha import AlphaBurstDetector
from thebox.events.base import Event
from thebox.events.blink import BlinkDetector
from thebox.events.bus import EventBus
from thebox.events.clench import ClenchDetector


async def main():
    config = TheBoxConfig()
    stream = EEGStream(duration=config.eeg_buffer_seconds)
    bus = EventBus()

    detectors = [
        BlinkDetector(config.blink_threshold, config.blink_window, config.blink_debounce),
        ClenchDetector(config.clench_threshold, config.clench_window, config.clench_debounce),
        AlphaBurstDetector(config.alpha_burst_ratio, config.alpha_baseline_seconds),
    ]

    def print_event(event: Event) -> None:
        print(f"  [{time.strftime('%H:%M:%S')}] {event}")

    bus.subscribe(None, print_event)

    conn = MuseConnection(config.device_name)
    conn.on_eeg(lambda ch, samples, ts: stream.append(ch, samples))

    await conn.connect()
    print("\nListening for events... Press Ctrl+C to stop.\n")

    running = True

    def stop():
        nonlocal running
        running = False

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, stop)

    try:
        while running:
            now = time.monotonic()
            for detector in detectors:
                for event in detector.detect(stream, now):
                    bus.publish(event)
            await asyncio.sleep(config.process_interval)
    finally:
        await conn.disconnect()
        print("Stopped.")


if __name__ == "__main__":
    asyncio.run(main())
