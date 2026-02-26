#!/usr/bin/env python3
"""Connectivity test — connect to Muse 2 and stream 5s of EEG.

Migrated from test_muse.py. Run with: python tests/test_connectivity.py
"""

import asyncio

from thebox.ble.connection import MuseConnection
from thebox.ble.protocol import CHANNEL_NAMES
from thebox.eeg.stream import EEGStream


async def main():
    stream = EEGStream(duration=10.0)
    sample_count = 0

    def on_eeg(channel: str, samples: list[float], ts: float) -> None:
        nonlocal sample_count
        sample_count += len(samples)
        avg = sum(samples) / len(samples) if samples else 0
        print(f"  {channel}: avg={avg:7.2f} µV  ({len(samples)} samples)")

    conn = MuseConnection("Muse-31A9")
    conn.on_eeg(on_eeg)

    await conn.connect()
    print("\nStreaming EEG for 5 seconds...\n")
    await asyncio.sleep(5)
    await conn.disconnect()

    print(f"\nDone! Received ~{sample_count} total samples across all channels.")


if __name__ == "__main__":
    asyncio.run(main())
