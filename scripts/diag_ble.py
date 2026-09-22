#!/usr/bin/env python3
"""Measure BLE packet loss from the Muse using the per-packet sequence number.

Every Muse EEG notification starts with a 16-bit packet counter; gaps in it
are packets the Pi never received. Loss doesn't depend on signal quality, so
the headband doesn't need to be worn.

    python scripts/diag_ble.py [seconds]
"""

import asyncio
import sys
import time

from thebox.ble.connection import MuseConnection
from thebox.config import TheBoxConfig


async def main(seconds: float) -> None:
    cfg = TheBoxConfig()
    conn = MuseConnection(cfg.device_name, max_retries=5)
    arrivals: list[float] = []
    conn.on_eeg(lambda *_: arrivals.append(time.monotonic()))

    await conn.connect()
    t0 = time.monotonic()
    try:
        await asyncio.sleep(seconds)
    finally:
        elapsed = time.monotonic() - t0
        await conn.disconnect()

    a = conn.aligner
    print(f"\n{elapsed:.1f}s streamed, expected ~{256 / 12 * elapsed:.0f} packets per channel\n")
    for ch in a.channels:
        total = a.received[ch] + a.lost[ch]
        print(f"  {ch:5s} received {a.received[ch]:5d} / {total:5d}  "
              f"({100 * a.loss_fraction(ch):5.1f}% lost)")
    print(f"\n  overall loss: {100 * a.loss_fraction():.1f}%   resyncs: {a.resyncs}")

    if len(arrivals) > 1:
        gaps = sorted(b - a for a, b in zip(arrivals, arrivals[1:]))
        print(f"  notifications/s: {len(arrivals) / elapsed:.1f} (lossless would be ~85)")
        print(f"  longest silences (ms): {[round(g * 1000) for g in gaps[-5:]]}")


if __name__ == "__main__":
    asyncio.run(main(float(sys.argv[1]) if len(sys.argv) > 1 else 15))
