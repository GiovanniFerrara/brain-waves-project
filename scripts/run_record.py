#!/usr/bin/env python3
"""Record EEG, save it to output/rec_<timestamp>.npz and plot a session report.

    python scripts/run_record.py [seconds]   # default 60

Re-plot a saved session later (on any machine) with scripts/plot_recording.py.
"""

import asyncio
import sys
from datetime import datetime

from thebox.ble.connection import MuseConnection
from thebox.config import TheBoxConfig
from thebox.eeg.recording import Recording
from thebox.report import WIN_SECONDS, SessionReport

DEFAULT_SECONDS = 60


async def record(config: TheBoxConfig, seconds: int) -> Recording:
    rec = Recording()
    conn = MuseConnection(
        config.device_name,
        scan_timeout=config.scan_timeout,
        connect_timeout=config.connect_timeout,
        max_retries=config.max_retries,
        retry_delay=config.retry_delay,
    )
    conn.on_eeg(rec.append)
    stopped = asyncio.Event()
    conn.on_disconnect(stopped.set)

    await conn.connect()
    print(f"\nRecording {seconds}s of EEG... sit still and relax.\n")
    try:
        for i in range(seconds, 0, -1):
            loss = 100 * conn.aligner.loss_fraction()
            print(f"  {i:3d}s remaining  (packets lost so far: {loss:4.1f}%)", end="\r")
            try:
                await asyncio.wait_for(stopped.wait(), timeout=1)
                print("\n  Connection lost — keeping what was recorded.")
                break
            except asyncio.TimeoutError:
                pass
        else:
            print("  Done!" + " " * 50)
    finally:
        await conn.disconnect()

    rec.meta = {
        "device": config.device_name,
        "received": conn.aligner.received,
        "lost": conn.aligner.lost,
    }
    return rec


if __name__ == "__main__":
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SECONDS
    rec = asyncio.run(record(TheBoxConfig(), seconds))
    if rec.n_samples == 0:
        sys.exit("No data received.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = rec.save(f"output/rec_{stamp}.npz")
    print(f"\nSaved {path} ({rec.duration:.1f}s)")
    if rec.duration < 2 * WIN_SECONDS:
        sys.exit("Too short to analyse.")

    report = SessionReport(rec)
    print(report.summary())
    for p in report.save_plots(f"output/rec_{stamp}"):
        print(f"Saved {p}")
