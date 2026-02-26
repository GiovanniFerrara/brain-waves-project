#!/usr/bin/env python3
"""Recording test — record 20s of EEG and generate plots.

Migrated from record_eeg.py. Run with: python tests/test_recording.py
"""

import asyncio
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thebox.ble.connection import MuseConnection
from thebox.ble.protocol import CHANNEL_NAMES, SAMPLE_RATE
from thebox.eeg.bands import ALL_BANDS, BAND_COLORS
from thebox.eeg.filters import bandpass
from thebox.eeg.stream import EEGStream

CHANNEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
DURATION = 20  # seconds


async def record() -> EEGStream:
    stream = EEGStream(duration=float(DURATION))
    conn = MuseConnection("Muse-31A9")
    conn.on_eeg(lambda ch, samples, ts: stream.append(ch, samples))

    try:
        await conn.connect()
        print(f"\nRecording {DURATION}s of EEG... sit still and relax.\n")
        for i in range(DURATION, 0, -1):
            print(f"  {i:2d}s remaining...", end="\r")
            await asyncio.sleep(1)
        print("  Done!              ")
    except Exception:
        total = stream.total_samples()
        if total > SAMPLE_RATE:
            print(f"\n  Disconnected, but saved {total} samples.")
        else:
            raise
    finally:
        await conn.disconnect()

    return stream


def plot(stream: EEGStream) -> None:
    os.makedirs("output", exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        "Muse 2 EEG — Raw (gray) vs Filtered 1-50 Hz (color)",
        fontsize=16, fontweight="bold",
    )

    for i, (name, color) in enumerate(zip(CHANNEL_NAMES, CHANNEL_COLORS)):
        raw = stream.get_window(name)
        n = len(raw)
        if n < SAMPLE_RATE:
            continue
        t = np.arange(n) / SAMPLE_RATE
        raw = raw - np.mean(raw)
        filtered = bandpass(raw, 1, 50)

        axes[i].plot(t, raw, color="lightgray", linewidth=0.3, label="Raw")
        axes[i].plot(t, filtered, color=color, linewidth=0.5, alpha=0.9, label="Filtered")
        axes[i].set_ylabel(f"{name}\n(µV)", fontsize=11)
        axes[i].set_xlim(0, t[-1])
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (seconds)", fontsize=12)
    plt.tight_layout()
    plt.savefig("output/eeg_filtered.png", dpi=150)
    print("Saved output/eeg_filtered.png")
    plt.close()

    # Frequency bands (AF7)
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "Muse 2 EEG Frequency Bands — AF7 (left forehead)",
        fontsize=16, fontweight="bold",
    )

    raw = stream.get_window("AF7")
    raw = raw - np.mean(raw)
    t = np.arange(len(raw)) / SAMPLE_RATE

    for i, (band, bcolor) in enumerate(zip(ALL_BANDS, BAND_COLORS)):
        filtered = bandpass(raw, band.low, band.high)
        axes[i].plot(t, filtered, color=bcolor, linewidth=0.5)
        axes[i].set_ylabel("µV", fontsize=10)
        axes[i].set_title(
            f"{band.name} ({band.low}-{int(band.high)} Hz)",
            fontsize=11, loc="left",
        )
        axes[i].set_xlim(0, t[-1])
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)", fontsize=12)
    plt.tight_layout()
    plt.savefig("output/eeg_bands.png", dpi=150)
    print("Saved output/eeg_bands.png")
    plt.close()


if __name__ == "__main__":
    eeg_stream = asyncio.run(record())
    total = eeg_stream.total_samples()
    if total > 0:
        print(f"\nCollected {total} total samples across {len(CHANNEL_NAMES)} channels.")
        plot(eeg_stream)
