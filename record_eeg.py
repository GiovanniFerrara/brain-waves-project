"""Record 20 seconds of Muse 2 EEG and plot raw + filtered brainwaves."""

import asyncio
import subprocess
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from bleak import BleakClient, BleakScanner

MUSE_NAME = "Muse-31A9"
DURATION = 20  # seconds
SAMPLE_RATE = 256  # Muse 2 EEG sample rate in Hz

EEG_UUIDS = [
    "273e0003-4c4d-454d-96be-f03bac821358",  # TP9
    "273e0004-4c4d-454d-96be-f03bac821358",  # AF7
    "273e0005-4c4d-454d-96be-f03bac821358",  # AF8
    "273e0006-4c4d-454d-96be-f03bac821358",  # TP10
]

CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]
CHANNEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# EEG frequency bands
BANDS = {
    "Delta (0.5-4 Hz)":  (0.5, 4),
    "Theta (4-8 Hz)":    (4, 8),
    "Alpha (8-13 Hz)":   (8, 13),
    "Beta (13-30 Hz)":   (13, 30),
    "Gamma (30-50 Hz)":  (30, 50),
}
BAND_COLORS = ["#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]

# Storage for samples
eeg_data = defaultdict(list)


def decode_eeg(packet: bytearray) -> list[float]:
    """Decode a 20-byte Muse EEG packet into 12 samples."""
    bit_buffer = 0
    bit_count = 0
    samples = []
    for byte in packet[2:]:
        bit_buffer = (bit_buffer << 8) | byte
        bit_count += 8
        while bit_count >= 12:
            bit_count -= 12
            raw = (bit_buffer >> bit_count) & 0xFFF
            samples.append(raw * 0.48828125)
    return samples


def make_callback(channel_name):
    def callback(sender, data):
        samples = decode_eeg(data)
        eeg_data[channel_name].extend(samples)
    return callback


def bandpass(data, low, high, order=4):
    """Apply a Butterworth bandpass filter."""
    nyq = SAMPLE_RATE / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)


async def record():
    print(f"Scanning for {MUSE_NAME}...")
    device = await BleakScanner.find_device_by_name(MUSE_NAME, timeout=10)
    if not device:
        print("Muse not found! Make sure it's in pairing mode.")
        return False

    print(f"Found: {device.name} ({device.address})")

    subprocess.run(
        ["bluetoothctl", "trust", device.address],
        capture_output=True, timeout=5,
    )

    for attempt in range(3):
        try:
            print(f"Connecting (attempt {attempt + 1}/3)...")
            async with BleakClient(device, timeout=30) as client:
                print(f"Connected: {client.is_connected}")

                for uuid, name in zip(EEG_UUIDS, CHANNEL_NAMES):
                    await client.start_notify(uuid, make_callback(name))

                control_uuid = "273e0001-4c4d-454d-96be-f03bac821358"
                await client.write_gatt_char(
                    control_uuid, bytearray([0x02, 0x64, 0x0a])
                )

                print(f"\nRecording {DURATION}s of EEG... sit still and relax.\n")
                try:
                    for i in range(DURATION, 0, -1):
                        print(f"  {i:2d}s remaining...", end="\r")
                        await asyncio.sleep(1)
                    print("  Done!              ")
                except Exception:
                    elapsed = DURATION - i
                    print(f"\n  Connection dropped after ~{elapsed}s.")

                # Try clean shutdown, but don't fail if disconnected
                try:
                    await client.write_gatt_char(
                        control_uuid, bytearray([0x02, 0x68, 0x0a])
                    )
                    for uuid in EEG_UUIDS:
                        await client.stop_notify(uuid)
                except Exception:
                    pass

            # If we got any data at all, consider it a success
            total = sum(len(v) for v in eeg_data.values())
            if total > 0:
                return True
            return False
        except Exception as e:
            # Check if we already collected data before the disconnect
            total = sum(len(v) for v in eeg_data.values())
            if total > SAMPLE_RATE:  # at least ~1 second of data
                print(f"  Disconnected, but saved {total} samples.")
                return True
            print(f"  Failed: {e}")
            if attempt < 2:
                print("  Retrying in 2s...")
                await asyncio.sleep(2)

    print("All connection attempts failed.")
    return False


def plot():
    # --- Page 1: Raw vs Filtered ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        "Muse 2 EEG — Raw (gray) vs Filtered 1-50 Hz (color)",
        fontsize=16, fontweight="bold",
    )

    for i, (name, color) in enumerate(zip(CHANNEL_NAMES, CHANNEL_COLORS)):
        raw = np.array(eeg_data[name])
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
    plt.savefig("eeg_filtered.png", dpi=150)
    print("Saved eeg_filtered.png")
    plt.close()

    # --- Page 2: Individual frequency bands (using AF7 as representative) ---
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "Muse 2 EEG Frequency Bands — AF7 (left forehead)",
        fontsize=16, fontweight="bold",
    )

    raw = np.array(eeg_data["AF7"])
    raw = raw - np.mean(raw)
    t = np.arange(len(raw)) / SAMPLE_RATE

    for i, ((band_name, (lo, hi)), bcolor) in enumerate(
        zip(BANDS.items(), BAND_COLORS)
    ):
        filtered = bandpass(raw, lo, hi)
        axes[i].plot(t, filtered, color=bcolor, linewidth=0.5)
        axes[i].set_ylabel("µV", fontsize=10)
        axes[i].set_title(band_name, fontsize=11, loc="left")
        axes[i].set_xlim(0, t[-1])
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)", fontsize=12)
    plt.tight_layout()
    plt.savefig("eeg_bands.png", dpi=150)
    print("Saved eeg_bands.png")
    plt.close()


if __name__ == "__main__":
    success = asyncio.run(record())
    if success:
        total = sum(len(v) for v in eeg_data.values())
        print(f"\nCollected {total} total samples across {len(CHANNEL_NAMES)} channels.")
        plot()
