#!/usr/bin/env python3
"""Quick EEG sonification — single-file, no pipeline overhead."""

import asyncio
import subprocess
import time

import numpy as np
import sounddevice as sd
from bleak import BleakClient, BleakScanner
from scipy.signal import butter, sosfilt

MUSE_NAME = "Muse-31A9"
SAMPLE_RATE = 256
AUDIO_SR = 44100
BLOCK_MS = 50
BLOCK_SIZE = int(AUDIO_SR * BLOCK_MS / 1000)  # 2205
DURATION = 120

EEG_UUIDS = {
    "TP9":  "273e0003-4c4d-454d-96be-f03bac821358",
    "AF7":  "273e0004-4c4d-454d-96be-f03bac821358",
    "AF8":  "273e0005-4c4d-454d-96be-f03bac821358",
    "TP10": "273e0006-4c4d-454d-96be-f03bac821358",
}
CONTROL_UUID = "273e0001-4c4d-454d-96be-f03bac821358"
CMD_RESUME = bytearray([0x02, 0x64, 0x0a])
CMD_HALT = bytearray([0x02, 0x68, 0x0a])

# --- EEG Ring Buffer ---
BUF_LEN = SAMPLE_RATE * 5
eeg_buf = {ch: np.zeros(BUF_LEN) for ch in EEG_UUIDS}
eeg_pos = {ch: 0 for ch in EEG_UUIDS}
eeg_total = {ch: 0 for ch in EEG_UUIDS}


def decode_eeg(packet: bytearray) -> list[float]:
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


def push_samples(ch, samples):
    buf = eeg_buf[ch]
    pos = eeg_pos[ch]
    for s in samples:
        buf[pos % BUF_LEN] = s
        pos += 1
    eeg_pos[ch] = pos
    eeg_total[ch] += len(samples)


def get_window(ch, seconds):
    n = min(int(SAMPLE_RATE * seconds), eeg_total[ch])
    if n == 0:
        return np.zeros(1)
    pos = eeg_pos[ch]
    return eeg_buf[ch][np.arange(pos - n, pos) % BUF_LEN]


def make_callback(ch):
    def cb(sender, data):
        push_samples(ch, decode_eeg(data))
    return cb


def band_power(data, low, high):
    if len(data) < SAMPLE_RATE:
        return 0.0
    nyq = SAMPLE_RATE / 2
    sos = butter(4, [low / nyq, high / nyq], btype="band", output="sos")
    return float(np.sqrt(np.mean(sosfilt(sos, data) ** 2)))


# --- Audio Synthesis ---
phase = 0.0


def generate_audio(alpha, beta, theta, gamma) -> np.ndarray:
    global phase
    t = np.arange(BLOCK_SIZE) / AUDIO_SR

    # Alpha → pitch (relaxed = lower)
    a_norm = min(1.0, alpha / 15.0)
    b_norm = min(1.0, beta / 15.0)
    th_norm = min(1.0, theta / 10.0)

    freq = 180 - a_norm * 60 + b_norm * 80  # ~120-260 Hz range

    # Phase accumulation
    inc = 2 * np.pi * freq / AUDIO_SR
    phases = phase + np.cumsum(np.full(BLOCK_SIZE, inc))
    phase = phases[-1] % (2 * np.pi)

    # Warm sine + gentle harmonics from beta
    audio = np.sin(phases)
    audio += 0.2 * b_norm * np.sin(2 * phases)
    audio += 0.08 * b_norm * np.sin(3 * phases)

    # Theta → slow pulsing
    audio *= 1.0 - 0.2 * th_norm * np.sin(2 * np.pi * 4 * t)

    # Volume from total power
    total = alpha + beta + theta
    vol = np.clip(total / 40.0, 0.03, 0.15)
    audio *= vol

    # Soft clip
    peak = np.max(np.abs(audio))
    if peak > 0.95:
        audio *= 0.95 / peak

    return audio.astype(np.float32)


# --- Main ---
async def main():
    print(f"Scanning for {MUSE_NAME}...")
    device = await BleakScanner.find_device_by_name(MUSE_NAME, timeout=10)
    if not device:
        print("Muse not found!")
        return

    print(f"Found: {device.name} ({device.address})")
    subprocess.run(["bluetoothctl", "trust", device.address],
                   capture_output=True, timeout=5)

    for attempt in range(3):
        try:
            print(f"Connecting (attempt {attempt + 1}/3)...")
            async with BleakClient(device, timeout=30) as client:
                print(f"Connected: {client.is_connected}")

                for ch, uuid in EEG_UUIDS.items():
                    await client.start_notify(uuid, make_callback(ch))
                await client.write_gatt_char(CONTROL_UUID, CMD_RESUME)

                print(f"\nSonifying for {DURATION}s — Ctrl+C to stop")
                print("  Relax → lower tone | Focus → brighter tone\n")

                await asyncio.sleep(1.5)

                # Start audio stream
                audio_queue = []

                def audio_cb(outdata, frames, time_info, status):
                    if audio_queue:
                        block = audio_queue.pop(0)
                        n = min(len(block), frames)
                        outdata[:n, 0] = block[:n]
                        if n < frames:
                            outdata[n:, 0] = 0.0
                    else:
                        outdata[:, 0] = 0.0

                stream = sd.OutputStream(
                    samplerate=AUDIO_SR, blocksize=BLOCK_SIZE,
                    channels=1, dtype="float32", callback=audio_cb,
                )
                stream.start()

                start = time.monotonic()
                chunk = 0
                try:
                    while time.monotonic() - start < DURATION:
                        af7 = get_window("AF7", 1.0)
                        af8 = get_window("AF8", 1.0)
                        n = min(len(af7), len(af8))
                        avg = (af7[:n] + af8[:n]) / 2

                        alpha = band_power(avg, 8, 13)
                        beta = band_power(avg, 13, 30)
                        theta = band_power(avg, 4, 8)
                        gamma = band_power(avg, 30, 50)

                        audio = generate_audio(alpha, beta, theta, gamma)
                        audio_queue.append(audio)

                        chunk += 1
                        if chunk % 20 == 0:
                            elapsed = time.monotonic() - start
                            print(
                                f"  [{elapsed:5.1f}s] "
                                f"α={alpha:5.1f} β={beta:5.1f} "
                                f"θ={theta:5.1f} γ={gamma:5.1f}"
                            )

                        await asyncio.sleep(BLOCK_MS / 1000)

                except KeyboardInterrupt:
                    print("\nStopping...")

                stream.stop()
                stream.close()

                try:
                    await client.write_gatt_char(CONTROL_UUID, CMD_HALT)
                except Exception:
                    pass

            print("Done!")
            return

        except Exception as e:
            if sum(eeg_total.values()) > SAMPLE_RATE:
                print(f"  Disconnected after streaming. Done!")
                return
            print(f"  Failed: {e}")
            if attempt < 2:
                await asyncio.sleep(2)

    print("All attempts failed.")


if __name__ == "__main__":
    asyncio.run(main())
