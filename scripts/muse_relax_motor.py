#!/usr/bin/env python3
"""Control motor speed with your relaxation level via Muse 2.

Relaxation is measured by the alpha/beta ratio on AF7/AF8.
  - Alert, focused → low alpha, high beta  → faster motor
  - Eyes closed, calm → high alpha, low beta → slower motor

Logs all EEG values to output/relax_motor.csv for tuning.

Usage:
    python scripts/muse_relax_motor.py
    python scripts/muse_relax_motor.py --max-speed 60
    python scripts/muse_relax_motor.py --direction l
"""

import argparse
import asyncio
import atexit
import csv
import os
import signal
import subprocess
import time
from collections import deque

import numpy as np
from bleak import BleakClient, BleakScanner
from gpiozero import PWMOutputDevice
from scipy.signal import butter, sosfilt

# --- Muse BLE ---
MUSE_NAME = "Muse-31A9"
SAMPLE_RATE = 256

EEG_UUIDS = {
    "AF7": "273e0004-4c4d-454d-96be-f03bac821358",
    "AF8": "273e0005-4c4d-454d-96be-f03bac821358",
}
CONTROL_UUID = "273e0001-4c4d-454d-96be-f03bac821358"
CMD_RESUME = bytearray([0x02, 0x64, 0x0A])
CMD_HALT = bytearray([0x02, 0x68, 0x0A])

# --- EEG Ring Buffer ---
BUF_LEN = SAMPLE_RATE * 5
eeg_buf = {ch: np.zeros(BUF_LEN) for ch in EEG_UUIDS}
eeg_pos = {ch: 0 for ch in EEG_UUIDS}
eeg_total = {ch: 0 for ch in EEG_UUIDS}


def decode_eeg(packet: bytearray) -> list[float]:
    bb = bc = 0
    samples = []
    for byte in packet[2:]:
        bb = (bb << 8) | byte
        bc += 8
        while bc >= 12:
            bc -= 12
            samples.append(((bb >> bc) & 0xFFF) * 0.48828125)
    return samples


def push_samples(ch: str, samples: list[float]):
    p = eeg_pos[ch]
    buf = eeg_buf[ch]
    for s in samples:
        buf[p % BUF_LEN] = s
        p += 1
    eeg_pos[ch] = p
    eeg_total[ch] += len(samples)


def get_window(ch: str, seconds: float) -> np.ndarray:
    n = min(int(SAMPLE_RATE * seconds), eeg_total[ch])
    if n == 0:
        return np.zeros(1)
    p = eeg_pos[ch]
    return eeg_buf[ch][np.arange(p - n, p) % BUF_LEN]


def make_callback(ch: str):
    def cb(_sender, data: bytearray):
        push_samples(ch, decode_eeg(data))
    return cb


def band_power(data: np.ndarray, low: float, high: float) -> float:
    if len(data) < SAMPLE_RATE:
        return 0.0
    nyq = SAMPLE_RATE / 2
    sos = butter(4, [low / nyq, high / nyq], btype="band", output="sos")
    return float(np.sqrt(np.mean(sosfilt(sos, data) ** 2)))


# --- Global motor refs for emergency cleanup ---
_rpwm = None
_lpwm = None


def emergency_stop():
    global _rpwm, _lpwm
    for pin in (_rpwm, _lpwm):
        if pin is not None:
            try:
                pin.value = 0
                pin.close()
            except Exception:
                pass
    _rpwm = _lpwm = None


atexit.register(emergency_stop)
signal.signal(signal.SIGTERM, lambda *_: (emergency_stop(), exit(0)))


# --- Main ---
async def run(rpwm_pin: int, lpwm_pin: int, max_speed: float, direction: str):
    global _rpwm, _lpwm

    # Setup CSV log
    os.makedirs("output", exist_ok=True)
    log_path = "output/relax_motor.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "time_s", "alpha", "beta", "theta", "delta", "gamma",
        "ratio", "ratio_min", "ratio_max", "pct", "smoothed", "motor_pct",
    ])
    print(f"Logging to {log_path}")

    print(f"Scanning for {MUSE_NAME}...")
    device = await BleakScanner.find_device_by_name(MUSE_NAME, timeout=15)
    if not device:
        print("Muse not found! Make sure it's on and in pairing mode.")
        log_file.close()
        return

    print(f"Found: {device.name} ({device.address})")
    subprocess.run(
        ["bluetoothctl", "trust", device.address],
        capture_output=True, timeout=5,
    )

    _rpwm = PWMOutputDevice(rpwm_pin)
    _lpwm = PWMOutputDevice(lpwm_pin)
    duty_max = max_speed / 100.0

    def set_motor(speed: float):
        _rpwm.value = 0
        _lpwm.value = 0
        if speed > 0.01:
            if direction == "r":
                _rpwm.value = speed
            else:
                _lpwm.value = speed

    for attempt in range(3):
        try:
            print(f"Connecting ({attempt + 1}/3)...")
            client = BleakClient(device, timeout=30)
            await client.connect()
            print(f"Connected: {client.is_connected}")

            for ch, uuid in EEG_UUIDS.items():
                await client.start_notify(uuid, make_callback(ch))
            await client.write_gatt_char(CONTROL_UUID, CMD_RESUME)

            print("EEG streaming. Calibrating (5s — stay neutral)...")
            await asyncio.sleep(5)

            # Use a rolling min/max window to adaptively scale the ratio
            # Seed with current readings
            ratio_history = deque(maxlen=200)  # ~20s of history at 10Hz
            for _ in range(20):
                af7 = get_window("AF7", 1.0)
                af8 = get_window("AF8", 1.0)
                n = min(len(af7), len(af8))
                avg = (af7[:n] + af8[:n]) / 2
                a = band_power(avg, 8, 13)
                b = band_power(avg, 13, 30)
                ratio_history.append(a / max(b, 0.1))
                await asyncio.sleep(0.05)

            baseline_ratio = np.median(list(ratio_history))
            print(f"Baseline α/β ratio: {baseline_ratio:.2f}")

            print(f"\nMotor speed reflects mental activity (max {max_speed}%).")
            print("  Alert, focused → faster")
            print("  Eyes closed, calm → slower")
            print("  Press Ctrl+C to stop\n")

            smoothed = 0.5
            start_time = time.monotonic()

            try:
                while True:
                    af7 = get_window("AF7", 1.0)
                    af8 = get_window("AF8", 1.0)
                    n = min(len(af7), len(af8))
                    if n < SAMPLE_RATE:
                        await asyncio.sleep(0.1)
                        continue

                    avg = (af7[:n] + af8[:n]) / 2
                    alpha = band_power(avg, 8, 13)
                    beta = band_power(avg, 13, 30)
                    theta = band_power(avg, 4, 8)
                    delta = band_power(avg, 0.5, 4)
                    gamma = band_power(avg, 30, 50)

                    ratio = alpha / max(beta, 0.1)
                    ratio_history.append(ratio)

                    # Adaptive normalization using rolling percentiles
                    hist = np.array(ratio_history)
                    r_min = np.percentile(hist, 10)
                    r_max = np.percentile(hist, 90)
                    spread = r_max - r_min
                    if spread < 0.05:
                        spread = 0.05  # minimum spread to avoid division issues

                    # 0 = low ratio (alert), 1 = high ratio (relaxed)
                    pct = np.clip((ratio - r_min) / spread, 0.0, 1.0)

                    # Smooth with responsive EMA
                    smoothed = 0.3 * pct + 0.7 * smoothed

                    # Invert: relaxed=slow, active=fast
                    motor_duty = (1.0 - smoothed) * duty_max
                    set_motor(motor_duty)

                    elapsed = time.monotonic() - start_time

                    # Log every sample
                    log_writer.writerow([
                        f"{elapsed:.2f}",
                        f"{alpha:.2f}", f"{beta:.2f}", f"{theta:.2f}",
                        f"{delta:.2f}", f"{gamma:.2f}",
                        f"{ratio:.3f}", f"{r_min:.3f}", f"{r_max:.3f}",
                        f"{pct:.3f}", f"{smoothed:.3f}",
                        f"{motor_duty * 100:.1f}",
                    ])
                    log_file.flush()

                    bar = "█" * int(smoothed * 20) + "░" * (20 - int(smoothed * 20))
                    print(
                        f"  α={alpha:5.1f} β={beta:5.1f} "
                        f"ratio={ratio:.2f} [{r_min:.2f}-{r_max:.2f}] "
                        f"relax={smoothed:.2f} "
                        f"[{bar}] "
                        f"motor={motor_duty * 100:.0f}%   ",
                        end="\r", flush=True,
                    )

                    await asyncio.sleep(0.1)

            except KeyboardInterrupt:
                print("\n\nStopping...")

            set_motor(0)
            try:
                await client.write_gatt_char(CONTROL_UUID, CMD_HALT)
            except Exception:
                pass
            await client.disconnect()
            emergency_stop()
            log_file.close()
            print(f"Log saved: {log_path}")
            print("Done!")
            return

        except Exception as e:
            print(f"  Failed: {e}")
            if attempt < 2:
                print("  Retrying in 2s...")
                await asyncio.sleep(2)

    emergency_stop()
    log_file.close()
    print("All connection attempts failed.")


def main():
    ap = argparse.ArgumentParser(description="Motor speed from relaxation level (Muse 2)")
    ap.add_argument("--rpwm", type=int, default=12, help="RPWM GPIO pin (default: 12)")
    ap.add_argument("--lpwm", type=int, default=13, help="LPWM GPIO pin (default: 13)")
    ap.add_argument("--max-speed", type=int, default=80, help="Max motor speed %% (default: 80)")
    ap.add_argument("--direction", choices=["r", "l"], default="r", help="Motor direction (default: r)")
    args = ap.parse_args()
    asyncio.run(run(args.rpwm, args.lpwm, args.max_speed, args.direction))


if __name__ == "__main__":
    main()
