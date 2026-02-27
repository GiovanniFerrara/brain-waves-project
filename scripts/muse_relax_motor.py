#!/usr/bin/env python3
"""Control motor speed with your relaxation level via Muse 2.

Uses theta/beta ratio on TP9/TP10 (temporal electrodes, away from eye artifacts).
  - Relaxed/drowsy → high theta, low beta  → ratio up → motor slower
  - Alert/focused  → low theta, high beta  → ratio down → motor faster

Artifacts (blinks, jaw clenches) are rejected by clamping outlier ratios.
Logs all EEG values to output/relax_motor_YYYYMMDD_HHMMSS.csv for tuning.
Generates a session plot (PNG) automatically on exit.

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
from datetime import datetime

import numpy as np
from bleak import BleakClient, BleakScanner
from gpiozero import PWMOutputDevice
from scipy.signal import butter, sosfilt

# --- Muse BLE ---
MUSE_NAME = "Muse-31A9"
SAMPLE_RATE = 256

# Use ALL 4 channels: temporal for relaxation, frontal for artifact rejection
EEG_UUIDS = {
    "TP9":  "273e0003-4c4d-454d-96be-f03bac821358",
    "AF7":  "273e0004-4c4d-454d-96be-f03bac821358",
    "AF8":  "273e0005-4c4d-454d-96be-f03bac821358",
    "TP10": "273e0006-4c4d-454d-96be-f03bac821358",
}
CONTROL_UUID = "273e0001-4c4d-454d-96be-f03bac821358"
CMD_RESUME = bytearray([0x02, 0x64, 0x0A])
CMD_HALT = bytearray([0x02, 0x68, 0x0A])

# --- EEG Ring Buffer ---
BUF_LEN = SAMPLE_RATE * 10  # 10 seconds
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
    lo = max(low / nyq, 0.001)
    hi = min(high / nyq, 0.999)
    sos = butter(4, [lo, hi], btype="band", output="sos")
    return float(np.sqrt(np.mean(sosfilt(sos, data) ** 2)))


def is_artifact(af7: np.ndarray, af8: np.ndarray, threshold: float = 150.0) -> bool:
    """Reject windows with blink/clench artifacts on frontal channels."""
    pp7 = np.max(af7) - np.min(af7) if len(af7) > 1 else 0
    pp8 = np.max(af8) - np.min(af8) if len(af8) > 1 else 0
    return max(pp7, pp8) > threshold


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

    # Setup CSV log with timestamp
    os.makedirs("output", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"output/relax_motor_{ts}.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "time_s", "theta_tp", "beta_tp", "alpha_tp",
        "artifact", "ratio", "ratio_p10", "ratio_p90",
        "pct", "smoothed", "motor_pct",
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

            print("EEG streaming. Calibrating (6s — stay neutral, eyes open)...")
            await asyncio.sleep(6)

            # Seed ratio history from calibration period
            ratio_history = deque(maxlen=300)  # ~30s at 10Hz
            for _ in range(30):
                tp9 = get_window("TP9", 2.0)
                tp10 = get_window("TP10", 2.0)
                n = min(len(tp9), len(tp10))
                avg = (tp9[:n] + tp10[:n]) / 2
                theta = band_power(avg, 4, 8)
                beta = band_power(avg, 13, 30)
                ratio_history.append(theta / max(beta, 0.1))
                await asyncio.sleep(0.05)

            baseline = np.median(list(ratio_history))
            print(f"Baseline θ/β ratio: {baseline:.2f}")

            print(f"\nMotor speed reflects mental activity (max {max_speed}%).")
            print("  Alert, eyes open  → faster")
            print("  Relaxed, eyes closed → slower")
            print("  Press Ctrl+C to stop\n")

            smoothed = 0.5
            start_time = time.monotonic()
            artifact_count = 0
            sample_count = 0
            prev_ratio = None
            stale_count = 0
            STALE_LIMIT = 15  # ~1.5s of identical readings = disconnected

            try:
                while True:
                    # Use 2s window on temporal channels (away from eyes)
                    tp9 = get_window("TP9", 2.0)
                    tp10 = get_window("TP10", 2.0)
                    n_tp = min(len(tp9), len(tp10))
                    if n_tp < SAMPLE_RATE:
                        await asyncio.sleep(0.1)
                        continue

                    avg_tp = (tp9[:n_tp] + tp10[:n_tp]) / 2

                    # Check frontal channels for artifact rejection
                    af7 = get_window("AF7", 0.5)
                    af8 = get_window("AF8", 0.5)
                    artf = is_artifact(af7, af8)

                    theta = band_power(avg_tp, 4, 8)
                    beta = band_power(avg_tp, 13, 30)
                    alpha = band_power(avg_tp, 8, 13)

                    ratio = theta / max(beta, 0.1)

                    # Detect stale data (Muse disconnected)
                    if prev_ratio is not None and abs(ratio - prev_ratio) < 1e-6:
                        stale_count += 1
                        if stale_count >= STALE_LIMIT:
                            print(f"\n\nMuse disconnected (stale data for {stale_count} cycles). Stopping motor.")
                            set_motor(0)
                            break
                    else:
                        stale_count = 0
                    prev_ratio = ratio

                    sample_count += 1

                    if artf:
                        # Skip artifact — hold current motor speed
                        artifact_count += 1
                        r_min = r_max = pct = 0.0
                    else:
                        # Clamp outliers before adding to history
                        hist = np.array(ratio_history) if ratio_history else np.array([ratio])
                        median = np.median(hist)
                        iqr = np.percentile(hist, 75) - np.percentile(hist, 25) if len(hist) > 5 else 0.1
                        # Only accept ratios within 3*IQR of median
                        if abs(ratio - median) < 3 * max(iqr, 0.05):
                            ratio_history.append(ratio)

                        hist = np.array(ratio_history)
                        r_min = np.percentile(hist, 5)
                        r_max = np.percentile(hist, 95)
                        spread = max(r_max - r_min, 0.02)

                        # 0 = low theta/beta (alert), 1 = high theta/beta (relaxed)
                        pct = np.clip((ratio - r_min) / spread, 0.0, 1.0)

                        # Smooth — relaxation changes slowly, so use heavy smoothing
                        smoothed = 0.15 * pct + 0.85 * smoothed

                    # Invert: relaxed=slow, active=fast
                    motor_duty = (1.0 - smoothed) * duty_max
                    set_motor(motor_duty)

                    elapsed = time.monotonic() - start_time

                    log_writer.writerow([
                        f"{elapsed:.2f}",
                        f"{theta:.2f}", f"{beta:.2f}", f"{alpha:.2f}",
                        "1" if artf else "0",
                        f"{ratio:.3f}",
                        f"{r_min:.3f}" if not artf else "",
                        f"{r_max:.3f}" if not artf else "",
                        f"{pct:.3f}",
                        f"{smoothed:.3f}",
                        f"{motor_duty * 100:.1f}",
                    ])
                    log_file.flush()

                    bar = "█" * int(smoothed * 20) + "░" * (20 - int(smoothed * 20))
                    art_str = " ARTIFACT" if artf else ""
                    print(
                        f"  θ={theta:5.1f} β={beta:5.1f} α={alpha:5.1f} "
                        f"θ/β={ratio:.2f} "
                        f"relax={smoothed:.2f} "
                        f"[{bar}] "
                        f"motor={motor_duty * 100:.0f}%"
                        f"{art_str}   ",
                        end="\r", flush=True,
                    )

                    await asyncio.sleep(0.1)

            except KeyboardInterrupt:
                pct_art = artifact_count / max(sample_count, 1) * 100
                print(f"\n\nStopping. {sample_count} samples, {artifact_count} artifacts ({pct_art:.0f}%)")

            set_motor(0)
            try:
                await client.write_gatt_char(CONTROL_UUID, CMD_HALT)
            except Exception:
                pass
            await client.disconnect()
            emergency_stop()
            log_file.close()
            print(f"Log saved: {log_path}")

            # Auto-generate session plot
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "plot_session",
                    os.path.join(os.path.dirname(__file__), "plot_session.py"),
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod.plot_session(log_path)
            except Exception as e:
                print(f"Plot generation failed: {e}")

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
