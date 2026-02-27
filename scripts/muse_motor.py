#!/usr/bin/env python3
"""Control a BTS7960 motor with your blinks via Muse 2.

Blink detection uses a z-score peak detector on AF7/AF8 frontal channels.
Eye blinks produce large transient spikes (100-300+ µV) that stand out
clearly from the ~10-30 µV background EEG.

Algorithm:
  1. Maintain a rolling window of peak-to-peak amplitudes (50ms sub-windows)
  2. Compute running mean and std of these amplitudes
  3. When current peak-to-peak exceeds mean + z_threshold * std → blink
  4. Debounce to avoid double-counting a single blink

Motor mapping:
  - Each blink toggles the motor on/off
  - Hold-blink (rapid blinks) increases speed

Usage:
    python scripts/muse_motor.py              # default GPIO 12/13
    python scripts/muse_motor.py --rpwm 18 --lpwm 19
    python scripts/muse_motor.py --test       # motor test only (no Muse)
"""

import argparse
import asyncio
import subprocess
import time
from collections import deque

import numpy as np
from bleak import BleakClient, BleakScanner
from gpiozero import PWMOutputDevice

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
BUF_LEN = SAMPLE_RATE * 5  # 5 seconds
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


# --- Blink Detector (z-score peak detection) ---
class BlinkDetector:
    """Detect blinks using adaptive z-score thresholding on peak-to-peak amplitude.

    Frontal EEG channels (AF7/AF8) pick up eye blink artifacts as sharp
    100-300+ µV spikes. Background EEG is typically 10-30 µV. We track
    peak-to-peak amplitude in short sub-windows and fire when it exceeds
    the running statistics by z_threshold standard deviations.
    """

    def __init__(
        self,
        z_threshold: float = 3.5,
        subwindow_ms: float = 60,
        history_seconds: float = 5.0,
        debounce_seconds: float = 0.35,
        min_warmup_seconds: float = 2.0,
    ):
        self.z_threshold = z_threshold
        self.subwindow_samples = int(SAMPLE_RATE * subwindow_ms / 1000)
        self.history_len = int(history_seconds / (subwindow_ms / 1000))
        self.debounce = debounce_seconds
        self.min_warmup = min_warmup_seconds

        # Rolling history of peak-to-peak amplitudes
        self.pp_history: deque[float] = deque(maxlen=self.history_len)
        self.last_blink_time = 0.0
        self._last_check_pos = {ch: 0 for ch in EEG_UUIDS}

    def _peak_to_peak(self, data: np.ndarray) -> float:
        """Peak-to-peak amplitude of a short window."""
        if len(data) < 3:
            return 0.0
        return float(np.max(data) - np.min(data))

    def check(self, now: float) -> bool:
        """Check for blink. Returns True if blink detected."""
        # Need enough data
        min_samples = int(SAMPLE_RATE * self.min_warmup)
        if any(eeg_total[ch] < min_samples for ch in EEG_UUIDS):
            return False

        # Debounce
        if now - self.last_blink_time < self.debounce:
            return False

        # Get latest sub-window from both frontal channels
        af7 = get_window("AF7", self.subwindow_samples / SAMPLE_RATE)
        af8 = get_window("AF8", self.subwindow_samples / SAMPLE_RATE)

        # Average frontal channels for robustness
        n = min(len(af7), len(af8))
        avg = (af7[:n] + af8[:n]) / 2.0

        pp = self._peak_to_peak(avg)

        # Update history
        self.pp_history.append(pp)

        # Need enough history for statistics
        if len(self.pp_history) < 10:
            return False

        # Compute z-score
        hist = np.array(self.pp_history)
        mean = np.mean(hist[:-1])  # exclude current
        std = np.std(hist[:-1])

        if std < 1.0:  # avoid division by tiny noise floor
            std = 1.0

        z = (pp - mean) / std

        if z > self.z_threshold:
            self.last_blink_time = now
            return True

        return False


# --- Motor Controller ---
class MotorController:
    """BTS7960 motor driver with smooth speed transitions."""

    def __init__(self, rpwm_pin: int = 12, lpwm_pin: int = 13):
        self.rpwm = PWMOutputDevice(rpwm_pin)
        self.lpwm = PWMOutputDevice(lpwm_pin)
        self._speed = 0.0
        self._target = 0.0
        self._direction = "r"  # r=right, l=left
        self._running = False

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def running(self) -> bool:
        return self._running

    def set_speed(self, speed: float, direction: str = "r"):
        """Set target speed (0.0-1.0) and direction."""
        self._target = max(0.0, min(1.0, speed))
        self._direction = direction

    def toggle(self, speed: float = 0.6, direction: str = "r"):
        """Toggle motor on/off."""
        if self._running:
            self._target = 0.0
            self._running = False
        else:
            self._target = speed
            self._direction = direction
            self._running = True

    def pulse(self, speed: float = 0.7, duration: float = 0.5, direction: str = "r"):
        """Immediate pulse — ramps up then schedules ramp down."""
        self._target = speed
        self._direction = direction
        self._running = True
        self._pulse_end = time.monotonic() + duration

    def update(self, dt: float):
        """Call each loop iteration to smooth speed transitions."""
        # Check pulse timeout
        if hasattr(self, '_pulse_end') and time.monotonic() > self._pulse_end:
            self._target = 0.0
            self._running = False
            del self._pulse_end

        # Smooth ramp (20% per step toward target)
        ramp_rate = 5.0  # per second
        diff = self._target - self._speed
        self._speed += np.clip(diff, -ramp_rate * dt, ramp_rate * dt)

        if self._speed < 0.01:
            self._speed = 0.0

        # Apply to GPIO
        self.rpwm.value = 0
        self.lpwm.value = 0
        if self._speed > 0:
            if self._direction == "r":
                self.rpwm.value = self._speed
            else:
                self.lpwm.value = self._speed

    def stop(self):
        self.rpwm.value = 0
        self.lpwm.value = 0
        self._speed = 0.0
        self._target = 0.0

    def close(self):
        self.stop()
        self.rpwm.close()
        self.lpwm.close()


# --- Main ---
async def run(rpwm_pin: int, lpwm_pin: int):
    print(f"Scanning for {MUSE_NAME}...")
    device = await BleakScanner.find_device_by_name(MUSE_NAME, timeout=15)
    if not device:
        print("Muse not found! Make sure it's on and in pairing mode.")
        return

    print(f"Found: {device.name} ({device.address})")
    subprocess.run(
        ["bluetoothctl", "trust", device.address],
        capture_output=True, timeout=5,
    )

    motor = MotorController(rpwm_pin, lpwm_pin)
    detector = BlinkDetector(z_threshold=3.5, debounce_seconds=0.35)
    blink_count = 0

    for attempt in range(3):
        try:
            print(f"Connecting ({attempt + 1}/3)...")
            client = BleakClient(device, timeout=30)
            await client.connect()
            print(f"Connected: {client.is_connected}")

            for ch, uuid in EEG_UUIDS.items():
                await client.start_notify(uuid, make_callback(ch))
                print(f"  Subscribed {ch}")
            await client.write_gatt_char(CONTROL_UUID, CMD_RESUME)

            print("\nEEG streaming. Warming up (2s)...")
            await asyncio.sleep(2)

            print("Ready! Blink to control the motor.")
            print("  Single blink  → motor pulse (0.5s)")
            print("  Press Ctrl+C to stop\n")

            last_time = time.monotonic()
            try:
                while True:
                    now = time.monotonic()
                    dt = now - last_time
                    last_time = now

                    if detector.check(now):
                        blink_count += 1
                        print(f"  *BLINK #{blink_count}* — motor pulse!")
                        motor.pulse(speed=0.8, duration=0.1)

                    motor.update(dt)

                    await asyncio.sleep(0.02)  # 50 Hz loop

            except KeyboardInterrupt:
                print(f"\nStopping. Total blinks detected: {blink_count}")

            motor.close()
            try:
                await client.write_gatt_char(CONTROL_UUID, CMD_HALT)
            except Exception:
                pass
            await client.disconnect()
            print("Done!")
            return

        except Exception as e:
            print(f"  Failed: {e}")
            if attempt < 2:
                print("  Retrying in 2s...")
                await asyncio.sleep(2)

    motor.close()
    print("All connection attempts failed.")


def test_motor(rpwm_pin: int, lpwm_pin: int):
    """Quick motor test without Muse."""
    print("=== Motor Test (no Muse needed) ===")
    motor = MotorController(rpwm_pin, lpwm_pin)
    try:
        print("Pulse RIGHT 0.5s at 60%...")
        motor.pulse(speed=0.6, duration=0.5)
        for _ in range(30):
            motor.update(0.02)
            time.sleep(0.02)

        time.sleep(0.3)

        print("Pulse LEFT 0.5s at 60%...")
        motor.pulse(speed=0.6, duration=0.5, direction="l")
        for _ in range(30):
            motor.update(0.02)
            time.sleep(0.02)

        print("Done!")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        motor.close()


def main():
    ap = argparse.ArgumentParser(description="Control motor with Muse 2 blinks")
    ap.add_argument("--test", action="store_true", help="Motor test only (no Muse)")
    ap.add_argument("--rpwm", type=int, default=12, help="RPWM GPIO pin (default: 12)")
    ap.add_argument("--lpwm", type=int, default=13, help="LPWM GPIO pin (default: 13)")
    ap.add_argument("--threshold", type=float, default=3.5,
                    help="Blink z-score threshold (default: 3.5, lower=more sensitive)")
    args = ap.parse_args()

    if args.test:
        test_motor(args.rpwm, args.lpwm)
    else:
        asyncio.run(run(args.rpwm, args.lpwm))


if __name__ == "__main__":
    main()
