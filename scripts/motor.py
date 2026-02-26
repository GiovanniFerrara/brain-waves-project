#!/usr/bin/env python3
"""Control a BTS7960 motor driver from the command line.

Usage:
    python motor.py --test                # quick connection test
    python motor.py -d 1d,2l,5d,7l
    python motor.py -d "1d, 2l, 5d, 7l" -s 80
    python motor.py -d 3l,3d --rpwm 18 --lpwm 19

Each token: <seconds><direction>
    d = destra (right / forward)
    l = left (reverse)

Default wiring (matches Pi 5 + BTS7960):
    RPWM  -> GPIO 12  (Pin 32)
    LPWM  -> GPIO 13  (Pin 33)
    R_EN  -> 3.3V     (Pin 1)   — always enabled
    L_EN  -> 3.3V     (Pin 17)  — always enabled
    VCC   -> 5V       (Pin 2)
    GND   -> GND      (Pin 6)
"""

import argparse
import re
import sys
import time

from gpiozero import PWMOutputDevice


def parse_sequence(raw: str) -> list[tuple[float, str]]:
    """Parse '1d,2l,5d,7l' into [(1.0, 'd'), (2.0, 'l'), ...]."""
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    seq = []
    for tok in tokens:
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([dlDL])", tok)
        if not m:
            print(f"ERROR: bad token '{tok}' — expected e.g. 3d or 2.5l")
            sys.exit(1)
        secs = float(m.group(1))
        direction = m.group(2).lower()
        seq.append((secs, direction))
    return seq


def test_connection(rpwm_pin, lpwm_pin):
    """Quick diagnostic: check GPIO pins and pulse the motor both ways."""
    print("=== BTS7960 Connection Test ===")
    print(f"Pins: RPWM=GPIO{rpwm_pin}  LPWM=GPIO{lpwm_pin}")
    print(f"R_EN/L_EN: wired to 3.3V (always on)\n")

    # 1. Open PWM pins
    print("[1/3] Opening GPIO pins...", end=" ", flush=True)
    try:
        rpwm = PWMOutputDevice(rpwm_pin)
        lpwm = PWMOutputDevice(lpwm_pin)
    except Exception as e:
        print(f"FAIL\n  -> {e}")
        sys.exit(1)
    print("OK")

    def stop():
        rpwm.value = 0
        lpwm.value = 0

    try:
        # 2. Pulse RIGHT 0.5s
        print("[2/3] Pulsing RIGHT (0.5s, 50%)...", end=" ", flush=True)
        rpwm.value = 0.5
        time.sleep(0.5)
        stop()
        print("OK — did the motor move?")

        time.sleep(0.3)

        # 3. Pulse LEFT 0.5s
        print("[3/3] Pulsing LEFT  (0.5s, 50%)...", end=" ", flush=True)
        lpwm.value = 0.5
        time.sleep(0.5)
        stop()
        print("OK — did the motor move?")

        print("\nIf nothing moved, check:")
        print("  - External power supply to BTS7960 (B+ / B-)")
        print("  - Motor wires on M+ / M-")
        print(f"  - RPWM wire -> Pi Pin 32 (GPIO{rpwm_pin})")
        print(f"  - LPWM wire -> Pi Pin 33 (GPIO{lpwm_pin})")
        print("  - R_EN -> Pi Pin 1 (3.3V)")
        print("  - L_EN -> Pi Pin 17 (3.3V)")
        print("  - GND shared between Pi and driver")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        stop()
        rpwm.close()
        lpwm.close()


def run(seq, rpwm_pin, lpwm_pin, speed):
    """Execute the motor sequence."""
    duty = speed / 100.0

    rpwm = PWMOutputDevice(rpwm_pin)
    lpwm = PWMOutputDevice(lpwm_pin)

    def stop():
        rpwm.value = 0
        lpwm.value = 0

    try:
        for secs, direction in seq:
            label = "RIGHT" if direction == "d" else "LEFT"
            print(f"  {label} for {secs}s at {speed}%")

            stop()
            if direction == "d":
                rpwm.value = duty
            else:
                lpwm.value = duty

            time.sleep(secs)

        stop()
        print("Done.")
    except KeyboardInterrupt:
        print("\nInterrupted — stopping motor.")
    finally:
        stop()
        rpwm.close()
        lpwm.close()


def main():
    ap = argparse.ArgumentParser(description="BTS7960 motor sequence driver")
    ap.add_argument("-d", "--drive",
                    help="Comma-separated sequence, e.g. 1d,2l,5d,7l")
    ap.add_argument("-t", "--test", action="store_true",
                    help="Run connection test (short pulse both directions)")
    ap.add_argument("-s", "--speed", type=int, default=100,
                    help="PWM duty cycle 1-100 (default: 100)")
    ap.add_argument("--rpwm", type=int, default=12, help="RPWM GPIO pin (default: 12)")
    ap.add_argument("--lpwm", type=int, default=13, help="LPWM GPIO pin (default: 13)")
    args = ap.parse_args()

    if args.test:
        test_connection(args.rpwm, args.lpwm)
        return

    if not args.drive:
        ap.error("one of --drive or --test is required")

    if not 1 <= args.speed <= 100:
        print("ERROR: speed must be 1-100")
        sys.exit(1)

    seq = parse_sequence(args.drive)

    total = sum(s for s, _ in seq)
    print(f"Sequence: {len(seq)} steps, {total}s total, speed={args.speed}%")
    print(f"Pins: RPWM=GPIO{args.rpwm} LPWM=GPIO{args.lpwm}")

    run(seq, args.rpwm, args.lpwm, args.speed)


if __name__ == "__main__":
    main()
