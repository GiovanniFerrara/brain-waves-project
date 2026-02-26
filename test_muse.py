"""Quick test: connect to Muse 2 and log a few seconds of EEG signal."""

import asyncio
import subprocess
from bleak import BleakClient, BleakScanner

MUSE_NAME = "Muse-31A9"

# Muse 2 EEG channel UUIDs (TP9, AF7, AF8, TP10, AUX)
EEG_UUIDS = [
    "273e0003-4c4d-454d-96be-f03bac821358",  # TP9
    "273e0004-4c4d-454d-96be-f03bac821358",  # AF7
    "273e0005-4c4d-454d-96be-f03bac821358",  # AF8
    "273e0006-4c4d-454d-96be-f03bac821358",  # TP10
    "273e0007-4c4d-454d-96be-f03bac821358",  # AUX
]

CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10", "AUX"]

sample_count = 0


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
            samples.append(raw * 0.48828125)  # 2000 / 4096
    return samples


def make_callback(channel_name):
    def callback(sender, data):
        global sample_count
        samples = decode_eeg(data)
        sample_count += len(samples)
        avg = sum(samples) / len(samples) if samples else 0
        print(f"  {channel_name}: avg={avg:7.2f} µV  ({len(samples)} samples)")
    return callback


async def main():
    global sample_count

    # Scan for the device
    print(f"Scanning for {MUSE_NAME}...")
    device = await BleakScanner.find_device_by_name(MUSE_NAME, timeout=10)
    if not device:
        print("Muse not found! Make sure it's in pairing mode.")
        return

    print(f"Found: {device.name} ({device.address})")

    # Trust the device via bluetoothctl to avoid auth issues
    subprocess.run(
        ["bluetoothctl", "trust", device.address],
        capture_output=True, timeout=5
    )
    print("Trusted device in BlueZ")

    # Connect with retries
    for attempt in range(3):
        try:
            print(f"Connecting (attempt {attempt + 1}/3)...")
            async with BleakClient(device, timeout=30) as client:
                print(f"Connected: {client.is_connected}")

                # Subscribe to EEG channels
                for uuid, name in zip(EEG_UUIDS, CHANNEL_NAMES):
                    await client.start_notify(uuid, make_callback(name))
                    print(f"  Subscribed to {name}")

                # Start streaming: send 'd' (resume) to the control channel
                control_uuid = "273e0001-4c4d-454d-96be-f03bac821358"
                await client.write_gatt_char(
                    control_uuid, bytearray([0x02, 0x64, 0x0a])
                )
                print("\nStreaming EEG for 5 seconds...\n")

                await asyncio.sleep(5)

                # Stop streaming: send 'h' (halt)
                await client.write_gatt_char(
                    control_uuid, bytearray([0x02, 0x68, 0x0a])
                )

                for uuid in EEG_UUIDS:
                    await client.stop_notify(uuid)

            print(
                f"\nDone! Received ~{sample_count} total samples across all channels."
            )
            return
        except Exception as e:
            print(f"  Failed: {e}")
            if attempt < 2:
                print("  Retrying in 2s...")
                await asyncio.sleep(2)
            else:
                print("All attempts failed.")


if __name__ == "__main__":
    asyncio.run(main())
