"""Muse 2 BLE protocol — UUIDs, commands, packet decoding."""

# GATT characteristic UUIDs
CONTROL_UUID = "273e0001-4c4d-454d-96be-f03bac821358"

EEG_UUIDS = {
    "TP9":  "273e0003-4c4d-454d-96be-f03bac821358",
    "AF7":  "273e0004-4c4d-454d-96be-f03bac821358",
    "AF8":  "273e0005-4c4d-454d-96be-f03bac821358",
    "TP10": "273e0006-4c4d-454d-96be-f03bac821358",
}

AUX_UUID = "273e0007-4c4d-454d-96be-f03bac821358"

CHANNEL_NAMES = list(EEG_UUIDS.keys())

# Control commands
CMD_RESUME = bytearray([0x02, 0x64, 0x0A])  # 'd' — start streaming
CMD_HALT = bytearray([0x02, 0x68, 0x0A])    # 'h' — stop streaming

# Muse 2 EEG parameters
SAMPLE_RATE = 256
SAMPLES_PER_PACKET = 12
SCALE_FACTOR = 0.48828125  # 2000 / 4096


def decode_packet(packet: bytearray) -> list[float]:
    """Decode a 20-byte Muse EEG packet into 12 µV samples.

    The packet has a 2-byte header followed by 18 bytes of 12-bit samples
    packed MSB-first.
    """
    bit_buffer = 0
    bit_count = 0
    samples = []
    for byte in packet[2:]:
        bit_buffer = (bit_buffer << 8) | byte
        bit_count += 8
        while bit_count >= 12:
            bit_count -= 12
            raw = (bit_buffer >> bit_count) & 0xFFF
            samples.append(raw * SCALE_FACTOR)
    return samples
