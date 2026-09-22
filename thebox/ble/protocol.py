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
TELEMETRY_UUID = "273e000b-4c4d-454d-96be-f03bac821358"

CHANNEL_NAMES = list(EEG_UUIDS.keys())

# Control commands
CMD_RESUME = bytearray([0x02, 0x64, 0x0A])  # 'd' — start streaming
CMD_HALT = bytearray([0x02, 0x68, 0x0A])    # 'h' — stop streaming

# Muse 2 EEG parameters
SAMPLE_RATE = 256
SAMPLES_PER_PACKET = 12
SCALE_FACTOR = 0.48828125  # 2000 µV / 4096 counts
ADC_OFFSET = 0x800         # 12-bit ADC is unsigned; 2048 is 0 µV
SEQ_MODULO = 1 << 16       # packet counter is a wrapping uint16


def parse_packet(packet: bytes | bytearray) -> tuple[int, list[float]]:
    """Split a 20-byte Muse EEG packet into (sequence number, 12 µV samples).

    Bytes 0-1 are a big-endian packet counter shared by all EEG channels;
    gaps in it are packets lost over BLE. Bytes 2-19 hold twelve 12-bit
    samples packed MSB-first.
    """
    seq = (packet[0] << 8) | packet[1]
    return seq, decode_packet(packet)


def decode_packet(packet: bytes | bytearray) -> list[float]:
    """Decode the 12 samples of a Muse EEG packet to µV, centred on zero."""
    bit_buffer = 0
    bit_count = 0
    samples = []
    for byte in packet[2:]:
        bit_buffer = (bit_buffer << 8) | byte
        bit_count += 8
        while bit_count >= 12:
            bit_count -= 12
            raw = (bit_buffer >> bit_count) & 0xFFF
            samples.append((raw - ADC_OFFSET) * SCALE_FACTOR)
    return samples


def parse_battery(packet: bytes | bytearray) -> float:
    """Battery percentage from a telemetry packet (bytes 2-3, 1/512 %)."""
    return int.from_bytes(packet[2:4], "big") / 512
