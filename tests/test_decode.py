"""Unit tests for Muse 2 packet decoding."""

import pytest

from thebox.ble.protocol import (
    CHANNEL_NAMES,
    CMD_HALT,
    CMD_RESUME,
    CONTROL_UUID,
    EEG_UUIDS,
    SAMPLE_RATE,
    SAMPLES_PER_PACKET,
    SCALE_FACTOR,
    decode_packet,
)


class TestDecodePacket:
    def test_returns_12_samples(self):
        """Each 20-byte packet should decode to 12 samples."""
        packet = bytearray(20)
        samples = decode_packet(packet)
        assert len(samples) == SAMPLES_PER_PACKET

    def test_zero_packet_gives_zero_samples(self):
        """All-zero payload should decode to all-zero samples."""
        packet = bytearray(20)
        samples = decode_packet(packet)
        assert all(s == 0.0 for s in samples)

    def test_max_values(self):
        """All-ones payload should decode to max scaled value."""
        packet = bytearray([0x00, 0x00] + [0xFF] * 18)
        samples = decode_packet(packet)
        max_val = 0xFFF * SCALE_FACTOR
        assert all(abs(s - max_val) < 0.01 for s in samples)

    def test_header_bytes_ignored(self):
        """First 2 bytes are header and should not affect samples."""
        packet_a = bytearray([0x00, 0x00] + [0x80] * 18)
        packet_b = bytearray([0xFF, 0xFF] + [0x80] * 18)
        assert decode_packet(packet_a) == decode_packet(packet_b)

    def test_known_pattern(self):
        """Verify a specific bit pattern decodes correctly."""
        # 12-bit value 0x800 = 2048 → 2048 * 0.48828125 = 1000.0
        # Pack two 12-bit values: 0x800, 0x800 = 0x800800 in 3 bytes
        payload = bytearray([0x80, 0x08, 0x00] * 6)
        packet = bytearray([0x00, 0x00]) + payload
        samples = decode_packet(packet)
        assert abs(samples[0] - 1000.0) < 0.01


class TestProtocolConstants:
    def test_channel_names(self):
        assert CHANNEL_NAMES == ["TP9", "AF7", "AF8", "TP10"]

    def test_eeg_uuids_match_channels(self):
        assert set(EEG_UUIDS.keys()) == set(CHANNEL_NAMES)

    def test_sample_rate(self):
        assert SAMPLE_RATE == 256

    def test_commands_are_bytearrays(self):
        assert isinstance(CMD_RESUME, bytearray)
        assert isinstance(CMD_HALT, bytearray)
