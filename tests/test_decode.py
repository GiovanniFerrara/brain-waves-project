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
    parse_packet,
)


class TestDecodePacket:
    def test_returns_12_samples(self):
        """Each 20-byte packet should decode to 12 samples."""
        packet = bytearray(20)
        samples = decode_packet(packet)
        assert len(samples) == SAMPLES_PER_PACKET

    def test_midscale_is_zero(self):
        """0x800 is the ADC midpoint and must decode to 0 µV."""
        packet = bytearray([0x00, 0x00] + [0x80, 0x08, 0x00] * 6)
        assert decode_packet(packet) == [0.0] * 12

    def test_full_scale_range(self):
        """0x000 → -1000 µV, 0xFFF → just under +1000 µV."""
        low = decode_packet(bytearray(20))
        high = decode_packet(bytearray([0x00, 0x00] + [0xFF] * 18))
        assert all(s == -1000.0 for s in low)
        assert all(abs(s - (0xFFF - 0x800) * SCALE_FACTOR) < 1e-9 for s in high)

    def test_header_bytes_ignored(self):
        """First 2 bytes are the sequence number and must not affect samples."""
        packet_a = bytearray([0x00, 0x00] + [0x80] * 18)
        packet_b = bytearray([0xFF, 0xFF] + [0x80] * 18)
        assert decode_packet(packet_a) == decode_packet(packet_b)

    def test_known_pattern(self):
        """Two samples 0x801, 0x7FF packed in 3 bytes → +1 and -1 count."""
        packet = bytearray([0x00, 0x00] + [0x80, 0x17, 0xFF] * 6)
        samples = decode_packet(packet)
        assert samples[0] == SCALE_FACTOR
        assert samples[1] == -SCALE_FACTOR

    def test_parse_packet_reads_big_endian_sequence(self):
        seq, samples = parse_packet(bytearray([0x12, 0x34] + [0x80, 0x08, 0x00] * 6))
        assert seq == 0x1234
        assert len(samples) == 12


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
