"""PacketAligner: lost packets must keep their place on the timeline."""

import numpy as np

from thebox.ble.aligner import PacketAligner


def _packet(value: float) -> list[float]:
    return [value] * 12


class TestPacketAligner:
    def test_in_order_packets_pass_through(self):
        a = PacketAligner(["AF7"])
        for seq in range(3):
            samples, valid = a.push("AF7", seq, _packet(seq))
            assert len(samples) == 12
            assert valid.all()
        assert a.lost["AF7"] == 0
        assert a.loss_fraction() == 0.0

    def test_gap_is_filled_and_flagged(self):
        a = PacketAligner(["AF7"])
        a.push("AF7", 10, _packet(0.0))
        samples, valid = a.push("AF7", 13, _packet(36.0))  # 11 and 12 lost
        assert len(samples) == 36
        assert not valid[:24].any() and valid[24:].all()
        # Linear ramp from the last real sample to the next one
        assert np.all(np.diff(samples[:25]) > 0)
        assert a.lost["AF7"] == 2
        assert a.loss_fraction() == 2 / 4

    def test_sequence_wraparound(self):
        a = PacketAligner(["AF7"])
        a.push("AF7", 65535, _packet(0))
        samples, valid = a.push("AF7", 0, _packet(0))
        assert len(samples) == 12 and valid.all()

    def test_duplicates_and_stale_packets_dropped(self):
        a = PacketAligner(["AF7"])
        a.push("AF7", 5, _packet(0))
        a.push("AF7", 6, _packet(0))
        samples, _ = a.push("AF7", 6, _packet(0))
        assert len(samples) == 0
        samples, _ = a.push("AF7", 4, _packet(0))
        assert len(samples) == 0

    def test_channels_share_one_origin(self):
        """A channel whose first packet was lost is padded so sample i lines up."""
        a = PacketAligner(["TP9", "AF7"])
        a.push("TP9", 100, _packet(1))
        samples, valid = a.push("AF7", 102, _packet(1))  # AF7's 100, 101 lost
        assert len(samples) == 36
        assert valid.sum() == 12

    def test_huge_jump_resyncs_instead_of_filling(self):
        a = PacketAligner(["AF7"], max_fill_seconds=1.0)
        a.push("AF7", 0, _packet(0))
        samples, valid = a.push("AF7", 5000, _packet(0))
        assert len(samples) == 12 and valid.all()
        assert a.resyncs == 1

    def test_reset_clears_state(self):
        a = PacketAligner(["AF7"])
        a.push("AF7", 0, _packet(0))
        a.push("AF7", 3, _packet(0))
        a.reset()
        assert a.lost["AF7"] == 0
        samples, valid = a.push("AF7", 500, _packet(0))
        assert len(samples) == 12 and valid.all()


def test_connection_callback_keeps_real_duration():
    """Raw BLE packets with 3 of 4 dropped still yield 1 s per 256 samples."""
    from thebox.ble.connection import MuseConnection
    from thebox.eeg.recording import Recording

    conn = MuseConnection("Muse-TEST")
    rec = Recording()
    conn.on_eeg(rec.append)
    callbacks = {ch: conn._make_notify_callback(ch) for ch in rec.channels}

    midscale = [0x80, 0x08, 0x00] * 6
    n_packets = 64  # 3 s at 256 Hz / 12 samples
    for seq in range(n_packets):
        if seq % 4 and seq != n_packets - 1:
            continue  # lose 3 of every 4 packets
        for ch, cb in callbacks.items():
            cb(None, bytearray([seq >> 8, seq & 0xFF] + midscale))

    assert rec.n_samples == n_packets * 12
    assert abs(rec.duration - 3.0) < 0.01
    assert 0.2 < rec.valid.mean() < 0.3
    assert abs(conn.aligner.loss_fraction() - 47 / 64) < 1e-9
