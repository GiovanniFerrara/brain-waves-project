"""MuseConnection — BLE lifecycle for the Muse 2 headband."""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable
from typing import Any

import numpy as np
from bleak import BleakClient, BleakScanner

from .aligner import PacketAligner
from .protocol import CMD_HALT, CMD_RESUME, CONTROL_UUID, EEG_UUIDS, parse_packet

# callback(channel, samples_µV, valid) — ``valid`` is False for samples
# interpolated over lost packets.
EEGCallback = Callable[[str, np.ndarray, np.ndarray], None]


class MuseConnection:
    """Manage scanning, connecting, and streaming from a Muse 2.

    Samples reach callbacks already aligned to a 256 Hz timeline: lost BLE
    packets are filled and flagged (see ``PacketAligner``).

    Usage::

        conn = MuseConnection("Muse-31A9")
        conn.on_eeg(my_callback)  # called with (channel, samples, valid)
        await conn.connect()
        await asyncio.sleep(20)
        await conn.disconnect()
        print(conn.aligner.loss_fraction())
    """

    def __init__(
        self,
        device_name: str,
        *,
        scan_timeout: float = 10.0,
        connect_timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.device_name = device_name
        self.scan_timeout = scan_timeout
        self.connect_timeout = connect_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.aligner = PacketAligner(list(EEG_UUIDS))

        self._callbacks: list[EEGCallback] = []
        self._disconnect_callbacks: list[Callable[[], None]] = []
        self._client: BleakClient | None = None
        self._device: Any = None
        self._closing = False

    @property
    def connected(self) -> bool:
        return self._client is not None and self._client.is_connected

    def on_eeg(self, callback: EEGCallback) -> None:
        """Register a callback: ``callback(channel_name, samples, valid)``."""
        self._callbacks.append(callback)

    def on_disconnect(self, callback: Callable[[], None]) -> None:
        """Register a callback for an unexpected loss of connection."""
        self._disconnect_callbacks.append(callback)

    def _make_notify_callback(self, channel_name: str):
        def callback(_sender: Any, data: bytearray) -> None:
            seq, decoded = parse_packet(data)
            samples, valid = self.aligner.push(channel_name, seq, decoded)
            if len(samples) == 0:
                return
            for cb in self._callbacks:
                cb(channel_name, samples, valid)
        return callback

    def _handle_disconnect(self, _client: BleakClient) -> None:
        if self._closing:
            return
        print(f"\n{self.device_name} disconnected unexpectedly.")
        for cb in self._disconnect_callbacks:
            cb()

    async def _scan(self) -> None:
        print(f"Scanning for {self.device_name}...")
        self._device = await BleakScanner.find_device_by_name(
            self.device_name, timeout=self.scan_timeout
        )
        if not self._device:
            raise RuntimeError(
                f"{self.device_name} not found. Is it on and in pairing mode?"
            )
        print(f"Found: {self._device.name} ({self._device.address})")

    async def _trust(self) -> None:
        """Trust the device via bluetoothctl to avoid BlueZ auth issues."""
        try:
            subprocess.run(
                ["bluetoothctl", "trust", self._device.address],
                capture_output=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # non-fatal (and absent on macOS)

    async def connect(self) -> None:
        """Scan, trust, and connect with retries. Starts EEG streaming."""
        await self._scan()
        await self._trust()
        self._closing = False

        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"Connecting (attempt {attempt}/{self.max_retries})...")
                self._client = BleakClient(
                    self._device,
                    timeout=self.connect_timeout,
                    disconnected_callback=self._handle_disconnect,
                )
                await self._client.connect()
                print(f"Connected: {self._client.is_connected}")

                self.aligner.reset()
                for name, uuid in EEG_UUIDS.items():
                    await self._client.start_notify(
                        uuid, self._make_notify_callback(name)
                    )

                await self._client.write_gatt_char(CONTROL_UUID, CMD_RESUME)
                return
            except Exception as e:
                print(f"  Failed: {e}")
                await self._safe_disconnect()
                if attempt < self.max_retries:
                    print(f"  Retrying in {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)

        raise RuntimeError("All connection attempts failed.")

    async def _safe_disconnect(self) -> None:
        if self._client is None:
            return
        self._closing = True
        try:
            await self._client.disconnect()
        except Exception:
            pass
        self._client = None

    async def disconnect(self) -> None:
        """Stop streaming and disconnect gracefully."""
        if not self._client:
            return
        self._closing = True
        try:
            await self._client.write_gatt_char(CONTROL_UUID, CMD_HALT)
            for uuid in EEG_UUIDS.values():
                await self._client.stop_notify(uuid)
        except Exception:
            pass  # already disconnected
        await self._safe_disconnect()
