"""MuseConnection — BLE lifecycle for the Muse 2 headband."""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable
from typing import Any

from bleak import BleakClient, BleakScanner

from .protocol import (
    CHANNEL_NAMES,
    CMD_HALT,
    CMD_RESUME,
    CONTROL_UUID,
    EEG_UUIDS,
    decode_packet,
)

EEGCallback = Callable[[str, list[float], float], None]


class MuseConnection:
    """Manage scanning, connecting, and streaming from a Muse 2.

    Usage::

        conn = MuseConnection("Muse-31A9")
        conn.on_eeg(my_callback)  # called with (channel, samples, timestamp)
        await conn.connect()
        await asyncio.sleep(20)
        await conn.disconnect()
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

        self._callbacks: list[EEGCallback] = []
        self._client: BleakClient | None = None
        self._device: Any = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected and self._client is not None and self._client.is_connected

    def on_eeg(self, callback: EEGCallback) -> None:
        """Register a callback: ``callback(channel_name, samples, timestamp)``."""
        self._callbacks.append(callback)

    def _make_notify_callback(self, channel_name: str):
        def callback(_sender: Any, data: bytearray) -> None:
            samples = decode_packet(data)
            ts = asyncio.get_event_loop().time()
            for cb in self._callbacks:
                cb(channel_name, samples, ts)
        return callback

    async def _scan(self) -> None:
        print(f"Scanning for {self.device_name}...")
        self._device = await BleakScanner.find_device_by_name(
            self.device_name, timeout=self.scan_timeout
        )
        if not self._device:
            raise RuntimeError(
                f"{self.device_name} not found. Is it in pairing mode?"
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
            pass  # non-fatal

    async def connect(self) -> None:
        """Scan, trust, and connect with retries. Starts EEG streaming."""
        await self._scan()
        await self._trust()

        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"Connecting (attempt {attempt}/{self.max_retries})...")
                self._client = BleakClient(
                    self._device, timeout=self.connect_timeout
                )
                await self._client.connect()
                self._connected = True
                print(f"Connected: {self._client.is_connected}")

                # Subscribe to EEG channels
                for name, uuid in EEG_UUIDS.items():
                    await self._client.start_notify(
                        uuid, self._make_notify_callback(name)
                    )

                # Start streaming
                await self._client.write_gatt_char(CONTROL_UUID, CMD_RESUME)
                return
            except Exception as e:
                self._connected = False
                print(f"  Failed: {e}")
                if attempt < self.max_retries:
                    print(f"  Retrying in {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)

        raise RuntimeError("All connection attempts failed.")

    async def disconnect(self) -> None:
        """Stop streaming and disconnect gracefully."""
        if not self._client:
            return

        try:
            await self._client.write_gatt_char(CONTROL_UUID, CMD_HALT)
            for uuid in EEG_UUIDS.values():
                await self._client.stop_notify(uuid)
        except Exception:
            pass  # already disconnected

        try:
            await self._client.disconnect()
        except Exception:
            pass

        self._connected = False
        self._client = None
