"""Recording — keep a whole session in memory and save/load it as .npz."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..ble.protocol import CHANNEL_NAMES, SAMPLE_RATE


@dataclass
class Recording:
    """All samples of a session, aligned across channels.

    ``data`` is (channels, samples) in µV; ``valid`` is False where samples
    were interpolated over lost packets. Build one live with ``append`` (it
    fits ``MuseConnection.on_eeg``) or read one back with ``load``.
    """

    channels: list[str] = field(default_factory=lambda: list(CHANNEL_NAMES))
    sample_rate: float = SAMPLE_RATE
    start_time: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)
    _chunks: dict[str, list[np.ndarray]] = field(default_factory=dict, repr=False)
    _valid_chunks: dict[str, list[np.ndarray]] = field(default_factory=dict, repr=False)
    _data: np.ndarray | None = field(default=None, repr=False)
    _valid: np.ndarray | None = field(default=None, repr=False)

    def append(self, channel: str, samples: np.ndarray, valid: np.ndarray) -> None:
        self._chunks.setdefault(channel, []).append(np.asarray(samples, dtype=np.float64))
        self._valid_chunks.setdefault(channel, []).append(np.asarray(valid, dtype=bool))
        self._data = self._valid = None

    def _build(self) -> None:
        if self._data is not None:
            return
        cat = {
            ch: np.concatenate(self._chunks[ch]) if self._chunks.get(ch) else np.array([])
            for ch in self.channels
        }
        vcat = {
            ch: np.concatenate(self._valid_chunks[ch]) if self._valid_chunks.get(ch) else np.array([], bool)
            for ch in self.channels
        }
        # Channels can differ by the last packet or so; trim to the shortest
        n = min(len(v) for v in cat.values())
        self._data = np.stack([cat[ch][:n] for ch in self.channels])
        self._valid = np.stack([vcat[ch][:n] for ch in self.channels])

    @property
    def data(self) -> np.ndarray:
        self._build()
        return self._data

    @property
    def valid(self) -> np.ndarray:
        self._build()
        return self._valid

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        return self.n_samples / self.sample_rate

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_samples) / self.sample_rate

    def channel(self, name: str) -> np.ndarray:
        return self.data[self.channels.index(name)]

    def channel_valid(self, name: str) -> np.ndarray:
        return self.valid[self.channels.index(name)]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            data=self.data,
            valid=self.valid,
            channels=np.array(self.channels),
            sample_rate=self.sample_rate,
            start_time=self.start_time,
            meta=np.array(json.dumps(self.meta)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Recording:
        with np.load(path) as f:
            rec = cls(
                channels=[str(c) for c in f["channels"]],
                sample_rate=float(f["sample_rate"]),
                start_time=float(f["start_time"]),
                meta=json.loads(str(f["meta"])) if "meta" in f else {},
            )
            rec._data = f["data"]
            rec._valid = f["valid"]
        return rec
