"""Event types and detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..eeg.stream import EEGStream


class EventType(Enum):
    BLINK = auto()
    CLENCH = auto()
    ALPHA_BURST_START = auto()
    ALPHA_BURST_END = auto()


@dataclass
class Event:
    type: EventType
    timestamp: float
    value: float = 0.0        # detector-specific magnitude
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Event({self.type.name}, value={self.value:.1f})"


class EventDetector(ABC):
    """Base class for all EEG event detectors."""

    @abstractmethod
    def detect(self, stream: EEGStream, now: float) -> list[Event]:
        """Examine the EEG stream and return any detected events.

        Called once per processing cycle (~20 Hz). Detectors maintain
        internal state for debouncing between calls.
        """
        ...
