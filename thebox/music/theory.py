"""Music theory helpers — notes, scales and the chord progression."""

from __future__ import annotations

from dataclasses import dataclass

ROOT = 62  # MIDI D4 — every pitched sample is tuned to D

# D natural minor: D E F G A Bb C
MINOR = [0, 2, 3, 5, 7, 8, 10]
# Minor pentatonic for melody: D F G A C — consonant over every chord below
PENTATONIC = [0, 3, 5, 7, 10]


def ratio(semitones: float) -> float:
    """Playback-rate ratio for a pitch shift."""
    return 2.0 ** (semitones / 12.0)


@dataclass(frozen=True)
class Chord:
    name: str
    root: int              # semitones above D
    tones: tuple[int, ...]  # semitones above D, voiced for the pad


# i – VI – III – VII, voiced close around D4 so pad copies stay near the sample's pitch
PROGRESSION = [
    Chord("Dm9", 0, (0, 3, 7, 10, 14)),
    Chord("Bbmaj7", -4, (-4, 0, 5, 9)),
    Chord("Fmaj7", 3, (3, 7, 10, 14)),
    Chord("Cadd9", -2, (-2, 2, 5, 10)),
]


def scale_note(degree: int, scale: list[int] = PENTATONIC) -> int:
    """Semitones above D for a (possibly negative or >len) scale degree."""
    octave, idx = divmod(degree, len(scale))
    return 12 * octave + scale[idx]
