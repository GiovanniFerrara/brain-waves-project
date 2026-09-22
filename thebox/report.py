"""Session report — plots and a text summary for a saved Recording."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from .eeg.bands import ALL_BANDS, epoch_band_powers
from .eeg.filters import clean
from .eeg.quality import assess, epoch_view
from .eeg.recording import Recording

CHANNEL_COLORS = {"TP9": "#1f77b4", "AF7": "#ff7f0e", "AF8": "#2ca02c", "TP10": "#d62728"}
EPOCH_SECONDS = 2.0
PSD_RANGE = (1.0, 45.0)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


class SessionReport:
    """Cleaned signal, epoch quality and band powers for one recording."""

    def __init__(self, rec: Recording, epoch_seconds: float = EPOCH_SECONDS):
        self.rec = rec
        self.fs = rec.sample_rate
        self.epoch_seconds = epoch_seconds
        self.cleaned = clean(rec.data, sample_rate=self.fs)
        self.quality = assess(self.cleaned, rec.valid, epoch_seconds, self.fs)
        self.epochs = epoch_view(self.cleaned, int(epoch_seconds * self.fs))
        # (channels, epochs, bands)
        self.band_powers = epoch_band_powers(self.epochs, self.fs)

    def mean_psd(self, ch: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Welch PSD averaged over the channel's good epochs only."""
        good = self.epochs[ch][self.quality.good[ch]]
        if len(good) == 0:
            return None
        freqs, psd = welch(good, fs=self.fs, nperseg=good.shape[-1], axis=-1)
        return freqs, psd.mean(axis=0)

    def summary(self) -> str:
        rec = self.rec
        lost = 1.0 - rec.valid.mean()
        lines = [
            f"Duration: {rec.duration:.1f}s  |  packets lost: {100 * lost:.1f}%",
            f"Epochs of {self.epoch_seconds:g}s usable per channel:",
        ]
        good = self.quality.good_fraction()
        for i, name in enumerate(rec.channels):
            psd = self.mean_psd(i)
            peak = ""
            if psd is not None:
                freqs, p = psd
                m = (freqs >= 7) & (freqs <= 14)
                peak = f"  alpha peak {freqs[m][np.argmax(p[m])]:.1f} Hz"
            lines.append(f"  {name:5s} {100 * good[i]:5.1f}%{peak}")
        return "\n".join(lines)

    # --- plots ---

    def plot_signal(self, path: Path) -> None:
        rec = self.rec
        t = rec.times
        fig, axes = plt.subplots(len(rec.channels), 1, figsize=(14, 10), sharex=True)
        fig.suptitle("EEG — cleaned (1–45 Hz, 50 Hz notch)", fontsize=15, fontweight="bold")
        for i, (name, ax) in enumerate(zip(rec.channels, axes)):
            y = self.cleaned[i]
            good = y[np.repeat(self.quality.good[i], int(self.epoch_seconds * self.fs))]
            spread = np.percentile(np.abs(good if len(good) else y), 99)
            lim = float(np.clip(1.5 * spread, 30, 300))
            ax.plot(t, y, color=CHANNEL_COLORS.get(name, "k"), linewidth=0.4)
            for e in np.flatnonzero(~self.quality.good[i]):
                ax.axvspan(e * self.epoch_seconds, (e + 1) * self.epoch_seconds,
                           color="orange", alpha=0.15, linewidth=0)
            for a, b in _runs(~rec.valid[i]):
                ax.axvspan(t[a], t[b - 1], color="red", alpha=0.3, linewidth=0)
            ax.set_ylim(-lim, lim)
            ax.set_ylabel(f"{name}\n(µV)")
            ax.grid(True, alpha=0.3)
        axes[0].plot([], [], color="orange", alpha=0.4, linewidth=6, label="rejected epoch")
        axes[0].plot([], [], color="red", alpha=0.5, linewidth=6, label="lost packets")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("Time (s)")
        axes[-1].set_xlim(0, t[-1] if len(t) else 1)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    def plot_spectrum(self, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))
        for band in ALL_BANDS:
            ax.axvspan(max(band.low, PSD_RANGE[0]), min(band.high, PSD_RANGE[1]),
                       color=band.color, alpha=0.08)
            ax.text((max(band.low, PSD_RANGE[0]) + min(band.high, PSD_RANGE[1])) / 2, 1.0,
                    band.name, transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=9, color=band.color)
        for i, name in enumerate(self.rec.channels):
            psd = self.mean_psd(i)
            if psd is None:
                continue
            freqs, p = psd
            m = (freqs >= PSD_RANGE[0]) & (freqs <= PSD_RANGE[1])
            ax.semilogy(freqs[m], p[m], color=CHANNEL_COLORS.get(name, "k"), label=name)
        ax.set_xlim(*PSD_RANGE)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (µV²/Hz)")
        ax.set_title("Power spectrum (good epochs only)", pad=18)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    def plot_bands(self, path: Path) -> None:
        groups = {"Frontal (AF7+AF8)": ["AF7", "AF8"], "Temporal (TP9+TP10)": ["TP9", "TP10"]}
        t = (np.arange(self.band_powers.shape[1]) + 0.5) * self.epoch_seconds
        fig, axes = plt.subplots(len(groups), 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"Relative band power per {self.epoch_seconds:g}s epoch",
                     fontsize=15, fontweight="bold")
        for ax, (title, names) in zip(axes, groups.items()):
            idx = [self.rec.channels.index(n) for n in names if n in self.rec.channels]
            bp = self.band_powers[idx]                      # (ch, epochs, bands)
            rel = bp / bp.sum(axis=-1, keepdims=True)
            good = self.quality.good[idx][..., None]
            with np.errstate(invalid="ignore"):
                avg = np.where(good, rel, 0).sum(axis=0) / good.sum(axis=0)
            for b, band in enumerate(ALL_BANDS):
                ax.plot(t, avg[:, b], color=band.color, label=band.name, marker=".", markersize=3)
            ax.set_title(title, loc="left")
            ax.set_ylabel("Share of 1–45 Hz power")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", ncol=5, fontsize=9)
        axes[-1].set_xlabel("Time (s) — gaps are epochs rejected on both channels")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    def save_plots(self, prefix: str | Path) -> list[Path]:
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        paths = [Path(f"{prefix}_{kind}.png") for kind in ("signal", "spectrum", "bands")]
        self.plot_signal(paths[0])
        self.plot_spectrum(paths[1])
        self.plot_bands(paths[2])
        return paths
