"""Session report — plots and a text summary for a saved Recording."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from .eeg.bands import ALL_BANDS, epoch_band_powers
from .eeg.features import above_background_db, alpha_peak, db_change, smooth_powers
from .eeg.filters import clean
from .eeg.quality import artifact_mask, assess, windows
from .eeg.recording import Recording

CHANNEL_COLORS = {"TP9": "#1f77b4", "AF7": "#ff7f0e", "AF8": "#2ca02c", "TP10": "#d62728"}
GROUPS = {"Temporal (TP9+TP10)": ["TP9", "TP10"], "Frontal (AF7+AF8)": ["AF7", "AF8"]}
WIN_SECONDS = 2.0      # epochs for the spectrum and quality summary
SEGMENT_SECONDS = 1.0  # band-power time course: 1 s segments...
STEP_SECONDS = 0.5     # ...every 0.5 s...
SMOOTH_SEGMENTS = 8    # ...averaged over the last 8 (4.5 s span)
PSD_RANGE = (1.0, 45.0)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


class SessionReport:
    """Cleaned signal, artifact rejection and spectral features for one recording."""

    def __init__(self, rec: Recording):
        self.rec = rec
        self.fs = rec.sample_rate
        self.cleaned = clean(rec.data, sample_rate=self.fs)
        self.artifacts = artifact_mask(self.cleaned, self.fs)
        # Non-overlapping epochs for the spectrum, sliding windows for time courses
        self.epoch_quality = assess(self.cleaned, rec.valid, WIN_SECONDS,
                                    sample_rate=self.fs, artifacts=self.artifacts)
        self.segment_quality = assess(self.cleaned, rec.valid, SEGMENT_SECONDS, STEP_SECONDS,
                                      sample_rate=self.fs, artifacts=self.artifacts)
        win = int(WIN_SECONDS * self.fs)
        self.epochs = windows(self.cleaned, win, win)
        seg, step = int(SEGMENT_SECONDS * self.fs), int(STEP_SECONDS * self.fs)
        # (channels, segments, bands), smoothed over clean segments only
        self.band_powers = smooth_powers(
            epoch_band_powers(windows(self.cleaned, seg, step), self.fs),
            self.segment_quality.good, SMOOTH_SEGMENTS,
        )

    def mean_psd(self, ch: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Welch PSD averaged over the channel's good epochs only."""
        good = self.epochs[ch][self.epoch_quality.good[ch]]
        if len(good) < 3:
            return None
        freqs, psd = welch(good, fs=self.fs, nperseg=good.shape[-1], axis=-1)
        return freqs, psd.mean(axis=0)

    def summary(self) -> str:
        rec = self.rec
        lines = [
            f"Duration: {rec.duration:.1f}s  |  packets lost: {100 * (1 - rec.valid.mean()):.1f}%",
            "Per channel: usable time, alpha peak above the 1/f background",
        ]
        good = self.epoch_quality.good_fraction()
        for i, name in enumerate(rec.channels):
            psd = self.mean_psd(i)
            if psd is None:
                peak = "not enough clean data"
            else:
                p = alpha_peak(*psd)
                peak = f"{p.freq:.1f} Hz, +{p.prominence_db:.1f} dB" if p else "none"
            lines.append(f"  {name:5s} {100 * good[i]:5.1f}%   alpha: {peak}")
        return "\n".join(lines)

    # --- plots ---

    def plot_signal(self, path: Path) -> None:
        rec = self.rec
        t = rec.times
        fig, axes = plt.subplots(len(rec.channels), 1, figsize=(14, 10), sharex=True)
        fig.suptitle("EEG — cleaned (1–45 Hz, 50 Hz notch)", fontsize=15, fontweight="bold")
        for i, (name, ax) in enumerate(zip(rec.channels, axes)):
            y = self.cleaned[i]
            clean_part = y[~self.artifacts[i]]
            spread = np.percentile(np.abs(clean_part if len(clean_part) else y), 99.5)
            lim = float(np.clip(2 * spread, 30, 300))
            ax.plot(t, y, color=CHANNEL_COLORS.get(name, "k"), linewidth=0.4)
            for a, b in _runs(self.artifacts[i]):
                ax.axvspan(t[a], t[b - 1], color="orange", alpha=0.25, linewidth=0)
            for a, b in _runs(~rec.valid[i]):
                ax.axvspan(t[a], t[b - 1], color="red", alpha=0.3, linewidth=0)
            ax.set_ylim(-lim, lim)
            ax.set_ylabel(f"{name}\n(µV)")
            ax.grid(True, alpha=0.3)
        axes[0].plot([], [], color="orange", alpha=0.5, linewidth=6, label="artifact")
        axes[0].plot([], [], color="red", alpha=0.5, linewidth=6, label="lost packets")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("Time (s)")
        axes[-1].set_xlim(0, t[-1] if len(t) else 1)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    def plot_spectrum(self, path: Path) -> None:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 2]})
        for a in (ax, ax2):
            for band in ALL_BANDS:
                a.axvspan(max(band.low, PSD_RANGE[0]), min(band.high, PSD_RANGE[1]),
                          color=band.color, alpha=0.08)
        for band in ALL_BANDS:
            ax.text((max(band.low, PSD_RANGE[0]) + min(band.high, PSD_RANGE[1])) / 2, 1.0,
                    band.name, transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=9, color=band.color)
        for i, name in enumerate(self.rec.channels):
            psd = self.mean_psd(i)
            if psd is None:
                continue
            freqs, p = psd
            m = (freqs >= PSD_RANGE[0]) & (freqs <= PSD_RANGE[1])
            color = CHANNEL_COLORS.get(name, "k")
            ax.semilogy(freqs[m], p[m], color=color, label=name)
            ax2.plot(freqs[m], above_background_db(freqs, p)[m], color=color)
            peak = alpha_peak(freqs, p)
            if peak:
                ax2.plot(peak.freq, peak.prominence_db, "o", color=color)
        ax.set_ylabel("Power (µV²/Hz)")
        ax.set_title("Power spectrum (clean epochs only)", pad=18)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        ax2.axhline(0, color="k", linewidth=0.8)
        ax2.set_ylabel("Above 1/f background (dB)")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_xlim(*PSD_RANGE)
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    def group_db_change(self, names: list[str]) -> np.ndarray:
        """(windows, bands) dB change vs session median, averaged over the
        group's channels that are clean in each window (NaN if none)."""
        idx = [self.rec.channels.index(n) for n in names if n in self.rec.channels]
        change = db_change(self.band_powers[idx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
            return np.nanmean(change, axis=0)

    def plot_bands(self, path: Path) -> None:
        # Each value covers the trailing SMOOTH_SEGMENTS segments; plot at the end
        t = self.segment_quality.centers + SEGMENT_SECONDS / 2
        fig, axes = plt.subplots(len(GROUPS), 1, figsize=(14, 8), sharex=True)
        span = SEGMENT_SECONDS + (SMOOTH_SEGMENTS - 1) * STEP_SECONDS
        fig.suptitle(
            f"Band power vs this session's median (clean {SEGMENT_SECONDS:g}s segments, "
            f"averaged over {span:g}s)", fontsize=15, fontweight="bold",
        )
        for ax, (title, names) in zip(axes, GROUPS.items()):
            change = self.group_db_change(names)
            for b, band in enumerate(ALL_BANDS):
                ax.plot(t, change[:, b], color=band.color, label=band.name, linewidth=1.2)
            ax.axhline(0, color="k", linewidth=0.8)
            ax.set_title(title, loc="left")
            ax.set_ylabel("dB (+3 = double)")
            ax.set_ylim(-10, 10)
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", ncol=5, fontsize=9)
        axes[-1].set_xlabel("Time (s) — gaps: too few clean segments on both channels")
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
