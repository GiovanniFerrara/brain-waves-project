#!/usr/bin/env python3
"""Plot EEG motor session from CSV log.

Usage:
    python scripts/plot_session.py                          # latest session
    python scripts/plot_session.py output/relax_motor.csv   # specific file
"""

import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def smooth(y, window=15):
    """Simple moving average."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def plot_session(csv_path: str):
    df = pd.read_csv(csv_path)
    t = df["time_s"].values

    # Detect stale data (all values identical = disconnected)
    if "theta_tp" in df.columns:
        theta = df["theta_tp"].values
        beta = df["beta_tp"].values
        alpha = df["alpha_tp"].values
    else:
        theta = df.get("theta", df.get("alpha", np.zeros(len(t)))).values
        beta = df.get("beta", df.get("beta_tp", np.zeros(len(t)))).values
        alpha = df.get("alpha", df.get("alpha_tp", np.zeros(len(t)))).values

    ratio = df["ratio"].values
    motor = df["motor_pct"].values
    smoothed_relax = df["smoothed"].values

    # Find where data goes stale (identical consecutive values)
    diffs = np.abs(np.diff(ratio))
    stale_mask = np.concatenate([[False], diffs == 0])
    # Find first long stale run (>10 consecutive identical)
    stale_run = 0
    cutoff = len(t)
    for i, is_stale in enumerate(stale_mask):
        if is_stale:
            stale_run += 1
            if stale_run > 10:
                cutoff = i - 10
                break
        else:
            stale_run = 0

    if cutoff < len(t):
        print(f"  Muse disconnected at ~{t[cutoff]:.1f}s — trimming stale data")

    # Trim stale data
    t = t[:cutoff]
    theta = theta[:cutoff]
    beta = beta[:cutoff]
    alpha = alpha[:cutoff]
    ratio = ratio[:cutoff]
    motor = motor[:cutoff]
    smoothed_relax = smoothed_relax[:cutoff]

    artifact = df["artifact"].values[:cutoff] if "artifact" in df.columns else np.zeros(len(t))

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"EEG Motor Session — {csv_path}", fontsize=13, fontweight="bold")

    # 1. Band powers
    ax = axes[0]
    ax.plot(t, theta, alpha=0.3, color="purple", linewidth=0.5)
    ax.plot(t, smooth(theta), color="purple", linewidth=2, label="Theta (4-8 Hz)")
    ax.plot(t, beta, alpha=0.3, color="orange", linewidth=0.5)
    ax.plot(t, smooth(beta), color="orange", linewidth=2, label="Beta (13-30 Hz)")
    ax.plot(t, alpha, alpha=0.3, color="green", linewidth=0.5)
    ax.plot(t, smooth(alpha), color="green", linewidth=2, label="Alpha (8-13 Hz)")
    # Mark artifacts
    art_idx = np.where(artifact > 0)[0]
    if len(art_idx) > 0:
        ax.scatter(t[art_idx], theta[art_idx], color="red", s=10, zorder=5, label="Artifact")
    ax.set_ylabel("Power (µV RMS)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Band Powers (raw + smoothed trend)")
    ax.grid(True, alpha=0.3)

    # 2. Theta/Beta ratio
    ax = axes[1]
    ax.plot(t, ratio, alpha=0.3, color="steelblue", linewidth=0.5)
    ax.plot(t, smooth(ratio), color="steelblue", linewidth=2, label="θ/β ratio")
    if "ratio_p10" in df.columns:
        p10 = df["ratio_p10"].values[:cutoff]
        p90 = df["ratio_p90"].values[:cutoff]
        # Replace empty strings with NaN
        p10 = pd.to_numeric(p10, errors="coerce")
        p90 = pd.to_numeric(p90, errors="coerce")
        valid = ~np.isnan(p10) & ~np.isnan(p90)
        if valid.any():
            ax.fill_between(t[valid], p10[valid], p90[valid], alpha=0.15, color="steelblue", label="10th-90th pct")
    ax.set_ylabel("Ratio")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Theta/Beta Ratio (relaxation indicator)")
    ax.grid(True, alpha=0.3)

    # 3. Relaxation level (smoothed)
    ax = axes[2]
    ax.fill_between(t, 0, smoothed_relax, alpha=0.3, color="teal")
    ax.plot(t, smoothed_relax, color="teal", linewidth=2, label="Smoothed relaxation")
    ax.plot(t, smooth(smoothed_relax, 30), color="darkgreen", linewidth=2, linestyle="--", label="Trend")
    ax.set_ylabel("Relaxation (0-1)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Relaxation Level")
    ax.grid(True, alpha=0.3)

    # 4. Motor speed
    ax = axes[3]
    ax.fill_between(t, 0, motor, alpha=0.3, color="coral")
    ax.plot(t, motor, color="coral", linewidth=2, label="Motor %")
    ax.plot(t, smooth(motor, 30), color="darkred", linewidth=2, linestyle="--", label="Trend")
    ax.set_ylabel("Motor (%)")
    ax.set_ylim(-2, 85)
    ax.set_xlabel("Time (seconds)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Motor Speed")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save next to CSV
    out_path = csv_path.replace(".csv", ".png")
    plt.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Find latest session CSV (timestamped names sort naturally)
        files = sorted(glob.glob("output/relax_motor*.csv"))
        if not files:
            print("No session CSV found in output/")
            return
        csv_path = files[-1]

    print(f"Plotting: {csv_path}")
    plot_session(csv_path)


if __name__ == "__main__":
    main()
