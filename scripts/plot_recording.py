#!/usr/bin/env python3
"""Re-analyse a saved session: python scripts/plot_recording.py output/rec_XXXX.npz"""

import sys
from pathlib import Path

from thebox.eeg.recording import Recording
from thebox.report import SessionReport

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    report = SessionReport(Recording.load(path))
    print(report.summary())
    for p in report.save_plots(path.with_suffix("")):
        print(f"Saved {p}")
