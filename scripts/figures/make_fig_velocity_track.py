from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import figure_style as fs

SRC = REPO / "results" / "full" / "k3_doppler_full" / "velocity_track" / "velocity_track.csv"
WIDTH = 3.5

def main() -> None:
    frame = pd.read_csv(SRC).sort_values("window_bursts")

    fs.apply_style()
    fig, ax = plt.subplots(figsize=(3.3, 2.45))
    ax.plot(frame["window_bursts"], frame["speed_error_abs_median_mps"], marker="o",
            color=fs.ARCH_COLORS["conv1d"], lw=1.7, ms=6.0, label="median speed error")
    for x, y in zip(frame["window_bursts"], frame["speed_error_abs_median_mps"]):
        ax.annotate("%.2f" % y, (float(x), float(y)), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=8.5,
                    color=fs.ARCH_COLORS["conv1d"])
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlabel("bursts in the track")
    ax.set_ylabel("median $|\\Delta v|$ [m/s]", rotation=90, labelpad=5)
    ax.set_ylim(0.0, 2.6)
    ax.set_xlim(0.85, 8.4)
    ax.grid(True, which="major", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    fig.subplots_adjust(left=0.24, right=0.97, top=0.96, bottom=0.22)
    fs.save(fig, "velocity_track")
    plt.close(fig)

if __name__ == "__main__":
    main()
