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
WIDTH = 7.6

def main() -> None:
    frame = pd.read_csv(SRC).sort_values("window_bursts")

    fs.apply_style()
    fig, ax = plt.subplots(figsize=(WIDTH, 3.8))
    ax.plot(frame["window_bursts"], frame["speed_error_abs_median_mps"], marker="o",
            color=fs.ARCH_COLORS["conv1d"], lw=1.7, ms=7.0, label="median speed error")
    ax.axhspan(0.0, 2.0, color=fs.BASELINE_FILL, zorder=0)
    ax.axhline(2.0, color=fs.TARGET_COLOR, lw=1.4, ls=(0, (6, 2)), zorder=2)
    ax.annotate("a few m/s: the track is readable", xy=(float(frame["window_bursts"].iloc[-1]), 2.0),
                xytext=(-6, 8), textcoords="offset points", ha="right", fontsize=10,
                color=fs.TARGET_COLOR)
    for x, y in zip(frame["window_bursts"], frame["speed_error_abs_median_mps"]):
        ax.annotate("%.2f" % y, (float(x), float(y)), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=9.5,
                    color=fs.ARCH_COLORS["conv1d"])
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlabel("bursts averaged into the track")
    ax.set_ylabel("median $|\\Delta v|$ [m/s]", rotation=90, labelpad=10)
    ax.set_title("Speed comes from the sequence of bursts", pad=8)
    ax.set_ylim(0.0, 2.6)
    ax.grid(True, which="major", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    fs.legend_below(ax, ncol=1)
    fs.save(fig, "velocity_track")
    plt.close(fig)

if __name__ == "__main__":
    main()
