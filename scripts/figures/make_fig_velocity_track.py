from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import figure_style as fs

SRC = REPO / "results" / "full" / "k3_doppler_full" / "velocity_track" / "velocity_track.csv"
COLOR = "#636EFA"


def main() -> None:
    frame = pd.read_csv(SRC).sort_values("window_bursts")
    windows = [int(v) for v in frame["window_bursts"]]
    values = [float(v) for v in frame["speed_error_abs_median_mps"]]

    fs.apply_style()
    fig, ax = plt.subplots(figsize=(3.0, 1.75))
    y = np.arange(len(windows))[::-1]
    bars = ax.barh(y, values, height=0.58, color=COLOR, alpha=0.85, zorder=3)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.2f}",
                    (value, bar.get_y() + bar.get_height() / 2.0),
                    textcoords="offset points", xytext=(4, 0), va="center",
                    fontsize=7.5, color=COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels([str(v) for v in windows], fontsize=8)
    ax.set_xlim(0.0, 2.35)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_ylim(-0.6, len(windows) - 0.4)
    ax.set_xlabel("median $|\\Delta v|$ [m/s]", labelpad=2)
    ax.set_ylabel("bursts", rotation=90, labelpad=3)
    ax.grid(True, which="major", axis="x", color=fs.GRID_COLOR, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", color="black", length=2.5)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)

    fig.tight_layout(pad=0.3)
    fs.save(fig, "velocity_track")
    plt.close(fig)


if __name__ == "__main__":
    main()
