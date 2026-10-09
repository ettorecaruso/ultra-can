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

SRC = REPO / "results" / "full" / "diagnostics" / "scene"
OBS = "#EF553B"
SCA = "#636EFA"
WIDTH = 11.8

def main() -> None:
    frame = pd.read_csv(SRC / "geometry.csv").sort_values("distance_m")
    summary = pd.read_csv(SRC / "summary.csv").iloc[0]

    fs.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.6))

    ax = axes[0]
    ax.plot(frame["distance_m"], frame["obstacle_amplitude"], marker="o",
            color=OBS, lw=1.7, ms=7.0, label="obstacle (own echo, $1/d^4$)")
    ax.plot(frame["distance_m"], frame["scatterer_amplitude"], marker="s",
            color=SCA, lw=1.7, ms=7.0, ls=(0, (6, 2)), label="scatterer (peer, link loss)")
    ax.set_xlabel("link distance [m]")
    ax.set_ylabel("echo amplitude before normalisation", rotation=90, labelpad=10)
    ax.set_title("The sky is a scene, not a die", pad=8)
    ax.grid(True, which="major", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)
    ax.annotate("1/d$^4$ in power", xy=(float(frame["distance_m"].iloc[-2]),
                                       float(frame["obstacle_amplitude"].iloc[-2])),
                xytext=(-10, 26), textcoords="offset points", ha="right", fontsize=10,
                color="#a33a27")

    ax = axes[1]
    scale = float(summary["n_symbols"]) / float(summary["n_symbols"])
    ax.plot(frame["distance_m"], frame["delay_samples"], marker="o", color=OBS,
            lw=1.7, ms=7.0, label="round-trip delay (15 m per sample)")
    ax.set_xlabel("link distance [m]")
    ax.set_ylabel("delay [samples]", rotation=90, labelpad=10)
    ax.set_title("Distance maps to the delay window", pad=8)
    ax.set_ylim(0, 34)
    ax.axhline(1.0, color=fs.GRID_COLOR, lw=1.0)
    ax.axhline(33.0, color=fs.TARGET_COLOR, lw=1.4, ls=(0, (6, 2)))
    ax.annotate("window 1-33 samples ($\\approx$ 500 m)", xy=(float(frame["distance_m"].iloc[-1]),
                33.0), xytext=(-6, -16), textcoords="offset points", ha="right",
                fontsize=10, color=fs.TARGET_COLOR)
    ax.grid(True, which="major", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

    fig.tight_layout()
    fs.legend_below_fig(fig, axes, ncol=3)
    fs.save(fig, "swarm_scene")
    plt.close(fig)

if __name__ == "__main__":
    main()
