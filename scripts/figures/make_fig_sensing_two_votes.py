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

FULL = REPO / "results" / "full"
PEAK_COLOR = "#7F7F7F"
K3_COLOR = "#636EFA"
IOD_COLOR = "#EF553B"
WIDTH = 11.8
SCENARIOS = (("k3_doppler_full", "K = 3, no peer", K3_COLOR, "-"),
             ("iod_peers", "K = 3, two peers", IOD_COLOR, (0, (6, 2))))

def main() -> None:
    frames = {name: pd.read_csv(FULL / name / "ranging_benchmark" / "conv1d.csv")
              for name, _, _, _ in SCENARIOS}

    fs.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.6))

    ax = axes[0]
    x = np.arange(len(SCENARIOS))
    width = 0.34
    first = [float(frames[name][frames[name]["gamma"] == 1.0]["medae_peak_samples"].iloc[0])
             for name, _, _, _ in SCENARIOS]
    second = [float(frames[name][frames[name]["gamma"] == 1.0]["medae_head_samples"].iloc[0])
              for name, _, _, _ in SCENARIOS]
    bars = ax.bar(x - width / 2.0, first, width=width, color=PEAK_COLOR, alpha=0.65,
                  edgecolor=PEAK_COLOR, linewidth=1.1, label="correlation peak", zorder=3)
    for bar, value in zip(bars, first):
        ax.annotate("%.2f" % value, (bar.get_x() + bar.get_width() / 2.0, value),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=9.5, color=PEAK_COLOR)
    bars = ax.bar(x + width / 2.0, second, width=width,
                  color=[color for _, _, color, _ in SCENARIOS], alpha=0.65,
                  edgecolor=[color for _, _, color, _ in SCENARIOS], linewidth=1.1,
                  label="sensing head", zorder=3)
    for bar, value in zip(bars, second):
        ax.annotate("%.2f" % value, (bar.get_x() + bar.get_width() / 2.0, value),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=9.5, color=bar.get_edgecolor())
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _, _ in SCENARIOS])
    ax.set_xlim(-0.6, len(SCENARIOS) - 0.4)
    ax.set_ylim(0.0, max(first + second) * 1.3)
    ax.set_ylabel("median $|\\hat{\\tau}-\\tau|$ [samples]", rotation=90, labelpad=10)
    ax.set_title("Vote 1: how wrong is the visible peak", pad=8)
    ax.grid(True, which="major", axis="y", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)

    ax = axes[1]
    for name, label, color, dash in SCENARIOS:
        order = frames[name].sort_values("gamma")
        ax.plot(order["gamma"], order["silence_fraction"], marker="o", color=color,
                lw=1.7, ms=7.0, ls=dash, label="silence, %s" % label)
        value = float(order["p_oracle"].iloc[0])
    ax.set_xlabel("abstention guard $\\gamma$")
    ax.set_ylabel("fraction", rotation=90, labelpad=10)
    ax.set_title("Vote 2: silence, and is the peak the target?", pad=8)
    ax.set_xticks([1.0, 1.25, 1.5, 2.0])
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="major", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

    fig.tight_layout()
    fs.legend_below_fig(fig, axes, ncol=4)
    fs.save(fig, "sensing_two_votes")
    plt.close(fig)

if __name__ == "__main__":
    main()
