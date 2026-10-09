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

SRC = REPO / "results" / "full" / "k3_ambiguity_mix" / "ambiguity"
ARCHS = ["conv1d", "qkv"]
ARCH_TITLE = {"conv1d": "Ultra-CAN (Conv1D)", "qkv": "Ultra-CAN-QKV"}
PILES = ["easy", "ambiguous"]
PILE_LABEL = {"easy": "unambiguous pile", "ambiguous": "ambiguous pile"}
PEAK_COLOR = "#7F7F7F"
WIDTH = 11.8

def main() -> None:
    frames = {arch: pd.read_csv(SRC / f"{arch}.csv").set_index("pile") for arch in ARCHS}

    fs.apply_style()
    fig, axes = plt.subplots(1, len(ARCHS), figsize=(WIDTH, 3.6), sharey=True)
    x = np.arange(len(PILES))
    width = 0.34

    for ax, arch in zip(axes, ARCHS):
        frame = frames[arch]
        peak = [float(frame.loc[pile, "medae_peak_samples"]) for pile in PILES]
        head = [float(frame.loc[pile, "medae_head_samples"]) for pile in PILES]
        bars = ax.bar(x - width / 2.0, peak, width=width, color=PEAK_COLOR, alpha=0.65,
                      edgecolor=PEAK_COLOR, linewidth=1.1, label="correlation peak",
                      zorder=3)
        for bar, value in zip(bars, peak):
            ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2.0, value),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=9.5, color=PEAK_COLOR)
        bars = ax.bar(x + width / 2.0, head, width=width, color=fs.ARCH_COLORS[arch],
                      alpha=0.65, edgecolor=fs.ARCH_COLORS[arch], linewidth=1.1,
                      label="sensing head", zorder=3)
        for bar, value in zip(bars, head):
            ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2.0, value),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=9.5, color=fs.ARCH_COLORS[arch])
        ax.set_xticks(x)
        ax.set_xticklabels([PILE_LABEL[pile] for pile in PILES])
        ax.set_xlim(-0.6, len(PILES) - 0.4)
        ax.set_ylim(0.0, max(peak + head) + 3.5)
        ax.set_xlabel("profile pile")
        ax.set_title(ARCH_TITLE[arch], pad=8)
        ax.grid(True, which="major", axis="y", color=fs.GRID_COLOR, lw=1.0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

    axes[0].set_ylabel("median $|\\hat{\\tau}-\\tau|$ [samples]", rotation=90, labelpad=10)
    fig.tight_layout()
    fs.legend_below_fig(fig, axes, ncol=2)
    fs.save(fig, "ambiguity_verdict")
    plt.close(fig)

if __name__ == "__main__":
    main()
