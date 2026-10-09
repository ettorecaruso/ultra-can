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

SRC = REPO / "results" / "full" / "peer_estimation"
ARCHS = ["conv1d", "qkv"]
ARCH_LABEL = {"conv1d": "Ultra-CAN (Conv1D)", "qkv": "Ultra-CAN-QKV"}
PEER_COLOR = "#7F7F7F"
RATIO_ORDER = ["peer_stronger", "comparable", "target_gt1.25", "target_gt2"]
RATIO_LABEL = {"peer_stronger": "peer stronger", "comparable": "comparable",
               "target_gt1.25": "obstacle $>$1.25$\\times$",
               "target_gt2": "obstacle $>$2$\\times$"}
TOL = 0.5
WIDTH = 11.8

def _spines(ax) -> None:
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

def main() -> None:
    guard = {}
    ratio = {}
    for arch in ARCHS:
        frame = pd.read_csv(SRC / arch / "peer_estimation.csv")
        guard[arch] = frame[frame["summary"] == "all"].sort_values("guard_factor")
        ratio[arch] = frame[(frame["summary"] == "amplitude_ratio")
                            & (frame["guard_factor"] == 1.0)].set_index("key_bin")

    fs.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.6))

    ax = axes[0]
    ref = guard["conv1d"]
    x_guard = [float(v) for v in ref["guard_factor"]]
    ax.plot(x_guard, 1.0 - ref["fallback_rate"], marker="o",
            color=fs.ARCH_COLORS["conv1d"], lw=1.7, ms=7.0,
            label="range reported (either proposed variant)")
    ax.plot(x_guard, ref["peer_false_rate"], marker="s", color=PEER_COLOR,
            lw=1.7, ms=7.0, ls=(0, (6, 2)),
            label="peer reported as the obstacle")
    ax.set_ylim(0.0, 0.8)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlim(x_guard[0] - 0.08, x_guard[-1] + 0.08)
    ax.set_xticks(x_guard)
    ax.set_xlabel("peer-exclusion guard factor")
    ax.set_ylabel("rate", rotation=0, labelpad=16)
    ax.set_title("Fail-safe: abstention is the knob", pad=8)
    ax.grid(True, which="major", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)
    ax.annotate("1.4%: never traded for coverage",
                xy=(x_guard[-1], float(ref["peer_false_rate"].iloc[-1])),
                xytext=(-10, 22), textcoords="offset points", fontsize=10,
                ha="right", color="#a33a27")
    _spines(ax)

    ax = axes[1]
    x = list(range(len(RATIO_ORDER)))
    width = 0.34
    for offset, arch in zip((-0.5, 0.5), ARCHS):
        values = [float(ratio[arch].loc[key, "mean_abs_err"]) for key in RATIO_ORDER]
        bars = ax.bar([xi + offset * width for xi in x], values, width=width,
                      color=fs.ARCH_COLORS[arch], alpha=0.65,
                      edgecolor=fs.ARCH_COLORS[arch], linewidth=1.1,
                      label=ARCH_LABEL[arch], zorder=3)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.1f}",
                        (bar.get_x() + bar.get_width() / 2.0, value),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=9.5, color=fs.ARCH_COLORS[arch])
    ax.set_xticks(x)
    ax.set_xticklabels([RATIO_LABEL[key] for key in RATIO_ORDER])
    ax.set_ylim(0.0, 7.6)
    ax.set_yticks(range(0, 8))
    ax.set_xlim(-0.6, 3.6)
    ax.set_xlabel("obstacle-to-peer amplitude ratio")
    ax.set_title("Obstacle range error [samples]", pad=8)
    ax.grid(True, which="major", axis="y", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)
    _spines(ax)

    fig.tight_layout()
    fs.legend_below_fig(fig, axes, ncol=4)
    fs.save(fig, "peer_aware_ranging")
    plt.close(fig)

if __name__ == "__main__":
    main()
