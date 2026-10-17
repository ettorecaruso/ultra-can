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

SRC = REPO / "results" / "full" / "iod_peers" / "ofdm_isac" / "ofdm_isac.csv"
SNR_REF = 21.0
CURVES = (
    ("chaotic_mf", "Chaotic matched filter", "#7F7F7F", "o", "-"),
    ("ofdm_60k_extcp_all", "OFDM-ISAC", "#00A37A", "^", (0, (4, 2))),
    ("head_conv1d", "Ultra-CAN", "#636EFA", "o", "-"),
)
BARS = (
    ("chaotic_mf", "Chaotic\nMF", "#7F7F7F"),
    ("ofdm_60k_extcp_all", "OFDM\n60 kHz", "#00A37A"),
    ("peak", "Peak\nbound", "#4D4D4D"),
    ("head_conv1d", "Ultra-\nCAN", "#636EFA"),
)
WIDTH = 7.2


def _spines(ax) -> None:
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def main() -> None:
    frame = pd.read_csv(SRC)
    frame = frame[frame["scenario"] == "iod_peers"]
    if frame.empty:
        raise SystemExit(f"no iod_peers rows in {SRC}")

    fs.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9),
                             gridspec_kw={"width_ratios": [1.45, 1.0]})

    ax = axes[0]
    for key, label, color, marker, ls in CURVES:
        sub = frame[frame["method"] == key].sort_values("snr_db")
        ax.plot(sub["snr_db"], 100.0 * sub["exact_tau"], label=label, color=color,
                marker=marker, ls=ls, lw=1.6, ms=5.0, zorder=3)
    ceiling = frame[frame["method"] == "peak_picker_ceiling"]
    if not ceiling.empty:
        value = 100.0 * float(ceiling["exact_tau"].iloc[0])
        ax.axhline(value, color="#4D4D4D", lw=1.3, ls=(0, (1, 1.8)), zorder=2,
                   label="peak-picker bound")

    ax.set_ylim(0.0, 80.0)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_xlim(-5.8, 21.8)
    ax.set_xticks([-5, -1, 3, 7, 11, 15, 21])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("exact obstacle [%]", rotation=90, labelpad=9)
    ax.set_title("(a) Against SNR", pad=7)
    ax.grid(True, which="major", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)
    _spines(ax)

    ax = axes[1]
    ref = frame[np.isclose(frame["snr_db"], SNR_REF)]
    values, labels, colors, hatches = [], [], [], []
    for key, label, color in BARS:
        name = "peak_picker_ceiling" if key == "peak" else key
        row = ref[ref["method"] == name]
        values.append(100.0 * float(row["exact_tau"].iloc[0]))
        labels.append(label)
        colors.append(color)
        hatches.append("//" if key == "peak" else "")

    x = np.arange(len(values))
    bars = ax.bar(x, values, width=0.62, color=colors, zorder=3,
                  edgecolor=["#111111" if h else "none" for h in hatches],
                  hatch=hatches, linewidth=0.9)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.0f}%", (bar.get_x() + bar.get_width() / 2.0, value),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.0, 80.0)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_xlabel("method")
    ax.set_title("(b) At 21 dB", pad=7)
    ax.grid(True, which="major", axis="y", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)
    _spines(ax)

    fig.tight_layout()
    fs.legend_below_fig(fig, [axes[0]], ncol=4)
    fs.save(fig, "sensing_identification")
    plt.close(fig)

    print("bars:", dict(zip([b[0] for b in BARS], np.round(values, 1))))


if __name__ == "__main__":
    main()
