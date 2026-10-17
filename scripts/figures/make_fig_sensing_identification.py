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
METHODS = {
    "chaotic_mf": ("Chaotic matched filter", "#7F7F7F", "o", "-"),
    "ofdm_60k_extcp_all": ("OFDM-ISAC 60 kHz", "#00A37A", "^", (0, (4, 2))),
    "head_conv1d": ("Ultra-CAN (Conv1D)", "#636EFA", "o", "-"),
    "head_qkv": ("Ultra-CAN-QKV", "#EF553B", "s", (0, (6, 2))),
}
BAR_ORDER = (
    ("chaotic_mf", "Chaotic matched filter", "#7F7F7F"),
    ("ofdm_60k_extcp_all", "OFDM-ISAC 60 kHz", "#00A37A"),
    ("ofdm_15k_all", "OFDM-ISAC 15 kHz", "#66C2A5"),
    ("peak", "Peak-picker bound", "#4D4D4D"),
    ("head_conv1d", "Ultra-CAN (Conv1D)", "#636EFA"),
    ("head_qkv", "Ultra-CAN-QKV", "#EF553B"),
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
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    for key, (label, color, marker, ls) in METHODS.items():
        sub = frame[frame["method"] == key].sort_values("snr_db")
        if sub.empty:
            continue
        ax.plot(sub["snr_db"], 100.0 * sub["exact_tau"], label=label,
                color=color, marker=marker, ls=ls, lw=1.6, ms=5.2, zorder=3)

    ceiling = frame[frame["method"] == "peak_picker_ceiling"]
    if not ceiling.empty:
        value = 100.0 * float(ceiling["exact_tau"].iloc[0])
        ax.axhline(value, color="#4D4D4D", lw=1.3, ls=(0, (1, 1.8)), zorder=2,
                   label="peak-picker bound")

    ax.set_ylim(30.0, 75.0)
    ax.set_yticks([30, 40, 50, 60, 70])
    ax.set_xlim(-5.8, 21.8)
    ax.set_xticks([-5, -1, 3, 7, 11, 15, 21])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("exact obstacle delay [%]", rotation=90, labelpad=7)
    ax.set_title("(a) Exact identification against SNR", pad=7)
    ax.grid(True, which="major", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)
    _spines(ax)

    ax = axes[1]
    ref = frame[np.isclose(frame["snr_db"], SNR_REF)]
    values, labels, colors, hatches = [], [], [], []
    for key, label, color in BAR_ORDER:
        name = "peak_picker_ceiling" if key == "peak" else key
        row = ref[ref["method"] == name]
        value = float(row["exact_tau"].iloc[0]) if not row.empty else np.nan
        values.append(100.0 * value)
        labels.append(label)
        colors.append(color)
        hatches.append("//" if key == "peak" else "")

    y = np.arange(len(values))[::-1]
    bars = ax.barh(y, values, height=0.68, color=colors, zorder=3,
                   edgecolor=["#111111" if h else "none" for h in hatches],
                   hatch=hatches, linewidth=0.9)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.0f}%", (value, bar.get_y() + bar.get_height() / 2.0),
                    textcoords="offset points", xytext=(4, 0), va="center",
                    fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0.0, 80.0)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("exact obstacle delay [%]")
    ax.set_title("(b) Comparison at 21 dB", pad=7)
    ax.grid(True, which="major", axis="x", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)
    _spines(ax)

    fig.tight_layout()
    fs.legend_below_fig(fig, [axes[0]], ncol=3)
    fs.save(fig, "sensing_identification")
    plt.close(fig)

    print("bars:", dict(zip([k for k, _, _ in BAR_ORDER], np.round(values, 1))))


if __name__ == "__main__":
    main()
