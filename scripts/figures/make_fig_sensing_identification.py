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
BARS = (
    ("chaotic_mf", "Chaotic\nmatched filter", "#7F7F7F"),
    ("ofdm_60k_extcp_all", "OFDM-ISAC\n60 kHz", "#00A37A"),
    ("peak", "Peak-picker\nbound", "#4D4D4D"),
    ("head_conv1d", "Ultra-CAN", "#636EFA"),
)
WIDTH = 3.35


def _spines(ax) -> None:
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)


def main() -> None:
    frame = pd.read_csv(SRC)
    frame = frame[frame["scenario"] == "iod_peers"]
    if frame.empty:
        raise SystemExit(f"no iod_peers rows in {SRC}")
    ref = frame[np.isclose(frame["snr_db"], SNR_REF)]

    values, labels, colors, hatches = [], [], [], []
    for key, label, color in BARS:
        name = "peak_picker_ceiling" if key == "peak" else key
        row = ref[ref["method"] == name]
        values.append(100.0 * float(row["exact_tau"].iloc[0]))
        labels.append(label)
        colors.append(color)
        hatches.append("//" if key == "peak" else "")

    fs.apply_style()
    fig, ax = plt.subplots(figsize=(WIDTH, 1.45))
    y = np.arange(len(values))[::-1]
    bars = ax.barh(y, values, height=0.62, color=colors, zorder=3,
                   edgecolor=["#111111" if h else "none" for h in hatches],
                   hatch=hatches, linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.0f}%", (value, bar.get_y() + bar.get_height() / 2.0),
                    textcoords="offset points", xytext=(4, 0), va="center",
                    fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0.0, 82.0)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("exact obstacle [%]", labelpad=3)
    ax.grid(True, which="major", axis="x", color=fs.GRID_COLOR, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", color="black", length=2.5)
    _spines(ax)

    fig.tight_layout(pad=0.3)
    fs.save(fig, "sensing_identification")
    plt.close(fig)
    print("bars:", dict(zip([b[0] for b in BARS], np.round(values, 1))))


if __name__ == "__main__":
    main()
