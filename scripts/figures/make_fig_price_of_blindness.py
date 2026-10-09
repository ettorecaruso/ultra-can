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

SRC = REPO / "results" / "full" / "k3_doppler_full" / "blind_vs_oracle" / "conv1d.csv"
WIDTH = 7.6

def main() -> None:
    frame = pd.read_csv(SRC).sort_values("snr_db")

    fs.apply_style()
    fig, ax = plt.subplots(figsize=(WIDTH, 3.8))
    ax.plot(frame["snr_db"], frame["ber_blind"], marker="o",
            color=fs.ARCH_COLORS["conv1d"], lw=1.7, ms=7.0, label="blind receiver (I/Q only)")
    ax.plot(frame["snr_db"], frame["ber_oracle"], marker="s",
            color=fs.ARCH_COLORS["lstm"], lw=1.7, ms=7.0, ls=(0, (6, 2)),
            label="oracle: the seed is known locally")
    for x, y in zip(frame["snr_db"], frame["ber_blind"]):
        ax.annotate("%.3f" % y, (float(x), float(y)), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=9.5,
                    color=fs.ARCH_COLORS["conv1d"])
    ax.annotate("0.000: the coherent bound", xy=(float(frame["snr_db"].iloc[-1]), 0.0),
                xytext=(-6, 12), textcoords="offset points", ha="right", fontsize=10,
                color=fs.ARCH_COLORS["lstm"])
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("BER", rotation=90, labelpad=10)
    ax.set_title("The price of blindness", pad=8)
    ax.set_xticks([float(v) for v in frame["snr_db"]])
    ax.set_ylim(-0.0008, 0.0062)
    ax.grid(True, which="major", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    fs.legend_below(ax, ncol=2)
    fs.save(fig, "price_of_blindness")
    plt.close(fig)

if __name__ == "__main__":
    main()
