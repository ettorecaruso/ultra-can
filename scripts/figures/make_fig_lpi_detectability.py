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

SRC = REPO / "results/full/diagnostics/lpi_detectability"
ORDER = ("proposed (CSK)", "BPSK", "OOK", "DSSS-BPSK")
LABELS = {"proposed (CSK)": "CSK", "BPSK": "BPSK", "OOK": "OOK",
          "DSSS-BPSK": "DSSS-BPSK"}
COLOURS = {"proposed (CSK)": "#636EFA", "BPSK": "#EF553B", "OOK": "#FFA15A",
           "DSSS-BPSK": "#00CC96"}
Y_LO, Y_HI = 1e-3, 1.0

def main() -> None:
    desc_path = SRC / "lpi_descriptors.csv"
    roc_path = SRC / "lpi_roc.csv"
    if not desc_path.is_file() or not roc_path.is_file():
        raise SystemExit(f"missing {desc_path}: run "
                         "scripts/diagnostics/lpi_detectability.py")
    desc = pd.read_csv(desc_path).set_index("waveform")
    roc = pd.read_csv(roc_path)

    fs.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    names = [n for n in ORDER if n in desc.index]
    y = np.arange(len(names), dtype=float)
    axes[0].barh(y, [float(desc.loc[n, "spectral_flatness"]) for n in names],
                 color=[COLOURS[n] for n in names], alpha=0.45,
                 edgecolor=[COLOURS[n] for n in names], linewidth=1.4, zorder=2)
    for yi, n in zip(y, names):
        axes[0].annotate(
            f"{float(desc.loc[n, 'spectral_flatness']):.3f}  "
            f"({float(desc.loc[n, 'peak_to_floor_db']):.1f} dB)",
            (float(desc.loc[n, "spectral_flatness"]) + 0.03, yi),
            ha="left", va="center", fontsize=9, color="#222222")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABELS[n] for n in names], fontsize=10)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.0, 1.45)
    axes[0].set_xlabel("spectral flatness (1 = white); peak-to-floor in brackets")
    axes[0].set_title("Spectral structure an interceptor can exploit", pad=8)
    axes[0].grid(True, which="major", axis="x", color=fs.GRID_COLOR, lw=1.0)

    fs.log_axis(axes[1], Y_LO, Y_HI)
    for n in names:
        sub = roc[roc.waveform == n].sort_values("snr_db")
        axes[1].plot(sub["snr_db"], np.clip(sub["pd_channelized"], Y_LO, Y_HI),
                     color=COLOURS[n], marker="o", ms=4.5, lw=1.4,
                     label=LABELS[n], zorder=3)
    axes[1].axhline(1e-2, color="0.4", ls=":", lw=1.2, zorder=1)
    axes[1].set_xlabel(r"total SNR in the window $\gamma$ (dB)")
    axes[1].set_ylabel(r"$P_d$", rotation=0, labelpad=16)
    axes[1].set_title(r"Channelized radiometer, $P_{fa}=10^{-2}$", pad=8)
    axes[1].set_axisbelow(True)

    fig.tight_layout()
    fs.legend_below_fig(fig, axes, ncol=4)
    fs.save(fig, "lpi_detectability")
    plt.close(fig)

if __name__ == "__main__":
    main()
