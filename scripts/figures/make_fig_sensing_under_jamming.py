from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import figure_style as fs

SRC = REPO / "results" / "full" / "jamming_interpretability"
SNR_DB = 21.0
ARCHS = ("conv1d", "qkv", "lstm", "mc_dlsk")
ARCH_LABELS = {
    "conv1d": "Ultra-CAN (Conv1D)",
    "qkv": "Ultra-CAN-QKV",
    "lstm": "LSTM-OFDM-DCSK",
    "mc_dlsk": "MC-DLCSK",
}
PANELS = (("cw", "CW tone"), ("partial_band", "Partial band"), ("barrage", "Barrage"))
CHANCE = 0.5
Y_BER_LO, Y_BER_HI = 1e-4, 1.0

def load(arch: str) -> pd.DataFrame:
    path = SRC / arch / "per_snr.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}: run the jamming_interpretability experiment")
    df = pd.read_csv(path)
    df = df[np.isclose(df["snr_db"], SNR_DB)]
    return df

def main() -> None:
    data: Dict[str, pd.DataFrame] = {a: load(a) for a in ARCHS}
    clean = {
        a: float(data[a].loc[data[a]["jammer"] == "clean", "corr_tau"].mean())
        for a in ARCHS
    }

    fs.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for ax, (jammer, title) in zip(axes, PANELS):
        rows = {
            a: data[a][data[a]["jammer"] == jammer].sort_values("jsr_db")
            for a in ARCHS
        }
        jsr = rows[ARCHS[0]]["jsr_db"].to_numpy(dtype=float)
        ax2 = ax.twinx()
        for arch in ARCHS:
            sub = rows[arch]
            kw = fs.series_kwargs(arch)
            ax.plot(jsr, sub["ber"].to_numpy(dtype=float), label=ARCH_LABELS[arch], **kw)
            ax2.plot(jsr, sub["corr_tau"].to_numpy(dtype=float), **{**kw, "ls": (0, (4, 2)), "alpha": 0.55})
            ax2.axhline(clean[arch], color=kw["color"], ls=":", lw=1.0, alpha=0.5)
        ax.axhline(CHANCE, color="#7F7F7F", ls=(0, (1, 1.6)), lw=1.2, zorder=1)
        fs.log_axis(ax, Y_BER_LO, Y_BER_HI, x_step=4.0)
        ax.set_xlim(jsr.min() - 0.5, jsr.max() + 0.5)
        ax.set_title(title)
        ax.set_xlabel("JSR (dB)")
        ax2.set_ylim(0.0, 1.0)
        ax2.set_ylabel("delay correlation")
        ax2.grid(False)
        ax2.tick_params(direction="out", color="black")
        series = ax.get_legend_handles_labels()
        legend_handles, legend_labels = series

    axes[0].set_ylabel("BER")
    fig.tight_layout()
    fs.legend_below_fig(fig, axes, ncol=4)
    fs.save(fig, "sensing_under_jamming")
    plt.close(fig)
    print("clean delay correlation:", {k: round(v, 3) for k, v in clean.items()})

if __name__ == "__main__":
    main()
