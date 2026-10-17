from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

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
ARCHS = {"conv1d": "Ultra-CAN (Conv1D)", "qkv": "Ultra-CAN-QKV"}
JAMMERS = (("cw", "CW", "-"), ("partial_band", "partial band", (0, (5, 2))))
CHANCE = 0.5
WIDTH = 3.5


def _curve(arch: str, jammer: str) -> pd.DataFrame:
    path = SRC / arch / "per_snr.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}: run the jamming_interpretability experiment")
    frame = pd.read_csv(path)
    frame = frame[np.isclose(frame["snr_db"], SNR_DB) & (frame["jammer"] == jammer)]
    return frame.sort_values("jsr_db")


def _spines(ax) -> None:
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def main() -> None:
    data: Dict[tuple, pd.DataFrame] = {
        (arch, jammer): _curve(arch, jammer)
        for arch in ARCHS for jammer, _, _ in JAMMERS
    }

    fs.apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 4.0), sharex=True)

    ax = axes[0]
    for arch, label in ARCHS.items():
        for jammer, jlabel, ls in JAMMERS:
            sub = data[(arch, jammer)]
            ax.plot(sub["jsr_db"], sub["ber"], color=fs.ARCH_COLORS[arch], ls=ls,
                    marker=fs.ARCH_MARKERS[arch], ms=4.2, lw=1.4,
                    label=f"{label}, {jlabel}")
    ax.axhline(CHANCE, color="#7F7F7F", ls=(0, (1, 1.6)), lw=1.1)
    fs.log_axis(ax, 1e-3, 1.0, x_step=4.0)
    ax.set_ylabel("BER", rotation=0, labelpad=9)
    ax.set_title("(a) Telemetry", pad=6)
    _spines(ax)

    ax = axes[1]
    for arch, label in ARCHS.items():
        for jammer, jlabel, ls in JAMMERS:
            sub = data[(arch, jammer)]
            ax.plot(sub["jsr_db"], sub["corr_tau"], color=fs.ARCH_COLORS[arch],
                    ls=ls, marker=fs.ARCH_MARKERS[arch], ms=4.2, lw=1.4,
                    label=f"{label}, {jlabel}")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("jammer-to-signal ratio (dB)")
    ax.set_ylabel("delay correlation", rotation=90, labelpad=7)
    ax.set_title("(b) Ranging", pad=6)
    ax.set_xlim(float(data[("conv1d", "cw")]["jsr_db"].min()) - 0.5,
                float(data[("conv1d", "cw")]["jsr_db"].max()) + 0.5)
    ax.grid(True, which="major", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)
    _spines(ax)

    fig.tight_layout()
    fs.legend_below_fig(fig, axes, ncol=2)
    fs.save(fig, "sensing_under_jamming")
    plt.close(fig)


if __name__ == "__main__":
    main()
