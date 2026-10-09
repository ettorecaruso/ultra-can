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

CLEAN = REPO / "results/full/jamming_interpretability/qkv/conditions.csv"
AWARE = REPO / "results/full/jamming_interpretability/jamming_aware_training/qkv/conditions.csv"
JAMMERS = [("cw", "CW"), ("barrage", "Barrage (control)"),
           ("partial_band", "Partial band")]

def _curve(df, jammer):
    c = df[(df.jammer == jammer)].dropna(subset=["jsr_db"]).sort_values("jsr_db")
    return c["jsr_db"].to_numpy(float), c["margin_mean"].to_numpy(float)

def main() -> None:
    if not AWARE.is_file():
        print("skipping jamaware_margin: %s not found; run "
              "scripts/tools/run_jam_aware_qkv.py first" % AWARE)
        return
    clean = pd.read_csv(CLEAN)
    aware = pd.read_csv(AWARE)
    clean_margin = float(clean.loc[clean.jammer == "clean", "margin_mean"].iloc[0])

    fs.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    for ax, (jammer, title) in zip(axes, JAMMERS):
        for df, name, label in ((clean, "clean_trained", "Clean-trained"),
                                (aware, "jamming_aware", "Jamming-aware")):
            x, y = _curve(df, jammer)
            ax.plot(x, y, label=label, zorder=3, **fs.series_kwargs(name))
        ax.axhline(clean_margin, color="0.45", ls=(0, (1, 1.6)), lw=1.2, zorder=1)
        ax.set_title(title, pad=8)
        ax.set_xlabel("JSR (dB)")
        ax.set_axisbelow(True)
        ax.grid(True, which="major", axis="y", color=fs.GRID_COLOR, lw=1.0)
    axes[0].set_ylabel("mean decision margin")
    axes[0].set_ylim(0.0, 3.2)
    axes[0].annotate("clean (no jammer)", (float(clean["jsr_db"].min()), clean_margin),
                     textcoords="offset points", xytext=(4, 5), fontsize=9,
                     color="0.35")
    fig.tight_layout()
    fs.legend_below(axes[1], ncol=2, y=-0.20)
    fs.save(fig, "jamaware_margin")
    plt.close(fig)

if __name__ == "__main__":
    main()
