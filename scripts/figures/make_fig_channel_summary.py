
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import figure_style as fs

DEFAULT_RUN_DIR = REPO / "results" / "full" / "channel_generalization"
ORACLE = REPO / "results" / "full" / "diagnostics" / "channel_label_observability" / "oracle.csv"
ARCHS = ("conv1d", "qkv")
ARCH_LABELS = {"conv1d": "Ultra-CAN (Conv1D)", "qkv": "Ultra-CAN-QKV"}
ORDER = ("nominal", "b_tdl_d_light", "b_tdl_d", "c_two_ray_jakes", "b_tdl_a")
TICKS = {"nominal": "nominal", "b_tdl_d_light": "TDL-D\n5%", "b_tdl_d": "TDL-D\n25%",
         "c_two_ray_jakes": "two-ray\n+Jakes", "b_tdl_a": "TDL-A\nNLOS"}
Y_LO, Y_HI = 1e-4, 4e-1

def _summary(run_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for arch in ARCHS:
        path = Path(run_dir) / arch / "summary.csv"
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"no summary.csv under {run_dir}")
    return pd.concat(frames, ignore_index=True)

def main(argv: Optional[List[str]] = None) -> None:
    run_dir = Path(argv[0]).resolve() if argv else DEFAULT_RUN_DIR
    data = _summary(run_dir)
    variants = [v for v in ORDER if v in set(data["channel_variant"])]
    x = np.arange(len(variants), dtype=float)

    fs.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    fs.log_axis(axes[0], Y_LO, Y_HI, x_step=1.0)
    for arch in ARCHS:
        sub = data[data.arch == arch].set_index("channel_variant")
        y = [float(sub.loc[v, "pooled_ber"]) for v in variants]
        lo = [float(sub.loc[v, "pooled_ber_ci_lo"]) for v in variants]
        hi = [float(sub.loc[v, "pooled_ber_ci_hi"]) for v in variants]
        kw = fs.series_kwargs(arch)
        axes[0].fill_between(x, np.clip(lo, Y_LO, Y_HI), np.clip(hi, Y_LO, Y_HI),
                             color=kw["color"], alpha=0.18, lw=0, zorder=2)
        axes[0].plot(x, y, label=ARCH_LABELS[arch], zorder=3, **kw)
    axes[0].set_title("Communication: pooled BER", pad=8)
    axes[0].set_ylabel("BER")

    oracle = pd.read_csv(ORACLE) if ORACLE.is_file() else None
    note = {str(r["channel_variant"]): float(r["p_label_is_strongest"])
            for _, r in oracle.iterrows()} if oracle is not None else {}
    for arch in ARCHS:
        sub = data[data.arch == arch].set_index("channel_variant")
        y = [float(sub.loc[v, "corr_tau_top"]) for v in variants]
        axes[1].plot(x, y, label=ARCH_LABELS[arch], zorder=3,
                     **fs.series_kwargs(arch))
    axes[1].set_title("Sensing: delay correlation of the dominant echo", pad=8)
    axes[1].set_ylabel(r"corr($\hat{\tau}$, $\tau$)")
    for xi, v in zip(x, variants):
        if v in note:
            axes[1].annotate(rf"$P_{{orb}}={note[v]:.2f}$", (xi, 0.02),
                             ha="center", va="bottom", fontsize=8.5, color="#444444")
    axes[1].set_ylim(-0.05, 0.9)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([TICKS.get(v, v) for v in variants], fontsize=9.5)
        ax.set_xlim(-0.4, len(variants) - 0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fs.legend_below_fig(fig, axes, ncol=2)
    fs.save(fig, "channel_generalization_summary")
    plt.close(fig)

if __name__ == "__main__":
    main(sys.argv[1:])
