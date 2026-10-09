
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import figure_style as fs

DEFAULT_RUN_DIR = REPO / "results" / "full" / "channel_generalization"
ARCHS = ("conv1d", "qkv")
ARCH_TITLES = {
    "conv1d": "Ultra-CAN (Conv1D)",
    "qkv": "Ultra-CAN-QKV",
    "lstm": "LSTM-OFDM-DCSK",
    "mc_dlsk": "MC-DLCSK",
}
VARIANT_LABELS = {
    "nominal": "Nominal",
    "b_tdl_d": "3GPP TDL-D",
    "b_tdl_d_light": "TDL-D, 5% clutter",
    "b_tdl_a": "3GPP TDL-A",
    "c_two_ray_jakes": "Two-ray + Jakes",
}
VARIANT_STYLE = {
    "nominal": ("#636EFA", "o", "-"),
    "b_tdl_d": ("#EF553B", "s", (0, (6, 2))),
    "b_tdl_d_light": ("#FFA15A", "P", (0, (3, 1, 1, 1))),
    "b_tdl_a": ("#00CC96", "D", (0, (4, 1.2, 1, 1.2))),
    "c_two_ray_jakes": ("#AB63FA", "^", (0, (1, 1.6))),
}
FALLBACK_VARIANTS = (
    "nominal", "b_tdl_d", "b_tdl_d_light", "b_tdl_a", "c_two_ray_jakes",
)
Y_LO, Y_HI = 1e-4, 4e-1

def _variants(run_dir: Path) -> List[str]:
    for arch in ARCHS:
        path = run_dir / arch / "summary.csv"
        if path.is_file():
            summary = pd.read_csv(path)
            names = [str(name) for name in summary["channel_variant"]]
            if names:
                return names
    return list(FALLBACK_VARIANTS)

def _curve(run_dir: Path, arch: str, variant: str) -> Optional[pd.DataFrame]:
    path = run_dir / arch / variant / "metrics.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    frame = frame[np.isfinite(frame["ber"]) & (frame["ber"] > 0.0)]
    return frame.sort_values("snr_db")

def main(argv: Optional[List[str]] = None) -> None:
    run_dir = Path(argv[0]).resolve() if argv else DEFAULT_RUN_DIR
    if not run_dir.is_dir():
        raise SystemExit(f"run directory not found: {run_dir}")
    variants = [name for name in _variants(run_dir) if name in VARIANT_STYLE]
    if not variants:
        raise SystemExit(f"no known channel variant under {run_dir}")

    fs.apply_style()
    fig, axes = plt.subplots(1, len(ARCHS), figsize=(10.6, 3.2), sharey=True)
    for ax, arch in zip(axes, ARCHS):
        fs.log_axis(ax, Y_LO, Y_HI, x_step=4.0)
        drawn = 0
        span = None
        ticks = set()
        for variant in variants:
            frame = _curve(run_dir, arch, variant)
            if frame is None or frame.empty:
                continue
            color, marker, linestyle = VARIANT_STYLE[variant]
            x = frame["snr_db"].to_numpy(dtype=float)
            y = frame["ber"].to_numpy(dtype=float)
            low = frame.get("ber_ci_lo", y)
            high = frame.get("ber_ci_hi", y)
            low = np.clip(np.asarray(low, dtype=float), Y_LO, Y_HI)
            high = np.clip(np.asarray(high, dtype=float), Y_LO, Y_HI)
            ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
            ax.plot(
                x, y,
                color=color, marker=marker, ls=linestyle,
                lw=fs.LINE_W, ms=fs.MARKER_SIZE,
                label=VARIANT_LABELS.get(variant, variant),
                zorder=3,
            )
            drawn += 1
            lo = float(x.min())
            hi = float(x.max())
            span = (lo, hi) if span is None else (min(span[0], lo), max(span[1], hi))
            ticks.update(float(value) for value in x)
        if drawn and span is not None:
            ax.set_xlim(span[0], span[1])
            ax.set_xticks(sorted(ticks))
        ax.set_title(ARCH_TITLES.get(arch, arch), pad=8)
        ax.set_xlabel("SNR (dB)")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("BER", rotation=0, labelpad=14)
    fig.suptitle(
        "Ultra-CAN receivers across channel models "
        "(shaded bands: 95% Wilson interval)",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    fs.legend_below_fig(fig, axes, ncol=5)
    fs.save(fig, "channel_generalization")
    plt.close(fig)

if __name__ == "__main__":
    main(sys.argv[1:])
