from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import figure_style as fs

SRC = REPO / "results" / "full" / "peer_estimation"
ARCHS = ("conv1d", "qkv")
ARCH_LABELS = {"conv1d": "Ultra-CAN (Conv1D)", "qkv": "Ultra-CAN-QKV"}
RATIO_ORDER = ("peer_stronger", "comparable", "target_gt1.25", "target_gt2")
RATIO_LABELS = {
    "peer_stronger": "peer\nstronger",
    "comparable": "comparable",
    "target_gt1.25": "target\nx1.25",
    "target_gt2": "target\nx2",
}
REFERENCE_GUARD = 1.5
OUTCOME = {
    "obstacle": ("#00CC96", "reports the obstacle"),
    "peer": ("#EF553B", "reports the peer (mirror failure)"),
    "elsewhere": ("#B0B0B0", "neither"),
}

def load() -> pd.DataFrame:
    frames = []
    for arch in ARCHS:
        path = SRC / arch / "peer_estimation.csv"
        if not path.is_file():
            raise SystemExit(f"missing {path}: run the peer_estimation experiment")
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    frame["key_bin"] = frame["key_bin"].astype(str)
    return frame

def _rates(frame: pd.DataFrame, summary: str, keys: List[str],
           guard: float = REFERENCE_GUARD) -> Dict[str, np.ndarray]:
    picked = frame[(frame["summary"] == summary)
                   & frame["key_bin"].isin(keys)
                   & np.isclose(frame["guard_factor"], float(guard))]
    obstacle, peer, counts = [], [], []
    for key in keys:
        sub = picked[picked["key_bin"] == key]
        obstacle.append(float(sub["target_id_rate"].mean()))
        peer.append(float(sub["peer_false_rate"].mean()))
        counts.append(int(sub["n"].mean()))
    obstacle = np.array(obstacle)
    peer = np.array(peer)
    return {
        "obstacle": obstacle,
        "peer": peer,
        "elsewhere": np.clip(1.0 - obstacle - peer, 0.0, None),
        "n": np.array(counts),
    }

def _rates_panel(ax, keys: List[str], labels: List[str], rates: Dict[str, np.ndarray],
                 xlabel: str, title: str) -> None:
    x = np.arange(len(keys))
    width = 0.38
    peak = float(np.nanmax(np.concatenate([rates["obstacle"], rates["peer"]])))
    top = max(0.05, peak) * 1.35
    for name, offset in (("obstacle", -width / 2), ("peer", width / 2)):
        color, label = OUTCOME[name]
        values = rates[name]
        bars = ax.bar(x + offset, values, width=width, color=color, label=label,
                      zorder=3)
        for bar, value in zip(bars, values):
            ax.annotate(f"{100 * value:.0f}%",
                        (bar.get_x() + bar.get_width() / 2.0, value),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=8.5, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Share of the samples")
    ax.set_ylim(0.0, top)
    ax.set_yticks([v for v in (0.0, 0.05, 0.10, 0.15) if v <= top])
    ax.set_yticklabels([f"{100 * v:.0f}%" for v in (0.0, 0.05, 0.10, 0.15) if v <= top])
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(True, axis="y", color=fs.GRID_COLOR, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)

def main() -> None:
    fs.apply_style()
    frame = load()

    offsets = sorted(int(value) for value in
                     frame.loc[frame["summary"] == "offset_samples", "key_bin"].unique())
    offset_labels = [("<=-4" if value == min(offsets) else
                      ">=+4" if value == max(offsets) else f"{value:+d}")
                     for value in offsets]
    offset_rates = _rates(frame, "offset_samples", [str(v) for v in offsets])
    ratio_rates = _rates(frame, "amplitude_ratio", list(RATIO_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    _rates_panel(axes[0], offsets, offset_labels, offset_rates,
                 "obstacle delay - nearest peer (samples)",
                 "A peer arriving earlier takes the report over")
    _rates_panel(axes[1], list(RATIO_ORDER), [RATIO_LABELS[k] for k in RATIO_ORDER],
                 ratio_rates, "obstacle / peer echo amplitude",
                 "A stronger obstacle is identified more often")

    fig.tight_layout()
    fs.legend_below_fig(fig, axes, ncol=2)
    fs.save(fig, "peer_estimation")
    plt.close(fig)

    overall = frame[(frame["summary"] == "all") & (frame["key_bin"] == "all")
                    & np.isclose(frame["guard_factor"], REFERENCE_GUARD)]
    print("overall identification per receiver:",
          {row["arch"]: round(float(row["target_id_rate"]), 4)
           for _, row in overall.iterrows()})
    print("worst receiver disagreement on the rates: %.2e"
          % float((frame[frame["summary"] == "all"].groupby("key_bin")
                   ["target_id_rate"].max()
                   - frame[frame["summary"] == "all"].groupby("key_bin")
                   ["target_id_rate"].min()).abs().max()))

if __name__ == "__main__":
    main()

