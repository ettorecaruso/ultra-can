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

SRC = REPO / "results" / "full" / "peer_estimation"
ARCHS = ("conv1d", "qkv")
ARCH_LABEL = {"conv1d": "Ultra-CAN (Conv1D)", "qkv": "Ultra-CAN-QKV"}
RATIO_ORDER = ("peer_stronger", "comparable", "target_gt1.25", "target_gt2")
RATIO_LABEL = {
    "peer_stronger": "peer\nstronger",
    "comparable": "similar",
    "target_gt1.25": "1.25$\\times$",
    "target_gt2": "2$\\times$",
}
RATIO_BINS = [0.0, 0.75, 1.25, 2.0, np.inf]
REF_GUARD = 1.5
PEER_COLOR = "#EF553B"
OBST_COLOR = "#00A37A"


def _spines(ax) -> None:
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)


def _samples(arch: str) -> pd.DataFrame:
    path = SRC / arch / "peer_samples.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}: run the peer_estimation experiment")
    return pd.read_csv(path)


def _rates(frame: pd.DataFrame, summary: str, keys, guard: float = REF_GUARD):
    picked = frame[(frame["summary"] == summary) & frame["key_bin"].isin(keys)
                   & np.isclose(frame["guard_factor"], guard)]
    obstacle, peer = [], []
    for key in keys:
        sub = picked[picked["key_bin"] == key]
        obstacle.append(float(sub["target_id_rate"].mean()))
        peer.append(float(sub["peer_false_rate"].mean()))
    return np.array(obstacle), np.array(peer)


def main() -> None:
    samples = {arch: _samples(arch) for arch in ARCHS}
    summary = {arch: pd.read_csv(SRC / arch / "peer_estimation.csv")
               for arch in ARCHS}
    fs.apply_style()

    # ---------------- double figure: range error + reported outcome ----------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.45))
    x = np.arange(len(RATIO_ORDER))
    width = 0.36

    ax = axes[0]
    for offset, arch in zip((-width / 2.0, width / 2.0), ARCHS):
        frame = samples[arch]
        frame = frame[np.isclose(frame["guard_factor"], REF_GUARD)]
        bins = pd.cut(frame["ratio"], bins=RATIO_BINS, labels=list(RATIO_ORDER),
                      right=False)
        values = [float(frame[(bins == key) & (~frame["fallback"])]["abs_err"].mean())
                  for key in RATIO_ORDER]
        bars = ax.bar(x + offset, values, width=width, color=fs.ARCH_COLORS[arch],
                      alpha=0.75, edgecolor=fs.ARCH_COLORS[arch], linewidth=0.8,
                      label=ARCH_LABEL[arch], zorder=3)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2.0, value),
                        textcoords="offset points", xytext=(0, 2), ha="center",
                        fontsize=7, color=fs.ARCH_COLORS[arch])
    ax.set_xticks(x)
    ax.set_xticklabels([RATIO_LABEL[k] for k in RATIO_ORDER], fontsize=7.5)
    ax.set_ylim(0.0, 8.0)
    ax.set_yticks(np.arange(0, 9, 2))
    ax.set_xlabel("obstacle / peer amplitude", labelpad=2)
    ax.set_ylabel("mean $|\\Delta\\tau|$ [samples]", rotation=90, labelpad=4)
    ax.set_title("(a) Range error of the answered bursts", pad=5)
    ax.grid(True, which="major", axis="y", color=fs.GRID_COLOR, lw=0.8)
    ax.set_axisbelow(True)
    _spines(ax)

    ax = axes[1]
    obstacle, peer = _rates(summary["conv1d"], "amplitude_ratio", list(RATIO_ORDER))
    ax.bar(x - width / 2.0, obstacle, width=width, color=OBST_COLOR, zorder=3,
           label="reports the obstacle")
    ax.bar(x + width / 2.0, peer, width=width, color=PEER_COLOR, zorder=3,
           label="reports a peer")
    for xi, value in zip(x, obstacle):
        ax.annotate(f"{100 * value:.0f}%", (xi - width / 2.0, value),
                    textcoords="offset points", xytext=(0, 2), ha="center",
                    fontsize=7, color=OBST_COLOR)
    for xi, value in zip(x, peer):
        if value > 0.02:
            ax.annotate(f"{100 * value:.0f}%", (xi + width / 2.0, value),
                        textcoords="offset points", xytext=(0, 2), ha="center",
                        fontsize=7, color=PEER_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([RATIO_LABEL[k] for k in RATIO_ORDER], fontsize=7.5)
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xlabel("obstacle / peer amplitude", labelpad=2)
    ax.set_ylabel("share of bursts", rotation=90, labelpad=4)
    ax.set_title("(b) Reported outcome", pad=5)
    ax.grid(True, which="major", axis="y", color=fs.GRID_COLOR, lw=0.8)
    ax.set_axisbelow(True)
    _spines(ax)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.32)
    fs.legend_below_fig(fig, axes, ncol=4, fontsize=7.2, gap=0.03)
    fs.save(fig, "peer_range_outcome")
    plt.close(fig)

    # ---------------- single figure: coverage and peer confusion ------------
    frame = samples["conv1d"]
    ref = frame.groupby("guard_factor").agg(
        answered=("fallback", lambda s: 1.0 - float(np.mean(s))))
    guards = [float(v) for v in ref.index]
    answered = 100.0 * ref["answered"].to_numpy(dtype=float)
    peer_ungated = 100.0 * float(frame["peer_false"].mean())
    peer_after = []
    for guard in guards:
        sub = frame[np.isclose(frame["guard_factor"], guard)]
        kept = ~sub["fallback"].to_numpy(dtype=bool)
        peer_after.append(100.0 * float(sub["peer_false"].to_numpy(dtype=bool)[kept].mean()))

    fig, ax = plt.subplots(figsize=(3.2, 1.95))
    ax.plot(guards, answered, marker="o", color=fs.ARCH_COLORS["conv1d"], lw=1.4,
            ms=4.5, label="range reported")
    ax.plot(guards, peer_after, marker="s", color="#7F7F7F", lw=1.4, ms=4.5,
            ls=(0, (5, 2)), label="peer reported")
    ax.axhline(peer_ungated, color="#a33a27", lw=0.9, ls=(0, (1, 1.5)))
    ax.text(1.55, 92.0, f"ungated: {peer_ungated:.1f}%", color="#a33a27",
            fontsize=6.5, va="top", ha="right")
    ax.set_xlim(guards[0] - 0.08, guards[-1] + 0.08)
    ax.set_xticks(guards)
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("peer-exclusion guard $\\gamma$", labelpad=2)
    ax.set_ylabel("rate [%]", rotation=90, labelpad=3)
    ax.grid(True, which="major", color=fs.GRID_COLOR, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", color="black", length=2.5)
    _spines(ax)
    fig.tight_layout(pad=0.3)
    fs.legend_below(ax, ncol=2, fontsize=6.5, y=-0.40)
    fs.save(fig, "peer_coverage")
    plt.close(fig)

    print("answered [%]:", np.round(answered, 1))
    print("peer after gating [%]:", np.round(peer_after, 2))


if __name__ == "__main__":
    main()

