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
    "target_gt1.25": "$>$1.25$\\times$",
    "target_gt2": "$>$2$\\times$",
}
RATIO_BINS = [0.0, 0.75, 1.25, 2.0, np.inf]
REF_GUARD = 1.5
WIDTH = 3.5


def _spines(ax) -> None:
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def _samples(arch: str) -> pd.DataFrame:
    path = SRC / arch / "peer_samples.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}: run the peer_estimation experiment")
    return pd.read_csv(path)


def main() -> None:
    frames = {arch: _samples(arch) for arch in ARCHS}

    fs.apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 5.2))

    # ---- (a) gate coverage and peer confusion -------------------------------
    ax = axes[0]
    frame = frames["conv1d"]
    ref = frame.groupby("guard_factor").agg(
        answered=("fallback", lambda s: 1.0 - float(np.mean(s))),
        peer_ungated=("peer_false", "mean"),
    )
    guards = [float(v) for v in ref.index]
    answered = 100.0 * ref["answered"].to_numpy(dtype=float)
    peer_ungated = 100.0 * float(ref["peer_ungated"].iloc[0])

    peer_after = []
    for guard in guards:
        sub = frame[np.isclose(frame["guard_factor"], guard)]
        kept = ~sub["fallback"].to_numpy(dtype=bool)
        peer_after.append(
            100.0 * float(sub["peer_false"].to_numpy(dtype=bool)[kept].mean())
        )

    ax.plot(guards, answered, marker="o", color=fs.ARCH_COLORS["conv1d"], lw=1.7,
            ms=7.0, label="range reported")
    ax.plot(guards, peer_after, marker="s", color="#7F7F7F", lw=1.7, ms=7.0,
            ls=(0, (6, 2)), label="peer reported as the obstacle")
    ax.axhline(peer_ungated, color="#a33a27", lw=1.2, ls=(0, (1, 1.6)))
    ax.text(1.02, 74.0, f"head before gating: {peer_ungated:.1f}%",
            color="#a33a27", fontsize=9, va="top")
    ax.set_xlim(guards[0] - 0.08, guards[-1] + 0.08)
    ax.set_xticks(guards)
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("peer-exclusion guard $\\gamma$")
    ax.set_ylabel("rate [%]", rotation=90, labelpad=8)
    ax.set_title("(a) Coverage and peer confusion", pad=7)
    ax.grid(True, which="major", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)
    _spines(ax)

    # ---- (b) range error of the answered bursts ----------------------------
    ax = axes[1]
    x = np.arange(len(RATIO_ORDER))
    width = 0.36
    for offset, arch in zip((-width / 2.0, width / 2.0), ARCHS):
        frame = frames[arch]
        frame = frame[np.isclose(frame["guard_factor"], REF_GUARD)]
        bins = pd.cut(frame["ratio"], bins=RATIO_BINS, labels=list(RATIO_ORDER),
                      right=False)
        values = []
        for key in RATIO_ORDER:
            sub = frame[(bins == key) & (~frame["fallback"])]
            values.append(float(sub["abs_err"].mean()))
        bars = ax.bar(x + offset, values, width=width, color=fs.ARCH_COLORS[arch],
                      alpha=0.75, edgecolor=fs.ARCH_COLORS[arch], linewidth=1.0,
                      label=ARCH_LABEL[arch], zorder=3)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.1f}",
                        (bar.get_x() + bar.get_width() / 2.0, value),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=8.5, color=fs.ARCH_COLORS[arch])
    ax.set_xticks(x)
    ax.set_xticklabels([RATIO_LABEL[key] for key in RATIO_ORDER], fontsize=8)
    ax.set_ylim(0.0, 8.0)
    ax.set_yticks(np.arange(0, 9, 2))
    ax.set_xlim(-0.6, len(RATIO_ORDER) - 0.4)
    ax.set_xlabel("obstacle-to-peer echo amplitude")
    ax.set_ylabel("mean $|\\Delta\\tau|$ [samples]", rotation=90, labelpad=8)
    ax.set_title("(b) Range error of the answered bursts", pad=7)
    ax.grid(True, which="major", axis="y", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)
    _spines(ax)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.55)
    fs.legend_below_fig(fig, axes, ncol=2)
    fs.save(fig, "peer_aware_ranging")
    plt.close(fig)

    print("answered [%]:", [round(v, 1) for v in answered])
    print("peer after gating [%]:", [round(v, 2) for v in peer_after])


if __name__ == "__main__":
    main()
