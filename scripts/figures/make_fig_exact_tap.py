"""Exact-tap sensing accuracy: head vs matched-filter peak picker.

Three panels (the sensing story of the paper):
  (a) exact-tap accuracy vs SNR, geometric scenarios;
  (b) supervision ladder at 21 dB: the head follows whatever supervision it is
      given, and only the geometric supervision beats the matched filter;
  (c) why: the gain is concentrated on the bursts in which the obstacle is
      *not* the strongest echo.
"""
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

FULL = REPO / "results" / "full"
ARCH = "conv1d"
PEAK_COLOR = "#7F7F7F"
WIDTH = 15.0

SCEN_LABEL = {
    "k3_doppler_full": "multi-echo, full Doppler",
    "iod_peers": "peer-interference scene",
}
LADDER = (
    ("k3_doppler_full", "geometric\n(obstacle tap)"),
    ("k3_ambiguity_mix", "matched-filter peak\n(ambiguity scene)"),
    ("k1_ablation_log_uniform", "matched-filter peak\n(log-uniform scene)"),
    ("k3_doppler_control", "legacy scalar delay"),
)


def _metrics(scen: str, arch: str = ARCH) -> pd.DataFrame:
    path = FULL / "ber_vs_snr" / scen / arch / "metrics.csv"
    return pd.read_csv(path)


def _panel_a(ax) -> None:
    for scen, color in (("k3_doppler_full", "#636EFA"), ("iod_peers", "#EF553B")):
        frame = _metrics(scen)
        ax.plot(frame["snr_db"], frame["exact_tau"], color=color, lw=1.8,
                marker="o", ms=4.5, label=f"sensing head, {SCEN_LABEL[scen]}")
        ax.plot(frame["snr_db"], frame["exact_tau_argmax"], color=color, lw=1.4,
                ls=(0, (4, 2)), marker=None, alpha=0.55,
                label=f"matched-filter peak, {SCEN_LABEL[scen]}")
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("exact-tap accuracy $P(|\\hat\\tau-\\tau|\\leq 0.5)$")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("(a) localizing the obstacle tap", pad=8)
    ax.grid(True, which="major", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)


def _panel_b(ax) -> None:
    head, peak, labels = [], [], []
    for scen, label in LADDER:
        frame = _metrics(scen)
        row = frame.loc[frame["snr_db"].sub(21.0).abs().idxmin()]
        head.append(float(row["exact_tau"]))
        peak.append(float(row["exact_tau_argmax"]))
        labels.append(label)
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2.0, peak, width, color=PEAK_COLOR, alpha=0.55,
           edgecolor=PEAK_COLOR, lw=1.0, label="matched-filter peak")
    bars = ax.bar(x + width / 2.0, head, width, color="#636EFA", alpha=0.85,
                  edgecolor="#636EFA", lw=1.0, label="sensing head")
    for bar, value in zip(bars, head):
        ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2.0, value),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8.5, color="#1F2A6E")
    for xi, value in zip(x - width / 2.0, peak):
        ax.annotate(f"{value:.3f}", (xi + width / 4.0, value),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8.0, color="#4D4D4D")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.2)
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("exact-tap accuracy at 21 dB")
    ax.set_title("(b) the head follows the supervision", pad=8)
    ax.grid(True, which="major", axis="y", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)


def _panel_c(ax) -> None:
    frame = pd.read_csv(FULL / "k3_doppler_full" / "ranging_benchmark"
                        / f"per_burst_{ARCH}.csv")
    frame = frame.loc[frame["snr_db"].sub(21.0).abs() < 1e-9]

    def acc(mask, pred, truth):
        sel = frame.loc[mask]
        if sel.empty:
            return float("nan")
        return float((np.abs(sel[pred] - sel[truth]) <= 0.5).mean())

    g1_hard = frame["silent_g1"].astype(str).str.lower() == "true"
    g2_hard = frame["silent_g2"].astype(str).str.lower() == "true"
    groups = (
        ("obstacle is the\nstrongest echo", ~g1_hard),
        (r"dominated $\geq1\times$", g1_hard),
        (r"dominated $\geq2\times$", g2_hard),
    )
    x = np.arange(len(groups))
    width = 0.36
    head = [acc(mask, "head_tau", "tau_true") for _, mask in groups]
    peak = [acc(mask, "peak_lag", "tau_true") for _, mask in groups]
    share = [float(mask.mean()) for _, mask in groups]
    ax.bar(x - width / 2.0, peak, width, color=PEAK_COLOR, alpha=0.55,
           edgecolor=PEAK_COLOR, lw=1.0, label="matched-filter peak")
    bars = ax.bar(x + width / 2.0, head, width, color="#636EFA", alpha=0.85,
                  edgecolor="#636EFA", lw=1.0, label="sensing head")
    for bar, value in zip(bars, head):
        ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2.0, value),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8.5, color="#1F2A6E")
    for xi, value in zip(x - width / 2.0, peak):
        ax.annotate(f"{value:.2f}", (xi + width / 4.0, value),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8.0, color="#4D4D4D")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{label}\n({s*100:.0f}% of bursts)"
                        for (label, _), s in zip(groups, share)], fontsize=8.2)
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("exact-tap accuracy at 21 dB")
    ax.set_title("(c) where the gain comes from", pad=8)
    ax.grid(True, which="major", axis="y", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)


def main() -> None:
    fs.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.7),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
    _panel_a(axes[0])
    _panel_b(axes[1])
    _panel_c(axes[2])
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)
    fig.tight_layout()
    fs.legend_below_fig(fig, axes, ncol=4)
    fs.save(fig, "exact_tap")
    plt.close(fig)


if __name__ == "__main__":
    main()
