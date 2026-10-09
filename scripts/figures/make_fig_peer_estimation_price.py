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

SRC = REPO / "results" / "full" / "peer_estimation"
ARCHS = ("conv1d", "qkv")
ARCH_LABELS = {"conv1d": "Ultra-CAN (Conv1D)", "qkv": "Ultra-CAN-QKV"}
RATIO_BINS = [0.0, 0.75, 1.25, 2.0, np.inf]
RATIO_ORDER = ("peer_stronger", "comparable", "target_gt1.25", "target_gt2")
RATIO_LABELS = {
    "peer_stronger": "peer\nstronger",
    "comparable": "comparable",
    "target_gt1.25": "target\nx1.25",
    "target_gt2": "target\nx2",
}
GUARD = 1.5
TOLERANCE = 0.5
WINDOW = 33.0

def load_samples() -> Dict[str, pd.DataFrame]:
    frames = {}
    for arch in ARCHS:
        path = SRC / arch / "peer_samples.csv"
        if not path.is_file():
            raise SystemExit(f"missing {path}: run the peer_estimation experiment")
        frame = pd.read_csv(path)
        frames[arch] = frame[np.isclose(frame["guard_factor"], GUARD)]
    return frames

def load_summary() -> pd.DataFrame:
    frames = []
    for arch in ARCHS:
        frames.append(pd.read_csv(SRC / arch / "peer_estimation.csv"))
    frame = pd.concat(frames, ignore_index=True)
    frame["key_bin"] = frame["key_bin"].astype(str)
    return frame

def _binned(frame: pd.DataFrame) -> pd.Series:
    return pd.cut(frame["ratio"], bins=RATIO_BINS, labels=list(RATIO_ORDER),
                  right=False).astype(str)

def _accuracy(samples: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for arch, frame in samples.items():
        binned = _binned(frame)
        mean, p90 = [], []
        for key in RATIO_ORDER:
            err = frame.loc[binned == key, "abs_err"].to_numpy(dtype=float)
            mean.append(float(np.mean(err)) if err.size else np.nan)
            p90.append(float(np.percentile(err, 90)) if err.size else np.nan)
        out[arch] = {"mean": np.array(mean), "p90": np.array(p90)}
    return out

def _panel_tolerance(ax, samples: Dict[str, pd.DataFrame]) -> None:
    grid = np.arange(TOLERANCE, 15.0 + 1e-9, 0.5)
    for arch in ARCHS:
        err = samples[arch]["abs_err"].to_numpy(dtype=float)
        share = np.array([float((err <= tol).mean()) for tol in grid])
        kw = fs.series_kwargs(arch)
        kw["ms"] = 4.0
        ax.plot(grid, share, label=ARCH_LABELS[arch], zorder=3, **kw)

    ax.axvline(TOLERANCE, color="#444444", ls=":", lw=1.4, zorder=2)
    at_tolerance = max(float((samples[arch]["abs_err"] <= TOLERANCE).mean())
                       for arch in ARCHS)
    ax.annotate("",
                (TOLERANCE, at_tolerance), textcoords="offset points",
                xytext=(8, 4), ha="left", va="bottom", fontsize=9, color="#333333")

    ax.set_xlim(0.0, 15.0)
    ax.set_xticks(np.arange(0.0, 15.1, 3.0))
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Tolerated delay error (samples)")
    ax.set_ylabel("Share of samples within the tolerance")
    ax.set_title("How accurate is the range estimate", fontsize=11, pad=8)
    ax.grid(True, axis="y", color=fs.GRID_COLOR, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)

def _panel_fallback(ax, summary: pd.DataFrame) -> float:
    guards = sorted(float(v) for v in
                    summary.loc[summary["summary"] == "co_range", "guard_factor"].unique())
    worst = 0.0
    for key, dashes, label in (("co_range", "-", "peer shares the obstacle range"),
                               ("separated", (0, (4, 2)), "peer well separated")):
        values = []
        for guard in guards:
            sub = summary[(summary["summary"] == "co_range")
                          & (summary["key_bin"] == key)
                          & np.isclose(summary["guard_factor"], guard)]
            values.append(float(sub["fallback_rate"].mean()))
            worst = max(worst, float(sub["fallback_rate"].max() - sub["fallback_rate"].min()))
        ax.plot(guards, values, ls=dashes, marker="o", ms=6, lw=1.8,
                color="#636EFA" if key == "co_range" else "#EF553B", label=label,
                zorder=3)
        for guard, value in zip(guards, values):
            if key == "separated":
                ax.annotate(f"{100 * value:.0f}%", (guard, value),
                            textcoords="offset points", xytext=(0, -14),
                            ha="center", fontsize=8.5, color="#EF553B")
    ax.set_xticks(guards)
    ax.set_xlim(min(guards) - 0.15, max(guards) + 0.15)
    ax.set_ylim(0.0, 1.1)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Guard factor of the link-budget test")
    ax.set_ylabel("Share of samples with the fallback tripped")
    ax.set_title("What the fail-safe costs", fontsize=11, pad=8)
    ax.grid(True, axis="y", color=fs.GRID_COLOR, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    return worst

def main() -> None:
    fs.apply_style()
    samples = load_samples()
    summary = load_summary()
    accuracy = _accuracy(samples)

    qkv = samples[ARCHS[-1]]["abs_err"]
    print("delay error (samples, window %.0f): mean %.2f, median %.2f, P90 %.2f"
          % (WINDOW, qkv.mean(), qkv.median(), np.percentile(qkv, 90)))
    print("fraction within 0.5 / 1 / 2 / 5 samples: "
          + " / ".join(f"{100 * float((qkv <= t).mean()):.1f}%" for t in (0.5, 1, 2, 5)))
    for arch in ARCHS:
        print(f"  {ARCH_LABELS[arch]:<18} mean {samples[arch]['abs_err'].mean():.2f} | "
              f"P90 {np.percentile(samples[arch]['abs_err'], 90):.2f}")
    print("mean delay error per amplitude bin (conv1d / qkv):",
          {key: (f"{accuracy['conv1d']['mean'][i]:.2f} / {accuracy['qkv']['mean'][i]:.2f}")
           for i, key in enumerate(RATIO_ORDER)})

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    _panel_tolerance(axes[0], samples)
    worst = _panel_fallback(axes[1], summary)
    print("worst receiver disagreement on the fallback rate: %.2e" % worst)

    fig.tight_layout()
    fs.legend_below_fig(fig, axes, ncol=2)
    fs.save(fig, "peer_estimation_price")
    plt.close(fig)

if __name__ == "__main__":
    main()

