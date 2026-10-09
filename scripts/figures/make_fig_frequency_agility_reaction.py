
from __future__ import annotations

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

DEFAULT_RUN_DIR = REPO / "results" / "full" / "frequency_agility"
ARCHS = ("conv1d", "qkv", "lstm", "mc_dlsk")
ARCH_LABELS = {
    "conv1d": "Ultra-CAN (Conv1D)",
    "qkv": "Ultra-CAN-QKV",
    "lstm": "LSTM-OFDM-DCSK",
    "mc_dlsk": "MC-DLCSK",
}
JAMMER_LABELS = {"sweep": "Sweeping (blind)", "follower": "Reactive follower"}
DWELL_JSR = 6.0
Y_HI = 1.0
LAW_TOLERANCE = 0.05

def _load(run_dir: Path, arch: str) -> Optional[pd.DataFrame]:
    path = Path(run_dir) / arch / "frequency_agility_vs_reaction.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    return frame[np.isfinite(frame["ber"])].sort_values("latency_us")

def _load_dwell(run_dir: Path, arch: str) -> Optional[pd.DataFrame]:
    path = Path(run_dir) / arch / "frequency_agility_vs_dwell.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    return frame[np.isfinite(frame["ber"])]

def _endpoints(frame: pd.DataFrame) -> tuple:
    frac = frame["jammed_fraction"].to_numpy(dtype=float)
    if frac.max() < 1.0 - 1e-9 or frac.min() > 1e-9:
        raise AssertionError(
            "the reaction sweep must contain both f = 1 (latency 0) and f = 0 "
            f"(latency >= dwell), found f in [{frac.min():.3f}, {frac.max():.3f}]"
        )
    jammed = float(frame.loc[np.isclose(frac, 1.0), "ber"].iloc[0])
    clean = float(frame.loc[np.isclose(frac, 0.0), "ber"].iloc[0])
    return jammed, clean

def _check_mixing_law(frames: Dict[str, pd.DataFrame]) -> float:
    worst = 0.0
    for frame in frames.values():
        jammed, clean = _endpoints(frame)
        predicted = frame["jammed_fraction"] * jammed + (1 - frame["jammed_fraction"]) * clean
        interior = (frame["jammed_fraction"] > 1e-9) & (frame["jammed_fraction"] < 1.0 - 1e-9)
        measured = frame.loc[interior, "ber"].to_numpy(dtype=float)
        rel = np.abs(measured - predicted[interior].to_numpy(dtype=float)) / np.maximum(
            predicted[interior].to_numpy(dtype=float), 1e-12
        )
        if rel.size:
            worst = max(worst, float(np.max(rel)))
    if worst > LAW_TOLERANCE:
        raise AssertionError(
            f"the mixing law does not describe the reaction sweep: worst relative "
            f"error {worst:.3f} > {LAW_TOLERANCE:.2f} on the interior points. The "
            "jump in jammed_fraction and the BER jump must come from the same "
            "geometry."
        )
    return worst

def _rate_label(value: float) -> str:
    return f"{value / 1000:.3g}k" if value >= 1000 else f"{value:.3g}"

def _panel_hop_rate(ax, dwell: Dict[str, pd.DataFrame]) -> None:
    ticks: List[float] = []
    for model in ("sweep", "follower"):
        rates = np.array([], dtype=float)
        stack = []
        for arch in ARCHS:
            frame = dwell.get(arch)
            if frame is None:
                continue
            sub = frame[(frame.jammer_model == model)
                        & np.isclose(frame.jsr_db, DWELL_JSR)].sort_values("hop_rate_hz")
            if sub.empty:
                continue
            rates = sub["hop_rate_hz"].to_numpy(dtype=float)
            stack.append(sub["ber"].to_numpy(dtype=float))
        if not stack:
            continue
        mean = np.vstack(stack).mean(axis=0)
        spread = float(np.max(np.abs(np.vstack(stack) - mean)))
        print(f"  {model}: max deviation of any receiver from the mean curve {spread:.4f}")
        kw = fs.series_kwargs(model)
        ax.plot(rates, mean, label=JAMMER_LABELS[model], zorder=3, **kw)
        ticks = sorted(set(ticks) | set(float(v) for v in rates))

    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([_rate_label(v) for v in ticks], fontsize=9.5)
    ax.set_xlim(min(ticks) * 0.75, max(ticks) * 1.35)
    ax.set_xlabel("Hop rate (hops per second)")
    ax.set_title("A tracker is defeated, a blind sweeper is not", fontsize=11, pad=8)
    ax.annotate("dwell of 1-2 bursts:\nshorter than the follower reaction,\n"
                "so it never jams",
                (max(ticks) * 1.2, 0.35), ha="right", va="center", fontsize=9,
                color="#333333")

def _panel_latency(ax, frames: Dict[str, pd.DataFrame], y_lo: float) -> None:
    for arch, frame in frames.items():
        kw = fs.series_kwargs(arch)
        ax.plot(frame["latency_us"], frame["ber"], label=ARCH_LABELS[arch],
                zorder=3, **kw)

    reference = next(iter(frames.values()))
    jsr = float(reference["jsr_db"].iloc[0])
    jammed, clean = _endpoints(reference)
    lat = reference["latency_us"].to_numpy(dtype=float)
    frac = reference["jammed_fraction"].to_numpy(dtype=float)
    ax.plot(lat, frac * jammed + (1 - frac) * clean,
            color="black", lw=2.2, ls=(0, (1, 1.4)), zorder=5,
            label=r"law $\,f\,$BER$_{jam}+(1-f)\,$BER$_{clean}$")

    for x, f in zip(lat, frac):
        ax.annotate(rf"$f={f:.3f}$", (x, y_lo * 1.08), ha="center", va="bottom",
                    fontsize=9, color="#444444")

    ax.set_xticks(sorted(float(v) for v in lat))
    ax.set_xlim(float(lat.min()) - 8.0, float(lat.max()) + 8.0)
    ax.set_xlabel(r"Jammer reaction latency $L$ ($\mu$s), dwell = "
                  f"{int(reference['dwell_bursts'].iloc[0])} bursts")
    ax.set_title(f"Time-sharing, not a soft regime (JSR = +{jsr:.0f} dB)",
                 fontsize=11, pad=8)
    half = reference[np.isclose(frac, 0.5)]
    if not half.empty:
        ax.annotate("half the bursts jammed,\nhalf the BER",
                    (float(half["latency_us"].iloc[0]),
                     float(half["ber"].iloc[0]) * 1.9),
                    ha="center", va="bottom", fontsize=9, color="#333333")

def main(argv: Optional[List[str]] = None) -> None:
    run_dir = Path(argv[0]).resolve() if argv else DEFAULT_RUN_DIR
    frames: Dict[str, pd.DataFrame] = {}
    dwell: Dict[str, pd.DataFrame] = {}
    for arch in ARCHS:
        frame = _load(run_dir, arch)
        if frame is not None and not frame.empty:
            frames[arch] = frame
        dwell_frame = _load_dwell(run_dir, arch)
        if dwell_frame is not None and not dwell_frame.empty:
            dwell[arch] = dwell_frame
    if not frames:
        raise SystemExit(
            f"no frequency_agility_vs_reaction.csv under {run_dir}: run the "
            "frequency agility experiment with frequency_hopping.reaction_sweep set"
        )
    if not dwell:
        raise SystemExit(
            f"no frequency_agility_vs_dwell.csv under {run_dir}: the hop-rate panel "
            "needs the dwell sweep"
        )

    worst = _check_mixing_law(frames)
    print(f"mixing-law check (all receivers, interior points): worst {worst:.4f}")

    fs.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    y_lo = fs.ber_floor(
        [frame["ber"].to_numpy(dtype=float) for frame in frames.values()]
        + [frame["ber"].to_numpy(dtype=float) for frame in dwell.values()],
        Y_HI,
    )
    print(f"BER axis: {y_lo:g} .. {Y_HI:g}")
    for ax in axes:
        fs.log_axis(ax, y_lo, Y_HI)
        ax.set_ylabel("BER")
        ax.set_axisbelow(True)

    _panel_hop_rate(axes[0], dwell)
    _panel_latency(axes[1], frames, y_lo)

    fig.tight_layout()
    fs.legend_below_fig(fig, axes, ncol=3)
    fs.save(fig, "frequency_agility_reaction")
    plt.close(fig)

if __name__ == "__main__":
    main(sys.argv[1:])

