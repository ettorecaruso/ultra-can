from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.data.dataset_generator import (
    _x0_from_seed,
    _iterate_map_batch,
    _center_normalize_batch,
    apply_channel_batch,
)
from src.models.blind_stat import blind_features, fit_lda, blind_decide
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH

SEQLEN = 100
MU = float(yaml.safe_load(
    Path(DEFAULT_BASE_CONFIG_PATH).read_text(encoding="utf-8"))["data"]["map_param"])
SNR_GRID = [-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
SCENARIOS = {
    "k1_doppler_full": (1, 8e-5),
    "k3_doppler_full": (3, 8e-5),
    "k3_doppler_limited": (3, 4e-5),
}
MODELS = ["conv1d", "qkv", "lstm", "mc_dlsk"]
LEGEND = {
    "conv1d": "Ultra-CAN (Conv1D)",
    "qkv": "Ultra-CAN (QKV)",
    "lstm": "LSTM-OFDM-DCSK",
    "mc_dlsk": "MC-DLCSK",
    "blind_stat": "Blind statistical",
}
PLOTLY_COLORS = {
    "conv1d": "#636EFA",
    "qkv": "#EF553B",
    "lstm": "#00CC96",
    "mc_dlsk": "#AB63FA",
    "blind_stat": "#7F7F7F",
}
PLOTLY_MARKERS = {"conv1d": "o", "qkv": "s", "lstm": "D", "mc_dlsk": "+"}
PLOTLY_DASHES = {
    "conv1d": "-",
    "qkv": (0, (6, 2)),
    "lstm": (0, (1, 1.6)),
    "mc_dlsk": (0, (4, 1.2, 1, 1.2)),
}
LINE_W = 1.5
MARKER_SIZE = 6.5
BLIND_STYLE = dict(color=PLOTLY_COLORS["blind_stat"], ls=(0, (1, 1.6)), lw=LINE_W)

def _model_kwargs(model: str) -> dict:
    return dict(color=PLOTLY_COLORS[model], marker=PLOTLY_MARKERS[model],
                ls=PLOTLY_DASHES[model], lw=LINE_W, ms=MARKER_SIZE)
SCENARIO_TITLES = {
    "k1_doppler_full": "Single echo (K = 1)",
    "k3_doppler_full": "Multi-echo (K = 3)",
    "k3_doppler_limited": "Multi-echo, low Doppler (K = 3)",
}
SCENARIO_SLUGS = {
    "k1_doppler_full": "k1_single_echo",
    "k3_doppler_full": "k3_multi_echo",
    "k3_doppler_limited": "k3_low_doppler",
}

def _results_root() -> Path:
    return REPO / "results" / "full"

SRC_FULL = _results_root() / "ber_vs_snr"

_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

def _sup_exp(exp: int) -> str:
    return "10" + str(exp).translate(_SUPERSCRIPT)

def _x_ticks(values):
    values = sorted(values)
    return values[1:-1] if len(values) > 2 else values

def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": "#D3D3D3",
            "grid.linewidth": 1.0,
            "legend.frameon": True,
            "legend.framealpha": 0.8,
            "legend.edgecolor": "gray",
            "legend.facecolor": "white",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": LINE_W,
            "lines.markersize": MARKER_SIZE,
        }
    )

def _siino_axis(ax, y_lo: float, y_hi: float, x_step: float = 2.0) -> None:
    ax.set_axisbelow(True)
    ax.set_yscale("log")
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, which="major", axis="both", color="#D3D3D3", lw=1.0)
    ax.grid(True, which="minor", axis="y", color="#D3D3D3", lw=1.0, ls=":")
    e0 = int(np.floor(np.log10(y_lo)))
    e1 = int(np.ceil(np.log10(y_hi)))
    exps = [e for e in range(e0, e1 + 1)
            if y_lo * 0.999 <= 10.0 ** e <= y_hi * 1.001]
    ax.set_yticks([10.0 ** e for e in exps])
    ax.set_yticklabels([_sup_exp(e) for e in exps])
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1), numticks=100)
    )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(x_step))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(direction="out", color="black")

def _load_curves(out_dir, scenario: str) -> dict:
    curves = {}
    blind = Path(out_dir) / scenario / "blind_stat" / "metrics.csv"
    if blind.exists():
        curves["blind_stat"] = pd.read_csv(blind).sort_values("snr_db")
    for model in MODELS:
        path = Path(out_dir) / scenario / model / "metrics.csv"
        if path.exists():
            curves[model] = pd.read_csv(path).sort_values("snr_db")
    return curves

def _decade_floor(curves: dict) -> float:
    mins = []
    for df in curves.values():
        ber = df["ber"].to_numpy(dtype=float)
        ber = ber[np.isfinite(ber) & (ber > 0)]
        if ber.size:
            mins.append(float(ber.min()))
    if not mins:
        return 1e-5
    return 10.0 ** float(np.floor(np.log10(min(mins))))

def make_cfg(max_doppler: float) -> dict:
    return {
        "data": {
            "sequence_length": SEQLEN,
            "max_delay": 33,
            "max_doppler": max_doppler,
            "doppler_direct_max": 1e-5,
            "rician_kappa_db": 10.0,
            "alpha_min": 0.05,
            "alpha_max": 0.3,
            "map_type": "logistic",
            "map_param": MU,
            "alpha_tau_coupling": False,
            "alpha_floor": 0.05,
        }
    }

def gen(rng, n, k, snr_db, cfg):
    bits = rng.integers(0, 2, size=n).astype(np.int64)
    seeds = rng.integers(0, 2 ** 31 - 1, size=n)
    x0l = np.array([_x0_from_seed(int(s), "logistic", MU) for s in seeds])
    x0b = np.array([_x0_from_seed(int(s), "bernoulli", MU) for s in seeds])
    seq_l = _center_normalize_batch(_iterate_map_batch("logistic", MU, x0l, SEQLEN))
    seq_b = _center_normalize_batch(_iterate_map_batch("bernoulli", MU, x0b, SEQLEN))
    tx = np.where((bits == 0)[:, None], seq_l, seq_b)
    y, _, _ = apply_channel_batch(tx, k, snr_db, cfg, rng)
    return bits, y

def blind_ber_at_snr(cfg, k, snr_db, n, seed):
    rng = np.random.default_rng(seed)
    bits, y = gen(rng, n, k, snr_db, cfg)
    half = n // 2
    w, thr = fit_lda(blind_features(y[:half]), bits[:half])
    pred = blind_decide(y[half:], w, thr)
    errs = int(np.sum(pred != bits[half:]))
    return errs, half

def evaluate_blind(scenario, max_doppler, n, out_dir):
    k, _ = SCENARIOS[scenario]
    cfg = make_cfg(max_doppler)
    rows = []
    for i, snr in enumerate(SNR_GRID):
        errs, tot = blind_ber_at_snr(cfg, k, snr, n, seed=44000 + i * 17)
        rows.append(
            {
                "snr_db": float(snr),
                "ber": errs / tot,
                "n_errors": errs,
                "n_symbols": tot,
            }
        )
    df = pd.DataFrame(rows).sort_values("snr_db").reset_index(drop=True)
    dest = out_dir / scenario / "blind_stat"
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / "metrics.csv", index=False)
    return df

def copy_stored_metrics(scenario, out_dir):
    for model in MODELS:
        src = SRC_FULL / scenario / model / "metrics.csv"
        if not src.exists():
            print(f"  [warn] missing metrics: {src}")
            continue
        dest = out_dir / scenario / model
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest / "metrics.csv")

def _draw_panel(ax, curves: dict, title: str, y_lo: float, y_hi: float) -> None:
    _siino_axis(ax, y_lo, y_hi)
    ax.axhline(0.5, color="#D3D3D3", lw=1.0, ls=":", zorder=0)
    for model in MODELS:
        if model not in curves:
            continue
        df = curves[model]
        ax.plot(df["snr_db"], df["ber"], label=LEGEND[model],
                zorder=3, **_model_kwargs(model))
    if "blind_stat" in curves:
        df = curves["blind_stat"]
        ax.plot(df["snr_db"], df["ber"], label=LEGEND["blind_stat"],
                marker="x", ms=MARKER_SIZE + 0.5, zorder=2, **BLIND_STYLE)
    ax.set_title(title, pad=8)
    ax.set_xlabel("SNR (dB)")
    xs = sorted({float(v) for df in curves.values() for v in df["snr_db"]})
    if xs:
        ax.set_xlim(xs[0], xs[-1])
        ax.set_xticks(_x_ticks(xs))

def _legend_below(ax, ncol: int, y: float = -0.17) -> None:
    handles, labels, seen = [], [], set()
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in seen:
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
              ncol=ncol, framealpha=0.8, edgecolor="gray", facecolor="white",
              borderpad=0.6, labelspacing=0.4, handlelength=2.6)

def _save(fig, out_dir, stem: str) -> None:
    targets = [REPO / "figures" / f"{stem}.pdf"]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, bbox_inches="tight", pad_inches=0.12)
        print("saved", target)

def plot_full_range(out_dir) -> None:
    _apply_style()
    out_dir = Path(out_dir)
    curves = {s: _load_curves(out_dir, s) for s in SCENARIOS}
    if not any(curves.values()):
        print("[warn] no BER curves found in", out_dir)
        return

    available = [c for c in curves.values() if c]
    y_lo = min(_decade_floor(c) for c in available)
    y_hi = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        _draw_panel(ax, curves.get(scenario, {}), SCENARIO_TITLES[scenario],
                    y_lo, y_hi)
    axes[0].set_ylabel("BER", rotation=0, labelpad=14)
    fig.tight_layout()
    _legend_below(axes[1], ncol=5)
    _save(fig, out_dir, "ber_full_linear")
    plt.close(fig)

    for scenario in SCENARIOS:
        if not curves.get(scenario):
            continue
        fig, ax = plt.subplots(figsize=(6.1, 6.1))
        _draw_panel(ax, curves[scenario], SCENARIO_TITLES[scenario],
                    _decade_floor(curves[scenario]), y_hi)
        ax.set_ylabel("BER")
        fig.tight_layout()
        _legend_below(ax, ncol=2)
        _save(fig, out_dir, f"ber_full_{SCENARIO_SLUGS[scenario]}")
        plt.close(fig)

def plot_full_range_plotly(out_dir) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from style import plotly_style as ps

    out_dir = Path(out_dir)
    curves = {s: _load_curves(out_dir, s) for s in SCENARIOS}
    if not any(curves.values()):
        print("[warn] no BER curves found in", out_dir)
        return

    available = [c for c in curves.values() if c]
    y_lo = min(_decade_floor(c) for c in available)
    y_hi = 1.0
    scenarios = [s for s in SCENARIOS if curves.get(s)]
    xs = sorted({float(v) for c in available for df in c.values()
                 for v in df["snr_db"]})
    ticks = ps.interior_ticks(xs)
    ydict = ps.log_yaxis(y_lo, y_hi)
    ydict_plain = {k: v for k, v in ydict.items() if k != "title"}

    def _add(fig, scenario, showlegend, **kw):
        for model in MODELS:
            if model in curves[scenario]:
                df = curves[scenario][model]
                fig.add_trace(ps.scatter(df["snr_db"], df["ber"], LEGEND[model],
                                         model, showlegend=showlegend), **kw)
        if "blind_stat" in curves[scenario]:
            df = curves[scenario]["blind_stat"]
            fig.add_trace(ps.scatter(df["snr_db"], df["ber"], LEGEND["blind_stat"],
                                     "blind_stat", symbol="x",
                                     showlegend=showlegend), **kw)

    fig = make_subplots(rows=1, cols=len(scenarios), shared_yaxes=True,
                        horizontal_spacing=0.04,
                        subplot_titles=[SCENARIO_TITLES[s] for s in scenarios])
    for col, scenario in enumerate(scenarios, start=1):
        _add(fig, scenario, showlegend=(col == 1), row=1, col=col)
    fig.update_layout(template="plotly_white", width=1240, height=460,
                      font=dict(size=13), legend=ps.legend("lower left"),
                      margin=dict(l=70, r=20, t=50, b=60),
                      shapes=ps.minor_grid_shapes(y_lo, y_hi) + [ps.frame_shape()])
    for col in range(1, len(scenarios) + 1):
        fig.update_yaxes(ydict if col == 1 else ydict_plain, row=1, col=col)
        fig.update_xaxes(ps.xaxis(ticks), row=1, col=col)
    ps.write(fig, REPO / "figures" / "plotly" / "ber_full_linear.pdf",
             REPO / "figures" / "plotly" / "ber_full_linear.html")
    ps.write(fig, out_dir / "plots" / "plotly" / "ber_full_linear.pdf",
             out_dir / "plots" / "plotly" / "ber_full_linear.html")

    for scenario in scenarios:
        floor = _decade_floor(curves[scenario])
        fig = go.Figure()
        _add(fig, scenario, showlegend=True)
        fig.update_layout(template="plotly_white", width=610, height=610,
                          font=dict(size=14), legend=ps.legend("lower left"),
                          margin=dict(l=70, r=20, t=40, b=60),
                          shapes=(ps.minor_grid_shapes(floor, y_hi)
                                  + [ps.frame_shape()]))
        fig.update_yaxes(ps.log_yaxis(floor, y_hi))
        fig.update_xaxes(ps.xaxis(ticks))
        stem = f"ber_full_{SCENARIO_SLUGS[scenario]}"
        ps.write(fig, REPO / "figures" / "plotly" / f"{stem}.pdf",
                 REPO / "figures" / "plotly" / f"{stem}.html")
        ps.write(fig, out_dir / "plots" / "plotly" / f"{stem}.pdf",
                 out_dir / "plots" / "plotly" / f"{stem}.html")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=REPO / "figures" / "_data")
    ap.add_argument("--symbols", type=int, default=20000)
    ap.add_argument("--skip-figure", action="store_true")
    ap.add_argument("--only-figure", action="store_true",
                    help="only regenerate the figure from the results already present in output-dir")
    ap.add_argument("--engine", choices=["matplotlib", "plotly"],
                    default="matplotlib",
                    help="plotly reproduces the paper-notebook look (PDF + HTML)")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    draw = plot_full_range_plotly if args.engine == "plotly" else plot_full_range

    if args.only_figure:
        draw(out_dir)
        print("output dir:", out_dir)
        return

    for scenario, (k, max_doppler) in SCENARIOS.items():
        print(f"[scenario] {scenario} (K={k}, fD_max={max_doppler:.0e})")
        copy_stored_metrics(scenario, out_dir)
        df = evaluate_blind(scenario, max_doppler, args.symbols, out_dir)
        print(df.to_string(index=False))

    if not args.skip_figure:
        draw(out_dir)
    print("output dir:", out_dir)

if __name__ == "__main__":
    main()

