from pathlib import Path
import argparse
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ARCHS = ["conv1d", "qkv", "lstm", "mc_dlsk"]
LABELS = {
    "conv1d": "Ultra-CAN (Conv1D)",
    "qkv": "Ultra-CAN (QKV)",
    "lstm": "LSTM-OFDM-DCSK",
    "mc_dlsk": "MC-DLCSK",
}
PLOTLY_COLORS = {
    "conv1d": "#636EFA",
    "qkv": "#EF553B",
    "lstm": "#00CC96",
    "mc_dlsk": "#AB63FA",
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

def _model_kwargs(arch: str) -> dict:
    return dict(color=PLOTLY_COLORS[arch], marker=PLOTLY_MARKERS[arch],
                ls=PLOTLY_DASHES[arch], lw=LINE_W, ms=MARKER_SIZE)
SCENARIOS = [
    ("k1_doppler_full", "k1_single_echo", "Single echo (K = 1)"),
    ("k3_doppler_full", "k3_multi_echo", "Multi-echo (K = 3)"),
    ("k3_doppler_limited", "k3_low_doppler", "Multi-echo, low Doppler (K = 3)"),
]
MIN_DB = 5.0
# Nessuna linea-obiettivo. Il confine dei 10^-4 non e' piu' un traguardo che il grafico
# possa mostrare: sul canale nuovo nessun ricevitore ci arriva. Il fondo dell'asse sta
# appena SOPRA 10^-4, cosi' il valore non compare nemmeno come tacca dell'asse y.
Y_BOTTOM = 1.1e-4
Y_TOP = 0.011999999999999999

def _results_root() -> Path:
    return REPO / "results" / "full"

RESULTS = _results_root() / "ber_vs_snr"

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

def _curve(scenario: str) -> dict:
    curves = {}
    for arch in ARCHS:
        path = RESULTS / scenario / arch / "metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["snr_db"] >= MIN_DB].sort_values("snr_db")
        if len(df):
            curves[arch] = df
    return curves

def _draw_panel(ax, curves: dict, title: str) -> None:
    _siino_axis(ax, Y_BOTTOM, Y_TOP)
    for arch in ARCHS:
        if arch not in curves:
            continue
        df = curves[arch]
        ax.plot(df["snr_db"], df["ber"], label=LABELS[arch],
                zorder=3, **_model_kwargs(arch))
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

def _save(fig, stem: str) -> None:
    fig_dir = REPO / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"{stem}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.12)
    print("saved", out)
    mirror = REPO / "results" / "figures" / f"{stem}.pdf"
    if mirror.parent.is_dir():
        fig.savefig(mirror, bbox_inches="tight", pad_inches=0.12)
        print("saved", mirror)

def plot_region_zoom_plotly() -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from style import plotly_style as ps

    curves = {scenario: _curve(scenario) for scenario, _, _ in SCENARIOS}
    if not any(curves.values()):
        print("[warn] no metrics found under", RESULTS)
        return

    scenarios = [(s, sl, t) for s, sl, t in SCENARIOS if curves.get(s)]
    xs = sorted({float(v) for c in curves.values() for df in c.values()
                 for v in df["snr_db"]})
    ticks = ps.interior_ticks(xs)
    ydict = ps.log_yaxis(Y_BOTTOM, Y_TOP)
    ydict_plain = {k: v for k, v in ydict.items() if k != "title"}

    def _add(fig, scenario, x_ticks, first, **kw):
        for arch in ARCHS:
            if arch in curves[scenario]:
                df = curves[scenario][arch]
                fig.add_trace(ps.scatter(df["snr_db"], df["ber"], LABELS[arch],
                                         arch, showlegend=first), **kw)

    fig = make_subplots(rows=1, cols=len(scenarios), shared_yaxes=True,
                        horizontal_spacing=0.04,
                        subplot_titles=[t for _, _, t in scenarios])
    for col, (scenario, _, _) in enumerate(scenarios, start=1):
        _add(fig, scenario, ticks, first=(col == 1), row=1, col=col)
    fig.update_layout(template="plotly_white", width=1240, height=460,
                      font=dict(size=13), legend=ps.legend("upper right"),
                      margin=dict(l=70, r=20, t=50, b=60),
                      shapes=(ps.minor_grid_shapes(Y_BOTTOM, Y_TOP)
                              + [ps.frame_shape()]))
    for col in range(1, len(scenarios) + 1):
        fig.update_yaxes(ydict if col == 1 else ydict_plain, row=1, col=col)
        fig.update_xaxes(ps.xaxis(ticks), row=1, col=col)
    ps.write(fig, REPO / "figures" / "plotly" / "ber_region_zoom.pdf",
             REPO / "figures" / "plotly" / "ber_region_zoom.html")

    for scenario, slug, _ in scenarios:
        fig = go.Figure()
        _add(fig, scenario, ticks, first=True)
        fig.update_layout(template="plotly_white", width=610, height=610,
                          font=dict(size=14), legend=ps.legend("upper right"),
                          margin=dict(l=70, r=20, t=40, b=60),
                          shapes=(ps.minor_grid_shapes(Y_BOTTOM, Y_TOP)
                                  + [ps.frame_shape()]))
        fig.update_yaxes(ydict)
        fig.update_xaxes(ps.xaxis(ticks))
        stem = f"ber_region_zoom_{slug}"
        ps.write(fig, REPO / "figures" / "plotly" / f"{stem}.pdf",
                 REPO / "figures" / "plotly" / f"{stem}.html")

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=["matplotlib", "plotly"],
                    default="matplotlib",
                    help="plotly reproduces the paper-notebook look (PDF + HTML)")
    args = ap.parse_args(argv)
    if args.engine == "plotly":
        plot_region_zoom_plotly()
        return

    _apply_style()
    curves = {scenario: _curve(scenario) for scenario, _, _ in SCENARIOS}
    if not any(curves.values()):
        print("[warn] no metrics found under", RESULTS)
        return

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), sharey=True)
    for ax, (scenario, _, title) in zip(axes, SCENARIOS):
        _draw_panel(ax, curves.get(scenario, {}), title)
    axes[0].set_ylabel("BER", rotation=0, labelpad=14)
    fig.tight_layout()
    _legend_below(axes[1], ncol=5)
    _save(fig, "ber_region_zoom")
    plt.close(fig)

    for scenario, slug, title in SCENARIOS:
        if not curves.get(scenario):
            continue
        fig, ax = plt.subplots(figsize=(6.1, 6.1))
        _draw_panel(ax, curves[scenario], title)
        ax.set_ylabel("BER")
        fig.tight_layout()
        _legend_below(ax, ncol=2)
        _save(fig, f"ber_region_zoom_{slug}")
        plt.close(fig)

if __name__ == "__main__":
    main()
