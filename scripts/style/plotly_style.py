
from __future__ import annotations

import math

PLOTLY_COLORS = {
    "conv1d": "#636EFA",
    "qkv": "#EF553B",
    "lstm": "#00CC96",
    "mc_dlsk": "#AB63FA",
    "blind_stat": "#7F7F7F",
}
PLOTLY_MARKERS = {
    "conv1d": "circle",
    "qkv": "square",
    "lstm": "diamond",
    "mc_dlsk": "cross",
}
PLOTLY_DASHES = {
    "conv1d": "solid",
    "qkv": "dash",
    "lstm": "dot",
    "mc_dlsk": "dashdot",
    "blind_stat": "dot",
}
LINE_W = 1.5
MARKER_SIZE = 8
_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

def sup_exp(exp: int) -> str:
    return "10" + str(exp).translate(_SUPERSCRIPT)

def scatter(xs, ys, name, key, *, dash=None, symbol=None, width=None,
            color=None, showlegend=True):
    import plotly.graph_objects as go

    col = color or PLOTLY_COLORS.get(key, "#7F7F7F")
    return go.Scatter(
        x=list(xs),
        y=list(ys),
        name=name,
        mode="lines+markers",
        showlegend=showlegend,
        line=dict(width=width or LINE_W,
                  dash=dash or PLOTLY_DASHES.get(key, "solid"), color=col),
        marker=dict(size=MARKER_SIZE,
                    symbol=symbol or PLOTLY_MARKERS.get(key, "circle"),
                    color=col),
    )

def minor_grid_shapes(y_lo: float, y_hi: float, xref: str = "paper"):
    shapes = []
    for exp in range(int(math.floor(math.log10(y_lo))) - 1,
                     int(math.ceil(math.log10(y_hi))) + 1):
        for mult in range(2, 10):
            y = mult * 10.0 ** exp
            if y_lo * 0.999 <= y <= y_hi * 1.001:
                shapes.append(dict(type="line", xref=xref, yref="y",
                                   x0=0, x1=1, y0=y, y1=y,
                                   line=dict(color="lightgray", width=1, dash="dot"),
                                   layer="below"))
    return shapes

def frame_shape():
    return dict(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1,
                line=dict(color="black", width=1))

def log_yaxis(y_lo: float, y_hi: float, title: str = "Bit Error Rate (BER)"):
    e0 = int(math.floor(math.log10(y_lo)))
    e1 = int(math.ceil(math.log10(y_hi)))
    exps = [e for e in range(e0, e1 + 1)
            if y_lo * 0.999 <= 10.0 ** e <= y_hi * 1.001]
    return dict(title=title, type="log",
                range=[math.log10(y_lo), math.log10(y_hi)],
                tickvals=[10.0 ** e for e in exps],
                ticktext=[sup_exp(e) for e in exps],
                showgrid=True, gridcolor="lightgray", gridwidth=1,
                ticks="outside", showline=True, linewidth=1, linecolor="black",
                mirror=True)

def linear_yaxis(title: str):
    return dict(title=title, showgrid=True, gridcolor="lightgray", gridwidth=1,
                ticks="outside", showline=True, linewidth=1, linecolor="black",
                mirror=True, zeroline=False)

def xaxis(ticks, title: str = "SNR (dB)"):
    return dict(title=title, showgrid=True, gridcolor="lightgray", gridwidth=1,
                tickvals=list(ticks), ticks="outside", showline=True,
                linewidth=1, linecolor="black", mirror=True,
                range=[min(ticks), max(ticks)])

def legend(loc: str = "lower left", title: str = "Models"):
    anchors = {
        "lower left": (0.02, 0.02, "left", "bottom"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, xa, ya = anchors[loc]
    return dict(x=x, y=y, xanchor=xa, yanchor=ya, title_text=title,
                bgcolor="rgba(255,255,255,0.8)", bordercolor="gray",
                borderwidth=1)

def interior_ticks(values):
    values = sorted(values)
    return values[1:-1] if len(values) > 2 else values

def write(fig, pdf_path, html_path=None):
    from pathlib import Path

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(pdf_path), format="pdf")
    print("saved", pdf_path)
    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        print("saved", html_path)
