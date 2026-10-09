
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO / "results" / "full" / "channel_generalization"

ARCHS = ("conv1d", "qkv")
ARCH_LABELS = {
    "conv1d": "Ultra-CAN (Conv1D)",
    "qkv": "Ultra-CAN-QKV",
    "lstm": "LSTM-OFDM-DCSK",
    "mc_dlsk": "MC-DLCSK",
}
VARIANT_LABELS = {
    "nominal": "nominal (aerial multi-echo)",
    "b_tdl_d": "B: 3GPP TDL-D (LOS)",
    "b_tdl_d_light": "B: 3GPP TDL-D, 5\\% clutter",
    "b_tdl_a": "B: 3GPP TDL-A (NLOS)",
    "c_two_ray_jakes": "C: two-ray + Jakes",
}

def _label(name: str) -> str:
    return VARIANT_LABELS.get(str(name), str(name).replace("_", "\\_"))

def _fmt(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"

def _min_snr_text(value: float) -> str:
    if value is None or not np.isfinite(float(value)):
        return "not reached"
    return f"${float(value):.0f}$"

def _sci(value: float) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    exponent = int(np.floor(np.log10(abs(float(value)))))
    mantissa = float(value) / 10.0 ** exponent
    return f"${mantissa:.2f}\\times 10^{{{exponent}}}$"

def _sci_math(value: float) -> str:
    rendered = _sci(value)
    return rendered[1:-1] if rendered.startswith("$") else rendered

def load(run_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for arch in ARCHS:
        path = Path(run_dir) / str(arch) / "summary.csv"
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"no summary.csv found under {run_dir}")
    frame = pd.concat(frames, ignore_index=True)
    frame["arch"] = pd.Categorical(frame["arch"], categories=list(ARCHS), ordered=True)
    return frame.sort_values(["channel_variant", "arch"]).reset_index(drop=True)

def latex(frame: pd.DataFrame) -> str:
    variants: List[str] = []
    for name in frame["channel_variant"]:
        if name not in variants:
            variants.append(str(name))
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Cross-channel generalization of the two Ultra-CAN variants (no "
        "retraining). Pooled BER over the operating region with its $95\\%$ Wilson "
        "interval, minimum SNR at the target BER, and the sensing metrics of the "
        "dominant echo. Every row is measured with the same protocol. \\emph{not "
        "reached} means the pooled BER never falls to the target inside the "
        "evaluated SNR grid.}",
        "\\label{tab:generalization}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Channel variant & Receiver & Pooled BER & min SNR & $\\rho_\\tau$ & MSE$_\\tau$ \\\\",
        "\\midrule",
    ]
    for variant in variants:
        subset = frame[frame["channel_variant"] == variant]
        first = True
        for _, row in subset.iterrows():
            label = _label(variant) if first else ""
            first = False
            pooled_low = row.get("pooled_ber_ci_lo", float("nan"))
            pooled_high = row.get("pooled_ber_ci_hi", float("nan"))
            if np.isfinite(float(pooled_low)) and np.isfinite(float(pooled_high)):
                pooled = (
                    f"{_sci(row['pooled_ber'])} "
                    f"$[{_sci_math(pooled_low)}, {_sci_math(pooled_high)}]$"
                )
            else:
                pooled = _sci(row["pooled_ber"])
            lines.append(
                f"{label} & {ARCH_LABELS.get(str(row['arch']), row['arch'])} & {pooled} & "
                f"{_min_snr_text(row.get('min_snr_at_target'))} & "
                f"{_fmt(row.get('corr_tau_top'), 3)} & "
                f"{_fmt(row.get('mse_tau_top'), 1)} \\\\"
            )
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.extend(["\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)

def main(argv: Optional[List[str]] = None) -> None:
    run_dir = Path(argv[0]).resolve() if argv else DEFAULT_RUN_DIR
    frame = load(run_dir)
    out_dir = Path(run_dir)
    csv_path = out_dir / "generalization_table.csv"
    tex_path = out_dir / "generalization_table.tex"
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(latex(frame), encoding="utf-8")
    print("saved", csv_path)
    print("saved", tex_path)
    print(frame.to_string(index=False))

if __name__ == "__main__":
    main(sys.argv[1:])
