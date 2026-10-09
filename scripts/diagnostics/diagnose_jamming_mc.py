from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.data_loader import _build_feature_matrix
from src.data.dataset_generator import generate_test_batch
from src.evaluation.metrics import bit_error_count
from src.experiments.run_jamming import (
    _add_jammer_at_jsr,
    apply_jamming,
    sample_jammer_waveform,
)
from src.experiments.runner import load_experiment_config
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH
from src.utils.model_io import load_model

JSR = [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
SNAPSHOTS = 5
SNR_DB = 15.0
BATCH = 1024

def _ber(model, x, bit, x_ref, feature_mode, feature_norm) -> float:
    feat = _build_feature_matrix(x, feature_mode, feature_norm, reference=x_ref)
    errors = 0
    total = 0
    for start in range(0, feat.shape[0], BATCH):
        end = min(start + BATCH, feat.shape[0])
        preds = model(tf.convert_to_tensor(feat[start:end]), training=False)
        n_err, _ = bit_error_count(preds["comm"].numpy(), bit[start:end])
        errors += int(n_err)
        total += int(end - start)
    return errors / max(1, total)

def _checkpoint(arch: str) -> Path:
    primary = _REPO / "results" / "full" / "jamming" / arch / "best_model.keras"
    if primary.is_file():
        return primary
    return (
        _REPO / "results" / "full" / "ber_vs_snr" / "k3_doppler_full" / arch
        / "best_model.keras"
    )

def _parse_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="qkv")
    parser.add_argument("--symbols", type=int, default=8000)
    parser.add_argument("--snapshots", type=int, default=SNAPSHOTS)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=_REPO / "results" / "full",
        help=(
            "Root the diagnostic is written to; the checkpoints and the dataset are "
            "still read from the repository (default: results/full)."
        ),
    )
    return parser.parse_args(argv)

def main(argv=None) -> None:
    args = _parse_args(argv)
    arch = str(args.arch)
    num_symbols = int(args.symbols)
    snapshots = int(args.snapshots)
    if num_symbols < 100 or snapshots < 1:
        raise ValueError("--symbols must be >= 100 and --snapshots >= 1")

    cfg = load_experiment_config(
        experiment_name="jamming",
        mode="full",
        experiments_yaml_path=_REPO / "configs" / "experiments.yaml",
        base_config_path=DEFAULT_BASE_CONFIG_PATH,
        cli_overrides=None,
    )
    ckpt = _checkpoint(arch)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found for {arch}: {ckpt}")
    t0 = time.time()
    model = load_model(ckpt)
    print(f"[diag] model {arch} loaded from {ckpt} ({time.time() - t0:.1f} s)", flush=True)
    feature_mode = str(cfg["data"].get("feature_mode", "iq"))
    feature_norm = str(cfg["data"].get("feature_norm", "none"))
    k = int(cfg["data"]["echoes"][-1])
    base_seed = int(cfg["general"]["seed"])

    t0 = time.time()
    batch = generate_test_batch(cfg, num_symbols, SNR_DB, k, np.random.default_rng(base_seed))
    x = np.asarray(batch["x"])
    bit = np.asarray(batch["bit"], dtype=np.int64)
    x_ref = np.asarray(batch["x_ref"])
    print(f"[diag] batch generated: {x.shape} K={k} in {time.time() - t0:.1f} s", flush=True)

    legacy = []
    for index, jsr in enumerate(JSR):
        seed = base_seed + (sum(ord(ch) for ch in "cw") % 10000) + index * 7
        jammed = apply_jamming(x, "cw", float(jsr), np.random.default_rng(seed))
        t0 = time.time()
        value = _ber(model, jammed, bit, x_ref, feature_mode, feature_norm)
        legacy.append(value)
        print(f"[diag] legacy  JSR={jsr:+6.1f} dB  BER={value:.5f}  ({time.time() - t0:.1f} s)",
              flush=True)

    rows = []
    for realization in range(snapshots):
        rng = np.random.default_rng(base_seed + realization * 100003)
        jammer = sample_jammer_waveform(
            x.shape, "cw", rng, realization=realization, n_realizations=snapshots
        )
        f_cw = (float(realization) + 0.5) / float(snapshots) * 0.5
        for jsr in JSR:
            jammed = _add_jammer_at_jsr(x, jammer, float(jsr))
            t0 = time.time()
            value = _ber(model, jammed, bit, x_ref, feature_mode, feature_norm)
            rows.append({"jsr_db": float(jsr), "realization": int(realization),
                         "f_cw": float(f_cw), "ber": float(value)})
            print(f"[diag] MC r={realization} f_cw={f_cw:.3f} JSR={jsr:+6.1f} dB  "
                  f"BER={value:.5f}  ({time.time() - t0:.1f} s)", flush=True)
    mc = pd.DataFrame(rows)
    summary = mc.groupby("jsr_db")["ber"].agg(["mean", "std", "min", "max"]).reset_index()
    summary = summary.rename(columns={
        "mean": "ber_mc", "std": "ber_mc_std", "min": "ber_mc_min", "max": "ber_mc_max",
    })
    table = pd.DataFrame({"jsr_db": JSR, "ber_legacy": legacy}).merge(summary, on="jsr_db")

    out = args.out_root / "diagnostics" / "jamming_artifact"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "ber_vs_jsr_cw_legacy_vs_mc.csv", index=False)
    mc.to_csv(out / "ber_vs_jsr_cw_realizations.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.semilogy(table["jsr_db"], table["ber_legacy"], marker="o", ls="-", color="#c44e52",
                label="Legacy: one realization per JSR point")
    ax.semilogy(table["jsr_db"], table["ber_mc"], marker="s", ls="-", color="#4c72b0",
                label=f"Monte Carlo mean over {snapshots} tones")
    ax.fill_between(
        table["jsr_db"].to_numpy(dtype=float),
        table["ber_mc_min"].clip(lower=1e-6), table["ber_mc_max"].clip(lower=1e-6),
        color="#4c72b0", alpha=0.18, linewidth=0,
    )
    ax.set_xlabel("JSR (dB)")
    ax.set_ylabel("BER")
    ax.set_title(f"CW jamming, {arch}: single-realization artifact vs Monte Carlo")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "jamming_artifact_cw.pdf", bbox_inches="tight")
    print("saved", out)
    print(table.to_string(index=False))

if __name__ == "__main__":
    main()
