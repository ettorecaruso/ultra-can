from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import src.data.dataset_generator as dg
from src.data.frequency_hopping import (
    build_hop_sequence,
    build_slot_ids,
    hop_config,
)
from src.utils.config_loader import (
    DEFAULT_BASE_CONFIG_PATH,
    load_config,
    with_channel_variant,
)

VARIANTS = ("nominal", "ablation_log_uniform", "b_tdl_d", "b_tdl_d_light",
            "b_tdl_a", "c_two_ray_jakes")
N_SYMBOLS = 4000
SNR_DB = 15.0
K = 3
MAX_LAG = 33
SEED = 20260925

_ORIGINAL_APPLY_OVERLAY = dg.apply_overlay
_CAPTURED: list = []

def _spy(config, geometry, rng, slot_ids=None, hop_channels=None):
    applied = _ORIGINAL_APPLY_OVERLAY(
        config, geometry, rng, slot_ids=slot_ids, hop_channels=hop_channels
    )
    _CAPTURED.append(applied)
    return applied

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def probe(config, variant: str) -> dict:
    variant_cfg = with_channel_variant(config, variant)
    hop = hop_config(variant_cfg)
    slot_ids = build_slot_ids(N_SYMBOLS, hop)
    hop_channels = build_hop_sequence(N_SYMBOLS, hop)

    _CAPTURED.clear()
    dg.apply_overlay = _spy
    try:
        batch = dg.generate_test_batch(
            variant_cfg, N_SYMBOLS, SNR_DB, K, np.random.default_rng(SEED),
            slot_ids=slot_ids, hop_channels=hop_channels,
        )
    finally:
        dg.apply_overlay = _ORIGINAL_APPLY_OVERLAY

    geometry = _CAPTURED[0]
    label = np.asarray(batch["tau"], dtype=float)
    amplitude = np.abs(geometry.amplitudes)
    valid = geometry.valid
    masked = np.where(valid, amplitude, -np.inf)
    strongest = np.argmax(masked, axis=1)
    strongest_tau = geometry.taus[np.arange(geometry.taus.shape[0]), strongest]
    return {
        "channel_variant": variant,
        "hold_mode": str((variant_cfg.get("channel") or {}).get("hold_mode", "")),
        "n_taps": int(geometry.taus.shape[1]),
        "n_labeled_taps": int(geometry.k_geometric),
        "p_label_is_strongest": float(np.mean(strongest < geometry.k_geometric)),
        "corr_label_strongest": _corr(label, strongest_tau),
        "var_tau": float(np.var(label)),
    }

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path,
                        default=_REPO / "results" / "full")
    args = parser.parse_args(argv)

    config = load_config(DEFAULT_BASE_CONFIG_PATH)
    config = dict(config)
    config["data"] = dict(config["data"])
    config["data"]["echoes"] = [K]
    config["data"]["max_delay"] = MAX_LAG
    config["data"]["max_doppler"] = 8.0e-5

    rows = [probe(config, variant) for variant in VARIANTS]

    out = args.out_root / "diagnostics" / "channel_label_observability"
    out.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with open(out / "oracle.csv", "w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(
                f"{row[key]:.6f}" if isinstance(row[key], float) else str(row[key])
                for key in columns
            ) + "\n")

    print(f"{'variant':18s} {'hold':10s} {'taps':>5s} {'lab':>4s} "
          f"{'P(label=peak)':>14s} {'corr':>8s} {'var(tau)':>9s}")
    for row in rows:
        print(f"{row['channel_variant']:18s} {row['hold_mode']:10s} "
              f"{row['n_taps']:5d} {row['n_labeled_taps']:4d} "
              f"{row['p_label_is_strongest']:14.3f} "
              f"{row['corr_label_strongest']:8.3f} {row['var_tau']:9.2f}")
    print(f"probe saved {out}")

if __name__ == "__main__":
    main()
