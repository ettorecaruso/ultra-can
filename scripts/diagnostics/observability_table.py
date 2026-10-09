"""Model-free observability of the sensing label, per channel variant.

The paper reports, next to the trained head, the probability that the label is
also the strongest return of the *received* waveform.  Two closely related
quantities are computed here, both without training:

* ``p_label_is_strongest`` -- the empirical channel geometry: the labeled
  geometric tap is the strongest of the generated taps (same definition as
  ``channel_label_observability.py``).  Kept for continuity with ``oracle.csv``.
* ``p_peak_hit`` -- the model-free oracle of the paper: the argmax of the
  matched-filter delay profile equals the labeled delay within half a sample
  (``oracle_hit_rate``, the same function the ranging benchmark reports as
  ``p_oracle``).

The script writes ``peak_hit.csv`` next to ``oracle.csv`` in
``results/full/diagnostics/channel_label_observability/`` so the paper table
can be rebuilt without retraining anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import src.data.dataset_generator as dg
from src.data.frequency_hopping import (
    build_hop_sequence,
    build_slot_ids,
    hop_config,
)
from src.evaluation.ranging import oracle_hit_rate
from src.experiments.peer_estimation import _matched_filter_profile
from src.utils.config_loader import (
    DEFAULT_BASE_CONFIG_PATH,
    load_config,
    with_channel_variant,
)

VARIANTS = (
    "nominal",
    "ablation_log_uniform",
    "b_tdl_d",
    "b_tdl_d_light",
    "b_tdl_a",
    "c_two_ray_jakes",
)
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
    amplitude = np.abs(geometry.amplitudes)
    valid = geometry.valid
    masked = np.where(valid, amplitude, -np.inf)
    strongest = np.argmax(masked, axis=1)
    label_is_strongest = float(np.mean(strongest < geometry.k_geometric))

    profile = _matched_filter_profile(batch["x"], batch["x_ref"], MAX_LAG)
    peak_lag = (np.argmax(profile, axis=1) + 1).astype(np.float64)
    tau_true = np.asarray(batch["tau"], dtype=np.float64)
    return {
        "channel_variant": variant,
        "n_symbols": int(N_SYMBOLS),
        "p_label_is_strongest": label_is_strongest,
        "p_peak_hit": float(oracle_hit_rate(peak_lag, tau_true)),
        "n_taps": int(geometry.taus.shape[1]),
        "k_geometric": int(geometry.k_geometric),
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
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "peak_hit.csv", index=False)

    print(f"{'variant':20s} {'taps':>5s} {'P(label=peak)':>14s} {'P(hit)':>8s}")
    for row in rows:
        print(f"{row['channel_variant']:20s} {row['n_taps']:5d} "
              f"{row['p_label_is_strongest']:14.3f} {row['p_peak_hit']:8.3f}")
    print(f"saved {out / 'peak_hit.csv'}")


if __name__ == "__main__":
    main()
