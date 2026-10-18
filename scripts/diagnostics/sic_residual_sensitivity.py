"""Sensitivity of the sensing head to residual self-interference.

The radar mode is full-duplex: the direct path of Eq. (3) is the
pre-cancellation coupling, which the receiver estimates and subtracts.  This
script scales that coupling by ``10 ** (-I_SI / 20)``, where ``I_SI`` is the
assumed total circulator plus SIC suppression, and evaluates the frozen
peer-trained Conv1D receiver on the same 4000-burst peer-laden sky at 21 dB.
No retraining is performed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.data_loader import _build_feature_matrix
from src.data.dataset_generator import generate_test_batch
from src.experiments.peer_estimation import (
    _expected_peer_alpha,
    _head_delay,
    _matched_filter_profile,
    _peer_settings,
)
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config
from src.utils.model_io import load_model

_RESULTS = _REPO_ROOT / "results" / "full"
_DEFAULT_CHECKPOINT = _RESULTS / "ber_vs_snr" / "iod_peers" / "conv1d" / "best_model.keras"
_TOL_SAMPLES = 0.5
_LEVELS_DB = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)


def _level_row(
    model: Any,
    config: Dict[str, Any],
    n_symbols: int,
    snr_db: float,
    guard: float,
    level_db: Optional[float],
) -> Dict[str, Any]:
    scale = 0.0 if level_db is None else 10.0 ** (-float(level_db) / 20.0)
    rng = np.random.default_rng(20260926)
    batch = generate_test_batch(
        config,
        n_symbols,
        snr_db,
        3,
        rng,
        direct_scale=scale,
    )
    peer_taus = np.asarray(batch["peer_taus"], dtype=np.float64)
    peer_distances = np.asarray(batch["peer_distances_m"], dtype=np.float64)
    tau_true = np.asarray(batch["tau"], dtype=np.float64)
    max_delay = int(config["data"]["max_delay"])

    profile = _matched_filter_profile(batch["x"], batch["x_ref"], max_delay)
    features = _build_feature_matrix(
        batch["x"],
        str(config["data"].get("feature_mode", "iq")),
        config["data"].get("feature_norm"),
        reference=batch["x_ref"],
    )
    tau_hat = _head_delay(model, features, float(max_delay))

    rows = np.arange(int(n_symbols))
    settings = _peer_settings(config)
    nearest = np.argmin(peer_distances, axis=1)
    predicted = _expected_peer_alpha(peer_distances, settings)
    lags = np.clip(np.rint(peer_taus).astype(np.int64), 1, max_delay)
    fallback = np.zeros(int(n_symbols), dtype=bool)
    for column in range(peer_taus.shape[1]):
        measured = profile[rows, lags[rows, column] - 1]
        fallback |= measured > float(guard) * predicted[rows, column]

    target_hit = np.abs(tau_hat - tau_true) <= _TOL_SAMPLES
    peer_hit = np.zeros(int(n_symbols), dtype=bool)
    for column in range(peer_taus.shape[1]):
        peer_hit |= np.abs(tau_hat - peer_taus[rows, column]) <= _TOL_SAMPLES
    peer_false = peer_hit & ~target_hit
    abs_err = np.abs(tau_hat - tau_true)
    answered = ~fallback

    def _mean(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(values[mask])) if bool(np.any(mask)) else float("nan")

    return {
        "sic_suppression_db": "ideal" if level_db is None else f"{float(level_db):.0f}",
        "direct_scale": float(scale),
        "n_symbols": int(n_symbols),
        "snr_db": float(snr_db),
        "guard_factor": float(guard),
        "target_id_ungated": _mean(target_hit, np.ones_like(target_hit)),
        "peer_false_ungated": _mean(peer_false, np.ones_like(peer_false)),
        "coverage": _mean(answered, np.ones_like(answered)),
        "target_id_answered": _mean(target_hit, answered),
        "peer_false_answered": _mean(peer_false, answered),
        "median_abs_err_answered": float(np.median(abs_err[answered]))
        if bool(np.any(answered))
        else float("nan"),
    }


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--n-symbols", type=int, default=4000)
    parser.add_argument("--snr-db", type=float, default=21.0)
    parser.add_argument("--guard", type=float, default=1.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=_RESULTS / "diagnostics" / "sic_residual_sensitivity.csv",
    )
    args = parser.parse_args(argv)

    config = dict(load_config(DEFAULT_BASE_CONFIG_PATH))
    config["peers"] = dict(config["peers"])
    config["peers"]["enable"] = True
    config["peers"]["n_peers"] = 2
    config["experiments"] = dict(config.get("experiments") or {})
    config["experiments"]["peer_estimation"] = {
        "n_symbols": int(args.n_symbols),
        "snr_db": float(args.snr_db),
        "echoes": 3,
        "seed": 20260926,
    }

    model = load_model(args.checkpoint)
    rows = [
        _level_row(
            model,
            config,
            int(args.n_symbols),
            float(args.snr_db),
            float(args.guard),
            level,
        )
        for level in (None, *_LEVELS_DB)
    ]
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
