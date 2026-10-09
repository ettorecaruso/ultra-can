
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data import channel_models
from src.data.data_loader import _build_feature_matrix
from src.data.dataset_generator import generate_test_batch
from src.evaluation.metrics import bit_error_count
from src.evaluation.stats import wilson_interval
from src.utils.config_loader import with_channel_variant
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PREDICT_BATCH = 1024
_OPERATING_SNR_DB = 5.0
_TARGET_BER = 1e-4

def variants_of(config: Dict[str, Any]) -> List[str]:
    section = (config.get("experiments") or {}).get("channel_generalization") or {}
    names = section.get("channel_variants")
    if names is None:
        raise KeyError(
            "missing required configuration key: "
            "experiments.channel_generalization.channel_variants"
        )
    if isinstance(names, str):
        names = [names]
    ordered = [str(name) for name in names]
    if not ordered:
        raise ValueError("channel_variants must not be empty")
    return ordered

def _predict(model: Any, feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    import tensorflow as tf

    logits_parts: List[np.ndarray] = []
    sensing_parts: List[np.ndarray] = []
    for start in range(0, feat.shape[0], _PREDICT_BATCH):
        end = min(start + _PREDICT_BATCH, feat.shape[0])
        preds = model(tf.convert_to_tensor(feat[start:end]), training=False)
        logits_parts.append(preds["comm"].numpy())
        sensing_parts.append(preds["sensing"].numpy())
    if not logits_parts:
        raise RuntimeError("empty prediction batch")
    return np.concatenate(logits_parts, axis=0), np.concatenate(sensing_parts, axis=0)

def _batch_seed(base_seed: int, variant_index: int, snr_db: float, batch_idx: int) -> int:
    value = (
        int(base_seed) * 1000003
        + int(variant_index) * 15485863
        + int(round(float(snr_db) * 100.0)) * 7919
        + int(batch_idx) * 104729
    )
    return int(abs(value) % (2**31 - 1))

def protocol_of(config: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = config.get("evaluation") or {}
    section = (config.get("experiments") or {}).get("channel_generalization") or {}
    return {
        "snr_test_range": [
            float(value) for value in (section.get("snr_test_range") or evaluation.get("snr_test_range") or [])
        ],
        "bit_error_threshold": int(
            section.get("bit_error_threshold", evaluation.get("bit_error_threshold"))
        ),
        "max_symbols_per_snr": int(
            section.get("max_symbols_per_snr", evaluation.get("max_symbols_per_snr"))
        ),
        "batch_symbols": int(evaluation.get("eval_batch_symbols")),
    }

def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def _evaluate_variant(
    model: Any,
    arch: str,
    variant_cfg: Dict[str, Any],
    variant_name: str,
    variant_index: int,
    protocol: Dict[str, Any],
    echoes: Sequence[int],
    feature_mode: str,
    feature_norm: str,
    base_seed: int,
    tau_max: float,
    fd_max: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for snr in protocol["snr_test_range"]:
        accum_err = 0
        accum_sym = 0
        batch_idx = 0
        tau_true: List[np.ndarray] = []
        tau_pred: List[np.ndarray] = []
        fd_true: List[np.ndarray] = []
        fd_pred: List[np.ndarray] = []
        fd_available = True
        while (
            accum_sym < protocol["max_symbols_per_snr"]
            and accum_err < protocol["bit_error_threshold"]
        ):
            k = int(echoes[batch_idx % len(echoes)])
            rng = np.random.default_rng(
                _batch_seed(base_seed, variant_index, float(snr), batch_idx)
            )
            n_batch = min(
                protocol["batch_symbols"],
                protocol["max_symbols_per_snr"] - accum_sym,
            )
            batch = generate_test_batch(variant_cfg, n_batch, float(snr), k, rng)
            bit = np.asarray(batch["bit"], dtype=np.int64)
            feat = _build_feature_matrix(
                np.asarray(batch["x"]),
                feature_mode,
                feature_norm,
                reference=np.asarray(batch["x_ref"]),
            )
            logits, sensing = _predict(model, feat)
            errors, _ = bit_error_count(logits, bit)
            accum_err += int(errors)
            accum_sym += int(n_batch)
            batch_idx += 1
            tau_true.append(np.asarray(batch["tau"], dtype=np.float64))
            fd_true.append(np.asarray(batch["f_d"], dtype=np.float64))
            sensing = np.asarray(sensing, dtype=np.float64)
            tau_pred.append(sensing[:, 0] * float(tau_max))
            if sensing.shape[1] == 2:
                fd_pred.append(sensing[:, 1] * float(fd_max))
            else:
                fd_available = False
                fd_pred.append(np.zeros(sensing.shape[0], dtype=np.float64))
        ci_lo, ci_hi = wilson_interval(accum_err, accum_sym)
        tau_p = np.concatenate(tau_pred)
        tau_t = np.concatenate(tau_true)
        fd_p = np.concatenate(fd_pred)
        fd_t = np.concatenate(fd_true)
        rows.append({
            "channel_variant": variant_name,
            "arch": arch,
            "snr_db": float(snr),
            "ber": accum_err / max(1, accum_sym),
            "n_errors": int(accum_err),
            "n_symbols": int(accum_sym),
            "ber_ci_lo": ci_lo,
            "ber_ci_hi": ci_hi,
            "mse_tau": float(np.mean((tau_p - tau_t) ** 2)),
            "corr_tau": _correlation(tau_p, tau_t),
            "mse_fd": float(np.mean((fd_p - fd_t) ** 2)) if fd_available else 0.0,
            "corr_fd": _correlation(fd_p, fd_t) if fd_available else 0.0,
        })
    return rows

def pooled_ber(rows: Sequence[Dict[str, Any]], min_snr_db: float) -> Tuple[float, float, float]:
    errors = sum(int(row["n_errors"]) for row in rows if float(row["snr_db"]) >= min_snr_db - 1e-9)
    symbols = sum(int(row["n_symbols"]) for row in rows if float(row["snr_db"]) >= min_snr_db - 1e-9)
    if symbols == 0:
        return (float("nan"), float("nan"), float("nan"))
    lo, hi = wilson_interval(errors, symbols)
    return (errors / symbols, lo, hi)

def min_snr_at_target(rows: Sequence[Dict[str, Any]], target: float) -> float:
    for row in sorted(rows, key=lambda item: float(item["snr_db"])):
        if float(row["ber"]) <= float(target):
            return float(row["snr_db"])
    return float("nan")

def evaluate_channel_generalization(
    model: Any,
    config: Dict[str, Any],
    output_dir: Path,
    arch: str,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol_of(config)
    names = variants_of(config)
    data = config["data"]
    evaluation = config.get("evaluation") or {}
    echoes = [int(value) for value in data["echoes"]]
    if not echoes:
        raise ValueError("data.echoes must not be empty")
    base_seed = int((config.get("general") or {}).get("seed", 0))
    feature_mode = str(data.get("feature_mode", "iq"))
    feature_norm = str(data.get("feature_norm", "none"))
    tau_max = float(data["max_delay"])
    fd_max = float(data["max_doppler"])
    operating_snr = float(evaluation.get("operating_snr_db", _OPERATING_SNR_DB))
    target_ber = float(evaluation.get("target_ber", _TARGET_BER))

    all_rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    for index, name in enumerate(names):
        variant_cfg = with_channel_variant(config, name)
        rows = _evaluate_variant(
            model=model,
            arch=arch,
            variant_cfg=variant_cfg,
            variant_name=name,
            variant_index=index,
            protocol=protocol,
            echoes=echoes,
            feature_mode=feature_mode,
            feature_norm=feature_norm,
            base_seed=base_seed,
            tau_max=tau_max,
            fd_max=fd_max,
        )
        variant_dir = output_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(variant_dir / "metrics.csv", index=False)
        all_rows.extend(rows)
        pooled, ci_lo, ci_hi = pooled_ber(rows, operating_snr)
        finite_corr = [
            float(row["corr_tau"]) for row in rows if np.isfinite(row["corr_tau"])
        ]
        summary.append({
            "arch": arch,
            "channel_variant": name,
            "channel_model": channel_models.channel_model(variant_cfg),
            "channel_fingerprint": channel_models.channel_fingerprint(variant_cfg),
            "hold_mode": str((variant_cfg.get("channel") or {}).get("hold_mode", "")),
            "pooled_ber": pooled,
            "pooled_ber_ci_lo": ci_lo,
            "pooled_ber_ci_hi": ci_hi,
            "min_snr_at_target": min_snr_at_target(rows, target_ber),
            "corr_tau_top": max(finite_corr) if finite_corr else float("nan"),
            "mse_tau_top": min(
                (float(row["mse_tau"]) for row in rows), default=float("nan")
            ),
            "n_symbols_total": int(sum(int(row["n_symbols"]) for row in rows)),
            "diagnostics": json.dumps(
                channel_models.diagnostics(variant_cfg), sort_keys=True
            ),
        })
        minimum = summary[-1]["min_snr_at_target"]
        logger.info(
            "[channel_generalization %s] %-18s pooled_ber=%.6f "
            "[%.6f, %.6f] min_snr@target=%s",
            arch, name, summary[-1]["pooled_ber"], ci_lo, ci_hi,
            f"{minimum:.1f}" if np.isfinite(minimum) else "n/a",
        )

    pd.DataFrame(all_rows).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(summary).to_csv(output_dir / "summary.csv", index=False)

    metadata = {
        "experiment": "channel_generalization",
        "arch": arch,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "channel_variants": names,
        "protocol": protocol,
        "operating_snr_db": operating_snr,
        "target_ber": target_ber,
        "echoes": echoes,
        "channels": {
            name: channel_models.channel_metadata(with_channel_variant(config, name))
            for name in names
        },
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "metrics_csv": str(output_dir / "metrics.csv"),
        "summary_csv": str(output_dir / "summary.csv"),
        "n_variants": len(names),
    }

