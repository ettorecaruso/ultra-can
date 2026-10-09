

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import pandas as pd

from src.data.data_loader import DataDict, _build_feature_matrix
from src.evaluation.metrics import bit_error_count
from src.evaluation.stats import mean_interval
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BATCH = 1024
_BASE_SEED = 42
_PARTIAL_BAND_FRACTION = 0.25

_CAPTURE_LAYERS: Dict[str, List[str]] = {
    "conv1d": ["conv1", "conv2", "pool_projection", "attn_weights", "attn_pool",
               "sensing_position", "sensing_delay_profile", "sensing_features"],
    "qkv": ["conv1", "conv2", "qkv_attention", "shared_gap", "pool_projection",
            "sensing_position", "sensing_delay_profile", "sensing_features"],
    "lstm": ["lstm_1", "dropout_lstm", "shared_gap", "lstm_proj",
             "sensing_position", "sensing_delay_profile", "sensing_features"],
    "mc_dlsk": ["mc_bilstm_1", "dropout_mc", "shared_gap", "mc_proj",
                "sensing_position", "sensing_delay_profile", "sensing_features"],
}
_QKV_LAYER_NAME = "qkv_attention"
_ATTN_POOL_LAYER = "attn_weights"

def _row_entropy(w: np.ndarray) -> np.ndarray:
    
    w = np.asarray(w, dtype=np.float64)
    if w.ndim == 3 and w.shape[-1] == 1:
        w = w[..., 0]
    eps = np.finfo(np.float64).eps
    return -np.sum(w * np.log(np.clip(w, eps, 1.0)), axis=-1)

def _pool_temporal(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    return arr.mean(axis=1) if arr.ndim == 3 else arr

def _build_probe(model: tf.keras.Model, arch: str) -> Tuple[tf.keras.Model, List[str]]:
    
    cap, outs = [], {}
    for name in _CAPTURE_LAYERS.get(arch, []):
        try:
            layer = model.get_layer(name)
            if layer.output is not None:
                outs[name] = layer.output
                cap.append(name)
        except Exception:
            logger.debug("jamming_interpretability %s: layer %s not present, skipping", arch, name)
    outs["comm_logits"] = model.output["comm"]
    outs["sensing"] = model.output["sensing"]
    probe = tf.keras.Model(model.inputs, outs)
    return probe, cap

def _run_condition(
    model: tf.keras.Model,
    probe: tf.keras.Model,
    cap: List[str],
    arch: str,
    xx: np.ndarray,
    bit: np.ndarray,
    tau: np.ndarray,
    snr_db: np.ndarray,
    x_ref: np.ndarray,
    clean_store: Optional[Dict[str, np.ndarray]],
    ret_idx: np.ndarray,
    tau_max: float,
    capture_clean: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, np.ndarray]]]:
    
    if not capture_clean and clean_store is None:
        raise ValueError("clean_store is required for non-clean conditions")
    n = xx.shape[0]
    pos = {int(i): p for p, i in enumerate(ret_idx)}
    acc_err, acc_sym = 0, 0
    margin_sum = 0.0
    tau_preds = []
    snr_groups: Dict[float, Dict[str, float]] = {}
    attn_ent_sum = attn_max_sum = attn_n = 0.0
    energy: Dict[str, float] = {k: 0.0 for k in cap}
    energy_n: Dict[str, float] = {k: 0.0 for k in cap}
    cos_acc: Dict[str, float] = {k: 0.0 for k in cap}
    cos_n: Dict[str, int] = {k: 0 for k in cap}
    collect: Optional[Dict[str, list]] = (
        {k: [] for k in cap if k != _ATTN_POOL_LAYER} if capture_clean else None
    )

    qkv_layer = model.get_layer(_QKV_LAYER_NAME) if arch == "qkv" else None
    if qkv_layer is not None:
        qkv_layer._store_attention_weights = True

    feat = _build_feature_matrix(xx, "iq", "none", reference=x_ref)

    for s in range(0, n, _BATCH):
        e = min(s + _BATCH, n)
        bx = tf.convert_to_tensor(feat[s:e])
        preds = probe(bx, training=False)
        comm_logits = preds["comm_logits"].numpy()
        sens = preds["sensing"].numpy()
        bits = bit[s:e]

        n_err, _ = bit_error_count(comm_logits, bits.astype(np.int64))
        acc_err += int(n_err)
        acc_sym += int(e - s)
        ar = np.arange(e - s)
        margin_sum += float(np.sum(comm_logits[ar, bits] - comm_logits[ar, 1 - bits]))

        tau_p = np.clip(sens[:, 0], 0.0, 1.0) * tau_max
        tau_preds.append(tau_p)
        for u in np.unique(snr_db[s:e]):
            m = snr_db[s:e] == u
            g = snr_groups.setdefault(float(u), {"err": 0.0, "sym": 0.0})
            g["err"] += float(np.count_nonzero(comm_logits[m].argmax(1) != bits[m]))
            g["sym"] += float(m.sum())

        if arch == "conv1d":
            alpha = preds[_ATTN_POOL_LAYER].numpy()
            attn_ent_sum += float(_row_entropy(alpha).sum())
            attn_max_sum += float(np.max(alpha[..., 0], axis=-1).sum())
            attn_n += float(e - s)
        elif qkv_layer is not None:
            a = qkv_layer.last_attention_weights
            if a is not None:
                an = a.numpy()
                attn_ent_sum += float(_row_entropy(an).sum())
                attn_max_sum += float(np.max(an, axis=-1).sum())
                attn_n += float(an.shape[0] * an.shape[1])

        for k in energy:
            act = preds[k].numpy()
            energy[k] += float(np.mean(act ** 2)) * float(e - s)
            energy_n[k] += float(e - s)
            if capture_clean:
                if k not in collect:
                    continue
                pooled = _pool_temporal(act)
                keep = [j for j in range(e - s) if int(s + j) in pos]
                if keep:
                    collect[k].append(pooled[keep].astype(np.float32))
            elif k in clean_store:
                pooled = _pool_temporal(act)
                keep = [j for j in range(e - s) if int(s + j) in pos]
                if keep:
                    pp = np.array([pos[int(s + j)] for j in keep], dtype=np.int64)
                    cl = clean_store[k][pp]
                    qq = pooled[keep]
                    denom = np.linalg.norm(cl, axis=1) * np.linalg.norm(qq, axis=1)
                    denom = np.where(denom < 1e-12, 1.0, denom)
                    cos_acc[k] += float(np.sum(np.sum(cl * qq, axis=1) / denom))
                    cos_n[k] += len(keep)

    clean_out: Optional[Dict[str, np.ndarray]] = None
    if capture_clean:
        clean_out = {k: np.concatenate(v, axis=0) for k, v in collect.items() if v}

    tau_pred = np.concatenate(tau_preds)
    corr_tau = float(np.corrcoef(tau, tau_pred)[0, 1]) if np.std(tau) > 0 else float("nan")

    per_snr: List[Dict[str, Any]] = []
    for u in sorted(snr_groups):
        m = snr_db == u
        corr_t = (float(np.corrcoef(tau[m], tau_pred[m])[0, 1])
                  if (m.sum() > 1 and np.std(tau[m]) > 0) else float("nan"))
        per_snr.append({"snr_db": u, "n_err": int(snr_groups[u]["err"]),
                        "n_sym": int(snr_groups[u]["sym"]),
                        "ber": snr_groups[u]["err"] / max(1.0, snr_groups[u]["sym"]),
                        "corr_tau": corr_t})

    res: Dict[str, Any] = {"n_sym": acc_sym, "n_err": acc_err,
                           "ber": acc_err / max(1, acc_sym),
                           "mse_tau": float(np.mean((tau - tau_pred) ** 2)),
                           "mse_fd": 0.0,
                           "corr_tau": corr_tau,
                           "margin_mean": margin_sum / max(1, acc_sym)}
    res["attn_entropy"] = attn_ent_sum / attn_n if attn_n else float("nan")
    res["attn_maxw"] = attn_max_sum / attn_n if attn_n else float("nan")
    for k in energy:
        res[f"{k}_energy"] = energy[k] / max(1.0, energy_n[k])
        if capture_clean:
            res[f"{k}_cos"] = 1.0 if k in clean_out else float("nan")
        else:
            has_clean = clean_store is not None and k in clean_store
            res[f"{k}_cos"] = (cos_acc[k] / max(1, cos_n[k])) if has_clean else float("nan")
    return res, per_snr, clean_out

def _aggregate_rows(
    rows: List[Dict[str, Any]],
    keys: Sequence[str],
    std_fields: Sequence[str],
    ci_fields: Sequence[str] = ("ber",),
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    aggregated: List[Dict[str, Any]] = []
    for group_key, group in grouped.items():
        agg: Dict[str, Any] = dict(zip(keys, group_key))
        fields = [
            field
            for field, value in group[0].items()
            if field not in keys
            and isinstance(value, (int, float, np.floating, np.integer))
            and not isinstance(value, bool)
        ]
        for field in fields:
            values = np.asarray([float(row[field]) for row in group], dtype=np.float64)
            finite = np.isfinite(values)
            agg[field] = float(values[finite].mean()) if bool(finite.any()) else float("nan")
            if field in std_fields:
                agg[f"{field}_std"] = (
                    float(values[finite].std(ddof=0)) if bool(finite.any()) else float("nan")
                )
            if field in ci_fields and int(finite.sum()) >= 1:
                interval = mean_interval(values[finite])
                agg[f"{field}_sem"] = interval["sem"]
                agg[f"{field}_ci_lo"] = interval["ci_lo"]
                agg[f"{field}_ci_hi"] = interval["ci_hi"]
                agg[f"{field}_q05"] = interval["q05"]
                agg[f"{field}_q95"] = interval["q95"]
        agg["n_realizations"] = len(group)
        aggregated.append(agg)
    return aggregated

def _subsample(
    test_data: DataDict,
    max_symbols: Optional[int],
    tag: str = "jamming_interpretability",
) -> DataDict:
    if max_symbols is None:
        return test_data
    cap = int(max_symbols)
    if cap < 1:
        raise ValueError(f"max_symbols must be >= 1, got {max_symbols!r}")
    n = int(np.asarray(test_data["x"]).shape[0])
    if cap >= n:
        return test_data
    index = np.linspace(0, n - 1, cap).astype(np.int64)
    out: Dict[str, Any] = {}
    for key, value in test_data.items():
        array = np.asarray(value)
        out[key] = array[index] if array.shape[:1] == (n,) else value
    logger.info(
        "[%s] test set thinned to %d of %d symbols per realization",
        tag, cap, n,
    )
    return out

def run_jamming_interpretability_probe(
    model: tf.keras.Model,
    arch: str,
    test_data: DataDict,
    config: Dict[str, Any],
    out_dir: Path,
    jsr_values: Sequence[float],
    jammer_types: Sequence[str],
    ret_subset: int = 3000,
    tag: Optional[str] = None,
    n_realizations: int = 1,
    max_symbols: Optional[int] = None,
) -> Dict[str, Any]:
    
    from src.experiments.run_jamming import sample_jammer_waveform, _add_jammer_at_jsr

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_data = _subsample(test_data, max_symbols)
    x = np.asarray(test_data["x"])
    bit, tau = np.asarray(test_data["bit"]), np.asarray(test_data["tau"])
    snr_db = np.asarray(test_data["snr_db"])
    x_ref = np.asarray(test_data["x_ref"])
    n = x.shape[0]
    ret_idx = np.linspace(0, n - 1, max(1, min(ret_subset, n))).astype(np.int64)

    probe, cap = _build_probe(model, arch)
    logger.info("[jamming_interpretability %s] probe ready: %d captured layers", arch, len(cap))

    tau_max = float(config["data"]["max_delay"])
    base_seed = int(config["general"].get("seed", 42))
    n_realizations = int(n_realizations)
    if n_realizations < 1:
        raise ValueError(f"n_realizations must be >= 1, got: {n_realizations}")

    cond_rows: List[Dict[str, Any]] = []
    per_snr_rows: List[Dict[str, Any]] = []
    realization_rows: List[Dict[str, Any]] = []
    clean_store: Optional[Dict[str, np.ndarray]] = None

    def _save() -> None:
        pd.DataFrame(cond_rows).to_csv(out_dir / "conditions.csv", index=False)
        pd.DataFrame(per_snr_rows).to_csv(out_dir / "per_snr.csv", index=False)
        if realization_rows:
            pd.DataFrame(realization_rows).to_csv(
                out_dir / "conditions_realizations.csv", index=False
            )

    t0_clean = time.time()
    clean_res, clean_per_snr, clean_out = _run_condition(
        model, probe, cap, arch, x, bit, tau, snr_db, x_ref,
        None, ret_idx, tau_max, capture_clean=True,
    )
    clean_store = clean_out
    clean_res.update({"arch": arch, "jammer": "clean", "jsr_db": float("nan")})
    cond_rows.append(clean_res)
    for row in clean_per_snr:
        row.update({"arch": arch, "jammer": "clean", "jsr_db": float("nan")})
        per_snr_rows.append(row)
    _save()
    logger.info(
        "[jamming_interpretability %s] clean | BER=%.4f corr_tau=%.3f margin=%.2f (%.0f s)",
        arch, clean_res["ber"], clean_res["corr_tau"], clean_res["margin_mean"],
        time.time() - t0_clean,
    )

    for jammer in jammer_types:
        jammer_records: List[Dict[str, Any]] = []
        jammer_per_snr: List[Dict[str, Any]] = []
        for realization in range(n_realizations):
            seed = (
                base_seed
                + (sum(ord(ch) for ch in jammer) % 10000)
                + realization * 100003
            )
            rng = np.random.default_rng(seed)
            jammer_wave = sample_jammer_waveform(
                x.shape,
                jammer,
                rng,
                realization=realization,
                n_realizations=n_realizations,
            )
            for jsr_db in jsr_values:
                t0c = time.time()
                xx = _add_jammer_at_jsr(x, jammer_wave, float(jsr_db))
                res, per_snr, _ = _run_condition(
                    model, probe, cap, arch, xx, bit, tau, snr_db, x_ref,
                    clean_store, ret_idx, tau_max, capture_clean=False,
                )
                res.update({
                    "arch": arch,
                    "jammer": jammer,
                    "jsr_db": float(jsr_db),
                    "realization": int(realization),
                })
                jammer_records.append(res)
                realization_rows.append(dict(res))
                for row in per_snr:
                    row.update({
                        "arch": arch,
                        "jammer": jammer,
                        "jsr_db": float(jsr_db),
                        "realization": int(realization),
                    })
                    jammer_per_snr.append(row)
                logger.info(
                    "[jamming_interpretability %s] %-11s JSR=%6.1f dB r=%d | BER=%.4f "
                    "corr_tau=%.3f margin=%.2f (%.0f s)",
                    arch, jammer, float(jsr_db), realization,
                    res["ber"], res["corr_tau"], res["margin_mean"], time.time() - t0c,
                )
        cond_rows.extend(
            _aggregate_rows(
                jammer_records, ("arch", "jammer", "jsr_db"), ("ber", "margin_mean")
            )
        )
        per_snr_rows.extend(
            _aggregate_rows(
                jammer_per_snr,
                ("arch", "jammer", "jsr_db", "snr_db"),
                ("ber", "margin_mean"),
            )
        )
        _save()

    metadata = {
        "experiment": "jamming_interpretability", "arch": arch,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tensorflow_version": tf.__version__,
        "jsr_grid": [float(v) for v in jsr_values],
        "jammer_types": list(jammer_types),
        "n_test_samples": int(n), "ret_subset": int(ret_subset),
        "batch_size": _BATCH,
        "n_conditions": len(cond_rows),
        "n_realizations": int(n_realizations),
        "tag": tag if tag is not None else arch,
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("[jamming_interpretability %s] completed -> %s (conditions=%d)", arch, out_dir, len(cond_rows))
    return {"conditions_csv": str(out_dir / "conditions.csv"),
            "per_snr_csv": str(out_dir / "per_snr.csv"),
            "conditions_realizations_csv": (
                str(out_dir / "conditions_realizations.csv") if realization_rows else None
            ),
            "n_conditions": len(cond_rows)}

if __name__ == "__main__":
    raise SystemExit("Module not directly executable: use the runner (jamming_interpretability).")
