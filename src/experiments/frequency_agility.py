
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data import channel_models
from src.data.data_loader import _build_feature_matrix
from src.data.dataset_generator import (
    channel_hold_mode,
    channel_time_block_slots,
    generate_test_batch,
)
from src.data.frequency_hopping import (
    HopConfig,
    build_hop_sequence,
    build_slot_ids,
    hop_config,
    hop_counters,
    with_dwell,
)
from src.evaluation.metrics import bit_error_count
from src.evaluation.stats import describe
from src.experiments.run_jamming import _add_jammer_at_jsr, sample_jammer_waveform
from src.utils.logger import get_logger

logger = get_logger(__name__)

_VALID_JAMMER_MODELS = frozenset({"barrage", "fixed_partial", "sweep", "follower"})
_MODALITIES = ("fh_off", "fh_off_blind", "fh_on")
_JAMMER_KNOWLEDGE = {
    "fh_off": "omniscient",
    "fh_off_blind": "band_agnostic",
    "fh_on": "model",
}
_PREDICT_BATCH = 1024
_JAMMER_STREAM_OFFSET = 7919
_MASK_STREAM_OFFSET = 50021
_REALIZATION_STRIDE = 100003

def jammed_mask(
    hop_channels: np.ndarray,
    slot_ids: np.ndarray,
    jammer_model: str,
    num_channels: int,
    params: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    if jammer_model not in _VALID_JAMMER_MODELS:
        raise ValueError(
            f"invalid jammer model: {jammer_model!r} "
            f"(expected: {sorted(_VALID_JAMMER_MODELS)})"
        )
    hop = np.asarray(hop_channels, dtype=np.int64)
    slots = np.asarray(slot_ids, dtype=np.int64)
    if hop.ndim != 1 or slots.shape != hop.shape:
        raise ValueError("hop_channels and slot_ids must be 1D arrays of equal length")
    if int(num_channels) < 1:
        raise ValueError(f"num_channels must be >= 1, got: {num_channels}")

    if jammer_model == "barrage":
        return np.ones(hop.shape, dtype=bool)

    if jammer_model == "fixed_partial":
        coverage = float(params.get("coverage_fraction", 0.25))
        if not (0.0 <= coverage <= 1.0):
            raise ValueError(f"coverage_fraction must be in [0, 1], got: {coverage}")
        n_jammed = int(round(coverage * int(num_channels)))
        if coverage > 0.0:
            n_jammed = max(1, n_jammed)
        n_jammed = min(n_jammed, int(num_channels))
        if n_jammed == 0:
            return np.zeros(hop.shape, dtype=bool)
        pool = rng.permutation(int(num_channels))[:n_jammed]
        return np.isin(hop, pool)

    if jammer_model == "sweep":
        jam_slots = max(1, int(params.get("sweep_jam_slots", 1)))
        order = rng.permutation(int(num_channels))
        swept = order[(slots // jam_slots) % int(num_channels)]
        return hop == swept

    reaction_bursts = max(0, int(params.get("follower_reaction_bursts", 2)))
    dwell = max(1, int(params.get("dwell_bursts", 1)))
    offsets = np.arange(hop.shape[0], dtype=np.int64) % dwell
    return offsets >= reaction_bursts

def _modality_inputs(
    n_bursts: int,
    hop_cfg: HopConfig,
    modality: str,
    jammer_model: str,
    params: Dict[str, Any],
    mask_rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if modality not in _MODALITIES:
        raise ValueError(f"invalid modality: {modality!r} (expected: {_MODALITIES})")
    n = int(n_bursts)
    slot_ids = build_slot_ids(n, hop_cfg)
    if modality == "fh_on":
        hop_channels = build_hop_sequence(n, hop_cfg)
        mask = jammed_mask(
            hop_channels, slot_ids, jammer_model,
            int(hop_cfg.num_channels), params, mask_rng,
        )
        return hop_channels, slot_ids, mask
    hop_channels = np.zeros(n, dtype=np.int64)
    if modality == "fh_off":
        return hop_channels, slot_ids, np.ones(n, dtype=bool)
    mask = jammed_mask(
        hop_channels, slot_ids, jammer_model,
        int(hop_cfg.num_channels), params, mask_rng,
    )
    return hop_channels, slot_ids, mask

def _add_jammer_masked(
    y: np.ndarray,
    jammer: np.ndarray,
    mask: np.ndarray,
    jsr_db: float,
) -> np.ndarray:
    if not bool(np.any(mask)):
        return y
    out = np.array(y, copy=True)
    out[mask] = _add_jammer_at_jsr(y[mask], jammer[mask], jsr_db)
    return out

def _predict_ber(
    model: tf.keras.Model,
    x: np.ndarray,
    bit: np.ndarray,
    x_ref: np.ndarray,
    feature_mode: str,
    feature_norm: str,
) -> float:
    feat = _build_feature_matrix(x, feature_mode, feature_norm, reference=x_ref)
    errors = 0
    total = 0
    for start in range(0, feat.shape[0], _PREDICT_BATCH):
        end = min(start + _PREDICT_BATCH, feat.shape[0])
        preds = model(tf.convert_to_tensor(feat[start:end]), training=False)
        logits = preds["comm"].numpy()
        n_err, _ = bit_error_count(logits, bit[start:end].astype(np.int64))
        errors += int(n_err)
        total += int(end - start)
    if total == 0:
        raise RuntimeError("empty prediction batch")
    return errors / float(total)

def _generate_batch(
    config: Dict[str, Any],
    n_bursts: int,
    snr_db: float,
    k: int,
    rng: np.random.Generator,
    slot_ids: Optional[np.ndarray],
    hop_channels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = generate_test_batch(
        config,
        n_bursts,
        snr_db,
        k,
        rng,
        slot_ids=slot_ids,
        hop_channels=hop_channels,
    )
    return (
        np.asarray(batch["x"]),
        np.asarray(batch["bit"], dtype=np.int64),
        np.asarray(batch["x_ref"]),
    )

def evaluate_frequency_agility(
    model: tf.keras.Model,
    config: Dict[str, Any],
    output_dir: Path,
    arch: str,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fa = dict(config.get("frequency_hopping") or {})
    if not bool(fa.get("enable", True)):
        logger.info("frequency_hopping.enable is false: skipping the agility evaluation")
        return {"skipped": True}

    hop_cfg = hop_config(config)
    jammer_models = list(
        fa.get("jammer_models") or ["barrage", "fixed_partial", "sweep", "follower"]
    )
    invalid = [m for m in jammer_models if m not in _VALID_JAMMER_MODELS]
    if invalid:
        raise ValueError(
            f"Invalid frequency_hopping.jammer_models: {invalid}. "
            f"Expected: {sorted(_VALID_JAMMER_MODELS)}"
        )
    in_channel_types = list(fa.get("in_channel_types") or ["cw", "partial_band"])
    jsr_values = [
        float(v) for v in (fa.get("jsr_values") or [-10.0, -6.0, -2.0, 2.0, 6.0, 10.0])
    ]
    n_bursts = int(fa.get("n_bursts_per_condition", 20000))
    n_realizations = int(fa.get("n_realizations", 5))
    if n_bursts < 1:
        raise ValueError(
            f"frequency_hopping.n_bursts_per_condition must be >= 1, got: {n_bursts}"
        )
    if n_realizations < 1:
        raise ValueError(
            f"frequency_hopping.n_realizations must be >= 1, got: {n_realizations}"
        )

    feature_mode = str(config["data"].get("feature_mode", "iq"))
    feature_norm = str(config["data"].get("feature_norm", "none"))
    echo_list = [int(value) for value in config["data"]["echoes"]]
    k = int(fa.get("num_echoes", echo_list[-1] if echo_list else 1))
    snr_db = float(fa.get("snr_db", float(config["evaluation"]["snr_test_range"][-1])))
    base_seed = int(config["general"].get("seed", 42))
    partial_band_fraction = float(
        config.get("jamming", {}).get("partial_band_fraction", 0.25)
    )
    burst_length = int(config["data"]["sequence_length"])

    params = {
        "coverage_fraction": float(fa.get("coverage_fraction", 0.25)),
        "sweep_jam_slots": int(fa.get("sweep_jam_slots", 1)),
        "follower_reaction_bursts": int(fa.get("follower_reaction_bursts", 2)),
        "dwell_bursts": int(hop_cfg.dwell_bursts),
    }

    logger.info(
        "[frequency_agility %s] channels=%d dwell=%d slot=%.1f us hop_rate=%.0f Hz "
        "bursts=%d realizations=%d snr=%.1f dB K=%d",
        arch, hop_cfg.num_channels, hop_cfg.dwell_bursts, hop_cfg.slot_duration_us,
        hop_cfg.hop_rate_hz, n_bursts, n_realizations, snr_db, k,
    )
    logger.info(
        "[frequency_agility %s] controlled comparison: fh_off and fh_on share the "
        "same per-slot channel process and the same transmission; only the hop "
        "sequence and the jammer coverage differ",
        arch,
    )

    rows: List[Dict[str, Any]] = []
    realization_rows: List[Dict[str, Any]] = []

    for modality in _MODALITIES:
        for jammer_model in jammer_models:
            for realization in range(n_realizations):
                seed = (
                    base_seed
                    + (sum(ord(ch) for ch in jammer_model) % 10000)
                    + realization * _REALIZATION_STRIDE
                )
                rng = np.random.default_rng(seed)
                mask_rng = np.random.default_rng(seed + _MASK_STREAM_OFFSET)
                hop_channels, slot_ids, mask = _modality_inputs(
                    n_bursts, hop_cfg, modality, jammer_model, params, mask_rng,
                )
                jammer_rng = np.random.default_rng(seed + _JAMMER_STREAM_OFFSET)
                x_clean, bit, x_ref = _generate_batch(
                    config, n_bursts, snr_db, k, rng, slot_ids, hop_channels
                )
                ber_clean = _predict_ber(
                    model, x_clean, bit, x_ref, feature_mode, feature_norm
                )
                for in_channel in in_channel_types:
                    jammer = sample_jammer_waveform(
                        (n_bursts, burst_length),
                        in_channel,
                        jammer_rng,
                        partial_band_fraction=partial_band_fraction,
                        realization=realization,
                        n_realizations=n_realizations,
                    )
                    for jsr_db in jsr_values:
                        t0c = time.time()
                        x_jammed = _add_jammer_masked(
                            x_clean, jammer, mask, float(jsr_db)
                        )
                        ber = _predict_ber(
                            model, x_jammed, bit, x_ref, feature_mode, feature_norm
                        )
                        row = {
                            "arch": arch,
                            "modality": modality,
                            "jammer_knowledge": _JAMMER_KNOWLEDGE[modality],
                            "jammer_model": jammer_model,
                            "in_channel": in_channel,
                            "jsr_db": float(jsr_db),
                            "realization": int(realization),
                            "ber": float(ber),
                            "jammed_fraction": float(np.mean(mask)),
                            "ber_clean": float(ber_clean),
                            "hop_rate_hz": float(hop_cfg.hop_rate_hz),
                            "dwell_bursts": int(hop_cfg.dwell_bursts),
                            "slot_duration_us": float(hop_cfg.slot_duration_us),
                            "num_channels": int(hop_cfg.num_channels),
                        }
                        rows.append(row)
                        realization_rows.append(dict(row))
                        logger.info(
                            "[frequency_agility %s] %s %-13s %-13s JSR=%6.1f dB r=%d "
                            "jammed=%.3f BER=%.4f (%.0f s)",
                            arch, modality, jammer_model, in_channel,
                            float(jsr_db), realization, float(np.mean(mask)), ber,
                            time.time() - t0c,
                        )
            if realization_rows:
                pd.DataFrame(realization_rows).to_csv(
                    output_dir / "frequency_agility_realizations.csv", index=False
                )

    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    keys = ("arch", "modality", "jammer_model", "in_channel", "jsr_db")
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    summary: List[Dict[str, Any]] = []
    for group_key, group in grouped.items():
        spread = describe([float(row["ber"]) for row in group])
        summary.append({
            "arch": group_key[0],
            "modality": group_key[1],
            "jammer_knowledge": _JAMMER_KNOWLEDGE[group_key[1]],
            "jammer_model": group_key[2],
            "in_channel": group_key[3],
            "jsr_db": float(group_key[4]),
            "ber": spread["mean"],
            "ber_std": spread["std"],
            "ber_sem": spread["sem"],
            "ber_ci_lo": spread["ci_lo"],
            "ber_ci_hi": spread["ci_hi"],
            "ber_min": spread["min"],
            "ber_max": spread["max"],
            "ber_q05": spread["q05"],
            "ber_q50": spread["q50"],
            "ber_q95": spread["q95"],
            "jammed_fraction": float(np.mean([r["jammed_fraction"] for r in group])),
            "ber_clean": float(np.mean([r["ber_clean"] for r in group])),
            "hop_rate_hz": float(group[0]["hop_rate_hz"]),
            "dwell_bursts": int(group[0]["dwell_bursts"]),
            "slot_duration_us": float(group[0]["slot_duration_us"]),
            "num_channels": int(group[0]["num_channels"]),
            "n_realizations": int(spread["n"]),
        })
    summary_df = pd.DataFrame(summary).sort_values(
        ["jammer_model", "in_channel", "modality", "jsr_db"]
    )
    summary_path = output_dir / "frequency_agility_vs_jsr.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("[frequency_agility %s] BER-vs-JSR summary saved to %s", arch, summary_path)

    dwell_values = [int(d) for d in (fa.get("dwell_sweep") or [1, 2, 4, 8, 16])]
    dwell_jsr = float(fa.get("dwell_sweep_jsr", 6.0))
    dwell_models = list(fa.get("dwell_sweep_models") or ["sweep", "follower"])
    dwell_rows: List[Dict[str, Any]] = []
    for dwell in dwell_values:
        if dwell < 1:
            raise ValueError(
                f"frequency_hopping.dwell_sweep entries must be >= 1, got: {dwell}"
            )
        dwell_cfg = with_dwell(hop_cfg, dwell)
        hop_channels = build_hop_sequence(n_bursts, dwell_cfg)
        slot_ids = build_slot_ids(n_bursts, dwell_cfg)
        dwell_params = dict(params)
        dwell_params["dwell_bursts"] = dwell
        for jammer_model in dwell_models:
            ber_samples: List[float] = []
            frac_samples: List[float] = []
            for realization in range(n_realizations):
                seed = (
                    base_seed
                    + (sum(ord(ch) for ch in jammer_model) % 10000)
                    + realization * _REALIZATION_STRIDE
                )
                rng = np.random.default_rng(seed)
                mask = jammed_mask(
                    hop_channels, slot_ids, jammer_model,
                    int(dwell_cfg.num_channels), dwell_params, rng,
                )
                in_channel = in_channel_types[0]
                jammer_rng = np.random.default_rng(seed + _JAMMER_STREAM_OFFSET)
                x_clean, bit, x_ref = _generate_batch(
                    config, n_bursts, snr_db, k, rng, slot_ids, hop_channels
                )
                jammer = sample_jammer_waveform(
                    (n_bursts, burst_length),
                    in_channel,
                    jammer_rng,
                    partial_band_fraction=partial_band_fraction,
                    realization=realization,
                    n_realizations=n_realizations,
                )
                x_jammed = _add_jammer_masked(x_clean, jammer, mask, dwell_jsr)
                ber_samples.append(
                    _predict_ber(model, x_jammed, bit, x_ref, feature_mode, feature_norm)
                )
                frac_samples.append(float(np.mean(mask)))
            spread = describe(ber_samples)
            frac_arr = np.asarray(frac_samples, dtype=np.float64)
            dwell_rows.append({
                "arch": arch,
                "jammer_model": jammer_model,
                "in_channel": in_channel,
                "dwell_bursts": dwell,
                "slot_duration_us": float(dwell_cfg.slot_duration_us),
                "hop_rate_hz": float(dwell_cfg.hop_rate_hz),
                "jsr_db": dwell_jsr,
                "ber": spread["mean"],
                "ber_std": spread["std"],
                "ber_sem": spread["sem"],
                "ber_ci_lo": spread["ci_lo"],
                "ber_ci_hi": spread["ci_hi"],
                "ber_min": spread["min"],
                "ber_max": spread["max"],
                "ber_q05": spread["q05"],
                "ber_q95": spread["q95"],
                "jammed_fraction": float(frac_arr.mean()),
                "n_realizations": int(spread["n"]),
            })
            logger.info(
                "[frequency_agility %s] dwell=%d (%.0f Hz) %-13s JSR=%.1f dB "
                "jammed=%.3f BER=%.4f [%.4f, %.4f]",
                arch, dwell, dwell_cfg.hop_rate_hz, jammer_model, dwell_jsr,
                float(frac_arr.mean()), spread["mean"], spread["ci_lo"], spread["ci_hi"],
            )
    dwell_df = pd.DataFrame(dwell_rows).sort_values(["jammer_model", "dwell_bursts"])
    dwell_path = output_dir / "frequency_agility_vs_dwell.csv"
    dwell_df.to_csv(dwell_path, index=False)
    logger.info("[frequency_agility %s] BER-vs-hop-rate summary saved to %s", arch, dwell_path)

    reaction_values = [int(value) for value in (fa.get("reaction_sweep") or [])]
    reaction_models = list(fa.get("reaction_sweep_models") or ["follower"])
    reaction_jsr = float(fa.get("reaction_sweep_jsr", dwell_jsr))
    reaction_dwell = int(fa.get("reaction_sweep_dwell", hop_cfg.dwell_bursts))
    reaction_path = output_dir / "frequency_agility_vs_reaction.csv"
    reaction_rows: List[Dict[str, Any]] = []
    if reaction_values:
        reaction_cfg = with_dwell(hop_cfg, reaction_dwell)
        hop_channels = build_hop_sequence(n_bursts, reaction_cfg)
        slot_ids = build_slot_ids(n_bursts, reaction_cfg)
        for reaction in reaction_values:
            if reaction < 0:
                raise ValueError(
                    f"frequency_hopping.reaction_sweep entries must be >= 0, got: {reaction}"
                )
            reaction_params = dict(params)
            reaction_params["follower_reaction_bursts"] = reaction
            reaction_params["dwell_bursts"] = reaction_dwell
            for jammer_model in reaction_models:
                ber_samples: List[float] = []
                frac_samples: List[float] = []
                for realization in range(n_realizations):
                    seed = (
                        base_seed
                        + (sum(ord(ch) for ch in jammer_model) % 10000)
                        + realization * _REALIZATION_STRIDE
                    )
                    rng = np.random.default_rng(seed)
                    mask = jammed_mask(
                        hop_channels, slot_ids, jammer_model,
                        int(reaction_cfg.num_channels), reaction_params, rng,
                    )
                    in_channel = in_channel_types[0]
                    jammer_rng = np.random.default_rng(seed + _JAMMER_STREAM_OFFSET)
                    x_clean, bit, x_ref = _generate_batch(
                        config, n_bursts, snr_db, k, rng, slot_ids, hop_channels
                    )
                    jammer = sample_jammer_waveform(
                        (n_bursts, burst_length),
                        in_channel,
                        jammer_rng,
                        partial_band_fraction=partial_band_fraction,
                        realization=realization,
                        n_realizations=n_realizations,
                    )
                    x_jammed = _add_jammer_masked(x_clean, jammer, mask, reaction_jsr)
                    ber_samples.append(
                        _predict_ber(model, x_jammed, bit, x_ref, feature_mode, feature_norm)
                    )
                    frac_samples.append(float(np.mean(mask)))
                spread = describe(ber_samples)
                reaction_rows.append({
                    "arch": arch,
                    "jammer_model": jammer_model,
                    "in_channel": in_channel,
                    "reaction_bursts": reaction,
                    "latency_us": float(reaction) * float(fa.get("burst_duration_us", 10.0)),
                    "dwell_bursts": reaction_dwell,
                    "hop_rate_hz": float(reaction_cfg.hop_rate_hz),
                    "jsr_db": reaction_jsr,
                    "ber": spread["mean"],
                    "ber_std": spread["std"],
                    "ber_sem": spread["sem"],
                    "ber_ci_lo": spread["ci_lo"],
                    "ber_ci_hi": spread["ci_hi"],
                    "ber_min": spread["min"],
                    "ber_max": spread["max"],
                    "jammed_fraction": float(np.mean(frac_samples)),
                    "n_realizations": int(spread["n"]),
                })
        pd.DataFrame(reaction_rows).sort_values(
            ["jammer_model", "reaction_bursts"]
        ).to_csv(reaction_path, index=False)
        logger.info(
            "[frequency_agility %s] BER-vs-reaction-latency summary saved to %s",
            arch, reaction_path,
        )

    metadata = {
        "experiment": "frequency_agility",
        "arch": arch,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tensorflow_version": tf.__version__,
        "modalities": list(_MODALITIES),
        "jammer_knowledge": dict(_JAMMER_KNOWLEDGE),
        "jammer_models": jammer_models,
        "in_channel_types": in_channel_types,
        "jsr_values": jsr_values,
        "n_bursts_per_condition": n_bursts,
        "n_realizations": n_realizations,
        "snr_db": snr_db,
        "num_echoes": k,
        "dwell_sweep": dwell_values,
        "dwell_sweep_jsr": dwell_jsr,
        "reaction_sweep": reaction_values,
        "reaction_sweep_models": reaction_models,
        "reaction_sweep_jsr": reaction_jsr,
        "reaction_sweep_dwell": reaction_dwell,
        "hop": hop_counters(n_bursts, hop_cfg),
        "channel": channel_models.channel_metadata(config),
        "shared_process": (
            "the modalities share the waveform stream and the slot ids; the hold "
            "mode of the channel decides how the realization is selected"
        ),
        "mask_rng_offset": _MASK_STREAM_OFFSET,
    }
    if channel_hold_mode(config) == "per_hop":
        metadata["time_block_slots"] = channel_time_block_slots(config)
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "summary_csv": str(summary_path),
        "dwell_csv": str(dwell_path),
        "n_conditions": len(summary),
        "n_rows": len(rows),
    }
