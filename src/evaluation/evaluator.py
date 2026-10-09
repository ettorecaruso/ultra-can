
from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.data_loader import (
    DataDict,
    _build_feature_matrix,
    build_reference_matrix,
    load_npz_files,
    verify_snr_balance,
)
from src.data.dataset_generator import build_snr_grid
from src.evaluation.metrics import (
    argmax_delay_from_profile,
    bit_error_count,
    mse_delay_doppler,
)
from src.models.baselines import build_baseline
from src.models.ultra_can import build_ultra_can
from src.models.ultra_can_qkv import build_ultra_can_qkv
from src.models.heads import sensing_output_units
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config, validate_config
from src.utils.logger import get_logger, log_config_summary, setup_logging
from src.utils.model_io import load_model

logger = get_logger(__name__)

_SNR_ROUND_DECIMALS = 6
_RANGE_TOL = 1e-9
_EXACT_TOL = 0.5
_MIN_BATCH_SIZE = 1
_MSE_NEG_TOL = 1e-9

def _validate_config(config: Dict[str, Any]) -> None:
    
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got: {type(config).__name__}")

    general = config.get("general")
    data = config.get("data")
    training = config.get("training")
    evaluation = config.get("evaluation")
    if not isinstance(general, dict):
        raise ValueError("'general' section missing or not a dict")
    if not isinstance(data, dict):
        raise ValueError("'data' section missing or not a dict")
    if not isinstance(training, dict):
        raise ValueError("'training' section missing or not a dict")
    if not isinstance(evaluation, dict):
        raise ValueError("'evaluation' section missing or not a dict")

    required_data = ("sequence_length", "max_delay", "max_doppler", "raw_dir", "echoes")
    for key in required_data:
        if key not in data:
            raise ValueError(f"missing key in data: {key}")

    required_eval = ("snr_test_range", "bit_error_threshold", "max_symbols_per_snr")
    for key in required_eval:
        if key not in evaluation:
            raise ValueError(f"missing key in evaluation: {key}")

    max_delay = data.get("max_delay")
    if not isinstance(max_delay, (int, float)) or max_delay <= 0:
        raise ValueError(f"data.max_delay must be > 0, got: {max_delay}")

    max_doppler = data.get("max_doppler")
    if not isinstance(max_doppler, (int, float)) or max_doppler <= 0:
        raise ValueError(f"data.max_doppler must be > 0, got: {max_doppler}")

    snr_test_range = evaluation.get("snr_test_range")
    if not isinstance(snr_test_range, (list, tuple)) or len(snr_test_range) == 0:
        raise ValueError(f"evaluation.snr_test_range must be a non-empty list, got: {snr_test_range}")

    bit_error_threshold = evaluation.get("bit_error_threshold")
    if not isinstance(bit_error_threshold, int) or bit_error_threshold <= 0:
        raise ValueError(f"evaluation.bit_error_threshold must be an int > 0, got: {bit_error_threshold}")

    max_symbols_per_snr = evaluation.get("max_symbols_per_snr")
    if not isinstance(max_symbols_per_snr, int) or max_symbols_per_snr <= 0:
        raise ValueError(f"evaluation.max_symbols_per_snr must be an int > 0, got: {max_symbols_per_snr}")

    batch_size = training.get("batch_size")
    if not isinstance(batch_size, int) or batch_size < _MIN_BATCH_SIZE:
        raise ValueError(f"training.batch_size must be an int >= {_MIN_BATCH_SIZE}, got: {batch_size}")

    logger.debug("Evaluator config validated")

def evaluate_model(
    model: tf.keras.Model,
    data: DataDict,
    config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    
    if not isinstance(model, tf.keras.Model):
        raise TypeError(f"model must be a tf.keras.Model, got: {type(model).__name__}")
    if not isinstance(data, dict):
        raise TypeError(f"data must be a dict, got: {type(data).__name__}")
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got: {type(config).__name__}")

    _validate_config(config)

    required_keys = ("x", "bit", "tau", "f_d", "snr_db", "seed")
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"data is missing the keys: {missing}")

    x = data["x"]
    bit = data["bit"]
    tau = data["tau"]
    f_d = data["f_d"]
    snr_db = data["snr_db"]
    seed = data["seed"]

    if x.shape[0] == 0:
        raise ValueError("data contains no samples (N=0)")

    eval_cfg = config["evaluation"]
    snr_test_range = eval_cfg["snr_test_range"]
    bit_error_threshold = eval_cfg["bit_error_threshold"]
    max_symbols_per_snr = eval_cfg["max_symbols_per_snr"]
    tau_max = float(config["data"]["max_delay"])
    fd_max = float(config["data"]["max_doppler"])
    batch_size = int(config["training"]["batch_size"])

    feature_mode = str(config["data"].get("feature_mode", "real"))
    feature_norm = str(config["data"].get("feature_norm", "none"))
    x_ref = data.get("x_ref")

    snr_values: List[float] = []
    ber_list: List[float] = []
    mse_tau_list: List[float] = []
    mse_fd_list: List[float] = []
    n_errors_list: List[int] = []
    n_symbols_list: List[int] = []
    corr_tau_list: List[float] = []
    corr_fd_list: List[float] = []
    mse_tau_argmax_list: List[float] = []
    corr_tau_argmax_list: List[float] = []
    exact_tau_list: List[float] = []
    exact_tau_argmax_list: List[float] = []

    logger.info("Starting evaluation over %d SNR points", len(snr_test_range))
    num_outputs = sensing_output_units(config)

    for snr_target in snr_test_range:
        mask = np.isclose(snr_db, snr_target, rtol=0, atol=1e-6)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            logger.warning("No samples for SNR=%.1f dB, skipping", snr_target)
            continue

        x_snr = None
        bit_snr = bit[idx]
        tau_snr = tau[idx]
        f_d_snr = f_d[idx]

        n_total = len(bit_snr)
        accum_errors = 0
        accum_symbols = 0
        mse_tau_acc = 0.0
        mse_fd_acc = 0.0
        tau_pred_all: List[float] = []
        fd_pred_all: List[float] = []
        tau_true_all: List[float] = []
        fd_true_all: List[float] = []
        tau_argmax_all: List[float] = []
        n_exact_tau = 0
        n_exact_argmax = 0
        n_out_range = 0
        n_out_range_total = 0
        n_out_range_min: Optional[float] = None
        n_out_range_max: Optional[float] = None

        start = 0
        while start < n_total and accum_symbols < max_symbols_per_snr and accum_errors < bit_error_threshold:
            end = min(start + batch_size, n_total)
            raw_idx = idx[start:end]
            batch_bit = bit[raw_idx]
            batch_tau = tau[raw_idx]
            batch_fd = f_d[raw_idx]
            batch_size_actual = len(batch_bit)

            if x_ref is not None:
                batch_ref = x_ref[raw_idx]
            else:
                batch_ref = build_reference_matrix(batch_bit, seed[raw_idx], config)
            batch_x = _build_feature_matrix(
                x[raw_idx], feature_mode, feature_norm, batch_ref
            )
            batch_argmax = argmax_delay_from_profile(
                x[raw_idx], batch_ref, int(tau_max)
            )

            pred = model(batch_x, training=False)
            comm_logits = pred["comm"].numpy()
            sensing_pred = pred["sensing"].numpy()

            if comm_logits.shape[0] != batch_size_actual or comm_logits.ndim != 2:
                raise ValueError(f"comm_logits shape {comm_logits.shape} is invalid")
            if sensing_pred.shape[0] != batch_size_actual or sensing_pred.ndim != 2 or sensing_pred.shape[1] != num_outputs:
                raise ValueError(f"sensing_pred shape {sensing_pred.shape} is invalid")
            if not np.all(np.isfinite(comm_logits)):
                raise RuntimeError("comm_logits contains NaN/Inf")
            if not np.all(np.isfinite(sensing_pred)):
                raise RuntimeError("sensing_pred contains NaN/Inf")

            n_errors_batch, _ = bit_error_count(
                comm_logits, np.asarray(batch_bit, dtype=np.int64)
            )
            accum_errors += int(n_errors_batch)
            accum_symbols += batch_size_actual

            n_out_batch = int(np.count_nonzero((sensing_pred < 0.0) | (sensing_pred > 1.0)))
            if n_out_batch:
                n_out_range += n_out_batch
                n_out_range_total += sensing_pred.size
                batch_min = float(np.min(sensing_pred))
                batch_max = float(np.max(sensing_pred))
                if n_out_range_min is None or batch_min < n_out_range_min:
                    n_out_range_min = batch_min
                if n_out_range_max is None or batch_max > n_out_range_max:
                    n_out_range_max = batch_max
                logger.debug(
                    "SNR=%.1f dB, batch: %d/%d sensing elements outside [0,1] -> clamped",
                    snr_target, n_out_batch, sensing_pred.size,
                )
            sensing_pred_clamped = np.clip(sensing_pred, 0.0, 1.0)
            tau_pred = sensing_pred_clamped[:, 0] * tau_max
            if num_outputs == 2:
                fd_pred = sensing_pred_clamped[:, 1] * fd_max
            else:
                fd_pred = np.zeros_like(tau_pred)

            tau_min = np.min(tau_pred)
            tau_max_pred = np.max(tau_pred)
            fd_min = np.min(fd_pred)
            fd_max_pred = np.max(fd_pred)

            if tau_min < 0.0 or tau_max_pred > tau_max + _RANGE_TOL:
                raise RuntimeError(
                    f"tau_pred out of range [0, {tau_max}] for SNR={snr_target}: "
                    f"min={tau_min:.4f}, max={tau_max_pred:.4f}"
                )
            if num_outputs == 2 and (fd_min < 0.0 or fd_max_pred > fd_max + _RANGE_TOL):
                raise RuntimeError(
                    f"fd_pred out of range [0, {fd_max}] for SNR={snr_target}: "
                    f"min={fd_min:.4e}, max={fd_max_pred:.4e}"
                )

            if num_outputs == 2:
                mse_tau_b, mse_fd_b = mse_delay_doppler(
                    tau_pred,
                    fd_pred,
                    np.asarray(batch_tau, dtype=np.float64),
                    np.asarray(batch_fd, dtype=np.float64),
                )
            else:
                mse_tau_b = float(
                    np.mean((tau_pred - np.asarray(batch_tau, dtype=np.float64)) ** 2)
                )
                mse_fd_b = 0.0
            mse_tau_acc += mse_tau_b * batch_size_actual
            mse_fd_acc += mse_fd_b * batch_size_actual
            tau_pred_all.extend(float(v) for v in tau_pred)
            fd_pred_all.extend(float(v) for v in fd_pred)
            tau_true_all.extend(float(v) for v in batch_tau)
            fd_true_all.extend(float(v) for v in batch_fd)
            tau_argmax_all.extend(float(v) for v in batch_argmax)
            batch_tau_np = np.asarray(batch_tau, dtype=np.float64)
            n_exact_tau += int(
                np.count_nonzero(np.abs(tau_pred - batch_tau_np) <= _EXACT_TOL)
            )
            n_exact_argmax += int(
                np.count_nonzero(np.abs(batch_argmax - batch_tau_np) <= _EXACT_TOL)
            )

            start = end

        if n_out_range > 0:
            logger.warning(
                "SNR=%.1f dB: %d/%d sensing elements outside [0,1] "
                "(min=%.4f, max=%.4f) -> clamped",
                snr_target, n_out_range, n_out_range_total,
                n_out_range_min, n_out_range_max,
            )

        if accum_symbols == 0:
            logger.warning("No symbols evaluated for SNR=%.1f dB, skipping", snr_target)
            continue

        ber = accum_errors / accum_symbols
        mse_tau = mse_tau_acc / accum_symbols
        mse_fd = mse_fd_acc / accum_symbols

        if not np.isfinite(ber) or ber < 0.0 or ber > 1.0:
            raise RuntimeError(f"invalid BER for SNR {snr_target}: {ber}")
        if not np.isfinite(mse_tau) or not np.isfinite(mse_fd):
            raise RuntimeError(f"MSE not finite for SNR {snr_target}: mse_tau={mse_tau}, mse_fd={mse_fd}")
        if mse_tau < -_MSE_NEG_TOL or mse_fd < -_MSE_NEG_TOL:
            logger.warning("negative MSE for SNR=%.1f: mse_tau=%.6f, mse_fd=%.6f", snr_target, mse_tau, mse_fd)
            mse_tau = max(0.0, mse_tau)
            mse_fd = max(0.0, mse_fd)

        def _pearson(a: np.ndarray, b: np.ndarray) -> float:
            if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        corr_tau_snr = _pearson(
            np.asarray(tau_pred_all), np.asarray(tau_true_all)
        )
        if num_outputs == 2:
            corr_fd_snr = _pearson(
                np.asarray(fd_pred_all), np.asarray(fd_true_all)
            )
        else:
            corr_fd_snr = 0.0
        tau_true_np = np.asarray(tau_true_all, dtype=np.float64)
        tau_argmax_np = np.asarray(tau_argmax_all, dtype=np.float64)
        mse_tau_argmax = float(np.mean((tau_argmax_np - tau_true_np) ** 2))
        corr_tau_argmax = _pearson(tau_argmax_np, tau_true_np)
        n_scored = max(1, int(tau_true_np.size))
        exact_tau = float(n_exact_tau) / n_scored
        exact_tau_argmax = float(n_exact_argmax) / n_scored

        snr_values.append(float(snr_target))
        ber_list.append(float(ber))
        mse_tau_list.append(float(mse_tau))
        mse_fd_list.append(float(mse_fd))
        n_errors_list.append(accum_errors)
        n_symbols_list.append(accum_symbols)
        corr_tau_list.append(corr_tau_snr)
        corr_fd_list.append(corr_fd_snr)
        mse_tau_argmax_list.append(mse_tau_argmax)
        corr_tau_argmax_list.append(corr_tau_argmax)
        exact_tau_list.append(exact_tau)
        exact_tau_argmax_list.append(exact_tau_argmax)

        logger.debug(
            "SNR=%.1f dB: BER=%.6f, MSE_tau=%.6f, MSE_fd=%.6e, "
            "corr(tau)=%.3f, corr(fD)=%.3f, n_errors=%d, n_symbols=%d",
            snr_target, ber, mse_tau, mse_fd, corr_tau_snr, corr_fd_snr,
            accum_errors, accum_symbols,
        )

    if not snr_values:
        raise RuntimeError("No SNR evaluated (empty dataset or no match)")

    results = {
        "snr_db": np.array(snr_values, dtype=np.float64),
        "ber": np.array(ber_list, dtype=np.float64),
        "mse_tau": np.array(mse_tau_list, dtype=np.float64),
        "mse_fd": np.array(mse_fd_list, dtype=np.float64),
        "n_errors": np.array(n_errors_list, dtype=np.int64),
        "n_symbols": np.array(n_symbols_list, dtype=np.int64),
        "corr_tau": np.array(corr_tau_list, dtype=np.float64),
        "corr_fd": np.array(corr_fd_list, dtype=np.float64),
        "mse_tau_argmax": np.array(mse_tau_argmax_list, dtype=np.float64),
        "corr_tau_argmax": np.array(corr_tau_argmax_list, dtype=np.float64),
        "exact_tau": np.array(exact_tau_list, dtype=np.float64),
        "exact_tau_argmax": np.array(exact_tau_argmax_list, dtype=np.float64),
    }

    logger.info("Evaluation completed for %d SNR points", len(snr_values))
    return results

def _online_batch_seed(seed_base: int, snr_db: float, batch_idx: int) -> int:
    
    digest = hashlib.sha256(f"{int(seed_base)}:{float(snr_db)}:{int(batch_idx)}".encode())
    return int.from_bytes(digest.digest()[:4], "big")

def _predict_online_chunked(
    model: tf.keras.Model,
    batch_x: np.ndarray,
    predict_batch: int,
    min_batch: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    
    n = int(batch_x.shape[0])
    if isinstance(predict_batch, bool) or int(predict_batch) < 1:
        raise ValueError(
            f"predict_batch must be an int >= 1, got: {predict_batch!r}"
        )
    if isinstance(min_batch, bool) or int(min_batch) < 1:
        raise ValueError(f"min_batch must be an int >= 1, got: {min_batch!r}")
    chunk = int(predict_batch)
    min_batch = int(min_batch)

    comm_parts: List[np.ndarray] = []
    sensing_parts: List[np.ndarray] = []
    start = 0
    while start < n:
        end = min(start + chunk, n)
        try:
            pred = model(batch_x[start:end], training=False)
        except tf.errors.ResourceExhaustedError:
            if chunk <= min_batch:
                logger.error(
                    "Persistent OOM in online forward even at chunk=%d "
                    "(batch=%d samples): aborting evaluation",
                    min_batch, n,
                )
                raise
            chunk = max(min_batch, chunk // 2)
            logger.warning(
                "OOM in online forward (chunk=%d samples): halving to %d and retrying",
                end - start, chunk,
            )
            gc.collect()
            continue
        comm_parts.append(pred["comm"].numpy())
        sensing_parts.append(pred["sensing"].numpy())
        start = end
    return np.concatenate(comm_parts, axis=0), np.concatenate(sensing_parts, axis=0)

def evaluate_model_online(
    model: tf.keras.Model,
    config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    
    if not isinstance(model, tf.keras.Model):
        raise TypeError(f"model must be a tf.keras.Model, got: {type(model).__name__}")
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got: {type(config).__name__}")

    _validate_config(config)

    eval_cfg = config["evaluation"]
    snr_test_range = eval_cfg["snr_test_range"]
    bit_error_threshold = eval_cfg["bit_error_threshold"]
    max_symbols_per_snr = eval_cfg["max_symbols_per_snr"]
    eval_batch_symbols = int(eval_cfg.get("eval_batch_symbols", 100_000))
    predict_batch = int(eval_cfg.get("eval_predict_batch_size", 1024))
    tau_max = float(config["data"]["max_delay"])
    fd_max = float(config["data"]["max_doppler"])
    num_outputs = sensing_output_units(config)
    echoes = [int(k) for k in config["data"]["echoes"]]
    if not echoes:
        raise ValueError("data.echoes is empty: cannot generate online batches")
    feature_mode = str(config["data"].get("feature_mode", "real"))
    feature_norm = str(config["data"].get("feature_norm", "none"))
    seed_base = int(config["general"].get("seed", 42))

    from src.data.dataset_generator import generate_test_batch

    snr_values: List[float] = []
    ber_list: List[float] = []
    mse_tau_list: List[float] = []
    mse_fd_list: List[float] = []
    n_errors_list: List[int] = []
    n_symbols_list: List[int] = []
    corr_tau_list: List[float] = []
    corr_fd_list: List[float] = []
    mse_tau_argmax_list: List[float] = []
    corr_tau_argmax_list: List[float] = []
    exact_tau_list: List[float] = []
    exact_tau_argmax_list: List[float] = []

    logger.info(
        "ONLINE evaluation over %d SNR points (batch=%d, max_symbols=%d, "
        "floor BER=%.1e)",
        len(snr_test_range), eval_batch_symbols, max_symbols_per_snr,
        1.0 / (2.0 * max_symbols_per_snr),
    )

    for snr_target in snr_test_range:
        accum_errors = 0
        accum_symbols = 0
        mse_tau_acc = 0.0
        mse_fd_acc = 0.0
        n_corr = 0
        tau_sum_t = 0.0; tau_sum_p = 0.0; tau_sum_t2 = 0.0; tau_sum_p2 = 0.0; tau_sum_tp = 0.0
        fd_sum_t = 0.0; fd_sum_p = 0.0; fd_sum_t2 = 0.0; fd_sum_p2 = 0.0; fd_sum_tp = 0.0
        mse_argmax_acc = 0.0
        argmax_sum_p = 0.0; argmax_sum_p2 = 0.0; argmax_sum_tp = 0.0
        n_exact_tau = 0
        n_exact_argmax = 0
        n_out_range = 0
        n_out_range_total = 0
        n_out_range_min: Optional[float] = None
        n_out_range_max: Optional[float] = None

        batch_idx = 0
        while (
            accum_symbols < max_symbols_per_snr
            and accum_errors < bit_error_threshold
        ):
            k = echoes[batch_idx % len(echoes)]
            rng = np.random.default_rng(
                _online_batch_seed(seed_base, snr_target, batch_idx)
            )
            batch_symbols = min(eval_batch_symbols, max_symbols_per_snr - accum_symbols)
            batch = generate_test_batch(
                config, batch_symbols, snr_target, k, rng
            )
            batch_bit = batch["bit"]
            batch_tau = batch["tau"]
            batch_fd = batch["f_d"]
            batch_size_actual = int(batch_bit.shape[0])

            batch_x = _build_feature_matrix(
                batch["x"], feature_mode, feature_norm, batch["x_ref"]
            )
            batch_argmax = argmax_delay_from_profile(
                batch["x"], batch["x_ref"], int(tau_max)
            )
            comm_logits, sensing_pred = _predict_online_chunked(
                model, batch_x, predict_batch
            )

            if comm_logits.shape[0] != batch_size_actual or comm_logits.ndim != 2:
                raise ValueError(f"comm_logits shape {comm_logits.shape} is invalid")
            if (
                sensing_pred.shape[0] != batch_size_actual
                or sensing_pred.ndim != 2
                or sensing_pred.shape[1] != num_outputs
            ):
                raise ValueError(f"sensing_pred shape {sensing_pred.shape} is invalid")
            if not np.all(np.isfinite(comm_logits)):
                raise RuntimeError("comm_logits contains NaN/Inf")
            if not np.all(np.isfinite(sensing_pred)):
                raise RuntimeError("sensing_pred contains NaN/Inf")

            n_errors_batch, _ = bit_error_count(
                comm_logits, np.asarray(batch_bit, dtype=np.int64)
            )
            accum_errors += int(n_errors_batch)
            accum_symbols += batch_size_actual

            n_out_batch = int(
                np.count_nonzero((sensing_pred < 0.0) | (sensing_pred > 1.0))
            )
            if n_out_batch:
                n_out_range += n_out_batch
                n_out_range_total += sensing_pred.size
                batch_min = float(np.min(sensing_pred))
                batch_max = float(np.max(sensing_pred))
                if n_out_range_min is None or batch_min < n_out_range_min:
                    n_out_range_min = batch_min
                if n_out_range_max is None or batch_max > n_out_range_max:
                    n_out_range_max = batch_max

            sensing_pred_clamped = np.clip(sensing_pred, 0.0, 1.0)
            tau_pred = sensing_pred_clamped[:, 0] * tau_max
            if num_outputs == 2:
                fd_pred = sensing_pred_clamped[:, 1] * fd_max
            else:
                fd_pred = np.zeros_like(tau_pred)

            tau_min = np.min(tau_pred)
            tau_max_pred = np.max(tau_pred)
            fd_min = np.min(fd_pred)
            fd_max_pred = np.max(fd_pred)
            if tau_min < 0.0 or tau_max_pred > tau_max + _RANGE_TOL:
                raise RuntimeError(
                    f"tau_pred out of range [0, {tau_max}] for SNR={snr_target}: "
                    f"min={tau_min:.4f}, max={tau_max_pred:.4f}"
                )
            if num_outputs == 2 and (fd_min < 0.0 or fd_max_pred > fd_max + _RANGE_TOL):
                raise RuntimeError(
                    f"fd_pred out of range [0, {fd_max}] for SNR={snr_target}: "
                    f"min={fd_min:.4e}, max={fd_max_pred:.4e}"
                )

            if num_outputs == 2:
                mse_tau_b, mse_fd_b = mse_delay_doppler(
                    tau_pred,
                    fd_pred,
                    np.asarray(batch_tau, dtype=np.float64),
                    np.asarray(batch_fd, dtype=np.float64),
                )
            else:
                mse_tau_b = float(
                    np.mean((tau_pred - np.asarray(batch_tau, dtype=np.float64)) ** 2)
                )
                mse_fd_b = 0.0
            mse_tau_acc += mse_tau_b * batch_size_actual
            mse_fd_acc += mse_fd_b * batch_size_actual
            tau_t = np.asarray(batch_tau, dtype=np.float64)
            tau_p = np.asarray(tau_pred, dtype=np.float64)
            fd_t = np.asarray(batch_fd, dtype=np.float64)
            fd_p = np.asarray(fd_pred, dtype=np.float64)
            n_corr += batch_size_actual
            batch_tau_np = np.asarray(batch_tau, dtype=np.float64)
            n_exact_tau += int(
                np.count_nonzero(np.abs(tau_p - batch_tau_np) <= _EXACT_TOL)
            )
            n_exact_argmax += int(
                np.count_nonzero(np.abs(batch_argmax - batch_tau_np) <= _EXACT_TOL)
            )
            mse_argmax_acc += float(np.sum((batch_argmax - batch_tau_np) ** 2))
            argmax_sum_p += float(batch_argmax.sum())
            argmax_sum_p2 += float((batch_argmax * batch_argmax).sum())
            argmax_sum_tp += float((batch_tau_np * batch_argmax).sum())
            tau_sum_t += float(tau_t.sum()); tau_sum_p += float(tau_p.sum())
            tau_sum_t2 += float((tau_t * tau_t).sum()); tau_sum_p2 += float((tau_p * tau_p).sum())
            tau_sum_tp += float((tau_t * tau_p).sum())
            fd_sum_t += float(fd_t.sum()); fd_sum_p += float(fd_p.sum())
            fd_sum_t2 += float((fd_t * fd_t).sum()); fd_sum_p2 += float((fd_p * fd_p).sum())
            fd_sum_tp += float((fd_t * fd_p).sum())

            batch_idx += 1

        if n_out_range > 0:
            logger.warning(
                "SNR=%.1f dB: %d/%d sensing elements outside [0,1] "
                "(min=%.4f, max=%.4f) -> clamped",
                snr_target, n_out_range, n_out_range_total,
                n_out_range_min, n_out_range_max,
            )

        if accum_symbols == 0:
            logger.warning("No symbols evaluated for SNR=%.1f dB, skipping", snr_target)
            continue

        ber = accum_errors / accum_symbols
        mse_tau = mse_tau_acc / accum_symbols
        mse_fd = mse_fd_acc / accum_symbols

        if not np.isfinite(ber) or ber < 0.0 or ber > 1.0:
            raise RuntimeError(f"invalid BER for SNR {snr_target}: {ber}")
        if not np.isfinite(mse_tau) or not np.isfinite(mse_fd):
            raise RuntimeError(
                f"MSE not finite for SNR {snr_target}: mse_tau={mse_tau}, mse_fd={mse_fd}"
            )
        if mse_tau < -_MSE_NEG_TOL or mse_fd < -_MSE_NEG_TOL:
            mse_tau = max(0.0, mse_tau)
            mse_fd = max(0.0, mse_fd)

        def _pearson_from_sums(
            n: int, sa: float, sa2: float, sb: float, sb2: float, sab: float
        ) -> float:
            if n < 2:
                return 0.0
            cov = n * sab - sa * sb
            va = n * sa2 - sa * sa
            vb = n * sb2 - sb * sb
            if va <= 1e-20 or vb <= 1e-20:
                return 0.0
            return float(cov / ((va * vb) ** 0.5))

        corr_tau_snr = _pearson_from_sums(
            n_corr, tau_sum_t, tau_sum_t2, tau_sum_p, tau_sum_p2, tau_sum_tp
        )
        if num_outputs == 2:
            corr_fd_snr = _pearson_from_sums(
                n_corr, fd_sum_t, fd_sum_t2, fd_sum_p, fd_sum_p2, fd_sum_tp
            )
        else:
            corr_fd_snr = 0.0
        mse_tau_argmax = mse_argmax_acc / max(1, accum_symbols)
        corr_tau_argmax = _pearson_from_sums(
            n_corr, tau_sum_t, tau_sum_t2, argmax_sum_p, argmax_sum_p2, argmax_sum_tp
        )
        n_scored = max(1, accum_symbols)
        exact_tau = float(n_exact_tau) / n_scored
        exact_tau_argmax = float(n_exact_argmax) / n_scored

        snr_values.append(float(snr_target))
        ber_list.append(float(ber))
        mse_tau_list.append(float(mse_tau))
        mse_fd_list.append(float(mse_fd))
        n_errors_list.append(accum_errors)
        n_symbols_list.append(accum_symbols)
        corr_tau_list.append(corr_tau_snr)
        corr_fd_list.append(corr_fd_snr)
        mse_tau_argmax_list.append(mse_tau_argmax)
        corr_tau_argmax_list.append(corr_tau_argmax)
        exact_tau_list.append(exact_tau)
        exact_tau_argmax_list.append(exact_tau_argmax)

        logger.debug(
            "SNR=%.1f dB: BER=%.3e (err=%d/%d), MSE_tau=%.6f, corr(tau)=%.3f",
            snr_target, ber, accum_errors, accum_symbols, mse_tau, corr_tau_snr,
        )

    if not snr_values:
        raise RuntimeError("No SNR evaluated (empty snr_test_range)")

    results = {
        "snr_db": np.array(snr_values, dtype=np.float64),
        "ber": np.array(ber_list, dtype=np.float64),
        "mse_tau": np.array(mse_tau_list, dtype=np.float64),
        "mse_fd": np.array(mse_fd_list, dtype=np.float64),
        "n_errors": np.array(n_errors_list, dtype=np.int64),
        "n_symbols": np.array(n_symbols_list, dtype=np.int64),
        "corr_tau": np.array(corr_tau_list, dtype=np.float64),
        "corr_fd": np.array(corr_fd_list, dtype=np.float64),
        "mse_tau_argmax": np.array(mse_tau_argmax_list, dtype=np.float64),
        "corr_tau_argmax": np.array(corr_tau_argmax_list, dtype=np.float64),
        "exact_tau": np.array(exact_tau_list, dtype=np.float64),
        "exact_tau_argmax": np.array(exact_tau_argmax_list, dtype=np.float64),
    }

    logger.info("Online evaluation completed for %d SNR points", len(snr_values))
    return results

def compute_ber_curve(results: Dict[str, np.ndarray]) -> pd.DataFrame:
    
    if not isinstance(results, dict):
        raise TypeError(f"results must be a dict, got: {type(results).__name__}")
    required = ("snr_db", "ber", "mse_tau", "mse_fd", "n_errors", "n_symbols")
    missing = [k for k in required if k not in results]
    if missing:
        raise ValueError(f"results is missing the keys: {missing}")

    df = pd.DataFrame({
        "snr_db": results["snr_db"],
        "ber": results["ber"],
        "mse_tau": results["mse_tau"],
        "mse_fd": results["mse_fd"],
        "n_errors": results["n_errors"],
        "n_symbols": results["n_symbols"],
        "corr_tau": results.get("corr_tau", np.zeros(len(results["snr_db"]))),
        "corr_fd": results.get("corr_fd", np.zeros(len(results["snr_db"]))),
        "mse_tau_argmax": results.get("mse_tau_argmax", np.zeros(len(results["snr_db"]))),
        "corr_tau_argmax": results.get("corr_tau_argmax", np.zeros(len(results["snr_db"]))),
        "exact_tau": results.get("exact_tau", np.zeros(len(results["snr_db"]))),
        "exact_tau_argmax": results.get("exact_tau_argmax", np.zeros(len(results["snr_db"]))),
    })

    ber = df["ber"].to_numpy(dtype=np.float64)
    n_symbols = df["n_symbols"].to_numpy(dtype=np.float64)
    zero_mask = (ber == 0.0) & (n_symbols > 0)
    if np.any(zero_mask):
        n_floored = int(np.count_nonzero(zero_mask))
        df["ber"] = np.where(zero_mask, 1.0 / (2.0 * n_symbols), ber)
        logger.info(
            "BER floor applied to %d points (no observed errors): "
            "1/(2*n_symbols)", n_floored,
        )

    df = df.sort_values("snr_db").reset_index(drop=True)
    logger.debug("Curves generated: %d points", len(df))
    return df

def compute_sensing_rmse(results: Dict[str, np.ndarray]) -> pd.DataFrame:
    
    if not isinstance(results, dict):
        raise TypeError(f"results must be a dict, got: {type(results).__name__}")
    required = ("snr_db", "mse_tau", "mse_fd")
    missing = [k for k in required if k not in results]
    if missing:
        raise ValueError(f"results is missing the keys: {missing}")

    rmse_tau = np.sqrt(np.maximum(results["mse_tau"], 0.0))
    rmse_fd = np.sqrt(np.maximum(results["mse_fd"], 0.0))

    df = pd.DataFrame({
        "snr_db": results["snr_db"],
        "rmse_tau": rmse_tau,
        "rmse_fd": rmse_fd,
    })
    df = df.sort_values("snr_db").reset_index(drop=True)
    logger.debug("Sensing RMSE computed for %d points", len(df))
    return df

def plot_ber_vs_snr(
    df: pd.DataFrame,
    output_dir: Path,
    plot_format: str,
    model_name: str,
) -> Path:
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pd.DataFrame, got: {type(df).__name__}")
    if "snr_db" not in df.columns or "ber" not in df.columns:
        raise ValueError("df must contain the 'snr_db' and 'ber' columns")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(df["snr_db"], df["ber"], marker="o", linestyle="-", linewidth=2, label=model_name)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title(f"BER vs SNR - {model_name}")
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.legend()

    min_ber = df["ber"].min()
    if min_ber > 0:
        y_lower = max(1e-6, min_ber / 10)
    else:
        y_lower = 1e-6
    ax.set_ylim([y_lower, 1.0])

    output_path = output_dir / f"ber_vs_snr_{model_name}.{plot_format}"
    plt.savefig(output_path, format=plot_format, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("BER plot saved to %s", output_path)
    return output_path

def plot_loss_curves(
    history: tf.keras.callbacks.History,
    output_dir: Path,
    plot_format: str,
    model_name: str,
) -> Path:
    
    if not isinstance(history, tf.keras.callbacks.History):
        raise TypeError(f"history must be a tf.keras.callbacks.History, got: {type(history).__name__}")
    if not history.history:
        raise ValueError("history.history is empty")
    if "loss" not in history.history:
        raise ValueError("history.history must contain the key 'loss'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = range(1, len(history.history["loss"]) + 1)
    ax.plot(epochs, history.history["loss"], label="Train Loss", marker="o", linestyle="-")
    if "val_loss" in history.history:
        ax.plot(epochs, history.history["val_loss"], label="Val Loss", marker="s", linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curves - {model_name}")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    output_path = output_dir / f"loss_curves_{model_name}.{plot_format}"
    plt.savefig(output_path, format=plot_format, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("Loss plot saved to %s", output_path)
    return output_path

def plot_sensing_error(
    df: pd.DataFrame,
    output_dir: Path,
    plot_format: str,
    model_name: str,
) -> Path:
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pd.DataFrame, got: {type(df).__name__}")
    required = ("snr_db", "rmse_tau", "rmse_fd")
    for col in required:
        if col not in df.columns:
            raise ValueError(f"df must contain the column '{col}'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.semilogy(df["snr_db"], df["rmse_tau"], marker="o", linestyle="-", color="blue")
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("RMSE τ (samples)")
    ax1.grid(True, linestyle="--", alpha=0.7)

    ax2.semilogy(df["snr_db"], df["rmse_fd"], marker="s", linestyle="-", color="orange")
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("RMSE fD (cycles/sample)")
    ax2.grid(True, linestyle="--", alpha=0.7)

    fig.suptitle(f"Sensing Error vs SNR - {model_name}")

    output_path = output_dir / f"sensing_rmse_{model_name}.{plot_format}"
    plt.savefig(output_path, format=plot_format, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("Sensing plot saved to %s", output_path)
    return output_path

def main(argv: Optional[Sequence[str]] = None) -> None:
    
    parser = argparse.ArgumentParser(
        description="Evaluate a Ultra-CAN (ISAC in IoD) model"
    )
    parser.add_argument("--config", required=True, help="path to the experiment config (YAML)")
    parser.add_argument(
        "--model",
        choices=["conv1d", "qkv", "lstm", "mc_dlsk"],
        default="conv1d",
        help="model to evaluate (default: conv1d)",
    )
    parser.add_argument(
        "--plot-format",
        default="pdf",
        help="plot format (default: pdf)",
    )
    parser.add_argument(
        "--allow-missing-model",
        action="store_true",
        help="if set, builds the model from scratch when the checkpoint does not exist (otherwise raises FileNotFoundError)",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path, DEFAULT_BASE_CONFIG_PATH)
    _validate_config(config)

    general = config.get("general", {})
    experiment_name = str(general.get("experiment_name", "ultra_can_isac"))
    log_dir = _REPO_ROOT / "results" / experiment_name / "logs"
    log_file = setup_logging(
        log_dir=log_dir,
        level=str(general.get("log_level", "INFO")),
        experiment_name=experiment_name,
    )
    log_config_summary(config, logger)
    logger.info("Log file: %s", log_file)

    from src.utils.dataset_utils import get_dataset_dir

    data_cfg = config["data"]
    data_dir = get_dataset_dir(config)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset not found in {data_dir}. Generate the data through the runner "
            "(src/experiments/runner.py) or scripts/generate_all_datasets.sh."
        )
    snr_grid = build_snr_grid(data_cfg["snr_range"], data_cfg["snr_step"])
    echoes = list(data_cfg["echoes"])

    logger.info("Loading test dataset from %s", data_dir)
    data = load_npz_files(data_dir, snr_grid, echoes, "test", config)
    logger.info("Test dataset loaded: %d samples", data["x"].shape[0])
    data["x_ref"] = build_reference_matrix(data["bit"], data["seed"], config)

    verify_snr_balance(data, snr_grid, echoes)

    model_dir = _REPO_ROOT / "results" / experiment_name / args.model / "models"
    checkpoint_candidates: List[Path] = [
        model_dir / "best_model.keras",
        model_dir / "best_model.h5",
    ]
    training_cfg = config.get("training")
    if isinstance(training_cfg, dict) and training_cfg.get("checkpoint_path"):
        ckpt_from_config = Path(str(training_cfg["checkpoint_path"]))
        if not ckpt_from_config.is_absolute():
            ckpt_from_config = _REPO_ROOT / ckpt_from_config
        checkpoint_candidates.append(ckpt_from_config)
    model_path = next(
        (p for p in checkpoint_candidates if p.is_file()),
        checkpoint_candidates[0],
    )
    if model_path.is_file():
        logger.info("Loading model from %s", model_path)
        model = load_model(model_path)
    else:
        if args.allow_missing_model:
            logger.warning("Model not found in %s, building from scratch without training", model_path)
            if args.model == "conv1d":
                model = build_ultra_can(config)
            elif args.model == "qkv":
                model = build_ultra_can_qkv(config)
            elif args.model in ("lstm", "mc_dlsk"):
                model = build_baseline(config, args.model)
            else:
                raise ValueError(f"Model not supported: {args.model}")
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                "Train the model before evaluating it or use --allow-missing-model to build it from scratch."
            )

    results = evaluate_model(model, data, config)

    df_ber = compute_ber_curve(results)
    df_sensing = compute_sensing_rmse(results)

    plot_dir = _REPO_ROOT / "results" / experiment_name / args.model / "plots"
    plot_ber_vs_snr(df_ber, plot_dir, args.plot_format, args.model)
    plot_sensing_error(df_sensing, plot_dir, args.plot_format, args.model)

    history_csv = _REPO_ROOT / "results" / experiment_name / "logs" / "history.csv"
    if history_csv.exists():
        try:
            history_df = pd.read_csv(history_csv)
            if "loss" in history_df.columns:
                history = tf.keras.callbacks.History()
                history.history = {col: history_df[col].tolist() for col in history_df.columns}
                plot_loss_curves(history, plot_dir, args.plot_format, args.model)
            else:
                logger.warning("history.csv does not contain the 'loss' column, skipping the loss-curve plots")
        except Exception as e:
            logger.warning("Unable to load history for the loss curves: %s", e)
    else:
        logger.info("history.csv not found, skipping the loss-curve plots")

    csv_path = plot_dir / "metrics.csv"
    df_ber.to_csv(csv_path, index=False)
    logger.info("Metrics saved to %s", csv_path)

    npz_path = plot_dir / "evaluation_results.npz"
    np.savez(npz_path, **results)
    logger.info("Results saved to %s", npz_path)

    logger.info("Evaluation completed. Results in %s", plot_dir)

if __name__ == "__main__":
    main()

