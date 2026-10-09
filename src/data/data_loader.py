
from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset_generator import build_snr_grid, resolve_n_per_combo
from src.utils.config_loader import (
    DEFAULT_BASE_CONFIG_PATH,
    load_config,
    save_config_snapshot,
)
from src.utils.logger import log_config_summary, setup_logging

logger = logging.getLogger(__name__)

_SPLITS = frozenset({"train", "val", "test"})
_FEATURE_MODES = frozenset({"real", "iq"})
_FEATURE_NORMS = frozenset({"none", "sign", "energy"})
_NPZ_KEYS = frozenset({"x", "bit", "tau", "f_d", "snr_db", "k", "seed"})
_NPZ_NAME_RE = re.compile(
    r"^(train|val|test)_snr(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)_echo(\d+)\.npz$"
)
_SNR_ROUND_DECIMALS = 6
_RANGE_TOL = 1e-9
_MIN_SEQUENCE_LENGTH = 1
_MIN_BATCH_SIZE = 1
_NUM_FEATURES_REAL = 1
_NUM_FEATURES_IQ = 2
_NUM_REFERENCE_CHANNELS = 1

DataDict = Dict[str, np.ndarray]

def model_input_channels(feature_mode: str) -> int:
    if feature_mode == "iq":
        return _NUM_FEATURES_IQ + _NUM_REFERENCE_CHANNELS
    if feature_mode == "real":
        return _NUM_FEATURES_REAL + _NUM_REFERENCE_CHANNELS
    raise ValueError(
        f"feature_mode must be one of {sorted(_FEATURE_MODES)}, got: {feature_mode!r}"
    )

def _parse_npz_name(path: Path) -> Tuple[str, float, int]:
    match = _NPZ_NAME_RE.match(path.name)
    if match is None:
        raise ValueError(
            f"file name not recognized: {path.name!r} "
            f"(expected: <split>_snr<snr>_echo<k>.npz)"
        )
    split, snr_str, k_str = match.groups()
    return split, float(snr_str), int(k_str)

def _validate_npz_arrays(
    x: np.ndarray,
    bit: np.ndarray,
    tau: np.ndarray,
    f_d: np.ndarray,
    sequence_length: int,
    max_delay: float,
    max_doppler: float,
) -> None:
    if x.ndim != 2 or x.shape[1] != sequence_length:
        raise ValueError(
            f"x must have shape (N, {sequence_length}), got: {x.shape}"
        )
    n = int(x.shape[0])
    if n == 0:
        raise ValueError("block without samples (n=0)")
    if not (len(bit) == len(tau) == len(f_d) == n):
        raise ValueError(
            "lunghezze incoerenti: "
            f"x={n}, bit={len(bit)}, tau={len(tau)}, f_d={len(f_d)}"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN/Inf")
    if not np.all(np.isin(bit, (0, 1))):
        raise ValueError("bit must contain only the values 0/1")
    if not np.all(np.isfinite(tau)) or np.any(tau < 0.0) or np.any(tau > max_delay + _RANGE_TOL):
        raise ValueError(f"tau out of range [0, {max_delay}] or not finite")
    if not np.all(np.isfinite(f_d)) or np.any(f_d < 0.0) or np.any(f_d > max_doppler + _RANGE_TOL):
        raise ValueError(f"f_d out of range [0, {max_doppler}] or not finite")

def load_npz_files(
    data_dir: Path,
    snr_values: List[float],
    echoes: List[int],
    split: str,
    config: Dict[str, Any],
) -> DataDict:
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {sorted(_SPLITS)}, got: {split!r}")
    if not snr_values:
        raise ValueError("snr_values cannot be empty")
    if not echoes:
        raise ValueError("echoes cannot be empty")
    if not all(math.isfinite(float(s)) for s in snr_values):
        raise ValueError(f"snr_values must contain only finite values: {snr_values!r}")
    if not all(isinstance(k, int) and not isinstance(k, bool) and k >= 0 for k in echoes):
        raise ValueError(f"echoes must contain ints >= 0, got: {echoes!r}")

    data = config["data"]
    sequence_length = int(data["sequence_length"])
    max_delay = float(data["max_delay"])
    max_doppler = float(data["max_doppler"])

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir does not exist or is not a directory: {data_dir}")

    combos: List[Tuple[float, int]] = [
        (float(snr), int(k)) for snr in snr_values for k in echoes
    ]

    files_by_combo: Dict[Tuple[float, int], Path] = {}
    for path in sorted(data_dir.glob(f"{split}_*.npz")):
        try:
            f_split, snr, k = _parse_npz_name(path)
        except ValueError as exc:
            logger.warning("file skipped: %s", exc)
            continue
        if f_split != split:
            continue
        if (snr, k) in files_by_combo:
            logger.warning("duplicate combination (snr=%s, k=%d): skipped %s", snr, k, path.name)
            continue
        files_by_combo[(snr, k)] = path

    if not files_by_combo:
        raise FileNotFoundError(
            f"no valid .npz file for split={split!r} in {data_dir}"
        )

    missing = [combo for combo in combos if combo not in files_by_combo]
    if missing:
        raise ValueError(
            f"(SNR, K) grid incomplete for split={split!r}: "
            f"missing files for {missing}"
        )

    x_parts: List[np.ndarray] = []
    bit_parts: List[np.ndarray] = []
    tau_parts: List[np.ndarray] = []
    f_d_parts: List[np.ndarray] = []
    snr_parts: List[np.ndarray] = []
    k_parts: List[np.ndarray] = []
    seed_parts: List[np.ndarray] = []

    for snr, k in combos:
        path = files_by_combo[(snr, k)]
        with np.load(path, allow_pickle=False) as npz:
            missing_keys = _NPZ_KEYS - set(npz.files)
            if missing_keys:
                raise ValueError(
                    f"missing keys in {path.name}: {sorted(missing_keys)}"
                )
            x_file = np.asarray(npz["x"])
            bit_file = np.asarray(npz["bit"])
            tau_file = np.asarray(npz["tau"])
            f_d_file = np.asarray(npz["f_d"])
            seed_file = np.asarray(npz["seed"])
            snr_file = float(np.asarray(npz["snr_db"]))
            k_file = int(np.asarray(npz["k"]))

        if abs(snr_file - snr) > 1e-6 or k_file != k:
            raise ValueError(
                f"{path.name}: snr_db={snr_file} or k={k_file} inconsistent with the file name"
            )

        _validate_npz_arrays(
            x_file,
            bit_file,
            tau_file,
            f_d_file,
            sequence_length=sequence_length,
            max_delay=max_delay,
            max_doppler=max_doppler,
        )
        n_file = int(x_file.shape[0])
        if seed_file.shape != (n_file,):
            raise ValueError(
                f"{path.name}: seed shape {seed_file.shape} inconsistent with n={n_file}"
            )
        x_parts.append(x_file)
        bit_parts.append(bit_file)
        tau_parts.append(tau_file)
        f_d_parts.append(f_d_file)
        snr_parts.append(np.full(n_file, snr_file, dtype=np.float64))
        k_parts.append(np.full(n_file, k_file, dtype=np.int64))
        seed_parts.append(seed_file)
        logger.debug(
            "loaded %s: %d samples, snr=%.4g, k=%d", path.name, n_file, snr_file, k_file,
        )

    data_out: DataDict = {
        "x": np.concatenate(x_parts, axis=0),
        "bit": np.concatenate(bit_parts, axis=0),
        "tau": np.concatenate(tau_parts, axis=0),
        "f_d": np.concatenate(f_d_parts, axis=0),
        "snr_db": np.concatenate(snr_parts, axis=0),
        "k": np.concatenate(k_parts, axis=0),
        "seed": np.concatenate(seed_parts, axis=0),
    }

    _validate_npz_arrays(
        data_out["x"],
        data_out["bit"],
        data_out["tau"],
        data_out["f_d"],
        sequence_length=sequence_length,
        max_delay=max_delay,
        max_doppler=max_doppler,
    )

    logger.info(
        "split '%s': %d total samples, x shape=%s, tau in [%.2f, %.2f], f_d in [%.3e, %.3e]",
        split,
        int(data_out["x"].shape[0]),
        data_out["x"].shape,
        float(np.min(data_out["tau"])),
        float(np.max(data_out["tau"])),
        float(np.min(data_out["f_d"])),
        float(np.max(data_out["f_d"])),
    )
    return data_out

def normalize_targets(
    tau: np.ndarray,
    f_d: np.ndarray,
    tau_max: float,
    fd_max: float,
    num_outputs: int = 2,
) -> np.ndarray:
    if num_outputs not in (1, 2):
        raise ValueError(f"num_outputs must be 1 or 2, got: {num_outputs!r}")
    tau_arr = np.asarray(tau, dtype=np.float64)
    f_d_arr = np.asarray(f_d, dtype=np.float64)
    if tau_arr.ndim != 1 or f_d_arr.ndim != 1:
        raise ValueError(
            f"tau and f_d must be 1D, got: {tau_arr.shape}, {f_d_arr.shape}"
        )
    if tau_arr.shape != f_d_arr.shape:
        raise ValueError(
            f"tau and f_d must have the same shape: {tau_arr.shape} vs {f_d_arr.shape}"
        )
    if not (math.isfinite(tau_max) and tau_max > 0.0):
        raise ValueError(f"tau_max must be finite and > 0, got: {tau_max!r}")
    if not (math.isfinite(fd_max) and fd_max > 0.0):
        raise ValueError(f"fd_max must be finite and > 0, got: {fd_max!r}")
    if not (np.all(np.isfinite(tau_arr)) and np.all(np.isfinite(f_d_arr))):
        raise ValueError("tau/f_d contain NaN/Inf")

    tau_norm = tau_arr / float(tau_max)
    if np.any(tau_norm < -_RANGE_TOL) or np.any(tau_norm > 1.0 + _RANGE_TOL):
        raise RuntimeError(
            f"tau_norm out of [0, 1]: range [{np.min(tau_norm)}, {np.max(tau_norm)}]"
        )
    columns = [tau_norm]
    if num_outputs == 2:
        f_d_norm = f_d_arr / float(fd_max)
        if np.any(f_d_norm < -_RANGE_TOL) or np.any(f_d_norm > 1.0 + _RANGE_TOL):
            raise RuntimeError(
                f"f_d_norm out of [0, 1]: range [{np.min(f_d_norm)}, {np.max(f_d_norm)}]"
            )
        columns.append(f_d_norm)

    targets = np.stack(columns, axis=-1).astype(np.float32)
    logger.debug(
        "normalised targets: shape=%s, tau_norm in [%.4f, %.4f]",
        targets.shape,
        float(np.min(tau_norm)),
        float(np.max(tau_norm)),
    )
    return targets

def regenerate_reference(bit: Any, seed: Any, config: Dict[str, Any]) -> np.ndarray:
    from src.data.dataset_generator import (
        _center_normalize_symbol,
        _map_type_for_bit,
        generate_chaotic_sequence,
    )

    data = config["data"]
    seq_len = int(data["sequence_length"])
    map_type = str(data["map_type"])
    map_param = float(data["map_param"])
    x = generate_chaotic_sequence(
        _map_type_for_bit(map_type, int(bit)), map_param, int(seed), seq_len
    )
    return _center_normalize_symbol(x)

def build_reference_matrix(
    bit: np.ndarray, seed: np.ndarray, config: Dict[str, Any]
) -> np.ndarray:
    bit_arr = np.asarray(bit)
    seed_arr = np.asarray(seed)
    if bit_arr.ndim != 1 or seed_arr.ndim != 1 or bit_arr.shape != seed_arr.shape:
        raise ValueError(
            f"bit/seed must be aligned 1D arrays, got: {bit_arr.shape}, {seed_arr.shape}"
        )
    n = int(bit_arr.shape[0])
    seq_len = int(config["data"]["sequence_length"])
    out = np.empty((n, seq_len), dtype=np.float32)
    for i in range(n):
        out[i] = regenerate_reference(bit_arr[i], seed_arr[i], config).astype(np.float32)
    logger.debug("reference matrix built: shape=%s", out.shape)
    return out

def _build_feature_matrix(
    y_complex: np.ndarray,
    feature_mode: str,
    feature_norm: Optional[str] = None,
    reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    y = np.asarray(y_complex)
    if y.ndim != 2:
        raise ValueError(f"y_complex must be 2D, got: {y.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError("y_complex contains NaN/Inf")
    if feature_mode not in _FEATURE_MODES:
        raise ValueError(
            f"feature_mode must be one of {sorted(_FEATURE_MODES)}, got: {feature_mode!r}"
        )
    if feature_norm is None:
        feature_norm = "none"
    if feature_norm not in _FEATURE_NORMS:
        raise ValueError(
            f"feature_norm must be one of {sorted(_FEATURE_NORMS)}, got: {feature_norm!r}"
        )

    if feature_mode == "real":
        features = np.real(y)[..., np.newaxis]
    else:
        features = np.stack([np.real(y), np.imag(y)], axis=-1)

    if feature_norm != "none":
        if feature_mode != "real":
            raise ValueError(
                "feature_norm != 'none' requires feature_mode='real' "
                f"(got feature_mode={feature_mode!r})"
            )
        x = features[..., 0]
        if feature_norm in ("sign", "energy"):
            x = np.where(x.mean(axis=-1, keepdims=True) < 0.0, -x, x)
        if feature_norm == "energy":
            norm = np.linalg.norm(x, axis=-1, keepdims=True)
            norm = np.where(norm == 0.0, 1.0, norm)
            x = x / norm
        features = x[..., np.newaxis]

    if reference is not None:
        ref = np.asarray(reference, dtype=np.float32)
        if ref.ndim != 2 or ref.shape[0] != features.shape[0]:
            raise ValueError(
                f"expected reference shape (N, N_seq) with N={features.shape[0]}, "
                f"got: {ref.shape}"
            )
        if ref.shape[1] != features.shape[1]:
            raise ValueError(
                f"reference length {ref.shape[1]} inconsistent with N_seq={features.shape[1]}"
            )
        features = np.concatenate([features, ref[..., np.newaxis]], axis=-1)

    features = features.astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise RuntimeError("features not finite after the float32 conversion")
    logger.debug(
        "features: mode=%s, norm=%s, shape=%s, dtype=%s",
        feature_mode, feature_norm, features.shape, features.dtype,
    )
    return features

def build_tf_dataset(
    data: DataDict,
    batch_size: int,
    config: Dict[str, Any],
    shuffle: bool = True,
    seed: Optional[int] = None,
    feature_mode: Optional[str] = None,
    feature_norm: Optional[str] = None,
) -> tf.data.Dataset:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < _MIN_BATCH_SIZE:
        raise ValueError(
            f"batch_size must be an int >= {_MIN_BATCH_SIZE}, got: {batch_size!r}"
        )
    if seed is None:
        seed_val = config["general"].get("seed")
        if seed_val is None:
            raise ValueError("missing key: general.seed")
        seed = int(seed_val)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be an int >= 0, got: {seed!r}")
    if feature_mode is None:
        feature_mode = config["data"].get("feature_mode")
        if not isinstance(feature_mode, str):
            raise ValueError("missing or non-string key: data.feature_mode")
    if feature_mode not in _FEATURE_MODES:
        raise ValueError(
            f"feature_mode must be one of {sorted(_FEATURE_MODES)}, got: {feature_mode!r}"
        )
    if feature_norm is None:
        feature_norm = config["data"].get("feature_norm", "none")
        if not isinstance(feature_norm, str):
            raise ValueError("missing or non-string key: data.feature_norm")
    if feature_norm not in _FEATURE_NORMS:
        raise ValueError(
            f"feature_norm must be one of {sorted(_FEATURE_NORMS)}, got: {feature_norm!r}"
        )
    if feature_norm != "none" and feature_mode != "real":
        raise ValueError(
            "feature_norm != 'none' requires feature_mode='real' "
            f"(got feature_mode={feature_mode!r})"
        )

    missing = [key for key in ("x", "bit", "tau", "f_d", "seed") if key not in data]
    if missing:
        raise ValueError(f"data is missing the keys: {missing}")
    n = int(data["x"].shape[0])
    if n < 1:
        raise ValueError("data contains no samples (n < 1)")

    sequence_length = int(config["data"]["sequence_length"])
    max_delay = float(config["data"]["max_delay"])
    max_doppler = float(config["data"]["max_doppler"])

    if "x_ref" in data and data["x_ref"] is not None:
        reference = data["x_ref"]
    else:
        reference = build_reference_matrix(data["bit"], data["seed"], config)
    features = _build_feature_matrix(data["x"], feature_mode, feature_norm, reference)
    from src.models.heads import sensing_output_units

    num_outputs = sensing_output_units(config)
    targets = normalize_targets(
        data["tau"], data["f_d"], tau_max=max_delay, fd_max=max_doppler,
        num_outputs=num_outputs,
    )
    bits = np.asarray(data["bit"], dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (features, {"comm": bits, "sensing": targets})
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=n, seed=seed, reshuffle_each_iteration=True
        )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    num_features = model_input_channels(feature_mode)
    features_spec = dataset.element_spec[0]
    comm_spec = dataset.element_spec[1]["comm"]
    sensing_spec = dataset.element_spec[1]["sensing"]
    if features_spec.shape.as_list() != [None, sequence_length, num_features]:
        raise ValueError(
            f"expected features shape [None, {sequence_length}, {num_features}], "
            f"got: {features_spec.shape}"
        )
    if comm_spec.shape.as_list() != [None]:
        raise ValueError(f"expected comm shape [None], got: {comm_spec.shape}")
    if sensing_spec.shape.as_list() != [None, num_outputs]:
        raise ValueError(
            f"expected sensing shape [None, {num_outputs}], got: {sensing_spec.shape}"
        )

    logger.info(
        "dataset built: n=%d, batch_size=%d, feature_mode=%s, shuffle=%s, seed=%d",
        n, batch_size, feature_mode, shuffle, seed,
    )
    logger.debug(
        "element_spec: features=%s, comm=%s, sensing=%s",
        features_spec, comm_spec, sensing_spec,
    )
    return dataset

def verify_snr_balance(
    data: DataDict,
    snr_values: List[float],
    echoes: List[int],
    expected_per_combo: Optional[int] = None,
) -> Dict[Tuple[float, int], int]:
    for key in ("snr_db", "k"):
        if key not in data:
            raise ValueError(f"data is missing the key '{key}' for the balancing check")
    snr_arr = np.asarray(data["snr_db"], dtype=np.float64)
    k_arr = np.asarray(data["k"], dtype=np.int64)
    if snr_arr.ndim != 1 or k_arr.ndim != 1:
        raise ValueError("snr_db and k must be per-sample 1D arrays")
    if len(snr_arr) != len(k_arr):
        raise ValueError(
            f"snr_db ({len(snr_arr)}) and k ({len(k_arr)}) must have the same length"
        )
    if expected_per_combo is not None:
        if (
            not isinstance(expected_per_combo, int)
            or isinstance(expected_per_combo, bool)
            or expected_per_combo <= 0
        ):
            raise ValueError(
                f"expected_per_combo must be an int > 0, got: {expected_per_combo!r}"
            )

    counts: Dict[Tuple[float, int], int] = Counter(
        (round(float(snr_arr[i]), _SNR_ROUND_DECIMALS), int(k_arr[i]))
        for i in range(len(snr_arr))
    )

    expected_combos = {
        (round(float(snr), _SNR_ROUND_DECIMALS), int(k)) for snr in snr_values for k in echoes
    }
    got_combos = set(counts)
    missing = sorted(expected_combos - got_combos)
    extra = sorted(got_combos - expected_combos)
    if missing or extra:
        raise ValueError(
            f"(SNR, K) grid unbalanced: missing={missing}, extra={extra}"
        )

    values = list(counts.values())
    if len(set(values)) != 1:
        raise ValueError(
            "dataset unbalanced per (SNR, K): non-uniform counts -> "
            f"{dict(sorted(counts.items()))}"
        )
    if expected_per_combo is not None and values[0] != expected_per_combo:
        raise ValueError(
            f"dataset unbalanced: per (SNR, K) count = {values[0]}, "
            f"expected = {expected_per_combo}"
        )

    logger.info(
        "balancing verified: %d samples for each of the %d (SNR, K) combinations",
        values[0], len(expected_combos),
    )
    return dict(counts)

def _expected_per_combo(config: Dict[str, Any], split: str, num_combos: int) -> int:
    return resolve_n_per_combo(config, split, num_combos)

def _validate_config(config: Dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got: {type(config).__name__}")
    general = config.get("general")
    data = config.get("data")
    training = config.get("training")
    if not isinstance(general, dict):
        raise ValueError("'general' section missing or not a dict")
    if not isinstance(data, dict):
        raise ValueError("'data' section missing or not a dict")
    if not isinstance(training, dict):
        raise ValueError("'training' section missing or not a dict")

    seed = general.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"general.seed must be an int >= 0, got: {seed!r}")
    experiment_name = general.get("experiment_name")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("general.experiment_name must be a non-empty string")

    sequence_length = data.get("sequence_length")
    if (
        not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or sequence_length < _MIN_SEQUENCE_LENGTH
    ):
        raise ValueError(
            f"data.sequence_length must be an int >= {_MIN_SEQUENCE_LENGTH}, "
            f"got: {sequence_length!r}"
        )

    feature_mode = data.get("feature_mode")
    if feature_mode not in _FEATURE_MODES:
        raise ValueError(
            f"data.feature_mode must be one of {sorted(_FEATURE_MODES)}, got: {feature_mode!r}"
        )

    for key in ("max_delay", "max_doppler"):
        value = data.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"data.{key} must be finite and > 0, got: {value!r}")
    if float(data["max_doppler"]) >= 0.5:
        raise ValueError("data.max_doppler must be < 0.5 (anti-aliasing guard)")

    snr_range = data.get("snr_range")
    if not isinstance(snr_range, (list, tuple)) or len(snr_range) != 2:
        raise ValueError(f"data.snr_range must be [min, max], got: {snr_range!r}")
    snr_min, snr_max = float(snr_range[0]), float(snr_range[1])
    if not (math.isfinite(snr_min) and math.isfinite(snr_max)) or snr_min >= snr_max:
        raise ValueError(f"data.snr_range invalid: {snr_range!r}")

    snr_step = data.get("snr_step")
    if (
        not isinstance(snr_step, (int, float))
        or isinstance(snr_step, bool)
        or not math.isfinite(float(snr_step))
        or float(snr_step) <= 0.0
    ):
        raise ValueError(f"data.snr_step must be > 0, got: {snr_step!r}")

    echoes = data.get("echoes")
    if not isinstance(echoes, (list, tuple)) or len(echoes) == 0:
        raise ValueError(f"data.echoes must be a non-empty list, got: {echoes!r}")
    for k in echoes:
        if not isinstance(k, int) or isinstance(k, bool) or k < 0:
            raise ValueError(f"data.echoes must contain ints >= 0, got: {k!r}")

    for key in ("num_symbols_train", "num_symbols_val", "num_symbols_test"):
        value = data.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"data.{key} must be an int > 0, got: {value!r}")

    raw_dir = data.get("raw_dir")
    if not isinstance(raw_dir, str) or not raw_dir.strip():
        raise ValueError("data.raw_dir must be a non-empty string")

    batch_size = training.get("batch_size")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < _MIN_BATCH_SIZE:
        raise ValueError(
            f"training.batch_size must be an int >= {_MIN_BATCH_SIZE}, got: {batch_size!r}"
        )

    logger.debug(
        "validated config: seq_len=%d, feature_mode=%s, max_delay=%d, max_doppler=%s, batch_size=%d",
        sequence_length,
        feature_mode,
        int(data["max_delay"]),
        data["max_doppler"],
        batch_size,
    )

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Data loader for the Ultra-CAN (ISAC in IoD)"
    )
    parser.add_argument("--config", required=True, help="path to the experiment config (YAML)")
    parser.add_argument(
        "--splits", default="train,val,test", help="splits to load, comma-separated"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="override the data directory (default: data.raw_dir)",
    )
    args = parser.parse_args(argv)

    config = load_config(
        config_path=Path(args.config),
        base_config_path=DEFAULT_BASE_CONFIG_PATH,
    )
    _validate_config(config)

    experiment_name = str(config["general"]["experiment_name"])
    log_dir = _REPO_ROOT / "results" / experiment_name / "logs"
    setup_logging(
        log_dir=log_dir,
        level=str(config["general"].get("log_level", "INFO")),
        experiment_name=experiment_name,
    )
    log_config_summary(config, logger)
    save_config_snapshot(config, log_dir)

    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not splits:
        raise ValueError("--splits does not contain valid splits")
    for split in splits:
        if split not in _SPLITS:
            raise ValueError(f"invalid split: {split!r} (expected: {sorted(_SPLITS)})")

    data_dir = Path(args.data_dir) if args.data_dir else Path(config["data"]["raw_dir"])
    snr_grid = build_snr_grid(config["data"]["snr_range"], config["data"]["snr_step"])
    echoes = [int(k) for k in config["data"]["echoes"]]
    num_combos = len(snr_grid) * len(echoes)
    batch_size = int(config["training"]["batch_size"])
    feature_mode = str(config["data"]["feature_mode"])

    logger.info(
        "starting the data loader: splits=%s, data_dir=%s, SNR grid=%s, echoes=%s",
        splits, data_dir, snr_grid, echoes,
    )
    for split in splits:
        data = load_npz_files(data_dir, snr_grid, echoes, split, config)
        expected = _expected_per_combo(config, split, num_combos)
        verify_snr_balance(data, snr_grid, echoes, expected_per_combo=expected)
        targets = normalize_targets(
            data["tau"],
            data["f_d"],
            tau_max=float(config["data"]["max_delay"]),
            fd_max=float(config["data"]["max_doppler"]),
        )
        dataset = build_tf_dataset(data, batch_size, config)
        logger.info(
            "split '%s' ready: %d samples, targets shape=%s, feature_mode=%s",
            split, int(data["x"].shape[0]), targets.shape, feature_mode,
        )
        del dataset
    logger.info("data loader completed")

if __name__ == "__main__":
    main()

