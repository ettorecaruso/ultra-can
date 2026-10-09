
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config_loader import (
    DEFAULT_BASE_CONFIG_PATH,
    load_config,
)
from src.utils.logger import log_config_summary, setup_logging

logger = logging.getLogger(__name__)

_DETECTOR_NAMES: Tuple[str, ...] = ("dcsk", "matched_filter", "energy_detector")
_DCSK_REQUIRED_KEYS: Tuple[str, ...] = ("correlation_length", "threshold")
_ENERGY_EPS = 1e-12

def _iter_config_leaves(config: Any, prefix: str = "config") -> Iterator[Tuple[str, Any]]:
    
    if isinstance(config, dict):
        for key, value in config.items():
            yield from _iter_config_leaves(value, f"{prefix}.{key}")
    elif isinstance(config, (list, tuple)):
        for index, value in enumerate(config):
            yield from _iter_config_leaves(value, f"{prefix}[{index}]")
    else:
        yield prefix, config

def _assert_config_finite(config: Any) -> None:
    
    for path, value in _iter_config_leaves(config):
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise ValueError(f"non-finite value in the config: {path} = {value!r}")

def _as_positive_int(value: Any, label: str) -> int:
    
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a positive integer, got: {value!r}")
    as_int = int(value)
    if float(value) != float(as_int) or as_int <= 0:
        raise ValueError(f"{label} must be a positive integer, got: {value!r}")
    return as_int

def _split_meta_symbol(y: np.ndarray, beta: int) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y must be a np.ndarray, got: {type(y).__name__}")
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2:
        raise ValueError(f"y must have shape (B, 2*beta) or (2*beta,), got: {y.shape}")
    if beta < 1:
        raise ValueError(f"beta must be >= 1, got: {beta!r}")
    if y.shape[1] != 2 * beta:
        raise ValueError(
            f"last dimension of y must be 2*beta={2 * beta}, "
            f"got: {y.shape[1]} (shape {y.shape})"
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN/Inf")
    ref = y[:, :beta]
    data = y[:, beta:]
    return ref, data

def dcsk_correlator_demodulate(
    y: np.ndarray,
    ref: Optional[np.ndarray] = None,
    threshold: float = 0.0,
) -> np.ndarray:
    
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y must be a np.ndarray, got: {type(y).__name__}")
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2:
        raise ValueError(f"y must have shape (B, 2*beta) or (2*beta,), got: {y.shape}")
    length = y.shape[1]
    if length < 2 or length % 2 != 0:
        raise ValueError(
            f"y must have even length >= 2 (meta-symbol [ref, data]), got: {length}"
        )
    beta = length // 2
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN/Inf")
    if not math.isfinite(float(threshold)):
        raise ValueError(f"threshold must be finite, got: {threshold!r}")

    ref_internal, data = _split_meta_symbol(y, beta)
    if ref is not None:
        if not isinstance(ref, np.ndarray):
            raise TypeError(f"ref must be a np.ndarray, got: {type(ref).__name__}")
        if ref.ndim == 1:
            ref = ref.reshape(1, -1)
        if ref.ndim != 2 or ref.shape[1] != beta:
            raise ValueError(
                f"ref must have shape (beta,) or (1, beta) with beta={beta}, got: {ref.shape}"
            )
        if ref.shape[0] not in (1, y.shape[0]):
            raise ValueError(
                f"incompatible ref batch: {ref.shape[0]} (expected 1 or {y.shape[0]})"
            )
        if not np.all(np.isfinite(ref)):
            raise ValueError("ref contains NaN/Inf")
        ref_internal = ref

    z = np.real(np.sum(np.conj(ref_internal) * data, axis=-1))
    if not np.all(np.isfinite(z)):
        raise RuntimeError(
            "DCSK correlation statistic not finite (NaN/Inf): "
            ""
        )
    bits = (z > threshold).astype(np.int8)
    logger.debug(
        "dcsk_correlator_demodulate: batch=%d, beta=%d, z in [%.4e, %.4e]",
        y.shape[0], beta, float(np.min(z)), float(np.max(z)),
    )
    return bits

def matched_filter_demodulate(
    y: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y must be a np.ndarray, got: {type(y).__name__}")
    if not isinstance(template, np.ndarray):
        raise TypeError(
            f"template must be a np.ndarray, got: {type(template).__name__}"
        )
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2:
        raise ValueError(f"y must have shape (B, beta) or (beta,), got: {y.shape}")
    if template.ndim != 1:
        raise ValueError(f"template must be 1D (beta,), got: {template.shape}")
    if y.shape[1] != template.shape[0]:
        raise ValueError(
            f"template ({template.shape[0]} samples) not aligned with the window "
            f"y ({y.shape[1]} samples)"
        )
    if y.shape[1] == 0:
        raise ValueError("y is empty (beta=0): no samples to correlate")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN/Inf")
    if not np.all(np.isfinite(template)):
        raise ValueError("template contains NaN/Inf")
    if not math.isfinite(float(threshold)):
        raise ValueError(f"threshold must be finite, got: {threshold!r}")

    template_energy = float(np.sum(np.abs(template) ** 2))
    if template_energy <= _ENERGY_EPS:
        logger.warning(
            "matched_filter_demodulate: degenerate template (energy %.3e <= %.1e) - "
            "possible bias between the two maps (logistic/Bernoulli)",
            template_energy, _ENERGY_EPS,
        )

    z = np.real(np.sum(np.conj(template) * y, axis=-1))
    if not np.all(np.isfinite(z)):
        raise RuntimeError(
            "matched filter statistic not finite (NaN/Inf): "
        )
    bits = (z > threshold).astype(np.int8)
    logger.debug(
        "matched_filter_demodulate: batch=%d, template_energy=%.4e, z in [%.4e, %.4e]",
        y.shape[0], template_energy, float(np.min(z)), float(np.max(z)),
    )
    return bits

def energy_detector_demodulate(
    y: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y must be a np.ndarray, got: {type(y).__name__}")
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2:
        raise ValueError(f"y must have shape (B, N) or (N,), got: {y.shape}")
    if y.shape[1] == 0:
        raise ValueError("y is empty (no samples)")
    if y.shape[0] == 0:
        raise ValueError("y is empty (batch without samples)")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN/Inf")
    if not math.isfinite(float(threshold)):
        raise ValueError(f"threshold must be finite, got: {threshold!r}")

    energy = np.sum(np.abs(y) ** 2, axis=-1)
    if not np.all(np.isfinite(energy)) or np.any(energy < 0.0):
        raise RuntimeError(
            "invalid energy (NaN/Inf or negative): STOP (Sec. 1.2)"
        )
    bits = (energy > threshold).astype(np.int8)
    logger.debug(
        "energy_detector_demodulate: batch=%d, energy in [%.4e, %.4e]",
        y.shape[0], float(np.min(energy)), float(np.max(energy)),
    )
    return bits

def _ber(bits_hat: np.ndarray, bits_true: np.ndarray) -> float:
    
    bits_hat_arr = np.asarray(bits_hat)
    bits_true_arr = np.asarray(bits_true)
    if bits_hat_arr.ndim != 1 or bits_true_arr.ndim != 1:
        raise ValueError("bits_hat and bits_true must be 1D")
    if bits_hat_arr.shape != bits_true_arr.shape:
        raise ValueError(
            f"shape not aligned: bits_hat {bits_hat_arr.shape} != "
            f"bits_true {bits_true_arr.shape}"
        )
    n_bits = int(bits_hat_arr.shape[0])
    if n_bits == 0:
        raise ValueError("n_bits = 0: BER undefined")

    try:
        from src.evaluation.metrics import ber as _metrics_ber
    except (ImportError, AttributeError):
        _metrics_ber = None
    if _metrics_ber is not None:
        return float(_metrics_ber(bits_true_arr, bits_hat_arr))

    n_errors = int(np.count_nonzero(bits_hat_arr != bits_true_arr))
    return float(n_errors / n_bits)

def _validate_correlator_config(
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], int, float]:
    
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got: {type(config).__name__}")
    baselines_cfg = config.get("baselines")
    if not isinstance(baselines_cfg, dict):
        raise ValueError("'baselines' section missing or not a dict in config")
    corr_cfg = baselines_cfg.get("dcsk_correlator")
    if not isinstance(corr_cfg, dict):
        raise ValueError("'baselines.dcsk_correlator' section missing or not a dict")
    missing = [key for key in _DCSK_REQUIRED_KEYS if key not in corr_cfg]
    if missing:
        raise ValueError(f"missing keys in baselines.dcsk_correlator: {missing}")
    _assert_config_finite(corr_cfg)

    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("'data' section missing or not a dict in config")
    if "sequence_length" not in data_cfg:
        raise ValueError("missing 'data.sequence_length' key in config")
    sequence_length = _as_positive_int(
        data_cfg["sequence_length"], "data.sequence_length"
    )

    beta = _as_positive_int(
        corr_cfg["correlation_length"],
        "baselines.dcsk_correlator.correlation_length",
    )
    if 2 * beta > sequence_length:
        raise ValueError(
            f"2*correlation_length ({2 * beta}) > sequence_length ({sequence_length}): "
            "the DCSK meta-symbol cannot be represented in the received frame"
        )
    threshold = float(corr_cfg["threshold"])
    if not math.isfinite(threshold):
        raise ValueError(
            "baselines.dcsk_correlator.threshold must be finite, "
            f"got: {threshold!r}"
        )
    logger.debug(
        "valid correlator config: beta=%d, threshold=%g, 2*beta=%d <= N_seq=%d",
        beta, threshold, 2 * beta, sequence_length,
    )
    return corr_cfg, beta, threshold

def evaluate_classical(
    y: np.ndarray,
    bits_true: np.ndarray,
    detector: str,
    config: Dict[str, Any],
    ref: Optional[np.ndarray] = None,
    template: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    
    if detector not in _DETECTOR_NAMES:
        raise ValueError(
            f"invalid detector: {detector!r} (expected: {list(_DETECTOR_NAMES)})"
        )
    _, beta, threshold = _validate_correlator_config(config)

    if not isinstance(y, np.ndarray):
        raise TypeError(f"y must be a np.ndarray, got: {type(y).__name__}")
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2:
        raise ValueError(f"y must have shape (B, N) or (N,), got: {y.shape}")

    bits_true_arr = np.asarray(bits_true)
    if bits_true_arr.ndim != 1:
        raise ValueError(f"bits_true must be 1D, got: {bits_true_arr.shape}")
    if bits_true_arr.shape[0] != y.shape[0]:
        raise ValueError(
            f"bits_true ({bits_true_arr.shape[0]}) not aligned with the batch "
            f"y ({y.shape[0]})"
        )
    if not np.all(np.isin(bits_true_arr, (0, 1))):
        raise ValueError("bits_true must contain only 0/1 values")

    if detector == "dcsk" and y.shape[1] != 2 * beta:
        raise ValueError(
            f"frame y ({y.shape[1]} samples) inconsistent with correlation_length "
            f"(2*beta={2 * beta}): expected [ref, data] frames of 2*correlation_length"
        )
    if detector == "matched_filter" and y.shape[1] != beta:
        raise ValueError(
            f"frame y ({y.shape[1]} samples) inconsistent with correlation_length "
            f"(beta={beta}): expected windows of correlation_length samples"
        )

    if detector == "dcsk":
        bits_hat = dcsk_correlator_demodulate(y, ref=ref, threshold=threshold)
    elif detector == "matched_filter":
        if template is None:
            raise ValueError(
                "detector 'matched_filter' requires 'template' "
                "(reference chaotic sequence)"
            )
        bits_hat = matched_filter_demodulate(y, template, threshold=threshold)
    else:
        bits_hat = energy_detector_demodulate(y, threshold=threshold)

    n_bits = int(bits_true_arr.shape[0])
    n_errors = int(np.count_nonzero(bits_hat != bits_true_arr))
    ber = _ber(bits_hat, bits_true_arr)
    logger.info(
        "evaluate_classical: detector=%s, n_bits=%d, n_errors=%d, BER=%.6f",
        detector, n_bits, n_errors, ber,
    )
    return {"ber": ber, "n_errors": n_errors, "n_bits": n_bits}

def count_trainable_params(*_ignored: Any, **_ignored_kw: Any) -> int:
    
    return 0

def main(argv: Optional[Sequence[str]] = None) -> None:
    
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test for the classical receivers "
            "(DCSK / matched filter / energy detector)"
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="path to the experiment config (YAML) used to read the parameters",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(
        config_path=config_path, base_config_path=DEFAULT_BASE_CONFIG_PATH
    )
    general_cfg = config.get("general")
    if not isinstance(general_cfg, dict):
        raise ValueError("'general' section missing or not a dict in config")
    experiment_name = str(general_cfg.get("experiment_name", "ultra_can_isac"))
    log_dir = _REPO_ROOT / "results" / experiment_name / "logs"
    setup_logging(
        log_dir=log_dir,
        level=str(general_cfg.get("log_level", "INFO")),
        experiment_name=experiment_name,
    )
    log_config_summary(config, logger)

    corr_cfg, beta, threshold = _validate_correlator_config(config)
    logger.info(
        "smoke DCSK: correlation_length=%d (beta), threshold=%g, 2*beta=%d",
        beta, threshold, 2 * beta,
    )

    from src.data.dataset_generator import generate_chaotic_sequence

    data_cfg = config["data"]
    seed = int(general_cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    template_seed = int(rng.integers(1, 2 ** 31 - 1))
    ref_template = generate_chaotic_sequence(
        str(data_cfg["map_type"]),
        float(data_cfg["map_param"]),
        template_seed,
        beta,
    )
    if ref_template.shape != (beta,) or not np.all(np.isfinite(ref_template)):
        raise RuntimeError(
            f"invalid reference template: shape {ref_template.shape}, "
            "expected (beta,) with finite values"
        )
    y0 = np.concatenate([ref_template, -ref_template])
    y1 = np.concatenate([ref_template, ref_template])
    y_smoke = np.stack([y0, y1])
    bits_true = np.array([0, 1], dtype=np.int8)

    bits_dcsk = dcsk_correlator_demodulate(y_smoke, threshold=threshold)
    ber_dcsk = _ber(bits_dcsk, bits_true)
    logger.info(
        "smoke DCSK: bit_hat=%s expected=%s, BER=%.6f",
        bits_dcsk.tolist(), bits_true.tolist(), ber_dcsk,
    )
    if ber_dcsk > 0.0:
        logger.warning(
            "smoke DCSK: BER=%g > 0 on clean frames - possible "
            "implementation error", ber_dcsk,
        )

    bits_mf = matched_filter_demodulate(
        y_smoke[:, beta:], ref_template, threshold=threshold
    )
    ber_mf = _ber(bits_mf, bits_true)
    logger.info(
        "smoke matched filter: bit_hat=%s expected=%s, BER=%.6f",
        bits_mf.tolist(), bits_true.tolist(), ber_mf,
    )
    if ber_mf > 0.0:
        logger.warning(
            "smoke matched filter: BER=%g > 0 on clean frames - possible "
            "di implementation error", ber_mf,
        )

    bits_energy = energy_detector_demodulate(y_smoke, threshold=threshold)
    ber_energy = _ber(bits_energy, bits_true)
    logger.info(
        "smoke energy detector: bit_hat=%s expected=%s, BER=%.6f "
        "(expected floor ~0.5: 'classical receivers fail')",
        bits_energy.tolist(), bits_true.tolist(), ber_energy,
    )

    params = count_trainable_params()
    if params != 0:
        raise RuntimeError(
            f"count_trainable_params must be 0 for a classical receiver, "
            f"got: {params}"
        )
    logger.info(
        "smoke completed: count_trainable_params=%d, config=%s", params, corr_cfg
    )

if __name__ == "__main__":
    main()

