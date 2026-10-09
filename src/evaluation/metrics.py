
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MSE_NEGATIVE_TOL = 1e-9
_ENERGY_EPS = 1e-12

def _validate_logits_labels(logits: np.ndarray, labels: np.ndarray) -> Tuple[int, int]:
    
    if not isinstance(logits, np.ndarray):
        raise TypeError(f"logits must be a np.ndarray, got: {type(logits).__name__}")
    if not isinstance(labels, np.ndarray):
        raise TypeError(f"labels must be a np.ndarray, got: {type(labels).__name__}")

    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (B, M), got: {logits.ndim}D")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D (B,), got: {labels.ndim}D")

    B, M = logits.shape
    if labels.shape[0] != B:
        raise ValueError(
            f"shape mismatch: logits.shape[0]={B}, labels.shape[0]={labels.shape[0]}"
        )
    if B == 0:
        raise ValueError("no samples (B=0)")

    if not np.all(np.isfinite(logits)):
        raise ValueError("logits contains NaN/Inf")
    if not np.all(np.isfinite(labels)):
        raise ValueError("labels contains NaN/Inf")

    if labels.min() < 0 or labels.max() >= M:
        raise ValueError(
            f"labels outside [0, {M-1}]: min={labels.min()}, max={labels.max()}"
        )

    return B, M

def ber_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    
    B, M = _validate_logits_labels(logits, labels)

    pred = np.argmax(logits, axis=-1)

    n_errors = int(np.count_nonzero(pred != labels))
    ber = n_errors / B

    logger.debug("ber_from_logits: B=%d, M=%d, n_errors=%d, BER=%.6f", B, M, n_errors, ber)
    return float(ber)

def ber(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    if not isinstance(y_true, np.ndarray):
        raise TypeError(f"y_true must be a np.ndarray, got: {type(y_true).__name__}")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f"y_pred must be a np.ndarray, got: {type(y_pred).__name__}")

    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D, got: {y_true.ndim}D")
    if y_pred.ndim != 1:
        raise ValueError(f"y_pred must be 1D, got: {y_pred.ndim}D")

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}"
        )
    B = y_true.shape[0]
    if B == 0:
        raise ValueError("no samples (B=0)")

    if not np.all(np.isfinite(y_true)):
        raise ValueError("y_true contains NaN/Inf")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred contains NaN/Inf")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0/1 values")
    if not np.all(np.isin(y_pred, [0, 1])):
        raise ValueError("y_pred must contain only 0/1 values")

    n_errors = int(np.count_nonzero(y_true != y_pred))
    ber = n_errors / B

    logger.debug("ber: B=%d, n_errors=%d, BER=%.6f", B, n_errors, ber)
    return float(ber)

def mse_delay_doppler(
    tau_pred: np.ndarray,
    fd_pred: np.ndarray,
    tau_true: np.ndarray,
    fd_true: np.ndarray,
) -> Tuple[float, float]:
    
    tau_pred = np.asarray(tau_pred)
    fd_pred = np.asarray(fd_pred)
    tau_true = np.asarray(tau_true)
    fd_true = np.asarray(fd_true)

    if tau_pred.ndim != 1:
        raise ValueError(f"tau_pred must be 1D, got: {tau_pred.ndim}D")
    if fd_pred.ndim != 1:
        raise ValueError(f"fd_pred must be 1D, got: {fd_pred.ndim}D")
    if tau_true.ndim != 1:
        raise ValueError(f"tau_true must be 1D, got: {tau_true.ndim}D")
    if fd_true.ndim != 1:
        raise ValueError(f"fd_true must be 1D, got: {fd_true.ndim}D")

    if len(tau_pred) != len(fd_pred):
        raise ValueError(
            f"predictor shape mismatch: tau_pred {len(tau_pred)}, fd_pred {len(fd_pred)}"
        )
    if len(tau_pred) != len(tau_true):
        raise ValueError(
            f"pred/true shape mismatch: tau_pred {len(tau_pred)}, tau_true {len(tau_true)}"
        )
    if len(tau_pred) != len(fd_true):
        raise ValueError(
            f"pred/true shape mismatch: tau_pred {len(tau_pred)}, fd_true {len(fd_true)}"
        )

    B = len(tau_pred)
    if B == 0:
        raise ValueError("no samples (B=0)")

    if not np.all(np.isfinite(tau_pred)):
        raise ValueError("tau_pred contains NaN/Inf")
    if not np.all(np.isfinite(fd_pred)):
        raise ValueError("fd_pred contains NaN/Inf")
    if not np.all(np.isfinite(tau_true)):
        raise ValueError("tau_true contains NaN/Inf")
    if not np.all(np.isfinite(fd_true)):
        raise ValueError("fd_true contains NaN/Inf")

    err_tau = tau_pred - tau_true
    err_fd = fd_pred - fd_true

    mse_tau = float(np.mean(err_tau ** 2))
    mse_fd = float(np.mean(err_fd ** 2))

    if not np.isfinite(mse_tau) or not np.isfinite(mse_fd):
        raise RuntimeError("MSE is not finite (NaN/Inf)")
    if mse_tau < -_MSE_NEGATIVE_TOL or mse_fd < -_MSE_NEGATIVE_TOL:
        raise RuntimeError(f"MSE negativo: mse_tau={mse_tau:.6f}, mse_fd={mse_fd:.6f}")
    if mse_tau < 0.0:
        mse_tau = 0.0
    if mse_fd < 0.0:
        mse_fd = 0.0

    logger.debug(
        "mse_delay_doppler: B=%d, MSE_tau=%.6f, MSE_fd=%.6f",
        B, mse_tau, mse_fd,
    )
    return mse_tau, mse_fd

def rmse_from_mse(mse: float) -> float:
    
    if not isinstance(mse, (int, float)):
        raise TypeError(f"mse must be int or float, got: {type(mse).__name__}")
    if not np.isfinite(mse):
        raise ValueError(f"MSE must be finite, got: {mse}")
    if mse < -_MSE_NEGATIVE_TOL:
        raise ValueError(f"MSE negativo: {mse:.6f}")
    if mse < 0.0:
        mse = 0.0

    rmse = float(np.sqrt(mse))
    logger.debug("rmse_from_mse: MSE=%.6f -> RMSE=%.6f", mse, rmse)
    return rmse

def bit_error_count(logits: np.ndarray, labels: np.ndarray) -> Tuple[int, int]:
    
    B, M = _validate_logits_labels(logits, labels)

    pred = np.argmax(logits, axis=-1)
    n_errors = int(np.count_nonzero(pred != labels))
    n_total = B

    logger.debug("bit_error_count: B=%d, M=%d, n_errors=%d, n_total=%d", B, M, n_errors, n_total)
    return n_errors, n_total

def argmax_delay_from_profile(
    y_complex: np.ndarray,
    x_ref: np.ndarray,
    max_delay: int,
) -> np.ndarray:
    
    reference = np.asarray(x_ref, dtype=np.float64)
    received = np.asarray(y_complex)
    if reference.ndim != 2 or received.ndim != 2:
        raise ValueError(
            f"y_complex and x_ref must be 2D, got {received.shape} and {reference.shape}"
        )
    if reference.shape != received.shape:
        raise ValueError(
            f"y_complex and x_ref must share the shape, got {received.shape} "
            f"and {reference.shape}"
        )
    if int(max_delay) < 1 or int(max_delay) >= int(reference.shape[1]):
        raise ValueError(
            f"max_delay must be in [1, {int(reference.shape[1]) - 1}], got: {max_delay!r}"
        )
    if not (np.all(np.isfinite(reference)) and np.all(np.isfinite(received))):
        raise ValueError("y_complex/x_ref contain NaN/Inf")

    length = int(reference.shape[1])
    energy = np.sum(reference * reference, axis=1)
    energy = np.where(energy < _ENERGY_EPS, 1.0, energy)
    direct = np.sum(reference * received, axis=1) / energy
    residual = received - direct[:, None] * reference

    lag_best = np.zeros(received.shape[0], dtype=np.float64)
    peak = np.full(received.shape[0], -np.inf)
    for lag in range(1, int(max_delay) + 1):
        value = np.abs(
            np.sum(reference[:, : length - lag] * residual[:, lag:], axis=1)
        )
        better = value > peak
        lag_best = np.where(better, float(lag), lag_best)
        peak = np.where(better, value, peak)

    logger.debug("argmax_delay_from_profile: B=%d, max_delay=%d", received.shape[0], max_delay)
    return lag_best

def exact_delay_rate(
    tau_pred: np.ndarray,
    tau_true: np.ndarray,
    tolerance: float = 0.5,
) -> float:
    
    pred = np.asarray(tau_pred, dtype=np.float64)
    true = np.asarray(tau_true, dtype=np.float64)
    if pred.ndim != 1 or true.ndim != 1:
        raise ValueError(f"tau arrays must be 1D, got {pred.shape} and {true.shape}")
    if pred.shape != true.shape:
        raise ValueError(f"tau arrays must share the shape, got {pred.shape} and {true.shape}")
    if pred.size == 0:
        raise ValueError("no samples to score")
    if not (np.all(np.isfinite(pred)) and np.all(np.isfinite(true))):
        raise ValueError("tau arrays contain NaN/Inf")
    if not np.isfinite(float(tolerance)) or float(tolerance) < 0.0:
        raise ValueError(f"tolerance must be finite and >= 0, got: {tolerance!r}")

    rate = float(np.mean(np.abs(pred - true) <= float(tolerance)))
    logger.debug("exact_delay_rate: B=%d, tol=%.3f, rate=%.6f", pred.size, tolerance, rate)
    return rate

