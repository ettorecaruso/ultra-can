
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

_Z_SCORE_95 = 1.959963984540054
_ENERGY_EPS = 1e-12

def _as_1d(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr

def profile_peak(profile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    prof = np.asarray(profile, dtype=np.float64)
    if prof.ndim != 2:
        raise ValueError(f"profile must be 2D (N, max_delay), got {prof.shape}")
    if prof.shape[1] < 1:
        raise ValueError("profile must expose at least one delay bin")
    magnitude = np.abs(prof)
    peak_bin = np.argmax(magnitude, axis=1)
    peak_amplitude = magnitude[np.arange(magnitude.shape[0]), peak_bin]
    peak_lag = (peak_bin + 1).astype(np.float64)
    return peak_lag, peak_amplitude

def abstention_mask(
    tau_peak: np.ndarray,
    peak_amplitude: np.ndarray,
    target_amplitude: np.ndarray,
    max_delay: int,
    gamma: float,
) -> np.ndarray:
    tau = _as_1d(tau_peak, "tau_peak")
    peak = _as_1d(peak_amplitude, "peak_amplitude")
    target = _as_1d(target_amplitude, "target_amplitude")
    if not (tau.shape == peak.shape == target.shape):
        raise ValueError(
            "tau_peak/peak_amplitude/target_amplitude must share a shape, got "
            f"{tau.shape}, {peak.shape}, {target.shape}"
        )
    if int(max_delay) < 1:
        raise ValueError(f"max_delay must be >= 1, got: {max_delay!r}")
    if not (math.isfinite(float(gamma)) and float(gamma) >= 1.0):
        raise ValueError(f"gamma must be finite and >= 1, got: {gamma!r}")
    outside = (tau < 1.0) | (tau > float(max_delay))
    dominated = peak > float(gamma) * target
    return outside | dominated

def silence_fraction(mask: np.ndarray) -> float:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 1:
        raise ValueError(f"mask must be 1D, got {arr.shape}")
    if arr.size == 0:
        raise ValueError("mask is empty")
    return float(np.mean(arr))

def scored_mae(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    pred_arr = _as_1d(pred, "pred")
    true_arr = _as_1d(true, "true")
    mask_arr = np.asarray(mask, dtype=bool)
    if not (pred_arr.shape == true_arr.shape == mask_arr.shape):
        raise ValueError("pred/true/mask must share a shape")
    scored = ~mask_arr
    if not np.any(scored):
        return float("nan")
    return float(np.mean(np.abs(pred_arr[scored] - true_arr[scored])))

def scored_median_ae(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    pred_arr = _as_1d(pred, "pred")
    true_arr = _as_1d(true, "true")
    mask_arr = np.asarray(mask, dtype=bool)
    if not (pred_arr.shape == true_arr.shape == mask_arr.shape):
        raise ValueError("pred/true/mask must share a shape")
    scored = ~mask_arr
    if not np.any(scored):
        return float("nan")
    return float(np.median(np.abs(pred_arr[scored] - true_arr[scored])))

def samples_to_meters(samples: np.ndarray, config: dict) -> np.ndarray:
    fs_hz = float(config["data"]["fs_hz"])
    if not (math.isfinite(fs_hz) and fs_hz > 0.0):
        raise ValueError(f"data.fs_hz must be > 0, got: {fs_hz!r}")
    speed = 299_792_458.0
    return np.asarray(samples, dtype=np.float64) * speed / (2.0 * fs_hz)

def oracle_hit_rate(
    tau_peak: np.ndarray, tau_true: np.ndarray, tolerance: float = 0.5
) -> float:
    peak = _as_1d(tau_peak, "tau_peak")
    true = _as_1d(tau_true, "tau_true")
    if peak.shape != true.shape:
        raise ValueError("tau_peak and tau_true must share a shape")
    if not (math.isfinite(float(tolerance)) and float(tolerance) >= 0.0):
        raise ValueError(f"tolerance must be finite and >= 0, got: {tolerance!r}")
    return float(np.mean(np.abs(peak - true) <= float(tolerance)))

def wilson_interval(
    n_errors: int, n_total: int, z: float = _Z_SCORE_95
) -> Tuple[float, float]:
    if isinstance(n_total, bool) or int(n_total) <= 0:
        raise ValueError(f"n_total must be an int > 0, got: {n_total!r}")
    if isinstance(n_errors, bool) or int(n_errors) < 0 or int(n_errors) > int(n_total):
        raise ValueError(f"n_errors must lie in [0, {int(n_total)}], got: {n_errors!r}")
    if not (math.isfinite(float(z)) and float(z) > 0.0):
        raise ValueError(f"z must be finite and > 0, got: {z!r}")
    total = float(int(n_total))
    p = float(int(n_errors)) / total
    denom = 1.0 + (float(z) ** 2) / total
    centre = (p + (float(z) ** 2) / (2.0 * total)) / denom
    half = (
        float(z)
        * math.sqrt(p * (1.0 - p) / total + (float(z) ** 2) / (4.0 * total ** 2))
        / denom
    )
    return max(0.0, centre - half), min(1.0, centre + half)

def gap_resolved(
    value_a: float, value_b: float, relative_threshold: float = 0.10
) -> bool:
    if not (math.isfinite(float(value_a)) and math.isfinite(float(value_b))):
        raise ValueError("value_a and value_b must be finite")
    if not (math.isfinite(float(relative_threshold)) and float(relative_threshold) >= 0.0):
        raise ValueError(
            f"relative_threshold must be finite and >= 0, got: {relative_threshold!r}"
        )
    reference = max(abs(float(value_a)), abs(float(value_b)))
    if reference <= _ENERGY_EPS:
        return False
    return abs(float(value_a) - float(value_b)) / reference > float(relative_threshold)

    return outside | dominated
