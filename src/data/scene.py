
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from src.data.channel_config import (
    as_float,
    as_float_list,
    as_str,
    section,
)

_C_LIGHT_M_S = 299_792_458.0
_POWER_EPS = 1e-12

DEFAULT_ECHO_GENERATOR = "log_uniform"

SCENE_GENERATORS: Tuple[str, ...] = ("geometry", "log_uniform")
CHANNEL_MODES: Tuple[str, ...] = ("monostatic", "bistatic")

def channel_section(config: Dict[str, Any]) -> Dict[str, Any]:
    channel = config.get("channel")
    if channel is None:
        return {}
    if not isinstance(channel, dict):
        raise ValueError("'channel' section missing or not a mapping")
    return channel

def echo_generator(config: Dict[str, Any]) -> str:
    channel = channel_section(config)
    if "generator" not in channel:
        return DEFAULT_ECHO_GENERATOR
    value = str(as_str(channel, "generator", "channel")).strip().lower()
    if value not in SCENE_GENERATORS:
        raise ValueError(
            f"channel.generator must be one of {sorted(SCENE_GENERATORS)}, got: {value!r}"
        )
    return value

def channel_mode(config: Dict[str, Any]) -> str:
    channel = channel_section(config)
    value = str(channel.get("mode", "monostatic")).strip().lower()
    if value not in CHANNEL_MODES:
        raise ValueError(
            f"channel.mode must be one of {sorted(CHANNEL_MODES)}, got: {value!r}"
        )
    return value

def require_monostatic(config: Dict[str, Any]) -> None:
    mode = channel_mode(config)
    if mode != "monostatic":
        raise ValueError(
            "the sensing path requires channel.mode='monostatic' (the seed of the "
            f"burst just transmitted); got channel.mode={mode!r}"
        )

def scene_section(config: Dict[str, Any]) -> Dict[str, Any]:
    return section(channel_section(config), "scene", "channel")

def _pair(section_cfg: Dict[str, Any], key: str, path: str) -> Tuple[float, float]:
    values = as_float_list(section_cfg, key, path)
    if len(values) != 2:
        raise ValueError(f"{path}.{key} must be a [min, max] pair, got: {values!r}")
    lo, hi = float(values[0]), float(values[1])
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"{path}.{key} must be finite, got: {values!r}")
    if not (0.0 < lo <= hi):
        raise ValueError(f"{path}.{key} must satisfy 0 < min <= max, got: {values!r}")
    return lo, hi

def _positive(section_cfg: Dict[str, Any], key: str, path: str) -> float:
    value = as_float(section_cfg, key, path)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{path}.{key} must be finite and > 0, got: {value!r}")
    return value

def _non_negative(section_cfg: Dict[str, Any], key: str, path: str) -> float:
    value = as_float(section_cfg, key, path)
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{path}.{key} must be finite and >= 0, got: {value!r}")
    return value

def delay_samples_from_distance(distance_m: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    fs_hz = float(config["data"]["fs_hz"])
    if not (math.isfinite(fs_hz) and fs_hz > 0.0):
        raise ValueError(f"data.fs_hz must be > 0, got: {fs_hz!r}")
    return 2.0 * np.asarray(distance_m, dtype=np.float64) * fs_hz / _C_LIGHT_M_S

def doppler_from_radial_speed(speed_mps: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    fc_hz = float(config["data"]["fc_hz"])
    if not (math.isfinite(fc_hz) and fc_hz > 0.0):
        raise ValueError(f"data.fc_hz must be > 0, got: {fc_hz!r}")
    wavelength = _C_LIGHT_M_S / fc_hz
    return 2.0 * np.abs(np.asarray(speed_mps, dtype=np.float64)) / wavelength

def _normalized_doppler(raw_doppler_hz: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    fs_hz = float(config["data"]["fs_hz"])
    return np.asarray(raw_doppler_hz, dtype=np.float64) / fs_hz

def amplitude_from_distance(
    distance_m: np.ndarray, reflectivity: float, reference_range_m: float
) -> np.ndarray:
    if not (math.isfinite(reflectivity) and 0.0 < reflectivity <= 1.0):
        raise ValueError(f"reflectivity must be in (0, 1], got: {reflectivity!r}")
    if not (math.isfinite(reference_range_m) and reference_range_m > 0.0):
        raise ValueError(f"reference_range_m must be > 0, got: {reference_range_m!r}")
    distance = np.maximum(np.asarray(distance_m, dtype=np.float64), _POWER_EPS)
    return reflectivity * (reference_range_m / distance) ** 2

def _scene_params(config: Dict[str, Any]) -> Dict[str, Any]:
    scene = scene_section(config)
    path = "channel.scene"
    params: Dict[str, Any] = {
        "altitude": _positive(scene, "node_altitude_m", path),
        "obstacle_range": _pair(scene, "obstacle_distance_m", path),
        "obstacle_reflectivity": _positive(scene, "obstacle_reflectivity", path),
        "scatterer_reflectivity": _positive(scene, "scatterer_reflectivity", path),
        "scatterer_range": _pair(scene, "scatterer_distance_m", path),
        "speed_range": _pair(scene, "radial_speed_mps", path),
        "speed_outlier": _non_negative(scene, "speed_outlier_mps", path),
        "outlier_probability": _non_negative(scene, "speed_outlier_probability", path),
        "reference_range_m": _positive(scene, "reference_range_m", path),
        "echo_power_fraction": _positive(scene, "echo_power_fraction", path),
    }
    if params["outlier_probability"] > 1.0:
        raise ValueError(
            f"{path}.speed_outlier_probability must lie in [0, 1], "
            f"got: {params['outlier_probability']!r}"
        )
    if not (0.0 < params["echo_power_fraction"] < 1.0):
        raise ValueError(
            f"{path}.echo_power_fraction must lie in (0, 1), "
            f"got: {params['echo_power_fraction']!r}"
        )
    return params

def validate_scene(config: Dict[str, Any]) -> None:
    _scene_params(config)

def sample_scene_taps(
    n: int,
    rng: np.random.Generator,
    config: Dict[str, Any],
    max_delay: int,
    n_scatterers: int,
) -> Dict[str, np.ndarray]:
    params = _scene_params(config)
    if isinstance(n_scatterers, bool) or int(n_scatterers) < 0:
        raise ValueError(f"n_scatterers must be an int >= 0, got: {n_scatterers!r}")
    n_scatterers = int(n_scatterers)
    if int(max_delay) < 1:
        raise ValueError(f"max_delay must be >= 1, got: {max_delay!r}")
    altitude = params["altitude"]
    obstacle_lo, obstacle_hi = params["obstacle_range"]
    obstacle_reflectivity = params["obstacle_reflectivity"]
    reference_range_m = params["reference_range_m"]
    speed_lo, speed_hi = params["speed_range"]
    speed_outlier = params["speed_outlier"]
    outlier_prob = params["outlier_probability"]
    echo_power_fraction = params["echo_power_fraction"]

    obstacle_distance = rng.uniform(obstacle_lo, obstacle_hi, size=n)
    obstacle_speed = rng.uniform(speed_lo, speed_hi, size=n)
    if outlier_prob > 0.0 and speed_outlier > 0.0:
        outlier = rng.random(n) < outlier_prob
        obstacle_speed = np.where(outlier, speed_outlier, obstacle_speed)

    amplitude = [
        amplitude_from_distance(obstacle_distance, obstacle_reflectivity, reference_range_m)[:, None]
    ]
    taus = [
        np.clip(
            np.rint(delay_samples_from_distance(obstacle_distance, config)).astype(np.int64),
            1,
            int(max_delay),
        )[:, None]
    ]
    dopplers = [
        _normalized_doppler(doppler_from_radial_speed(obstacle_speed, config), config)[:, None]
    ]

    if n_scatterers > 0:
        scatterer_lo, scatterer_hi = params["scatterer_range"]
        scatterer_reflectivity = params["scatterer_reflectivity"]
        scatterer_distance = rng.uniform(scatterer_lo, scatterer_hi, size=(n, int(n_scatterers)))
        scatterer_speed = rng.uniform(speed_lo, speed_hi, size=(n, int(n_scatterers)))
        amplitude.append(
            amplitude_from_distance(scatterer_distance, scatterer_reflectivity, reference_range_m)
        )
        taus.append(
            np.clip(
                np.rint(delay_samples_from_distance(scatterer_distance, config)).astype(np.int64),
                1,
                int(max_delay),
            )
        )
        dopplers.append(
            _normalized_doppler(doppler_from_radial_speed(scatterer_speed, config), config)
        )

    amplitude_arr = np.concatenate(amplitude, axis=1)
    tau_arr = np.concatenate(taus, axis=1)
    doppler_arr = np.concatenate(dopplers, axis=1)
    energy = np.sqrt(np.sum(amplitude_arr ** 2, axis=1, keepdims=True))
    if not np.all(np.isfinite(energy)) or np.any(energy <= _POWER_EPS):
        raise RuntimeError("scene geometry produced a degenerate amplitude budget")
    amplitude_arr = math.sqrt(echo_power_fraction) * amplitude_arr / energy
    if not np.all(np.isfinite(amplitude_arr)):
        raise RuntimeError("scene amplitudes are not finite after normalization")

    return {
        "taus": tau_arr.astype(np.float64),
        "dopplers": doppler_arr.astype(np.float64),
        "alphas": amplitude_arr.astype(np.float64),
        "echo_power": (amplitude_arr[:, 0] ** 2).astype(np.float64),
        "k_geometric": 1,
        "node_altitude_m": np.full(n, altitude, dtype=np.float64),
        "obstacle_distance_m": obstacle_distance.astype(np.float64),
        "obstacle_alpha": amplitude_arr[:, 0].astype(np.float64),
        "echo_power_fraction": float(echo_power_fraction),
    }

