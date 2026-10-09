
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.data.channel_config import as_float, as_int, as_float_or_none, data_section, section
from src.data.channel_models import TapGeometry, add_taps

SPEED_OF_LIGHT = 299792458.0
MAX_POWER_FRACTION = 1.0

@dataclass(frozen=True)
class TwoRayParameters:
    h_tx_m: float
    h_rx_m: float
    d_m: float
    reflection_coeff: float
    velocity_mps: float
    n_cisoids: int
    min_delay_samples: float
    power_fraction: Optional[float]

def parameters(config: Mapping[str, Any]) -> TwoRayParameters:
    channel = section(config, "channel", "config")
    two_ray = section(channel, "two_ray", "channel")
    params = TwoRayParameters(
        h_tx_m=as_float(two_ray, "h_tx_m", "channel.two_ray"),
        h_rx_m=as_float(two_ray, "h_rx_m", "channel.two_ray"),
        d_m=as_float(two_ray, "d_m", "channel.two_ray"),
        reflection_coeff=as_float(two_ray, "reflection_coeff", "channel.two_ray"),
        velocity_mps=as_float(two_ray, "velocity_mps", "channel.two_ray"),
        n_cisoids=as_int(two_ray, "n_cisoids", "channel.two_ray"),
        min_delay_samples=as_float(two_ray, "min_delay_samples", "channel.two_ray"),
        power_fraction=as_float_or_none(two_ray, "power_fraction", "channel.two_ray"),
    )
    for name in ("h_tx_m", "h_rx_m", "d_m"):
        value = getattr(params, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"channel.two_ray.{name} must be finite and > 0, got {value!r}")
    if not 0.0 <= params.reflection_coeff <= 1.0:
        raise ValueError(
            "channel.two_ray.reflection_coeff must lie in [0, 1], got "
            f"{params.reflection_coeff}"
        )
    if not math.isfinite(params.velocity_mps) or params.velocity_mps < 0.0:
        raise ValueError(
            f"channel.two_ray.velocity_mps must be finite and >= 0, got {params.velocity_mps}"
        )
    if params.n_cisoids < 1:
        raise ValueError(f"channel.two_ray.n_cisoids must be >= 1, got {params.n_cisoids}")
    if not math.isfinite(params.min_delay_samples) or params.min_delay_samples <= 0.0:
        raise ValueError(
            "channel.two_ray.min_delay_samples must be finite and > 0, got "
            f"{params.min_delay_samples}"
        )
    share = power_fraction(params)
    if not 0.0 < share < MAX_POWER_FRACTION:
        raise ValueError(
            f"channel.two_ray power share must lie in (0, 1), got {share}"
        )
    return params

def derived_power_fraction(params: TwoRayParameters) -> float:
    delta = 2.0 * params.h_tx_m * params.h_rx_m / params.d_m
    return params.reflection_coeff ** 2 * (params.d_m / (params.d_m + delta)) ** 2

def power_fraction(params: TwoRayParameters) -> float:
    if params.power_fraction is not None:
        return float(params.power_fraction)
    return derived_power_fraction(params)

def excess_delay_seconds(params: TwoRayParameters) -> float:
    return 2.0 * params.h_tx_m * params.h_rx_m / (params.d_m * SPEED_OF_LIGHT)

def doppler_hz(params: TwoRayParameters, fc_hz: float) -> float:
    return params.velocity_mps * float(fc_hz) / SPEED_OF_LIGHT

def slot_seconds(config: Mapping[str, Any]) -> float:
    hopping = config.get("frequency_hopping")
    if not isinstance(hopping, dict):
        raise ValueError("'frequency_hopping' section missing or not a mapping")
    burst_us = float(hopping["burst_duration_us"])
    dwell = max(1, int(hopping["dwell_bursts"]))
    return burst_us * 1e-6 * dwell

def jakes_gain(
    f_d_hz: float,
    t_seconds: np.ndarray,
    rng: np.random.Generator,
    n_cisoids: int,
) -> np.ndarray:
    if int(n_cisoids) < 1:
        raise ValueError(f"n_cisoids must be >= 1, got {n_cisoids!r}")
    if not math.isfinite(float(f_d_hz)) or float(f_d_hz) < 0.0:
        raise ValueError(f"f_d_hz must be finite and >= 0, got {f_d_hz!r}")
    t = np.asarray(t_seconds, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError(f"t_seconds must be 1D, got {t.shape}")
    n = int(n_cisoids)
    alpha = rng.uniform(0.0, 2.0 * math.pi, size=n)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=n)
    freq = float(f_d_hz) * np.cos(alpha)
    rotation = np.exp(1j * (2.0 * math.pi * np.outer(t, freq) + phase[None, :]))
    return np.sum(rotation, axis=1) / math.sqrt(float(n))

def report(config: Mapping[str, Any]) -> Dict[str, Any]:
    params = parameters(config)
    data = data_section(config)
    fc_hz = float(data["fc_hz"])
    fs_hz = float(data["fs_hz"])
    delay_s = excess_delay_seconds(params)
    return {
        "excess_delay_ns": delay_s * 1e9,
        "excess_delay_samples": delay_s * fs_hz,
        "doppler_hz": doppler_hz(params, fc_hz),
        "power_fraction": power_fraction(params),
        "power_fraction_derived": derived_power_fraction(params),
        "slot_seconds": slot_seconds(config),
        "min_delay_samples": params.min_delay_samples,
    }

def validate(config: Mapping[str, Any]) -> None:
    params = parameters(config)
    data = data_section(config)
    tau_samples = excess_delay_seconds(params) * float(data["fs_hz"])
    if tau_samples < params.min_delay_samples:
        raise ValueError(
            f"two_ray excess delay of {tau_samples:.3f} samples is below "
            f"channel.two_ray.min_delay_samples={params.min_delay_samples}: the "
            "configured geometry does not produce a resolvable reflection"
        )
    if tau_samples > float(data["max_delay"]):
        raise ValueError(
            f"two_ray excess delay of {tau_samples:.3f} samples exceeds "
            f"data.max_delay={int(data['max_delay'])}"
        )
    slot_seconds(config)

def overlay(
    config: Mapping[str, Any],
    geometry: TapGeometry,
    rng: np.random.Generator,
    slot_ids: Any = None,
    hop_channels: Any = None,
) -> TapGeometry:
    validate(config)
    params = parameters(config)
    data = data_section(config)
    share = power_fraction(params)
    tau_samples = excess_delay_seconds(params) * float(data["fs_hz"])
    n = geometry.n_symbols
    f_d = doppler_hz(params, float(data["fc_hz"]))
    if slot_ids is None:
        t = np.zeros(n, dtype=np.float64)
    else:
        t = np.asarray(slot_ids, dtype=np.float64) * slot_seconds(config)
    gains = jakes_gain(f_d, t, rng, params.n_cisoids)
    amplitudes = (math.sqrt(share) * gains)[:, None]
    return add_taps(
        geometry,
        taus=np.full((n, 1), tau_samples, dtype=np.float64),
        dopplers=np.zeros((n, 1), dtype=np.float64),
        amplitudes=amplitudes,
        total_power=np.full(n, share, dtype=np.float64),
    )
