
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from src.data.channel_config import (
    as_choice,
    as_float,
    as_float_list,
    as_int,
    as_str,
    data_section,
    section,
)
from src.data.channel_models import TapGeometry, add_taps

PROFILE_NAMES = (
    "TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E",
    "CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E",
)
LOS_PROFILES = ("TDL-D", "TDL-E", "CDL-D", "CDL-E")
PDP_KINDS = ("exponential", "normative")
CLUTTER_FADING = ("rayleigh", "rician")
MAX_DOPPLER_RATIO = 0.5

def profile_is_los(profile: str) -> bool:
    name = str(profile).strip().upper()
    if name not in PROFILE_NAMES:
        raise ValueError(
            f"channel.pdp.profile must be one of {sorted(PROFILE_NAMES)}, got {profile!r}"
        )
    return name in LOS_PROFILES

@dataclass(frozen=True)
class PdpParameters:
    profile: str
    kind: str
    n_taps: int
    rms_delay_spread_ns: float
    decay_db_per_ds: float
    delays_ns: Tuple[float, ...]
    powers_db: Tuple[float, ...]
    clutter_power_fraction: float
    clutter_fading: str
    clutter_kappa_db: float
    clutter_doppler_max: float
    max_pdp_truncation_fraction: float

    @property
    def los(self) -> bool:
        return profile_is_los(self.profile)

def parameters(config: Mapping[str, Any]) -> PdpParameters:
    channel = section(config, "channel", "config")
    pdp = section(channel, "pdp", "channel")
    params = PdpParameters(
        profile=as_str(pdp, "profile", "channel.pdp"),
        kind=as_choice(pdp, "kind", "channel.pdp", PDP_KINDS),
        n_taps=as_int(pdp, "n_taps", "channel.pdp"),
        rms_delay_spread_ns=as_float(pdp, "rms_delay_spread_ns", "channel.pdp"),
        decay_db_per_ds=as_float(pdp, "decay_db_per_ds", "channel.pdp"),
        delays_ns=tuple(as_float_list(pdp, "delays_ns", "channel.pdp")),
        powers_db=tuple(as_float_list(pdp, "powers_db", "channel.pdp")),
        clutter_power_fraction=as_float(channel, "clutter_power_fraction", "channel"),
        clutter_fading=as_choice(channel, "clutter_fading", "channel", CLUTTER_FADING),
        clutter_kappa_db=as_float(channel, "clutter_kappa_db", "channel"),
        clutter_doppler_max=as_float(channel, "clutter_doppler_max", "channel"),
        max_pdp_truncation_fraction=as_float(
            channel, "max_pdp_truncation_fraction", "channel"
        ),
    )
    profile_is_los(params.profile)
    if params.n_taps < 1:
        raise ValueError(f"channel.pdp.n_taps must be >= 1, got {params.n_taps}")
    if params.rms_delay_spread_ns <= 0.0:
        raise ValueError(
            "channel.pdp.rms_delay_spread_ns must be > 0, got "
            f"{params.rms_delay_spread_ns}"
        )
    if params.decay_db_per_ds < 0.0:
        raise ValueError(
            f"channel.pdp.decay_db_per_ds must be >= 0, got {params.decay_db_per_ds}"
        )
    if not 0.0 <= params.clutter_power_fraction < 1.0:
        raise ValueError(
            "channel.clutter_power_fraction must lie in [0, 1), got "
            f"{params.clutter_power_fraction}"
        )
    if params.clutter_kappa_db < 0.0:
        raise ValueError(
            f"channel.clutter_kappa_db must be >= 0, got {params.clutter_kappa_db}"
        )
    if not 0.0 <= params.clutter_doppler_max < MAX_DOPPLER_RATIO:
        raise ValueError(
            f"channel.clutter_doppler_max must lie in [0, {MAX_DOPPLER_RATIO}), got "
            f"{params.clutter_doppler_max}"
        )
    if not 0.0 <= params.max_pdp_truncation_fraction < 1.0:
        raise ValueError(
            "channel.max_pdp_truncation_fraction must lie in [0, 1), got "
            f"{params.max_pdp_truncation_fraction}"
        )
    if params.kind == "normative" and (
        not params.delays_ns or len(params.delays_ns) != len(params.powers_db)
    ):
        raise ValueError(
            "channel.pdp.delays_ns and channel.pdp.powers_db must be equal-length "
            "non-empty lists for the normative kind"
        )
    return params

def power_delay_profile(params: PdpParameters) -> Tuple[np.ndarray, np.ndarray]:
    if params.kind == "normative":
        delays = np.asarray(params.delays_ns, dtype=np.float64)
        powers = 10.0 ** (np.asarray(params.powers_db, dtype=np.float64) / 10.0)
    else:
        index = np.arange(params.n_taps, dtype=np.float64)
        delays = index * params.rms_delay_spread_ns
        powers = 10.0 ** (-index * params.decay_db_per_ds / 10.0)
    if delays.ndim != 1 or delays.shape != powers.shape or delays.size == 0:
        raise ValueError("power delay profile must be a non-empty 1D pair")
    if np.any(delays < 0.0) or np.any(powers <= 0.0):
        raise ValueError("power delay profile delays must be >= 0 and powers > 0")
    if np.any(np.diff(delays) < 0.0):
        raise ValueError("power delay profile delays must be non-decreasing")
    return delays, powers / np.sum(powers)

def _window(
    params: PdpParameters,
    fs_hz: float,
    max_delay: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    delays, powers = power_delay_profile(params)
    if params.los and delays.size > 1:
        delays = delays[1:]
        powers = powers[1:] / np.sum(powers[1:])
    tau_samples = delays * 1e-9 * float(fs_hz)
    inside = tau_samples <= float(max_delay)
    if not np.any(inside):
        raise ValueError(
            "no power delay profile tap lies inside the observation window: "
            f"the first tap is at {float(np.min(tau_samples)):.3f} samples with "
            f"max_delay={max_delay}"
        )
    dropped = 1.0 - float(np.sum(powers[inside]) / np.sum(powers))
    return tau_samples[inside], powers[inside] / np.sum(powers[inside]), dropped

def window_report(config: Mapping[str, Any]) -> Dict[str, Any]:
    params = parameters(config)
    data = data_section(config)
    tau, weights, dropped = _window(params, float(data["fs_hz"]), int(data["max_delay"]))
    return {
        "profile": params.profile,
        "pdp_kind": params.kind,
        "los_tap_folded_into_direct_path": float(params.los),
        "rms_delay_spread_ns": params.rms_delay_spread_ns,
        "n_taps_in_window": float(tau.size),
        "dropped_power_fraction": float(dropped),
        "max_tap_delay_samples": float(np.max(tau)),
        "min_nonzero_tap_delay_samples": float(np.min(tau[tau > 0.0])) if np.any(tau > 0.0) else 0.0,
        "mean_tap_power": float(np.mean(weights)),
    }

def pdp_frequency_correlation(
    delays_ns: np.ndarray,
    powers: np.ndarray,
    delta_f_hz: float,
) -> complex:
    tau = np.asarray(delays_ns, dtype=np.float64) * 1e-9
    weights = np.asarray(powers, dtype=np.float64)
    return complex(np.sum(weights * np.exp(-2j * math.pi * float(delta_f_hz) * tau)))

def validate(config: Mapping[str, Any]) -> None:
    params = parameters(config)
    data = data_section(config)
    _tau, _weights, dropped = _window(
        params, float(data["fs_hz"]), int(data["max_delay"])
    )
    if dropped > params.max_pdp_truncation_fraction:
        raise ValueError(
            f"power delay profile truncated by {dropped:.3f}, above "
            "channel.max_pdp_truncation_fraction="
            f"{params.max_pdp_truncation_fraction}: raise max_delay or lower the RMS "
            "delay spread"
        )
def overlay(
    config: Mapping[str, Any],
    geometry: TapGeometry,
    rng: np.random.Generator,
    slot_ids: Any = None,
    hop_channels: Any = None,
) -> TapGeometry:
    params = parameters(config)
    if params.clutter_power_fraction <= 0.0:
        return geometry
    data = data_section(config)
    tau_samples, weights, _dropped = _window(
        params, float(data["fs_hz"]), int(data["max_delay"])
    )
    n = geometry.n_symbols
    k = int(tau_samples.size)
    if params.clutter_fading == "rayleigh":
        gains = (
            rng.standard_normal((n, k)) + 1j * rng.standard_normal((n, k))
        ) / math.sqrt(2.0)
    else:
        kappa_lin = 10.0 ** (params.clutter_kappa_db / 10.0)
        los = math.sqrt(kappa_lin / (kappa_lin + 1.0))
        scat = math.sqrt(1.0 / (kappa_lin + 1.0))
        theta = rng.uniform(0.0, 2.0 * math.pi, size=(n, k))
        diffuse = (
            rng.standard_normal((n, k)) + 1j * rng.standard_normal((n, k))
        ) / math.sqrt(2.0)
        gains = los * np.exp(1j * theta) + scat * diffuse
    amplitudes = np.sqrt(weights)[None, :] * gains
    dopplers = rng.uniform(0.0, params.clutter_doppler_max, size=(n, k))
    return add_taps(
        geometry,
        taus=np.tile(tau_samples[None, :], (n, 1)),
        dopplers=dopplers,
        amplitudes=amplitudes,
        total_power=np.full(n, params.clutter_power_fraction, dtype=np.float64),
    )
