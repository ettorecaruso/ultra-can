
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import numpy as np

DEFAULT_CHANNEL_MODEL = "aerial_multiecho"

CHANNEL_MODELS = ("aerial_multiecho", "tdl_3gpp", "two_ray_jakes")

_POWER_EPS = 1e-12

LABEL_EXCLUDED = -np.inf

@dataclass(frozen=True)
class TapGeometry:

    taus: np.ndarray
    dopplers: np.ndarray
    amplitudes: np.ndarray
    dominance: np.ndarray
    valid: np.ndarray
    echo_power: np.ndarray
    k_geometric: int
    is_peer: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        for name in ("taus", "dopplers", "amplitudes", "dominance", "valid"):
            if getattr(self, name).ndim != 2:
                raise ValueError(
                    f"{name} must be 2D (N, K), got {getattr(self, name).shape}"
                )
        shapes = {
            getattr(self, name).shape
            for name in ("taus", "dopplers", "amplitudes", "dominance", "valid")
        }
        if len(shapes) != 1:
            raise ValueError(f"tap arrays must share one shape, got {sorted(shapes)}")
        if self.echo_power.shape != (self.taus.shape[0],):
            raise ValueError(
                f"echo_power must have shape ({self.taus.shape[0]},), "
                f"got {self.echo_power.shape}"
            )
        if self.is_peer is not None:
            if self.is_peer.shape != self.taus.shape:
                raise ValueError(
                    f"is_peer must match the tap shape {self.taus.shape}, "
                    f"got {self.is_peer.shape}"
                )
            if self.is_peer.dtype != bool:
                raise ValueError(f"is_peer must be boolean, got {self.is_peer.dtype}")

    @property
    def peer_mask(self) -> np.ndarray:
        if self.is_peer is None:
            return np.zeros(self.taus.shape, dtype=bool)
        return self.is_peer

    @property
    def k_total(self) -> int:
        return int(self.taus.shape[1])

    @property
    def n_symbols(self) -> int:
        return int(self.taus.shape[0])

    def with_selection(self, index: np.ndarray) -> "TapGeometry":
        idx = np.asarray(index, dtype=np.int64)
        return replace(
            self,
            taus=self.taus[idx],
            dopplers=self.dopplers[idx],
            amplitudes=self.amplitudes[idx],
            dominance=self.dominance[idx],
            valid=self.valid[idx],
            echo_power=self.echo_power[idx],
            is_peer=None if self.is_peer is None else self.is_peer[idx],
        )

def channel_model(config: Dict[str, Any]) -> str:
    channel = config.get("channel")
    if channel is None:
        return DEFAULT_CHANNEL_MODEL
    if not isinstance(channel, dict):
        raise ValueError(
            f"'channel' section must be a dict, got {type(channel).__name__}"
        )
    name = str(channel.get("model", DEFAULT_CHANNEL_MODEL)).strip().lower()
    if name not in CHANNEL_MODELS:
        raise ValueError(
            f"channel.model must be one of {sorted(CHANNEL_MODELS)}, got {name!r}"
        )
    return name

def add_taps(
    geometry: TapGeometry,
    taus: np.ndarray,
    dopplers: np.ndarray,
    amplitudes: np.ndarray,
    total_power: np.ndarray,
    valid: Optional[np.ndarray] = None,
) -> TapGeometry:
    n = geometry.n_symbols
    taus_arr = np.asarray(taus, dtype=np.float64)
    if taus_arr.ndim != 2 or taus_arr.shape[0] != n:
        raise ValueError(f"clutter taus must have shape ({n}, K), got {taus_arr.shape}")
    if taus_arr.shape[1] == 0:
        return geometry
    share = np.asarray(total_power, dtype=np.float64).reshape(n)
    if np.any(share < 0.0) or np.any(share >= 1.0):
        raise ValueError(
            "clutter power share must lie in [0, 1), "
            f"got [{float(np.min(share))}, {float(np.max(share))}]"
        )
    base_scale = np.sqrt(1.0 - share)[:, None]
    eta = np.sqrt(share)[:, None]

    existing_power = np.sum(
        np.where(geometry.valid, np.abs(geometry.amplitudes) ** 2, 0.0), axis=1
    )
    if np.any(existing_power * (1.0 - share) + share >= 1.0 - _POWER_EPS):
        raise ValueError("tap power budget exceeded: sum(alpha^2) would reach one")

    new_valid = (
        np.ones(taus_arr.shape, dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if new_valid.shape != taus_arr.shape:
        raise ValueError("clutter validity mask must match the tap shape")
    if not np.any(new_valid):
        raise ValueError("at least one clutter tap must be valid")
    unit = np.where(new_valid, np.asarray(amplitudes), 0.0)
    unit_norm = np.sqrt(np.sum(np.abs(unit) ** 2, axis=1))
    if not np.all(np.isfinite(unit_norm)) or np.any(unit_norm <= _POWER_EPS):
        raise RuntimeError("clutter taps must carry a non-zero share of the power")
    new_amplitudes = (eta / unit_norm[:, None]) * unit
    clutter_power = np.sum(np.abs(new_amplitudes) ** 2, axis=1)
    if not np.all(np.isfinite(clutter_power)):
        raise RuntimeError("non-finite clutter power after normalization")

    return TapGeometry(
        taus=np.concatenate([geometry.taus, taus_arr], axis=1),
        dopplers=np.concatenate(
            [geometry.dopplers, np.asarray(dopplers, dtype=np.float64)], axis=1
        ),
        amplitudes=np.concatenate(
            [geometry.amplitudes * base_scale, new_amplitudes], axis=1
        ),
        dominance=np.concatenate(
            [geometry.dominance * base_scale, np.full(taus_arr.shape, LABEL_EXCLUDED)],
            axis=1,
        ),
        valid=np.concatenate([geometry.valid, new_valid], axis=1),
        echo_power=existing_power * (1.0 - share) + clutter_power,
        k_geometric=int(geometry.k_geometric),
        is_peer=(
            None
            if geometry.is_peer is None
            else np.concatenate(
                [geometry.is_peer, np.zeros(taus_arr.shape, dtype=bool)], axis=1
            )
        ),
    )

def apply_overlay(
    config: Dict[str, Any],
    geometry: TapGeometry,
    rng: np.random.Generator,
    slot_ids: Optional[np.ndarray] = None,
    hop_channels: Optional[np.ndarray] = None,
) -> TapGeometry:
    name = channel_model(config)
    if name == DEFAULT_CHANNEL_MODEL:
        return geometry
    if name == "tdl_3gpp":
        from src.data import channel_3gpp

        return channel_3gpp.overlay(config, geometry, rng, slot_ids, hop_channels)
    if name == "two_ray_jakes":
        from src.data import channel_two_ray

        return channel_two_ray.overlay(config, geometry, rng, slot_ids, hop_channels)
    raise ValueError(f"unhandled channel model: {name!r}")

def fingerprint_of(channel_section: Dict[str, Any]) -> str:
    payload = json.dumps(
        {str(key): value for key, value in channel_section.items()},
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:8]

def resolved_channel_section(config: Dict[str, Any]) -> Dict[str, Any]:
    section = dict(config.get("channel") or {})
    section["model"] = channel_model(config)
    return section

def channel_fingerprint(config: Dict[str, Any]) -> str:
    return fingerprint_of(resolved_channel_section(config))

def channel_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    channel = dict(config.get("channel") or {})
    return {
        "model": channel_model(config),
        "fingerprint": channel_fingerprint(config),
        "hold_mode": str(channel.get("hold_mode", "per_symbol")),
        "parameters": {
            key: value
            for key, value in channel.items()
            if key not in ("model", "hold_mode")
        },
    }

def power_split_valid(config: Dict[str, Any]) -> bool:
    channel = dict(config.get("channel") or {})
    share = float(channel.get("clutter_power_fraction", 0.0))
    return math.isfinite(share) and 0.0 <= share < 1.0

def band_spacing_hz(config: Dict[str, Any]) -> float:
    hopping = config.get("frequency_hopping") or {}
    num_channels = max(1, int(hopping.get("num_channels", 1)))
    fs_hz = float((config.get("data") or {}).get("fs_hz", 1.0))
    return fs_hz / float(num_channels)

def validate_overlay(config: Dict[str, Any]) -> None:
    name = channel_model(config)
    if name == "tdl_3gpp":
        from src.data import channel_3gpp

        channel_3gpp.validate(config)
    elif name == "two_ray_jakes":
        from src.data import channel_two_ray

        channel_two_ray.validate(config)

def diagnostics(config: Dict[str, Any]) -> Dict[str, Any]:
    name = channel_model(config)
    channel = dict(config.get("channel") or {})
    if name == "tdl_3gpp":
        from src.data import channel_3gpp

        params = channel_3gpp.parameters(config)
        report = channel_3gpp.window_report(config)
        delays, powers = channel_3gpp.power_delay_profile(params)
        report["clutter_power_fraction"] = params.clutter_power_fraction
        report["clutter_fading"] = params.clutter_fading
        report["band_spacing_hz"] = band_spacing_hz(config)
        report["frequency_correlation_at_spacing"] = abs(
            channel_3gpp.pdp_frequency_correlation(
                delays, powers, band_spacing_hz(config)
            )
        )
        if "time_block_slots" in channel:
            report["time_block_slots"] = int(channel["time_block_slots"])
        return report
    if name == "two_ray_jakes":
        from src.data import channel_two_ray

        report = channel_two_ray.report(config)
        report["band_spacing_hz"] = band_spacing_hz(config)
        if "time_block_slots" in channel:
            report["time_block_slots"] = int(channel["time_block_slots"])
        return report
    return {"model": name}
