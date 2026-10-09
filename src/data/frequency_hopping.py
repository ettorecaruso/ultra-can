
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import numpy as np

_DEFAULT_NUM_CHANNELS = 8
_DEFAULT_DWELL_BURSTS = 1
_DEFAULT_BURST_DURATION_US = 10.0
_DEFAULT_SEED = 42

@dataclass(frozen=True)
class HopConfig:

    num_channels: int
    dwell_bursts: int
    burst_duration_us: float
    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.num_channels, bool) or not isinstance(
            self.num_channels, (int, np.integer)
        ):
            raise ValueError(f"num_channels must be an int, got: {self.num_channels!r}")
        if int(self.num_channels) < 1:
            raise ValueError(f"num_channels must be >= 1, got: {self.num_channels!r}")
        if isinstance(self.dwell_bursts, bool) or not isinstance(
            self.dwell_bursts, (int, np.integer)
        ):
            raise ValueError(f"dwell_bursts must be an int, got: {self.dwell_bursts!r}")
        if int(self.dwell_bursts) < 1:
            raise ValueError(f"dwell_bursts must be >= 1, got: {self.dwell_bursts!r}")
        if not math.isfinite(float(self.burst_duration_us)) or float(self.burst_duration_us) <= 0.0:
            raise ValueError(
                f"burst_duration_us must be finite and > 0, got: {self.burst_duration_us!r}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError(f"seed must be an int, got: {self.seed!r}")

    @property
    def slot_duration_us(self) -> float:
        return float(int(self.dwell_bursts)) * float(self.burst_duration_us)

    @property
    def hop_rate_hz(self) -> float:
        return 1.0e6 / self.slot_duration_us

def hop_config_from_dict(
    cfg: Optional[Dict[str, Any]],
    default_seed: int = _DEFAULT_SEED,
) -> HopConfig:
    cfg = cfg or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"frequency_hopping section must be a dict, got: {type(cfg).__name__}")
    num_channels = cfg.get("num_channels", _DEFAULT_NUM_CHANNELS)
    if isinstance(num_channels, bool) or not isinstance(num_channels, (int, np.integer)):
        raise ValueError(
            f"frequency_hopping.num_channels must be an int, got: {num_channels!r}"
        )
    dwell_bursts = cfg.get("dwell_bursts", _DEFAULT_DWELL_BURSTS)
    if isinstance(dwell_bursts, bool) or not isinstance(dwell_bursts, (int, np.integer)):
        raise ValueError(
            f"frequency_hopping.dwell_bursts must be an int, got: {dwell_bursts!r}"
        )
    seed = cfg.get("sequence_seed")
    if seed is None:
        seed = default_seed
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(
            f"frequency_hopping.sequence_seed must be an int or null, got: {seed!r}"
        )
    return HopConfig(
        num_channels=int(num_channels),
        dwell_bursts=int(dwell_bursts),
        burst_duration_us=float(cfg.get("burst_duration_us", _DEFAULT_BURST_DURATION_US)),
        seed=int(seed),
    )

def hop_config(config: Dict[str, Any]) -> HopConfig:
    default_seed = int((config.get("general") or {}).get("seed", _DEFAULT_SEED))
    return hop_config_from_dict(config.get("frequency_hopping"), default_seed)

def with_dwell(cfg: HopConfig, dwell_bursts: int) -> HopConfig:
    return replace(cfg, dwell_bursts=int(dwell_bursts))

def with_channels(cfg: HopConfig, num_channels: int) -> HopConfig:
    return replace(cfg, num_channels=int(num_channels))

def num_slots(num_bursts: int, cfg: HopConfig) -> int:
    if isinstance(num_bursts, bool) or int(num_bursts) < 0:
        raise ValueError(f"num_bursts must be an int >= 0, got: {num_bursts!r}")
    n = int(num_bursts)
    if n == 0:
        return 0
    dwell = int(cfg.dwell_bursts)
    return (n + dwell - 1) // dwell

def build_slot_ids(num_bursts: int, cfg: HopConfig) -> np.ndarray:
    if isinstance(num_bursts, bool) or int(num_bursts) < 0:
        raise ValueError(f"num_bursts must be an int >= 0, got: {num_bursts!r}")
    n = int(num_bursts)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    return np.arange(n, dtype=np.int64) // int(cfg.dwell_bursts)

def build_hop_sequence(num_bursts: int, cfg: HopConfig) -> np.ndarray:
    if isinstance(num_bursts, bool) or int(num_bursts) < 0:
        raise ValueError(f"num_bursts must be an int >= 0, got: {num_bursts!r}")
    n = int(num_bursts)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    n_channels = int(cfg.num_channels)
    dwell = int(cfg.dwell_bursts)
    n_slots = num_slots(n, cfg)
    n_cycles = (n_slots + n_channels - 1) // n_channels

    rng = np.random.default_rng(int(cfg.seed))
    base = np.arange(n_channels, dtype=np.int64)
    cycles = np.empty((n_cycles, n_channels), dtype=np.int64)
    for cycle in range(n_cycles):
        cycles[cycle] = rng.permutation(base)

    slot_channels = cycles.reshape(-1)[:n_slots]
    hop = np.repeat(slot_channels, dwell)[:n]
    if not np.all(np.isfinite(hop)) or np.any(hop < 0) or np.any(hop >= n_channels):
        raise RuntimeError("hop sequence out of range")
    return hop.astype(np.int64)

def build_slot_channels(num_bursts: int, cfg: HopConfig) -> np.ndarray:
    n = int(num_bursts)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    slot_ids = build_slot_ids(n, cfg)
    hop = build_hop_sequence(n, cfg)
    n_slots = int(slot_ids[-1]) + 1
    slot_channels = np.empty(n_slots, dtype=np.int64)
    slot_channels[slot_ids] = hop
    return slot_channels

def hop_counters(num_bursts: int, cfg: HopConfig) -> Dict[str, float]:
    n = int(num_bursts)
    if n == 0:
        raise ValueError("num_bursts must be > 0")
    slot_ids = build_slot_ids(n, cfg)
    return {
        "num_bursts": float(n),
        "num_slots": float(int(slot_ids[-1]) + 1),
        "num_channels": float(int(cfg.num_channels)),
        "dwell_bursts": float(int(cfg.dwell_bursts)),
        "slot_duration_us": float(cfg.slot_duration_us),
        "hop_rate_hz": float(cfg.hop_rate_hz),
    }
