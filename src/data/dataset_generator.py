
from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config, save_config_snapshot
from src.utils.logger import log_config_summary, setup_logging
from src.data.channel_models import (
    LABEL_EXCLUDED,
    TapGeometry,
    apply_overlay,
    channel_model,
    validate_overlay,
)
from src.data.frequency_hopping import build_hop_sequence, hop_config
from src.data.scene import (
    channel_mode,
    echo_generator,
    require_monostatic,
    sample_scene_taps,
    validate_scene,
)

logger = logging.getLogger(__name__)

_LOGISTIC_MU_MIN = 3.57
_LOGISTIC_MU_MAX = 4.0
_ALPHA_COUPLING_FLOOR = 1e-3
_MAP_TYPES = frozenset({"logistic", "bernoulli"})
_MAP_RETRIES = 5
_X0_MIN = 1e-9
_X0_MAX = 1.0 - 1e-9
_ENERGY_EPS = 1e-12
_POWER_EPS = 1e-9
_FIXED_POINT_TOL = 1e-12
_SPREAD_EPS = 1e-9
_DRONE_SEED_BASE = 1_000_003
_DRONE_SEED_STRIDE = 10_000_019
_SPLITS = frozenset({"train", "val", "test"})
_SEED_HIGH = 2**63 - 1

_ECHO_FADING_TYPES = frozenset({"none", "rayleigh", "rician"})
_NUM_ECHOES_MODES = frozenset({"fixed", "poisson"})
_CHANNEL_HOLD_MODES = frozenset({"per_symbol", "per_slot", "per_hop"})
_POISSON_ECHOES_MEAN_DEFAULT = 2.0
_FADING_POWER_EPS = 1e-12

_BERNOULLI_MULTIPLIER = 5
_BERNOULLI_MASK = (1 << 64) - 1
_BERNOULLI_SCALE = float(1 << 64)

@dataclass(frozen=True)
class EchoParams:

    tau: int
    f_doppler: float
    alpha: float
    is_peer: bool = False
    peer_distance_m: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.tau, int) or self.tau < 1:
            raise ValueError(f"tau must be an int >= 1, got: {self.tau!r}")
        if not math.isfinite(self.f_doppler) or not (0.0 <= self.f_doppler < 0.5):
            raise ValueError(f"f_doppler must be in [0, 0.5), got: {self.f_doppler!r}")
        if not math.isfinite(self.alpha) or not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got: {self.alpha!r}")
        if not isinstance(self.is_peer, bool):
            raise ValueError(f"is_peer must be a bool, got: {self.is_peer!r}")
        if not math.isfinite(self.peer_distance_m) or self.peer_distance_m < 0.0:
            raise ValueError(
                "peer_distance_m must be finite and >= 0, got: "
                f"{self.peer_distance_m!r}"
            )

@dataclass(frozen=True)
class DirectPathParams:

    h_c: complex
    f_dc: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.h_c.real) and math.isfinite(self.h_c.imag)):
            raise ValueError(f"h_c not finite: {self.h_c!r}")
        if not math.isfinite(self.f_dc) or not (0.0 <= self.f_dc < 0.5):
            raise ValueError(f"f_dc must be in [0, 0.5), got: {self.f_dc!r}")

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

_REQUIRED_DATA_KEYS: Tuple[str, ...] = (
    "sequence_length", "map_type", "map_param", "fc_hz", "fs_hz",
    "snr_range", "snr_step", "echoes", "max_delay", "max_doppler",
    "alpha_min", "alpha_max", "num_symbols_train", "num_symbols_val",
    "num_symbols_test", "raw_dir", "processed_dir",
    "rician_kappa_db", "doppler_direct_max",
)

def _channel_section(config: Dict[str, Any]) -> Dict[str, Any]:
    channel = config.get("channel")
    if channel is None:
        return {}
    if not isinstance(channel, dict):
        raise ValueError(f"'channel' section must be a dict, got: {type(channel).__name__}")
    return channel

def channel_echo_fading(config: Dict[str, Any]) -> str:
    return str(_channel_section(config).get("echo_fading", "none")).strip().lower()

def channel_echo_fading_kappa_db(config: Dict[str, Any]) -> float:
    value = _channel_section(config).get("echo_fading_kappa_db")
    if value is None:
        return 0.0
    return float(value)

def channel_direct_kappa_db(config: Dict[str, Any]) -> float:
    value = _channel_section(config).get("direct_fading_kappa_db")
    if value is None:
        return float(config["data"]["rician_kappa_db"])
    return float(value)

def channel_num_echoes_mode(config: Dict[str, Any]) -> str:
    return str(_channel_section(config).get("num_echoes_model", "fixed")).strip().lower()

def channel_poisson_echoes_mean(config: Dict[str, Any]) -> float:
    channel = _channel_section(config)
    return float(channel.get("poisson_echoes_mean", _POISSON_ECHOES_MEAN_DEFAULT))

def channel_hold_mode(config: Dict[str, Any]) -> str:
    return str(_channel_section(config).get("hold_mode", "per_symbol")).strip().lower()

def channel_hold_dwell(config: Dict[str, Any]) -> int:
    cfg = config.get("frequency_hopping") or {}
    if not isinstance(cfg, dict):
        return 1
    return max(1, int(cfg.get("dwell_bursts", 1)))

def channel_time_block_slots(config: Dict[str, Any]) -> int:
    channel = _channel_section(config)
    if "time_block_slots" in channel:
        blocks = int(channel["time_block_slots"])
        path = "channel.time_block_slots"
    else:
        hopping = config.get("frequency_hopping") or {}
        if not isinstance(hopping, dict) or "time_block_slots" not in hopping:
            raise KeyError(
                "missing required configuration key: "
                "frequency_hopping.time_block_slots (or the channel.time_block_slots "
                "override)"
            )
        blocks = int(hopping["time_block_slots"])
        path = "frequency_hopping.time_block_slots"
    if blocks < 1:
        raise ValueError(f"{path} must be >= 1, got {blocks}")
    return blocks

def _validate_channel_config(config: Dict[str, Any]) -> None:
    channel = _channel_section(config)
    if not channel:
        return
    echo_fading = channel_echo_fading(config)
    if echo_fading not in _ECHO_FADING_TYPES:
        raise ValueError(
            f"channel.echo_fading must be one of {sorted(_ECHO_FADING_TYPES)}, got: {echo_fading!r}"
        )
    kappa_db = channel_echo_fading_kappa_db(config)
    if not math.isfinite(kappa_db) or kappa_db < 0.0:
        raise ValueError(f"channel.echo_fading_kappa_db must be finite and >= 0, got: {kappa_db!r}")
    direct_kappa_db = channel_direct_kappa_db(config)
    if not math.isfinite(direct_kappa_db) or direct_kappa_db < 0.0:
        raise ValueError(
            f"channel.direct_fading_kappa_db must be finite and >= 0, got: {direct_kappa_db!r}"
        )
    num_echoes_mode = channel_num_echoes_mode(config)
    if num_echoes_mode not in _NUM_ECHOES_MODES:
        raise ValueError(
            f"channel.num_echoes_model must be one of {sorted(_NUM_ECHOES_MODES)}, "
            f"got: {num_echoes_mode!r}"
        )
    if num_echoes_mode == "poisson":
        mean = channel_poisson_echoes_mean(config)
        if not math.isfinite(mean) or mean <= 0.0:
            raise ValueError(f"channel.poisson_echoes_mean must be finite and > 0, got: {mean!r}")
    hold_mode = channel_hold_mode(config)
    if hold_mode not in _CHANNEL_HOLD_MODES:
        raise ValueError(
            f"channel.hold_mode must be one of {sorted(_CHANNEL_HOLD_MODES)}, got: {hold_mode!r}"
        )
    if hold_mode == "per_hop":
        channel_time_block_slots(config)
    channel_model(config)
    validate_overlay(config)
    generator = echo_generator(config)
    channel_mode(config)
    if generator == "geometry":
        validate_scene(config)

def _hold_representatives(
    hold_mode: str,
    slot_ids: Optional[np.ndarray],
    hop_channels: Optional[np.ndarray],
    n: int,
    config: Dict[str, Any],
) -> Optional[np.ndarray]:
    if hold_mode == "per_symbol":
        return None
    if slot_ids is None:
        raise ValueError(f"channel.hold_mode is {hold_mode!r} but slot_ids is None")
    slots = np.asarray(slot_ids, dtype=np.int64)
    if hold_mode == "per_slot":
        return _slot_representatives(slots, n)
    if hop_channels is None:
        raise ValueError("channel.hold_mode is 'per_hop' but hop_channels is None")
    hop = np.asarray(hop_channels, dtype=np.int64)
    if hop.shape != slots.shape:
        raise ValueError(f"hop_channels must have shape {slots.shape}, got {hop.shape}")
    num_channels = max(
        1, int((config.get("frequency_hopping") or {}).get("num_channels", 1))
    )
    blocks = channel_time_block_slots(config)
    return _slot_representatives((slots // blocks) * num_channels + hop, n)

def _validate_config(config: Dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got: {type(config).__name__}")
    _assert_config_finite(config)
    data = config.get("data")
    general = config.get("general")
    if not isinstance(data, dict):
        raise ValueError("'data' section missing or not a dict in the config")
    if not isinstance(general, dict):
        raise ValueError("'general' section missing or not a dict in the config")
    missing = [key for key in _REQUIRED_DATA_KEYS if key not in data]
    if missing:
        raise ValueError(f"missing keys in config['data']: {missing}")
    if "seed" not in general:
        raise ValueError("missing key: general.seed")

    sequence_length = int(data["sequence_length"])
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be > 0, got: {sequence_length}")

    map_type = str(data["map_type"])
    if map_type not in _MAP_TYPES:
        raise ValueError(f"map_type must be one of {sorted(_MAP_TYPES)}, got: {map_type!r}")

    map_param = float(data["map_param"])
    if not math.isfinite(map_param):
        raise ValueError(f"map_param not finite: {map_param!r}")
    if map_type == "logistic" and not (_LOGISTIC_MU_MIN <= map_param <= _LOGISTIC_MU_MAX):
        raise ValueError(
            f"map_param (mu) outside the chaotic regime "
            f"[{_LOGISTIC_MU_MIN}, {_LOGISTIC_MU_MAX}]: {map_param}"
        )

    for key in ("fc_hz", "fs_hz", "max_doppler", "doppler_direct_max", "alpha_min", "alpha_max"):
        value = float(data[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"data.{key} must be finite and > 0, got: {value!r}")

    if float(data["max_doppler"]) >= 0.5:
        raise ValueError("data.max_doppler must be < 0.5 (anti-aliasing guard)")
    if float(data["doppler_direct_max"]) > float(data["max_doppler"]):
        raise ValueError("data.doppler_direct_max must be <= data.max_doppler")

    if float(data["alpha_min"]) >= float(data["alpha_max"]):
        raise ValueError("data.alpha_min must be < data.alpha_max")
    peer_echo_count(config)

    rician_kappa_db = float(data["rician_kappa_db"])
    if not math.isfinite(rician_kappa_db) or rician_kappa_db < 0.0:
        raise ValueError(f"data.rician_kappa_db must be >= 0, got: {rician_kappa_db!r}")

    snr_range = data["snr_range"]
    if not isinstance(snr_range, (list, tuple)) or len(snr_range) != 2:
        raise ValueError(f"data.snr_range must be [min, max], got: {snr_range!r}")
    snr_min, snr_max = float(snr_range[0]), float(snr_range[1])
    if not (math.isfinite(snr_min) and math.isfinite(snr_max)) or snr_min >= snr_max:
        raise ValueError(f"data.snr_range invalid: {snr_range!r}")

    snr_step = float(data["snr_step"])
    if not math.isfinite(snr_step) or snr_step <= 0.0:
        raise ValueError(f"data.snr_step must be > 0, got: {snr_step!r}")

    max_delay = int(data["max_delay"])
    if max_delay <= 0:
        raise ValueError(f"data.max_delay must be > 0, got: {max_delay}")
    if max_delay > sequence_length:
        raise ValueError(
            f"data.max_delay ({max_delay}) must be <= sequence_length ({sequence_length})"
        )

    echoes = data["echoes"]
    if not isinstance(echoes, (list, tuple)) or len(echoes) == 0:
        raise ValueError(f"data.echoes must be a non-empty list, got: {echoes!r}")
    for k in echoes:
        if not isinstance(k, (int,)) or isinstance(k, bool) or k < 0:
            raise ValueError(f"data.echoes must contain ints >= 0, got: {k!r}")
        if k > max_delay:
            raise ValueError(
                f"k={k} > max_delay={max_delay}: cannot sample {k} distinct delays"
            )

    for key in ("num_symbols_train", "num_symbols_val", "num_symbols_test"):
        value = data[key]
        if not isinstance(value, (int,)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"data.{key} must be an int > 0, got: {value!r}")

    for key in ("raw_dir", "processed_dir"):
        value = data[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"data.{key} must be a non-empty string, got: {value!r}")

    seed = general["seed"]
    if not isinstance(seed, (int,)) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"general.seed must be an int >= 0, got: {seed!r}")

    _validate_channel_config(config)

    logger.debug(
        "validated config: seq_len=%d, map_type=%s, mu=%s, max_delay=%d, max_doppler=%s",
        sequence_length, map_type, map_param, max_delay, data["max_doppler"],
    )

def _iterate_map(map_type: str, map_param: float, x0: float, sequence_length: int) -> np.ndarray:
    seq = np.empty(sequence_length, dtype=np.float64)
    if map_type == "logistic":
        x = x0
        for i in range(sequence_length):
            seq[i] = x
            if i + 1 == sequence_length:
                break
            x = map_param * x * (1.0 - x)
    else:
        state = int(np.float64(x0).view(np.uint64)) & _BERNOULLI_MASK
        for i in range(sequence_length):
            seq[i] = state / _BERNOULLI_SCALE
            state = (state * _BERNOULLI_MULTIPLIER) & _BERNOULLI_MASK
    return seq

def generate_chaotic_sequence(
    map_type: str,
    map_param: float,
    seed: int,
    sequence_length: int,
) -> np.ndarray:
    if not isinstance(map_type, str) or map_type not in _MAP_TYPES:
        raise ValueError(f"map_type must be one of {sorted(_MAP_TYPES)}, got: {map_type!r}")
    if not isinstance(map_param, (int, float)) or not math.isfinite(float(map_param)):
        raise ValueError(f"map_param must be a finite float, got: {map_param!r}")
    if map_type == "logistic" and not (_LOGISTIC_MU_MIN <= float(map_param) <= _LOGISTIC_MU_MAX):
        raise ValueError(
            f"map_param (mu) outside the chaotic regime "
            f"[{_LOGISTIC_MU_MIN}, {_LOGISTIC_MU_MAX}]: {map_param}"
        )
    if not isinstance(seed, (int,)) or isinstance(seed, bool) or int(seed) < 0:
        raise ValueError(f"seed must be an int >= 0, got: {seed!r}")
    if not isinstance(sequence_length, (int,)) or int(sequence_length) <= 0:
        raise ValueError(f"sequence_length must be an int > 0, got: {sequence_length!r}")

    n = int(sequence_length)
    mu = float(map_param)
    x0 = _x0_from_seed(int(seed), map_type, mu)
    seq = _iterate_map(map_type, mu, x0, n)
    energy = float(np.sum(seq ** 2))
    spread = float(np.max(seq) - np.min(seq))
    valid = (
        np.all(np.isfinite(seq))
        and np.all(seq >= 0.0)
        and np.all(seq <= 1.0)
        and energy > _ENERGY_EPS
        and spread > _SPREAD_EPS
        and np.count_nonzero(seq) >= max(2, n // 10)
    )
    if valid:
        logger.debug(
            "map %s generated (x0=%.6f, energy=%.3e)",
            map_type, x0, energy,
        )
        return np.asarray(seq, dtype=np.float64)

    raise RuntimeError(
        f"map {map_type!r} degenerate: NaN/Inf or zero energy"
    )

_C_LIGHT_M_S = 299_792_458.0

def _peer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    
    peers_cfg = config.get("peers")
    if peers_cfg is None:
        return {}
    if not isinstance(peers_cfg, dict):
        raise ValueError(
            f"'peers' section must be a dict, got: {type(peers_cfg).__name__}"
        )
    return peers_cfg

def drone_symbol_seeds(drone_id: int, n: int) -> np.ndarray:
    
    if isinstance(drone_id, bool) or not isinstance(drone_id, int) or int(drone_id) < 0:
        raise ValueError(f"drone_id must be an int >= 0, got: {drone_id!r}")
    if isinstance(n, bool) or not isinstance(n, int) or int(n) < 0:
        raise ValueError(f"n must be an int >= 0, got: {n!r}")
    start = _DRONE_SEED_BASE + int(drone_id) * _DRONE_SEED_STRIDE
    return start + np.arange(int(n), dtype=np.int64)

def peer_echo_count(config: Dict[str, Any]) -> int:
    
    peers_cfg = _peer_config(config)
    if not bool(peers_cfg.get("enable", False)):
        return 0
    n_peers = peers_cfg.get("n_peers", 0)
    if isinstance(n_peers, bool) or not isinstance(n_peers, (int, float)):
        raise ValueError(
            f"peers.n_peers must be a non-negative integer, got: {n_peers!r}"
        )
    if (
        not math.isfinite(float(n_peers))
        or not float(n_peers).is_integer()
        or float(n_peers) < 0.0
    ):
        raise ValueError(
            f"peers.n_peers must be a non-negative integer, got: {n_peers!r}"
        )
    return int(n_peers)

def peer_delay_samples(distances_m: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    
    fs_hz = float(config["data"]["fs_hz"])
    if not (math.isfinite(fs_hz) and fs_hz > 0.0):
        raise ValueError(f"data.fs_hz must be > 0, got: {fs_hz!r}")
    return 2.0 * np.asarray(distances_m, dtype=np.float64) * fs_hz / _C_LIGHT_M_S

def _sample_peer_taps(
    config: Dict[str, Any],
    rng: np.random.Generator,
    n: int,
    max_delay: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    n_peers = peer_echo_count(config)
    if n_peers == 0:
        empty = np.zeros((n, 0), dtype=np.float64)
        return empty, empty, empty, empty
    peers_cfg = _peer_config(config)
    fd_max = float(config["data"]["max_doppler"])
    range_cfg = peers_cfg.get("peer_range_m", [10.0, 120.0])
    if not isinstance(range_cfg, (list, tuple)) or len(range_cfg) != 2:
        raise ValueError(
            f"peers.peer_range_m must be a [min, max] pair, got: {range_cfg!r}"
        )
    range_lo = float(range_cfg[0])
    range_hi = float(range_cfg[1])
    if not (math.isfinite(range_lo) and math.isfinite(range_hi)):
        raise ValueError(f"peers.peer_range_m must be finite, got: {range_cfg!r}")
    if not (0.0 < range_lo < range_hi):
        raise ValueError(
            f"peers.peer_range_m must satisfy 0 < min < max, got: {range_cfg!r}"
        )
    alpha_ref = float(peers_cfg.get("peer_alpha_ref", 0.25))
    alpha_cap = float(peers_cfg.get("peer_alpha_max", 0.6))
    range_ref = float(peers_cfg.get("peer_range_ref_m", 30.0))
    fd_fraction = float(peers_cfg.get("peer_doppler_fraction", 0.5))
    if not (0.0 < alpha_ref < 1.0) or not (0.0 < alpha_cap <= 1.0):
        raise ValueError(
            "peers.peer_alpha_ref must be in (0, 1) and peers.peer_alpha_max in "
            f"(0, 1], got: {alpha_ref!r} / {alpha_cap!r}"
        )
    if not (math.isfinite(range_ref) and range_ref > 0.0):
        raise ValueError(f"peers.peer_range_ref_m must be > 0, got: {range_ref!r}")
    if not (0.0 <= fd_fraction <= 1.0):
        raise ValueError(
            f"peers.peer_doppler_fraction must be in [0, 1], got: {fd_fraction!r}"
        )
    distances = rng.uniform(range_lo, range_hi, size=(n, n_peers))
    taus = np.clip(
        np.rint(peer_delay_samples(distances, config)).astype(np.int64),
        1,
        int(max_delay),
    )
    alphas = np.clip(alpha_ref * (range_ref / distances) ** 2, 0.0, alpha_cap)
    dopplers = rng.uniform(0.0, fd_max * fd_fraction, size=(n, n_peers))
    if not (np.all(np.isfinite(alphas)) and np.all(np.isfinite(dopplers))):
        raise RuntimeError("peer geometry is not finite")
    return distances, taus, alphas, dopplers

def sample_echo_parameters(
    k: int,
    rng: np.random.Generator,
    config: Dict[str, Any],
) -> List[EchoParams]:
    if not isinstance(k, (int,)) or isinstance(k, bool) or int(k) < 0:
        raise ValueError(f"k must be an int >= 0, got: {k!r}")
    k = int(k)
    if k == 0:
        return []
    data = config["data"]
    sequence_length = int(data["sequence_length"])
    max_delay = int(data["max_delay"])
    max_doppler = float(data["max_doppler"])
    alpha_min = float(data["alpha_min"])
    alpha_max = float(data["alpha_max"])
    if max_delay > sequence_length:
        raise ValueError(f"max_delay ({max_delay}) > sequence_length ({sequence_length})")
    if k > max_delay:
        raise ValueError(
            f"k ({k}) > max_delay ({max_delay}): cannot sample "
            f"{k} distinct integer delays in [1, max_delay]"
        )

    taus = rng.choice(np.arange(1, max_delay + 1, dtype=np.int64), size=k, replace=False)
    dopplers = rng.uniform(0.0, max_doppler, size=k)
    if bool(data.get("alpha_tau_coupling", False)):
        floor = float(data.get("alpha_floor", _ALPHA_COUPLING_FLOOR))
        alphas = np.clip(alpha_max / np.maximum(taus.astype(np.float64), 1.0),
                         floor, alpha_max)
    else:
        alphas = 10.0 ** rng.uniform(math.log10(alpha_min), math.log10(alpha_max), size=k)

    echoes = [
        EchoParams(tau=int(taus[i]), f_doppler=float(dopplers[i]), alpha=float(alphas[i]))
        for i in range(k)
    ]
    peer_distances, peer_taus, peer_alphas, peer_dopplers = _sample_peer_taps(
        config, rng, 1, max_delay
    )
    for index in range(int(peer_taus.shape[1])):
        echoes.append(
            EchoParams(
                tau=int(peer_taus[0, index]),
                f_doppler=float(peer_dopplers[0, index]),
                alpha=float(peer_alphas[0, index]),
                is_peer=True,
                peer_distance_m=float(peer_distances[0, index]),
            )
        )
    if not all(math.isfinite(e.f_doppler) and math.isfinite(e.alpha) for e in echoes):
        raise ValueError("echo parameters not finite during sampling")
    logger.debug(
        "sampled %d echoes: tau=%s, fD=%s, alpha=%s",
        k, [e.tau for e in echoes], [e.f_doppler for e in echoes], [e.alpha for e in echoes],
    )
    return echoes

def sample_echo_counts(
    n: int,
    rng: np.random.Generator,
    config: Dict[str, Any],
) -> np.ndarray:
    if isinstance(n, bool) or int(n) <= 0:
        raise ValueError(f"n must be an int > 0, got: {n!r}")
    n = int(n)
    max_delay = int(config["data"]["max_delay"])
    mean = channel_poisson_echoes_mean(config)
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError(f"channel.poisson_echoes_mean must be finite and > 0, got: {mean!r}")
    counts = rng.poisson(mean, size=n).astype(np.int64)
    counts = np.clip(counts, 0, max_delay)
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise RuntimeError("poisson echo counts not valid")
    return counts

def sample_direct_path(rng: np.random.Generator, config: Dict[str, Any]) -> DirectPathParams:
    data = config["data"]
    kappa_db = float(data["rician_kappa_db"])
    if not math.isfinite(kappa_db) or kappa_db < 0.0:
        raise ValueError(f"data.rician_kappa_db must be >= 0, got: {kappa_db!r}")
    doppler_direct_max = float(data["doppler_direct_max"])

    kappa_lin = 10.0 ** (kappa_db / 10.0)
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    g = (rng.standard_normal() + 1j * rng.standard_normal()) / math.sqrt(2.0)
    h_c = (
        math.sqrt(kappa_lin / (kappa_lin + 1.0)) * np.exp(1j * theta)
        + math.sqrt(1.0 / (kappa_lin + 1.0)) * g
    )
    f_dc = float(rng.uniform(0.0, doppler_direct_max))
    if not (math.isfinite(h_c.real) and math.isfinite(h_c.imag) and math.isfinite(f_dc)):
        raise RuntimeError("direct path sampling not finite")
    logger.debug("path diretto: abs(h_c)=%.4f (kappa=%.1f dB), f_Dc=%.3e", abs(h_c), kappa_db, f_dc)
    return DirectPathParams(h_c=h_c, f_dc=f_dc)

def apply_aerial_channel(
    x: np.ndarray,
    h_c: complex,
    f_dc: float,
    echoes: List[EchoParams],
    snr_db: float,
    rng: np.random.Generator,
    *,
    echo_fading: Optional[Tuple[str, float]] = None,
) -> Tuple[np.ndarray, float]:
    if not isinstance(echoes, list):
        raise TypeError(f"echoes must be list[EchoParams], got: {type(echoes).__name__}")
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1 or x_arr.size == 0:
        raise ValueError(f"x must be a non-empty 1D vector, got: shape={x_arr.shape}")
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x contains NaN/Inf")
    if not (math.isfinite(f_dc) and 0.0 <= f_dc < 0.5):
        raise ValueError(f"f_dc must be in [0, 0.5), got: {f_dc!r}")
    if not (math.isfinite(h_c.real) and math.isfinite(h_c.imag)):
        raise ValueError(f"h_c not finite: {h_c!r}")
    if not math.isfinite(float(snr_db)):
        raise ValueError(f"snr_db must be finite, got: {snr_db!r}")

    fading_kind = "none"
    fading_kappa_db = 0.0
    if echo_fading is not None:
        if not isinstance(echo_fading, (tuple, list)) or len(echo_fading) != 2:
            raise ValueError(
                f"echo_fading must be a (kind, kappa_db) pair, got: {echo_fading!r}"
            )
        fading_kind = str(echo_fading[0]).strip().lower()
        fading_kappa_db = float(echo_fading[1])
        if fading_kind not in _ECHO_FADING_TYPES:
            raise ValueError(
                f"echo_fading kind must be one of {sorted(_ECHO_FADING_TYPES)}, "
                f"got: {fading_kind!r}"
            )
        if not math.isfinite(fading_kappa_db) or fading_kappa_db < 0.0:
            raise ValueError(
                f"echo_fading kappa_db must be finite and >= 0, got: {fading_kappa_db!r}"
            )

    n_samples = x_arr.size
    n_idx = np.arange(n_samples, dtype=np.float64)

    echo_power = 0.0
    for echo in echoes:
        if not isinstance(echo, EchoParams):
            raise TypeError(f"element of echoes is not an EchoParams: {echo!r}")
        if echo.tau > n_samples:
            raise ValueError(
                f"echo with tau={echo.tau} beyond the observation window (N={n_samples})"
            )
        echo_power += float(echo.alpha) ** 2
    if echo_power >= 1.0 - _POWER_EPS:
        raise ValueError(
            f"sum of alpha_k^2 = {echo_power:.6f} >= 1: normalization Eq. (4) is impossible"
        )

    if echoes and fading_kind in ("rayleigh", "rician"):
        alpha_arr = np.asarray([[float(e.alpha) for e in echoes]], dtype=np.float64)
        mask_arr = np.ones_like(alpha_arr, dtype=bool)
        faded = _apply_echo_fading(
            alpha_arr, mask_arr, fading_kind, fading_kappa_db, rng
        )
        echo_amplitudes: List[complex] = [complex(v) for v in faded[0]]
    else:
        echo_amplitudes = [float(e.alpha) for e in echoes]

    h_c_eff = h_c * math.sqrt(1.0 - echo_power)

    y_clean = h_c_eff * x_arr * np.exp(1j * 2.0 * math.pi * f_dc * n_idx)
    for index, echo in enumerate(echoes):
        x_delayed = np.zeros(n_samples, dtype=np.float64)
        x_delayed[echo.tau:] = x_arr[: n_samples - echo.tau]
        y_clean = y_clean + echo_amplitudes[index] * x_delayed * np.exp(
            1j * 2.0 * math.pi * echo.f_doppler * n_idx
        )

    signal_power = float(np.mean(np.abs(y_clean) ** 2))
    if not math.isfinite(signal_power) or signal_power <= 0.0:
        raise RuntimeError(f"invalid signal power: {signal_power!r} (degenerate symbol)")

    noise_var = signal_power * 10.0 ** (-float(snr_db) / 10.0)
    w = math.sqrt(noise_var / 2.0) * (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    )
    y = y_clean + w

    if not np.all(np.isfinite(y)) or not math.isfinite(noise_var):
        raise RuntimeError("channel output not finite (NaN/Inf)")

    realized_snr = 10.0 * math.log10(signal_power / noise_var) if noise_var > 0.0 else math.inf
    logger.debug(
        "channel: P_s=%.3e, noise_var=%.3e, realized SNR=%.2f dB (requested %.1f dB)",
        signal_power, noise_var, realized_snr, snr_db,
    )
    return y, noise_var

def apply_echo_only_channel(
    x: np.ndarray,
    echoes: List[EchoParams],
) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be a np.ndarray, got: {type(x).__name__}")
    if not isinstance(echoes, list):
        raise TypeError(f"echoes must be list[EchoParams], got: {type(echoes).__name__}")
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1 or x_arr.size == 0:
        raise ValueError(f"x must be a non-empty 1D vector, got: shape={x_arr.shape}")
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x contiene NaN/Inf")

    n_samples = x_arr.size
    n_idx = np.arange(n_samples, dtype=np.float64)

    y_clean = x_arr.astype(np.complex128)

    for echo in echoes:
        if not isinstance(echo, EchoParams):
            raise TypeError(f"element of echoes is not an EchoParams: {echo!r}")
        if echo.tau > n_samples:
            raise ValueError(
                f"echo with tau={echo.tau} beyond the observation window (N={n_samples})"
            )
        x_delayed = np.zeros(n_samples, dtype=np.float64)
        x_delayed[echo.tau:] = x_arr[: n_samples - echo.tau]
        y_clean = y_clean + echo.alpha * x_delayed * np.exp(
            1j * 2.0 * math.pi * echo.f_doppler * n_idx
        )

    if not np.all(np.isfinite(y_clean)):
        raise RuntimeError(
            "echo-only channel output not finite (NaN/Inf)"
        )

    logger.debug(
        "echo-only channel applied: N=%d, K=%d echoes, finite output",
        n_samples, len(echoes),
    )
    return y_clean

def build_snr_grid(snr_range: List[float], snr_step: float) -> List[float]:
    if not isinstance(snr_range, (list, tuple)) or len(snr_range) != 2:
        raise ValueError(f"snr_range must be [min, max], got: {snr_range!r}")
    snr_min, snr_max = float(snr_range[0]), float(snr_range[1])
    step = float(snr_step)
    if not (math.isfinite(snr_min) and math.isfinite(snr_max) and math.isfinite(step)):
        raise ValueError(f"non-finite values in snr_range/snr_step: {snr_range!r}, {snr_step!r}")
    if snr_min >= snr_max:
        raise ValueError(f"snr_min ({snr_min}) must be < snr_max ({snr_max})")
    if step <= 0.0:
        raise ValueError(f"snr_step must be > 0, got: {step}")

    n_points = int(math.ceil((snr_max - snr_min) / step - 1e-9))
    if n_points < 1:
        raise ValueError(f"empty SNR grid for {snr_range!r} with step {step}")
    points = [snr_min + i * step for i in range(n_points)]
    if abs(points[-1] - snr_max) > 1e-9:
        points.append(snr_max)
    logger.debug("inclusive SNR grid: %s", points)
    return points

def _map_type_for_bit(map_type: str, bit: int) -> str:
    if map_type not in _MAP_TYPES:
        raise ValueError(f"map_type must be one of {sorted(_MAP_TYPES)}, got: {map_type!r}")
    if bit not in (0, 1):
        raise ValueError(f"bit must be 0 or 1, got: {bit!r}")
    if bit == 0:
        return map_type
    return "bernoulli" if map_type == "logistic" else "logistic"

def _dominant_echo_index(echoes: List[EchoParams]) -> int:
    if not echoes:
        raise ValueError("empty echoes: no dominant echo")
    return int(np.argmax([echo.alpha for echo in echoes]))

def resolve_n_per_combo(
    config: Dict[str, Any],
    split: str,
    num_combos: int,
) -> int:
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {sorted(_SPLITS)}, got: {split!r}")
    if not isinstance(num_combos, int) or isinstance(num_combos, bool) or num_combos <= 0:
        raise ValueError(f"num_combos must be an int > 0, got: {num_combos!r}")

    data = config["data"]
    if split == "train":
        total = int(data["num_symbols_train"])
        if total % num_combos != 0:
            nearest_down = num_combos * (total // num_combos)
            nearest_up = nearest_down + num_combos
            raise ValueError(
                f"num_symbols_train ({total}) not divisible by num_combos ({num_combos}): "
                "balancing per (SNR, K) is impossible. "
                f"Valid nearby values: {nearest_down} or {nearest_up}."
            )
        n_per_combo = total // num_combos
    elif split == "val":
        n_per_combo = int(data["num_symbols_val"])
    else:
        n_per_combo = int(data["num_symbols_test"])

    if n_per_combo <= 0:
        raise ValueError(
            f"symbols per combination must be > 0, got: {n_per_combo}"
        )
    if n_per_combo % 2 != 0:
        if split == "train":
            unit = 2 * num_combos
            suggestion = unit * ((total + unit - 1) // unit)
            hint = (
                f"use a num_symbols_train multiple of "
                f"2*num_combos={unit} (valid nearby value: {suggestion})"
            )
        else:
            key = "num_symbols_val" if split == "val" else "num_symbols_test"
            hint = f"for split '{split}' use {key} even (e.g. {n_per_combo + 1})"
        raise ValueError(
            f"n_per_combo ({n_per_combo}) must be even to balance the 0/1 bits "
            f"for each (SNR, K) point: {hint}"
        )
    return n_per_combo

def _center_normalize_symbol(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    x_centered = x_arr - np.mean(x_arr)
    energy = float(np.sqrt(np.sum(x_centered ** 2)))
    if not math.isfinite(energy) or energy <= _ENERGY_EPS:
        raise RuntimeError(
            "degenerate symbol after centering/normalization (zero energy)"
        )
    return x_centered / energy

def _x0_from_seed(seed: int, map_type: str, map_param: float) -> float:
    if map_type not in _MAP_TYPES:
        raise ValueError(
            f"map_type must be one of {sorted(_MAP_TYPES)}, got: {map_type!r}"
        )
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError(f"seed must be an int >= 0, got: {seed!r}")
    seed = int(seed)
    mu = float(map_param)
    if map_type == "logistic":
        forbidden = (0.25, 0.5, 0.75, 1.0 - 1.0 / mu)
    else:
        forbidden = (0.0, 0.5, 1.0)

    rng_map = np.random.default_rng(seed)
    for _attempt in range(1, _MAP_RETRIES + 1):
        x0 = float(rng_map.uniform(_X0_MIN, _X0_MAX))
        if any(abs(x0 - point) < _FIXED_POINT_TOL for point in forbidden):
            continue
        return x0

    raise RuntimeError(
        f"map {map_type!r} degenerate after {_MAP_RETRIES} attempts: x0 on a fixed point"
    )

def _iterate_map_batch(
    map_type: str,
    map_param: float,
    x0: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    x0_arr = np.asarray(x0, dtype=np.float64)
    if x0_arr.ndim != 1:
        raise ValueError(f"x0 must be 1D, got: {x0_arr.shape}")
    n = int(x0_arr.shape[0])
    if n == 0:
        return np.empty((0, sequence_length), dtype=np.float64)
    seq = np.empty((n, sequence_length), dtype=np.float64)

    if map_type == "logistic":
        x = x0_arr.copy()
        for i in range(sequence_length):
            seq[:, i] = x
            if i + 1 == sequence_length:
                break
            x = map_param * x * (1.0 - x)
    else:
        state = np.asarray(x0_arr, dtype=np.float64).view(np.uint64) & _BERNOULLI_MASK
        for i in range(sequence_length):
            seq[:, i] = state / _BERNOULLI_SCALE
            state = (state * _BERNOULLI_MULTIPLIER) & _BERNOULLI_MASK
    return seq

def _map_types_for_bits(map_type: str, bits: np.ndarray) -> np.ndarray:
    if map_type not in _MAP_TYPES:
        raise ValueError(
            f"map_type must be one of {sorted(_MAP_TYPES)}, got: {map_type!r}"
        )
    bits_arr = np.asarray(bits)
    if not np.all(np.isin(bits_arr, (0, 1))):
        raise ValueError(f"bits must contain only 0/1, got: {bits_arr!r}")
    if map_type == "logistic":
        return np.where(bits_arr == 0, "logistic", "bernoulli")
    return np.where(bits_arr == 0, "bernoulli", "logistic")

def _center_normalize_batch(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"x must be 2D (N, N_seq), got: {x_arr.shape}")
    x_centered = x_arr - np.mean(x_arr, axis=1, keepdims=True)
    energy = np.sqrt(np.sum(x_centered ** 2, axis=1, keepdims=True))
    if (not np.all(np.isfinite(x_centered))) or np.any(energy <= _ENERGY_EPS):
        raise RuntimeError(
            "degenerate batch after centering/normalization (zero energy or NaN/Inf)"
        )
    return x_centered / energy

def generate_transmitted_batch(
    config: Dict[str, Any],
    bits: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray:
    data = config["data"]
    seq_len = int(data["sequence_length"])
    map_type = str(data["map_type"])
    map_param = float(data["map_param"])

    bits_arr = np.asarray(bits, dtype=np.int64)
    seeds_arr = np.asarray(seeds, dtype=np.int64)
    if bits_arr.ndim != 1 or seeds_arr.ndim != 1 or bits_arr.shape != seeds_arr.shape:
        raise ValueError(
            f"bits/seeds must be aligned 1D arrays, got: "
            f"{bits_arr.shape}, {seeds_arr.shape}"
        )
    n = int(bits_arr.shape[0])
    if n == 0:
        return np.empty((0, seq_len), dtype=np.float64)

    maps = _map_types_for_bits(map_type, bits_arr)
    out = np.empty((n, seq_len), dtype=np.float64)
    for mt in sorted(_MAP_TYPES):
        idx = np.where(maps == mt)[0]
        if idx.size == 0:
            continue
        x0 = np.array(
            [_x0_from_seed(int(s), mt, map_param) for s in seeds_arr[idx]],
            dtype=np.float64,
        )
        out[idx] = _iterate_map_batch(mt, map_param, x0, seq_len)

    return _center_normalize_batch(out)

def generate_transmitted_batch_fast(
    config: Dict[str, Any],
    bits: np.ndarray,
    rng: np.random.Generator,
    seeds: Optional[np.ndarray] = None,
) -> np.ndarray:
    data = config["data"]
    seq_len = int(data["sequence_length"])
    map_type = str(data["map_type"])
    map_param = float(data["map_param"])

    bits_arr = np.asarray(bits, dtype=np.int64)
    if bits_arr.ndim != 1:
        raise ValueError(f"bits must be 1D, got: {bits_arr.shape}")
    n = int(bits_arr.shape[0])
    if n == 0:
        return np.empty((0, seq_len), dtype=np.float64)

    seeds_arr: Optional[np.ndarray] = None
    if seeds is not None:
        seeds_arr = np.asarray(seeds, dtype=np.int64)
        if seeds_arr.shape != (n,):
            raise ValueError(f"seeds must have shape ({n},), got {seeds_arr.shape}")
        if np.any(seeds_arr < 0):
            raise ValueError("seeds must be non-negative")

    maps = _map_types_for_bits(map_type, bits_arr)
    out = np.empty((n, seq_len), dtype=np.float64)
    for mt in sorted(_MAP_TYPES):
        idx = np.where(maps == mt)[0]
        if idx.size == 0:
            continue
        if seeds_arr is None:
            x0 = rng.uniform(_X0_MIN, _X0_MAX, size=idx.size)
        else:
            x0 = np.array(
                [
                    _x0_from_seed(int(seed), mt, float(map_param))
                    for seed in seeds_arr[idx]
                ],
                dtype=np.float64,
            )
        if mt == "logistic":
            forbidden = (0.25, 0.5, 0.75, 1.0 - 1.0 / float(map_param))
        else:
            forbidden = (0.0, 0.5, 1.0)
        for point in forbidden:
            near = np.abs(x0 - point) < _FIXED_POINT_TOL
            if np.any(near):
                x0 = x0 + np.where(near, _FIXED_POINT_TOL * 10.0, 0.0)
        out[idx] = _iterate_map_batch(mt, map_param, x0, seq_len)

    return _center_normalize_batch(out)

def _apply_echo_fading(
    alphas: np.ndarray,
    valid: np.ndarray,
    echo_fading: str,
    kappa_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    alphas = np.asarray(alphas, dtype=np.float64)
    if alphas.ndim != 2:
        raise ValueError(f"alphas must be 2D (N, K), got: {alphas.shape}")
    n, k = alphas.shape
    if k == 0:
        return np.zeros((n, 0), dtype=np.complex128)
    if echo_fading == "rayleigh":
        gains = (
            rng.standard_normal((n, k)) + 1j * rng.standard_normal((n, k))
        ) / math.sqrt(2.0)
    elif echo_fading == "rician":
        kappa_lin = 10.0 ** (kappa_db / 10.0)
        los = math.sqrt(kappa_lin / (kappa_lin + 1.0))
        scat = math.sqrt(1.0 / (kappa_lin + 1.0))
        theta = rng.uniform(0.0, 2.0 * math.pi, size=(n, k))
        diffuse = (
            rng.standard_normal((n, k)) + 1j * rng.standard_normal((n, k))
        ) / math.sqrt(2.0)
        gains = los * np.exp(1j * theta) + scat * diffuse
    else:
        raise ValueError(f"invalid echo_fading: {echo_fading!r}")

    mask = np.asarray(valid, dtype=bool)
    if mask.shape != gains.shape:
        mask = np.ones(gains.shape, dtype=bool)

    faded = alphas * gains
    nominal = np.sum(np.where(mask, alphas ** 2, 0.0), axis=1, keepdims=True)
    realized = np.sum(np.where(mask, np.abs(faded) ** 2, 0.0), axis=1, keepdims=True)
    scale = np.sqrt(nominal / np.where(realized < _FADING_POWER_EPS, 1.0, realized))
    faded = faded * scale
    faded = np.where(mask, faded, 0.0)
    if not np.all(np.isfinite(faded)):
        raise RuntimeError("echo fading output not finite (NaN/Inf)")
    return faded

def _slot_representatives(slot_ids: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(slot_ids)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"slot_ids must have shape ({n},), got: {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"slot_ids must be integers, got: {arr.dtype}")
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if np.any(arr < 0):
        raise ValueError("slot_ids must be non-negative")
    order = np.argsort(arr, kind="stable")
    ordered = arr[order]
    first = np.empty(ordered.shape, dtype=bool)
    first[0] = True
    if ordered.size > 1:
        first[1:] = ordered[1:] != ordered[:-1]
    first_positions = order[first]
    unique_first = arr[first_positions]
    positions = np.searchsorted(unique_first, arr)
    reps = first_positions[positions].astype(np.int64)
    if not np.all(arr[reps] == arr):
        raise RuntimeError("slot representative mapping is inconsistent")
    return reps

def apply_channel_batch(
    x_norm: np.ndarray,
    k: int,
    snr_db: float,
    config: Dict[str, Any],
    rng: np.random.Generator,
    slot_ids: Optional[np.ndarray] = None,
    hop_channels: Optional[np.ndarray] = None,
    return_peer_info: bool = False,
) -> Tuple:
    x_arr = np.asarray(x_norm, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"x_norm must be 2D (N, N_seq), got: {x_arr.shape}")
    if isinstance(k, bool) or int(k) < 0:
        raise ValueError(f"k must be an int >= 0, got: {k!r}")
    k = int(k)
    if not math.isfinite(float(snr_db)):
        raise ValueError(f"snr_db must be finite, got: {snr_db!r}")

    data = config["data"]
    seq_len = int(data["sequence_length"])
    max_delay = int(data["max_delay"])
    max_doppler = float(data["max_doppler"])
    alpha_min = float(data["alpha_min"])
    alpha_max = float(data["alpha_max"])
    kappa_db = channel_direct_kappa_db(config)
    doppler_direct_max = float(data["doppler_direct_max"])
    echo_fading = channel_echo_fading(config)
    echo_fading_kappa_db = channel_echo_fading_kappa_db(config)
    hold_mode = channel_hold_mode(config)
    poisson_echoes = channel_num_echoes_mode(config) == "poisson"

    n = int(x_arr.shape[0])
    if int(x_arr.shape[1]) != seq_len:
        raise ValueError(
            f"x_norm second dimension must equal sequence_length={seq_len}, "
            f"got: {x_arr.shape[1]}"
        )
    if poisson_echoes:
        counts = sample_echo_counts(n, rng, config)
        k_max = int(counts.max()) if counts.size else 0
    else:
        counts = np.full(n, k, dtype=np.int64)
        k_max = k
    if k_max > max_delay:
        raise ValueError(
            f"k ({k_max}) > max_delay ({max_delay}): cannot sample "
            f"{k_max} distinct integer delays in [1, max_delay]"
        )
    if hold_mode != "per_symbol" and slot_ids is None:
        raise ValueError(f"channel.hold_mode is {hold_mode!r} but slot_ids is None")
    if hold_mode == "per_hop" and hop_channels is None:
        raise ValueError("channel.hold_mode is 'per_hop' but hop_channels is None")

    n_idx = np.arange(seq_len, dtype=np.float64)[None, :]
    n_idx_i = np.arange(seq_len, dtype=np.int64)[None, :]

    generator = echo_generator(config)
    k_geometric = k_max
    obstacle_alpha = np.zeros(n, dtype=np.float64)
    obstacle_distance_m = np.zeros(n, dtype=np.float64)
    echo_power_fraction = None
    if k_max > 0:
        if generator == "geometry":
            scene_taps = sample_scene_taps(n, rng, config, max_delay, max(0, k_max - 1))
            taus = scene_taps["taus"].astype(np.int64)
            dopplers = scene_taps["dopplers"]
            alphas = scene_taps["alphas"]
            k_geometric = int(scene_taps["k_geometric"])
            obstacle_alpha = scene_taps["obstacle_alpha"]
            obstacle_distance_m = scene_taps["obstacle_distance_m"]
            echo_power_fraction = float(scene_taps["echo_power_fraction"])
            valid = np.ones(taus.shape, dtype=bool)
            echo_power = np.sum(alphas ** 2, axis=1)
        else:
            pool = np.tile(np.arange(1, max_delay + 1, dtype=np.int64)[None, :], (n, 1))
            taus = rng.permuted(pool, axis=1)[:, :k_max]
            dopplers = rng.uniform(0.0, max_doppler, size=(n, k_max))
            if bool(data.get("alpha_tau_coupling", False)):
                floor = float(data.get("alpha_floor", _ALPHA_COUPLING_FLOOR))
                alphas = np.clip(alpha_max / np.maximum(taus.astype(np.float64), 1.0),
                                 floor, alpha_max)
            else:
                alphas = 10.0 ** rng.uniform(
                    math.log10(alpha_min), math.log10(alpha_max), size=(n, k_max)
                )
            valid = np.arange(k_max, dtype=np.int64)[None, :] < counts[:, None]
            if bool(np.all(valid)):
                echo_power = np.sum(alphas ** 2, axis=1)
            else:
                echo_power = np.sum(np.where(valid, alphas ** 2, 0.0), axis=1)
        if np.any(echo_power >= 1.0 - _POWER_EPS):
            raise ValueError("sum of alpha_k^2 >= 1: normalization Eq. (4) is impossible")
    else:
        taus = np.empty((n, 0), dtype=np.int64)
        dopplers = np.empty((n, 0), dtype=np.float64)
        alphas = np.empty((n, 0), dtype=np.float64)
        valid = np.zeros((n, 0), dtype=bool)
        echo_power = np.zeros(n, dtype=np.float64)

    kappa_lin = 10.0 ** (kappa_db / 10.0)
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    g = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2.0)
    h_c = (
        math.sqrt(kappa_lin / (kappa_lin + 1.0)) * np.exp(1j * theta)
        + math.sqrt(1.0 / (kappa_lin + 1.0)) * g
    )
    f_dc = rng.uniform(0.0, doppler_direct_max, size=n)

    peer_distances, peer_taus, peer_alphas, peer_dopplers = _sample_peer_taps(
        config, rng, n, max_delay
    )
    n_peers = int(peer_taus.shape[1])
    is_peer = None
    peers_cfg = _peer_config(config)
    if n_peers > 0:
        n_obstacle = int(taus.shape[1])
        taus = np.concatenate([taus, peer_taus], axis=1)
        dopplers = np.concatenate([dopplers, peer_dopplers], axis=1)
        alphas = np.concatenate([alphas, peer_alphas], axis=1)
        valid = np.concatenate([valid, np.ones((n, n_peers), dtype=bool)], axis=1)
        echo_power = echo_power + np.sum(peer_alphas ** 2, axis=1)
        if echo_power_fraction is not None:
            total_power = np.sum(np.where(valid, alphas ** 2, 0.0), axis=1, keepdims=True)
            rescale = np.sqrt(
                echo_power_fraction / np.maximum(total_power, _POWER_EPS)
            )
            alphas = alphas * rescale
            echo_power = np.sum(np.where(valid, alphas ** 2, 0.0), axis=1)
        if np.any(echo_power >= 1.0 - _POWER_EPS):
            raise ValueError(
                "sum of alpha_k^2 >= 1 with the peer taps: normalization Eq. (4) "
                "is impossible"
            )
        is_peer = np.concatenate(
            [
                np.zeros((n, n_obstacle), dtype=bool),
                np.ones((n, n_peers), dtype=bool),
            ],
            axis=1,
        )

    labeled = np.zeros(np.asarray(taus).shape, dtype=bool)
    if np.asarray(taus).shape[1] > 0:
        labeled[:, : int(k_geometric)] = True
    if echo_fading == "none":
        amplitudes = alphas
        dominance = np.where(labeled, alphas, LABEL_EXCLUDED)
    else:
        amplitudes = _apply_echo_fading(
            alphas, valid, echo_fading, echo_fading_kappa_db, rng
        )
        dominance = np.where(labeled, np.abs(amplitudes), LABEL_EXCLUDED)
    if is_peer is not None:
        dominance = np.where(is_peer, LABEL_EXCLUDED, dominance)
    if generator == "geometry" and np.asarray(amplitudes).shape[1] > 0:
        obstacle_alpha = np.abs(np.asarray(amplitudes))[:, 0].astype(np.float64)

    geometry = TapGeometry(
        taus=np.asarray(taus, dtype=np.float64),
        dopplers=np.asarray(dopplers, dtype=np.float64),
        amplitudes=np.asarray(amplitudes, dtype=np.complex128),
        dominance=np.asarray(dominance, dtype=np.float64),
        valid=np.asarray(valid, dtype=bool),
        echo_power=np.asarray(echo_power, dtype=np.float64),
        k_geometric=int(k_max),
        is_peer=None if is_peer is None else np.asarray(is_peer, dtype=bool),
    )
    geometry = apply_overlay(
        config, geometry, rng, slot_ids=slot_ids, hop_channels=hop_channels
    )
    reps = _hold_representatives(hold_mode, slot_ids, hop_channels, n, config)
    if reps is not None:
        geometry = geometry.with_selection(reps)
        counts = counts[reps]
        h_c = h_c[reps]
        f_dc = f_dc[reps]
        obstacle_alpha = obstacle_alpha[reps]
        obstacle_distance_m = obstacle_distance_m[reps]
    taus = geometry.taus
    dopplers = geometry.dopplers
    amplitudes = geometry.amplitudes
    dominance = geometry.dominance
    valid = geometry.valid
    echo_power = geometry.echo_power
    k_max = geometry.k_total

    h_c_eff = h_c * np.sqrt(1.0 - echo_power)

    y = h_c_eff[:, None] * x_arr * np.exp(1j * 2.0 * math.pi * f_dc[:, None] * n_idx)
    integer_delays = bool(np.all(taus == np.round(taus)))
    for j in range(k_max):
        if integer_delays:
            delay_idx = n_idx_i - taus[:, j, None].astype(np.int64)
            delay_valid = delay_idx >= 0
            clip_idx = np.clip(delay_idx, 0, seq_len - 1)
            x_delayed = np.where(
                delay_valid, np.take_along_axis(x_arr, clip_idx, axis=1), 0.0
            )
        else:
            delay_low = np.floor(taus[:, j, None])
            delay_frac = taus[:, j, None] - delay_low
            low_idx = delay_low.astype(np.int64)
            x_low = np.where(
                low_idx >= 0,
                np.take_along_axis(x_arr, np.clip(low_idx, 0, seq_len - 1), axis=1),
                0.0,
            )
            x_high = np.where(
                low_idx + 1 >= 0,
                np.take_along_axis(x_arr, np.clip(low_idx + 1, 0, seq_len - 1), axis=1),
                0.0,
            )
            x_delayed = (1.0 - delay_frac) * x_low + delay_frac * x_high
        phase = np.exp(1j * 2.0 * math.pi * dopplers[:, j, None] * n_idx)
        term = amplitudes[:, j, None] * x_delayed * phase
        if bool(np.all(valid[:, j])):
            y = y + term
        else:
            y = y + np.where(valid[:, j, None], term, 0.0)

    signal_power = np.mean(np.abs(y) ** 2, axis=1)
    if np.any(signal_power <= 0.0) or not np.all(np.isfinite(signal_power)):
        bad = np.flatnonzero(~np.isfinite(signal_power) | (signal_power <= 0.0))
        raise RuntimeError(
            "invalid signal power (degenerate symbol) at rows "
            f"{bad[:8].tolist()} of {n}: power in "
            f"[{float(np.nanmin(signal_power)):.3e}, {float(np.nanmax(signal_power)):.3e}], "
            f"echo power in [{float(np.min(echo_power)):.3e}, {float(np.max(echo_power)):.3e}], "
            f"|h_c| in [{float(np.min(np.abs(h_c))):.3e}, {float(np.max(np.abs(h_c))):.3e}], "
            f"taps={k_max}"
        )
    noise_var = signal_power * 10.0 ** (-float(snr_db) / 10.0)
    w = np.sqrt(noise_var[:, None] / 2.0) * (
        rng.standard_normal((n, seq_len)) + 1j * rng.standard_normal((n, seq_len))
    )
    y = y + w

    direct_interference_power = np.zeros(n, dtype=np.float64)
    if bool(peers_cfg.get("direct_interference", False)) and n_peers > 0:
        gain_ref = float(peers_cfg.get("direct_gain_ref", 1.0))
        if not (0.0 <= gain_ref <= 100.0):
            raise ValueError(
                f"peers.direct_gain_ref must be in [0, 100], got: {gain_ref!r}"
            )
        reference_range = float(peers_cfg.get("peer_range_ref_m", 30.0))
        for column in range(n_peers):
            peer_seeds = drone_symbol_seeds(column + 1, n)
            x_peer = generate_transmitted_batch_fast(
                config, peer_seeds % 2, rng, seeds=peer_seeds
            )
            lag = np.clip(
                np.rint(peer_taus[:, column]).astype(np.int64), 0, seq_len - 1
            )
            shifted_index = n_idx_i - lag[:, None]
            shifted = np.where(
                shifted_index >= 0,
                np.take_along_axis(
                    x_peer, np.clip(shifted_index, 0, seq_len - 1), axis=1
                ),
                0.0,
            )
            gain = gain_ref * (
                reference_range / np.maximum(peer_distances[:, column], 1e-9)
            )
            phase = np.exp(
                1j * 2.0 * math.pi * peer_dopplers[:, column, None] * n_idx
            )
            term = gain[:, None] * shifted * phase
            y = y + term
            direct_interference_power += np.mean(np.abs(term) ** 2, axis=1)

    if not np.all(np.isfinite(y)):
        raise RuntimeError("channel output not finite (NaN/Inf)")

    if k_max > 0:
        dominance_masked = np.where(valid, dominance, -np.inf)
        dom_idx = np.argmax(dominance_masked, axis=1)
        tau_labels = taus[np.arange(n), dom_idx].astype(np.float64)
        f_d_labels = dopplers[np.arange(n), dom_idx].astype(np.float64)
        no_echo = counts <= 0
        if bool(np.any(no_echo)):
            tau_labels[no_echo] = 0.0
            f_d_labels[no_echo] = 0.0
    else:
        tau_labels = np.zeros(n, dtype=np.float64)
        f_d_labels = np.zeros(n, dtype=np.float64)

    if not bool(return_peer_info):
        return y, tau_labels, f_d_labels
    peer_info = {
        "peer_taus": np.asarray(peer_taus, dtype=np.float64),
        "peer_distances_m": np.asarray(peer_distances, dtype=np.float64),
        "peer_alphas": np.asarray(peer_alphas, dtype=np.float64),
        "peer_dopplers": np.asarray(peer_dopplers, dtype=np.float64),
        "is_peer": None if is_peer is None else np.asarray(is_peer, dtype=bool),
        "direct_interference_power": np.asarray(
            direct_interference_power, dtype=np.float64
        ),
        "obstacle_alpha": np.asarray(obstacle_alpha, dtype=np.float64),
        "obstacle_distance_m": np.asarray(obstacle_distance_m, dtype=np.float64),
    }
    return y, tau_labels, f_d_labels, peer_info

def generate_test_batch(
    config: Dict[str, Any],
    num_symbols: int,
    snr_db: float,
    k: int,
    rng: np.random.Generator,
    slot_ids: Optional[np.ndarray] = None,
    slot_offset: int = 0,
    hop_channels: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    if isinstance(num_symbols, bool) or int(num_symbols) <= 0:
        raise ValueError(
            f"num_symbols must be an int > 0, got: {num_symbols!r}"
        )
    n = int(num_symbols)
    if isinstance(k, bool) or int(k) < 0:
        raise ValueError(f"k must be an int >= 0, got: {k!r}")
    if isinstance(slot_offset, bool) or int(slot_offset) < 0:
        raise ValueError(f"slot_offset must be an int >= 0, got: {slot_offset!r}")

    if slot_ids is None and channel_hold_mode(config) in ("per_slot", "per_hop"):
        dwell = channel_hold_dwell(config)
        slot_ids = (int(slot_offset) + np.arange(n, dtype=np.int64)) // dwell
    if channel_hold_mode(config) == "per_hop" and hop_channels is None:
        hop_channels = build_hop_sequence(n, hop_config(config))

    bits = rng.integers(0, 2, size=n, dtype=np.int64)
    peers_cfg = _peer_config(config)
    per_drone = (
        bool(peers_cfg.get("enable", False))
        and str(peers_cfg.get("sequence_id_scheme", "shared")) == "per_drone"
    )
    if per_drone:
        seeds = drone_symbol_seeds(0, n)
        x_ref = generate_transmitted_batch_fast(config, bits, rng, seeds=seeds)
    else:
        seeds = rng.integers(0, _SEED_HIGH, size=n, dtype=np.int64)
        x_ref = generate_transmitted_batch_fast(config, bits, rng, seeds=seeds)
    y, tau, f_d, peer_info = apply_channel_batch(
        x_ref,
        int(k),
        float(snr_db),
        config,
        rng,
        slot_ids=slot_ids,
        hop_channels=hop_channels,
        return_peer_info=True,
    )
    batch = {
        "x": y,
        "bit": bits,
        "tau": tau,
        "f_d": f_d,
        "seed": seeds,
        "x_ref": x_ref.astype(np.float32),
        "obstacle_alpha": peer_info["obstacle_alpha"],
        "obstacle_distance_m": peer_info["obstacle_distance_m"],
    }
    if peer_info["is_peer"] is not None:
        batch.update(
            {
                "peer_taus": peer_info["peer_taus"],
                "peer_distances_m": peer_info["peer_distances_m"],
                "peer_alphas": peer_info["peer_alphas"],
                "peer_dopplers": peer_info["peer_dopplers"],
                "is_peer": peer_info["is_peer"],
                "direct_interference_power": peer_info["direct_interference_power"],
            }
        )
    return batch

def generate_dataset(
    config: Dict[str, Any],
    split: str,
    output_dir: Path,
) -> List[Path]:
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {sorted(_SPLITS)}, got: {split!r}")
    _validate_config(config)
    data = config["data"]
    snr_grid = build_snr_grid(data["snr_range"], data["snr_step"])
    k_list = [int(k) for k in data["echoes"]]
    num_combos = len(snr_grid) * len(k_list)

    n_per_combo = resolve_n_per_combo(config, split, num_combos)

    max_delay = int(data["max_delay"])
    max_doppler = float(data["max_doppler"])

    rng_root = np.random.default_rng(int(config["general"]["seed"]))
    combo_seeds = rng_root.integers(0, _SEED_HIGH, size=num_combos)

    hold_mode = channel_hold_mode(config)
    slot_ids = None
    hop_channels = None
    if hold_mode != "per_symbol":
        slot_ids = np.arange(n_per_combo, dtype=np.int64) // channel_hold_dwell(config)
        if hold_mode == "per_hop":
            hop_channels = build_hop_sequence(n_per_combo, hop_config(config))

    paths: List[Path] = []
    combo_index = 0

    for snr in snr_grid:
        for k in k_list:
            rng = np.random.default_rng(int(combo_seeds[combo_index]))
            combo_index += 1

            bits = np.array([0] * (n_per_combo // 2) + [1] * (n_per_combo // 2), dtype=np.uint8)
            rng.shuffle(bits)

            seed_sym = rng.integers(0, _SEED_HIGH, size=n_per_combo, dtype=np.int64)
            x_ref = generate_transmitted_batch(config, bits, seed_sym)
            x_arr, tau, f_d = apply_channel_batch(
                x_ref,
                k,
                float(snr),
                config,
                rng,
                slot_ids=slot_ids,
                hop_channels=hop_channels,
            )

            if not np.all(np.isfinite(x_arr)):
                raise RuntimeError(f"array x not finite for {split} snr={snr} k={k}")
            if np.any(tau < 0.0) or np.any(tau > float(max_delay)):
                raise RuntimeError(f"label tau out of range for {split} snr={snr} k={k}")
            if np.any(f_d < 0.0) or np.any(f_d > max_doppler):
                raise RuntimeError(f"label f_d out of range for {split} snr={snr} k={k}")
            if not np.all(np.isin(bits, (0, 1))):
                raise RuntimeError(f"bit outside 0/1 for {split} snr={snr} k={k}")

            file_name = f"{split}_snr{snr:g}_echo{k}.npz"
            out_path = Path(output_dir) / file_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                out_path,
                x=x_arr,
                bit=bits,
                tau=tau,
                f_d=f_d,
                snr_db=float(snr),
                k=k,
                seed=seed_sym,
            )
            logger.info(
                "saved %s: %d symbols, x shape=%s, tau in [%.0f, %.0f], f_d in [%.3e, %.3e]",
                out_path, n_per_combo, x_arr.shape,
                float(np.min(tau)), float(np.max(tau)),
                float(np.min(f_d)), float(np.max(f_d)),
            )
            paths.append(out_path)

    logger.info(
        "split '%s': %d files generated, %d symbols per combination (combos=%d)",
        split, len(paths), n_per_combo, num_combos,
    )
    return paths

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Dataset generator for the Ultra-CAN (ISAC in IoD)"
    )
    parser.add_argument("--config", required=True, help="path to the experiment config (YAML)")
    parser.add_argument(
        "--splits", default="train,val,test", help="splits to generate, comma-separated"
    )
    parser.add_argument(
        "--output-dir", default=None, help="override the output directory (default: data.raw_dir)"
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path=config_path, base_config_path=DEFAULT_BASE_CONFIG_PATH)
    _validate_config(config)

    from src.utils.dataset_utils import get_dataset_dir

    experiment_name = str(config["general"].get("experiment_name", "ultra_can_isac"))
    output_dir = Path(args.output_dir) if args.output_dir else get_dataset_dir(config)
    log_dir = (
        output_dir / "logs"
        if args.output_dir
        else _REPO_ROOT / "results" / experiment_name / "logs"
    )
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

    logger.info(
        "starting the dataset generation: splits=%s, output_dir=%s, seed=%s",
        splits, output_dir, config["general"]["seed"],
    )
    for split in splits:
        generate_dataset(config, split, output_dir)
    logger.info("dataset generation completed for %s", splits)

if __name__ == "__main__":
    main()

