
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

_DATASET_VERSION = 5

def get_dataset_dir(config: Dict[str, Any]) -> Path:
    
    data = config["data"]
    echoes = tuple(sorted(data["echoes"]))
    max_delay = data["max_delay"]
    max_doppler = data["max_doppler"]
    snr_range = tuple(data["snr_range"])
    snr_step = data["snr_step"]
    feature_mode = data.get("feature_mode", "real")
    alpha_min = data.get("alpha_min", 1e-6)
    alpha_max = data.get("alpha_max", 0.01)
    coupling = bool(data.get("alpha_tau_coupling", False))
    floor = data.get("alpha_floor", 1e-3)
    map_type = data.get("map_type", "logistic")
    map_param = data.get("map_param")
    sequence_length = data.get("sequence_length")
    kappa_db = data.get("rician_kappa_db")
    doppler_direct = data.get("doppler_direct_max")
    n_train = data.get("num_symbols_train")
    n_val = data.get("num_symbols_val")
    n_test = data.get("num_symbols_test")
    fc_hz = data.get("fc_hz")
    fs_hz = data.get("fs_hz")
    params_str = (
        f"v{_DATASET_VERSION}_e{echoes}_d{max_delay}_D{max_doppler}_S{snr_range}"
        f"_s{snr_step}_f{feature_mode}_a{alpha_min}_{alpha_max}"
        f"_C{coupling}_F{floor}_m{map_type}{map_param}_n{sequence_length}"
        f"_k{kappa_db}_p{doppler_direct}"
        f"_N{n_train}_{n_val}_{n_test}_c{fc_hz}_{fs_hz}"
        f"{_channel_suffix(config)}{_peers_suffix(config)}{_hopping_suffix(config)}"
    )
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    raw_dir = Path(data.get("raw_dir", "data/raw"))
    return raw_dir / params_hash

def _channel_suffix(config: Dict[str, Any]) -> str:
    from src.data.channel_models import (
        DEFAULT_CHANNEL_MODEL,
        fingerprint_of,
        resolved_channel_section,
    )
    from src.utils.config_loader import DEFAULT_CHANNEL_VARIANT, load_channels

    nominal = load_channels()[DEFAULT_CHANNEL_VARIANT]
    nominal.setdefault("model", DEFAULT_CHANNEL_MODEL)
    section = resolved_channel_section(config)
    digest = fingerprint_of(section)
    if digest == fingerprint_of(nominal):
        return ""
    return f"_c{digest}"

def _peers_suffix(config: Dict[str, Any]) -> str:
    from src.data.channel_models import fingerprint_of

    peers = config.get("peers")
    if not isinstance(peers, dict) or not bool(peers.get("enable", False)):
        return ""
    return f"_P{fingerprint_of({str(key): value for key, value in peers.items()})}"

def _hopping_suffix(config: Dict[str, Any]) -> str:
    from dataclasses import asdict

    from src.data.channel_models import fingerprint_of
    from src.data.frequency_hopping import hop_config

    hopping = config.get("frequency_hopping")
    if not isinstance(hopping, dict) or not bool(hopping.get("enable", False)):
        return ""
    resolved = asdict(hop_config(config))
    return f"_H{fingerprint_of({str(key): value for key, value in resolved.items()})}"
