
from __future__ import annotations

from typing import Dict, Tuple

CANONICAL_MODELS: Tuple[str, ...] = ("conv1d", "qkv", "lstm", "mc_dlsk")

REFERENCE_MODELS: Tuple[str, ...] = ("blind_stat", "dcsk")

_ALIASES: Dict[str, str] = {
    "conv1d": "conv1d",
    "conv_1d": "conv1d",
    "ultra_can": "conv1d",
    "ultra_can_conv1d": "conv1d",
    "qkv": "qkv",
    "qkv_attention": "qkv",
    "ultra_can_qkv": "qkv",
    "lstm": "lstm",
    "lstm_baseline": "lstm",
    "lstm_ofdm_dcsk": "lstm",
    "mc_dlsk": "mc_dlsk",
    "mc_dlcsk": "mc_dlsk",
    "mc_dlcs": "mc_dlsk",
    "mc_dlsk_baseline": "mc_dlsk",
    "mc_dlcsk_baseline": "mc_dlsk",
    "blind_stat": "blind_stat",
    "blind_stats": "blind_stat",
    "dcsk": "dcsk",
    "dcsk_correlator": "dcsk",
}

_DISPLAY: Dict[str, str] = {
    "conv1d": "Ultra-CAN (Conv1D)",
    "qkv": "Ultra-CAN (QKV)",
    "lstm": "LSTM-OFDM-DCSK",
    "mc_dlsk": "MC-DLCSK",
    "blind_stat": "Blind statistical",
    "dcsk": "DCSK correlator",
    "dcsk_correlator": "DCSK correlator",
}

def _normalise(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")

def canonical_model_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"model name must be a string, got: {type(name).__name__}")
    key = _normalise(name)
    if key in _ALIASES:
        return _ALIASES[key]
    if key in CANONICAL_MODELS:
        return key
    raise ValueError(
        f"unknown model name: {name!r} (accepted: {sorted(_ALIASES)})"
    )

def display_model_name(name: str) -> str:
    try:
        return _DISPLAY.get(canonical_model_name(name), str(name))
    except (TypeError, ValueError):
        return _DISPLAY.get(str(name), str(name))

def canonical_or_none(name: object) -> str:
    if not isinstance(name, str):
        return name
    try:
        return canonical_model_name(name)
    except ValueError:
        return name
