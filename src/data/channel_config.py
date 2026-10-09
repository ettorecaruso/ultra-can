
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

_MISSING = object()

def section(mapping: Mapping[str, Any], key: str, path: str) -> Dict[str, Any]:
    value = mapping.get(key, _MISSING)
    if value is _MISSING:
        raise KeyError(f"missing required configuration section: {path}.{key}")
    if not isinstance(value, dict):
        raise ValueError(
            f"{path}.{key} must be a mapping, got {type(value).__name__}"
        )
    return value

def optional_section(
    mapping: Mapping[str, Any], key: str, path: str
) -> Dict[str, Any]:
    value = mapping.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"{path}.{key} must be a mapping, got {type(value).__name__}"
        )
    return value

def required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    value = mapping.get(key, _MISSING)
    if value is _MISSING or value is None:
        raise KeyError(f"missing required configuration key: {path}.{key}")
    return value

def as_float(mapping: Mapping[str, Any], key: str, path: str) -> float:
    return float(required(mapping, key, path))

def as_int(mapping: Mapping[str, Any], key: str, path: str) -> int:
    value = required(mapping, key, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{key} must be an integer, got {value!r}")
    return int(value)

def as_bool(mapping: Mapping[str, Any], key: str, path: str) -> bool:
    value = required(mapping, key, path)
    if not isinstance(value, bool):
        raise ValueError(f"{path}.{key} must be a boolean, got {value!r}")
    return value

def as_str(mapping: Mapping[str, Any], key: str, path: str) -> str:
    value = required(mapping, key, path)
    if not isinstance(value, str):
        raise ValueError(f"{path}.{key} must be a string, got {value!r}")
    return value

def as_choice(
    mapping: Mapping[str, Any], key: str, path: str, allowed: Sequence[str]
) -> str:
    value = str(as_str(mapping, key, path)).strip().lower()
    if value not in allowed:
        raise ValueError(
            f"{path}.{key} must be one of {sorted(allowed)}, got {value!r}"
        )
    return value

def as_float_or_none(
    mapping: Mapping[str, Any], key: str, path: str
) -> Optional[float]:
    value = mapping.get(key)
    if value is None:
        return None
    return float(value)

def as_float_list(mapping: Mapping[str, Any], key: str, path: str) -> list:
    value = mapping.get(key)
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path}.{key} must be a list, got {type(value).__name__}")
    return [float(item) for item in value]

def frequency_hopping(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    hopping = mapping.get("frequency_hopping")
    if hopping is None:
        return {}
    if not isinstance(hopping, dict):
        raise ValueError("frequency_hopping must be a mapping")
    return hopping

def data_section(config: Mapping[str, Any]) -> Dict[str, Any]:
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("'data' section missing or not a mapping")
    return data
