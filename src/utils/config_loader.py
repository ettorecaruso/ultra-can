
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import yaml

logger = logging.getLogger(__name__)

DEFAULT_BASE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "base_config.yaml"

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def _iter_config_items(config: Any, prefix: str = "config") -> Iterator[Tuple[str, Any]]:
    
    if isinstance(config, dict):
        for key, value in config.items():
            path = f"{prefix}.{key}"
            yield from _iter_config_items(value, path)
    elif isinstance(config, (list, tuple)):
        for index, value in enumerate(config):
            path = f"{prefix}[{index}]"
            yield from _iter_config_items(value, path)
    else:
        yield prefix, config

def load_config(
    config_path: Path,
    base_config_path: Path = DEFAULT_BASE_CONFIG_PATH,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    
    base_path = Path(base_config_path)
    exp_path = Path(config_path)
    for path, label in ((base_path, "base_config"), (exp_path, "experiment config")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    try:
        with open(base_path, "r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh)
        with open(exp_path, "r", encoding="utf-8") as fh:
            experiment = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML file: {exc}") from exc
    if not isinstance(base, dict):
        raise ValueError(f"{base_path} does not contain a top-level dict")
    if not isinstance(experiment, dict):
        raise ValueError(f"{exp_path} does not contain a top-level dict")

    merged = _deep_merge(base, experiment)
    if cli_overrides is not None:
        if not isinstance(cli_overrides, dict):
            raise TypeError(
                f"cli_overrides must be a dict, got: {type(cli_overrides).__name__}"
            )
        merged = _deep_merge(merged, cli_overrides)

    logger.debug("config loaded: %s <- %s (CLI overrides: %s)", base_path, exp_path, bool(cli_overrides))
    return merged

def _key_exists(config: Dict[str, Any], dotted_path: str) -> bool:
    
    node: Any = config
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True

def validate_config(config: Dict[str, Any], required_keys: Sequence[str]) -> None:
    
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got: {type(config).__name__}")
    missing = [key for key in required_keys if not _key_exists(config, key)]
    if missing:
        raise ValueError(f"required keys missing in the config: {missing}")
    logger.debug("validate_config OK: %d keys checked", len(required_keys))

def _coerce_cli_value(raw_value: str) -> Any:
    
    value = raw_value.strip()
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value

def parse_cli_overrides(raw: Sequence[str]) -> Dict[str, Any]:
    
    overrides: Dict[str, Any] = {}
    for token in raw:
        item = token.strip()
        if not item or item == "--set":
            continue
        if item.startswith("--set="):
            item = item[len("--set="):]
        elif item.startswith("--set "):
            item = item[len("--set "):].strip()
        if "=" not in item:
            raise ValueError(f"malformed CLI override (expected path.to.key=value): {token!r}")
        dotted_path, raw_value = item.split("=", 1)
        parts = [part for part in dotted_path.split(".") if part]
        if not parts:
            raise ValueError(f"empty override path: {token!r}")
        value = _coerce_cli_value(raw_value)
        node: Dict[str, Any] = overrides
        for part in parts[:-1]:
            child = node.get(part)
            if child is None:
                child = {}
                node[part] = child
            elif not isinstance(child, dict):
                raise ValueError(f"type collision on path {dotted_path!r}")
            node = child
        node[parts[-1]] = value
    logger.debug("parsed CLI overrides: %s", overrides)
    return overrides

def save_config_snapshot(config: Dict[str, Any], output_dir: Path) -> Path:
    
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got: {type(config).__name__}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = output / "config_used.yaml"
    with open(snapshot_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)
    logger.info("config snapshot saved: %s", snapshot_path)
    return snapshot_path

DEFAULT_CHANNELS_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "channels.yaml"
)

DEFAULT_CHANNEL_VARIANT = "nominal"

def load_channels(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    target = Path(path) if path is not None else DEFAULT_CHANNELS_PATH
    if not target.is_file():
        raise FileNotFoundError(f"channels file not found: {target}")
    try:
        with open(target, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid channels YAML: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{target} must contain a non-empty mapping of variants")
    for name, section in data.items():
        if not isinstance(section, dict):
            raise ValueError(
                f"channel variant {name!r} must be a mapping, got {type(section).__name__}"
            )
    return data

def channel_variant(name: str, path: Optional[Path] = None) -> Dict[str, Any]:
    variants = load_channels(path)
    if name not in variants:
        raise KeyError(
            f"unknown channel variant {name!r}; available: {sorted(variants)}"
        )
    return variants[name]

def with_channel_variant(
    config: Dict[str, Any],
    name: Optional[str] = None,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got {type(config).__name__}")
    selected = str(name) if name is not None else DEFAULT_CHANNEL_VARIANT
    section = channel_variant(selected, path)
    merged = _deep_merge(dict(config.get("channel") or {}), section)
    out = dict(config)
    out["channel"] = merged
    out["channel_variant"] = selected
    return out

