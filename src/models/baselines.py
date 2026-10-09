
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.heads import (
    _assert_config_finite,
    _as_positive_int,
    _head_input_dim,
    _position_feature,
    _slice_received,
    build_communication_head,
    build_sensing_features,
    build_sensing_head,
    sensing_output_units,
)
from src.models.ultra_can import (
    _SMOKE_BATCH_SIZES,
    _expected_att_length,
    _smoke_check_forward,
    count_trainable_params,
)
from src.utils.config_loader import (
    DEFAULT_BASE_CONFIG_PATH,
    load_config,
)
from src.utils.logger import log_config_summary, setup_logging
from src.utils.model_names import canonical_model_name

logger = logging.getLogger(__name__)

_HEAD_INPUT_DIM = 64
_MODEL_LSTM_NAME = "lstm_baseline"
_MODEL_MC_DLSK_NAME = "mc_dlsk_baseline"
_VALID_BASELINE_NAMES: Tuple[str, ...] = ("lstm", "mc_dlsk")
_VALID_SIZES: Tuple[str, ...] = ("full", "micro")
_VALID_FEATURE_MODES: Tuple[str, ...] = ("real", "iq")
_LSTM_REQUIRED_KEYS: Tuple[str, ...] = ("units", "dropout", "size")
_MC_DLSK_REQUIRED_KEYS: Tuple[str, ...] = ("units", "dropout", "size")

def _validate_baseline_config(
    config: Dict[str, Any],
    name: str,
) -> Tuple[Dict[str, Any], int, int]:
    
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got: {type(config).__name__}")

    baselines_cfg = config.get("baselines")
    if not isinstance(baselines_cfg, dict):
        raise ValueError("'baselines' section missing or not a dict in config")

    enabled = baselines_cfg.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError(f"baselines.enabled must be a bool, got: {enabled!r}")
    if not enabled:
        logger.warning(
            "baselines.enabled is false: baseline building is disabled"
        )
        raise ValueError(
            "baselines.enabled is false: baseline building is disabled"
        )

    if name not in _VALID_BASELINE_NAMES:
        raise ValueError(
            f"invalid baseline: {name!r} (expected: {sorted(_VALID_BASELINE_NAMES)})"
        )

    baseline_cfg = baselines_cfg.get(name)
    if not isinstance(baseline_cfg, dict):
        raise ValueError(f"'baselines.{name}' section missing or not a dict in the config")

    _assert_config_finite(baseline_cfg)

    required = _LSTM_REQUIRED_KEYS if name == "lstm" else _MC_DLSK_REQUIRED_KEYS
    missing = [key for key in required if key not in baseline_cfg]
    if missing:
        raise ValueError(f"missing keys in baselines.{name}: {missing}")

    size = baseline_cfg["size"]
    if not isinstance(size, str) or size not in _VALID_SIZES:
        raise ValueError(
            f"baselines.{name}.size must be one of {list(_VALID_SIZES)}, "
            f"got: {size!r}"
        )

    if name in ("lstm", "mc_dlsk"):
        units = baseline_cfg["units"]
        if not isinstance(units, (list, tuple)) or len(units) == 0:
            raise ValueError(
                f"baselines.{name}.units must be a non-empty list of "
                "positive integers (recurrent layers), got: %r" % (units,)
            )
        for index, unit in enumerate(units):
            _as_positive_int(unit, f"baselines.{name}.units[{index}]")
        dropout = baseline_cfg["dropout"]
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(float(dropout))
            or not (0.0 <= float(dropout) < 1.0)
        ):
            raise ValueError(
                f"baselines.{name}.dropout must be in [0, 1), got: {dropout!r}"
            )

    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("'data' section missing or not a dict in config")
    _assert_config_finite(data_cfg)

    seq_len = _as_positive_int(data_cfg.get("sequence_length"), "data.sequence_length")
    feature_mode = data_cfg.get("feature_mode", "real")
    if not isinstance(feature_mode, str) or feature_mode not in _VALID_FEATURE_MODES:
        raise ValueError(
            f"data.feature_mode must be one of {list(_VALID_FEATURE_MODES)}, "
            f"got: {feature_mode!r}"
        )
    num_features = 1 if feature_mode == "real" else 2

    head_input_dim = _head_input_dim(config)
    if head_input_dim != _HEAD_INPUT_DIM:
        raise ValueError(
            f"the shared heads expect v in R^{_HEAD_INPUT_DIM}, but "
            f"heads._head_input_dim derives {head_input_dim} from the backbone "
            "(model.conv_filters[1]/model.attention_dim): align the config "
            "before building the baselines"
        )

    logger.debug(
        "baseline config '%s' validated: size=%s, seq_len=%d, num_features=%d",
        name,
        size,
        seq_len,
        num_features,
    )
    return baseline_cfg, seq_len, num_features

def build_lstm_baseline(config: Dict[str, Any]) -> tf.keras.Model:
    
    baseline_cfg, seq_len, num_features = _validate_baseline_config(config, "lstm")
    size = str(baseline_cfg["size"])
    units = [int(unit) for unit in baseline_cfg["units"]]
    dropout = float(baseline_cfg["dropout"])

    if size == "micro":
        logger.warning(
            "baselines.lstm.size='micro': the shared heads "
            "built by heads.py are fixed at 10724 params -> the target is not "
            "reachable from this module"
        )

    model_input = tf.keras.layers.Input(
        shape=(seq_len, num_features + 1), name="isac_input"
    )
    received = tf.keras.layers.Lambda(
        _slice_received,
        arguments={"num_features": num_features},
        name="received_slice",
    )(model_input)

    lstm_out = received
    for index, unit in enumerate(units):
        lstm_out = tf.keras.layers.LSTM(
            units=unit,
            return_sequences=True,
            name=f"lstm_{index + 1}",
        )(lstm_out)

    lstm_out = tf.keras.layers.Dropout(rate=dropout, name="dropout_lstm")(lstm_out)

    v, v_sensing = build_sensing_features(lstm_out, config, received_input=model_input)

    lstm_dim = int(units[-1])
    if lstm_dim != _HEAD_INPUT_DIM:
        logger.warning(
            "baselines.lstm.units[-1]=%d != v contract in R^%d: "
            "adding Dense(%d) projection before the shared heads",
            lstm_dim,
            _HEAD_INPUT_DIM,
            _HEAD_INPUT_DIM,
        )
        v = tf.keras.layers.Dense(
            units=_HEAD_INPUT_DIM,
            activation="relu",
            kernel_initializer="he_normal",
            name="lstm_proj",
        )(v)
        sensing_cfg = config.get("model", {}).get("sensing_head") or {}
        use_position = bool(sensing_cfg.get("use_position_feature", False))
        use_profile = bool(sensing_cfg.get("use_reference_profile", True))
        parts: list = [v]
        if use_position:
            v_max = tf.keras.layers.Lambda(_position_feature, name="sensing_position")(lstm_out)
            parts.append(v_max)
        else:
            v_max = tf.keras.layers.GlobalMaxPooling1D(name="sensing_max_pool")(lstm_out)
            parts.append(v_max)
        profile = None
        profile_max_lag = 0
        if use_profile:
            from src.models.heads import (
                _SENSING_PEAK_SHARPNESS_DEFAULT,
                _SENSING_PEAK_SHARPNESS_KEY,
                _SENSING_PROFILE_LAG_KEY,
                _dca_mf_profile,
                _delay_peak_enabled,
                _soft_argmax_lag,
            )

            profile_max_lag = int(sensing_cfg.get(
                _SENSING_PROFILE_LAG_KEY, int(config.get("data", {}).get("max_delay", 33))
            ))
            nf = 2 if str(config.get("data", {}).get("feature_mode", "iq")) == "iq" else 1
            profile = tf.keras.layers.Lambda(
                _dca_mf_profile,
                arguments={"max_delay": profile_max_lag, "num_features": nf},
                name="sensing_delay_profile",
            )(model_input)
            parts.append(profile)
        if profile is not None and _delay_peak_enabled(sensing_cfg):
            parts.append(tf.keras.layers.Lambda(
                _soft_argmax_lag,
                arguments={
                    "max_delay": profile_max_lag,
                    "sharpness": float(sensing_cfg.get(
                        _SENSING_PEAK_SHARPNESS_KEY, _SENSING_PEAK_SHARPNESS_DEFAULT
                    )),
                },
                name="sensing_delay_peak",
            )(profile))
        v_sensing = tf.keras.layers.Concatenate(name="sensing_features")(parts)

    comm_head = build_communication_head(config)
    sensing_head = build_sensing_head(config, input_dim=int(v_sensing.shape[-1]))
    comm = comm_head(v)
    sensing = sensing_head(v_sensing)
    modulation_order = int(comm_head.output_shape[-1])

    if tuple(v.shape) != (None, _HEAD_INPUT_DIM):
        raise ValueError(
            f"expected v shape (None, {_HEAD_INPUT_DIM}), got: {tuple(v.shape)}"
        )
    if tuple(comm.shape) != (None, modulation_order):
        raise ValueError(
            f"expected comm shape (None, {modulation_order}), "
            f"got: {tuple(comm.shape)}"
        )
    sensing_units = sensing_output_units(config)
    if tuple(sensing.shape) != (None, sensing_units):
        raise ValueError(
            f"expected sensing shape (None, {sensing_units}), "
            f"got: {tuple(sensing.shape)}"
        )

    model = tf.keras.Model(
        inputs=model_input,
        outputs={"comm": comm, "sensing": sensing},
        name=_MODEL_LSTM_NAME,
    )
    _smoke_check_forward(
        model,
        _SMOKE_BATCH_SIZES,
        expected={
            "comm": (None, modulation_order),
            "sensing": (None, sensing_units),
        },
    )
    comm_params = count_trainable_params(comm_head)
    sensing_params = count_trainable_params(sensing_head)
    total_params = count_trainable_params(model)
    logger.info(
        "model '%s' built: backbone=%d params, comm=%d params, "
        "sensing=%d params, TOTAL=%d params",
        _MODEL_LSTM_NAME,
        total_params - comm_params - sensing_params,
        comm_params,
        sensing_params,
        total_params,
    )
    return model

def build_mc_dlsk_baseline(config: Dict[str, Any]) -> tf.keras.Model:
    
    baseline_cfg, seq_len, num_features = _validate_baseline_config(config, "mc_dlsk")
    size = str(baseline_cfg["size"])
    units = [int(unit) for unit in baseline_cfg["units"]]
    dropout = float(baseline_cfg["dropout"])

    if size == "micro":
        logger.warning(
            "baselines.mc_dlsk.size='micro': the shared "
            "heads built by heads.py are fixed -> the target is not "
            "reachable from this module"
        )

    model_input = tf.keras.layers.Input(
        shape=(seq_len, num_features + 1), name="isac_input"
    )
    received = tf.keras.layers.Lambda(
        _slice_received,
        arguments={"num_features": num_features},
        name="received_slice",
    )(model_input)

    mc_out = received
    for index, unit in enumerate(units):
        mc_out = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=unit, return_sequences=True),
            name=f"mc_bilstm_{index + 1}",
        )(mc_out)

    mc_out = tf.keras.layers.Dropout(rate=dropout, name="dropout_mc")(mc_out)

    v, v_sensing = build_sensing_features(mc_out, config, received_input=model_input)

    mc_dim = int(2 * units[-1])
    if mc_dim != _HEAD_INPUT_DIM:
        logger.warning(
            "baselines.mc_dlsk: 2*units[-1]=%d != v contract in R^%d: "
            "adding Dense(%d) projection before the shared heads",
            mc_dim,
            _HEAD_INPUT_DIM,
            _HEAD_INPUT_DIM,
        )
        v = tf.keras.layers.Dense(
            units=_HEAD_INPUT_DIM,
            activation="relu",
            kernel_initializer="he_normal",
            name="mc_proj",
        )(v)
        sensing_cfg = config.get("model", {}).get("sensing_head") or {}
        use_position = bool(sensing_cfg.get("use_position_feature", False))
        use_profile = bool(sensing_cfg.get("use_reference_profile", True))
        parts: list = [v]
        if use_position:
            v_max = tf.keras.layers.Lambda(_position_feature, name="sensing_position")(mc_out)
            parts.append(v_max)
        else:
            v_max = tf.keras.layers.GlobalMaxPooling1D(name="sensing_max_pool")(mc_out)
            parts.append(v_max)
        profile = None
        profile_max_lag = 0
        if use_profile:
            from src.models.heads import (
                _SENSING_PEAK_SHARPNESS_DEFAULT,
                _SENSING_PEAK_SHARPNESS_KEY,
                _SENSING_PROFILE_LAG_KEY,
                _dca_mf_profile,
                _delay_peak_enabled,
                _soft_argmax_lag,
            )

            profile_max_lag = int(sensing_cfg.get(
                _SENSING_PROFILE_LAG_KEY, int(config.get("data", {}).get("max_delay", 33))
            ))
            nf = 2 if str(config.get("data", {}).get("feature_mode", "iq")) == "iq" else 1
            profile = tf.keras.layers.Lambda(
                _dca_mf_profile,
                arguments={"max_delay": profile_max_lag, "num_features": nf},
                name="sensing_delay_profile",
            )(model_input)
            parts.append(profile)
        if profile is not None and _delay_peak_enabled(sensing_cfg):
            parts.append(tf.keras.layers.Lambda(
                _soft_argmax_lag,
                arguments={
                    "max_delay": profile_max_lag,
                    "sharpness": float(sensing_cfg.get(
                        _SENSING_PEAK_SHARPNESS_KEY, _SENSING_PEAK_SHARPNESS_DEFAULT
                    )),
                },
                name="sensing_delay_peak",
            )(profile))
        v_sensing = tf.keras.layers.Concatenate(name="sensing_features")(parts)

    comm_head = build_communication_head(config)
    sensing_head = build_sensing_head(config, input_dim=int(v_sensing.shape[-1]))
    comm = comm_head(v)
    sensing = sensing_head(v_sensing)
    modulation_order = int(comm_head.output_shape[-1])

    if tuple(v.shape) != (None, _HEAD_INPUT_DIM):
        raise ValueError(
            f"expected v shape (None, {_HEAD_INPUT_DIM}), got: {tuple(v.shape)}"
        )
    if tuple(comm.shape) != (None, modulation_order):
        raise ValueError(
            f"expected comm shape (None, {modulation_order}), "
            f"got: {tuple(comm.shape)}"
        )
    sensing_units = sensing_output_units(config)
    if tuple(sensing.shape) != (None, sensing_units):
        raise ValueError(
            f"expected sensing shape (None, {sensing_units}), "
            f"got: {tuple(sensing.shape)}"
        )

    model = tf.keras.Model(
        inputs=model_input,
        outputs={"comm": comm, "sensing": sensing},
        name=_MODEL_MC_DLSK_NAME,
    )
    _smoke_check_forward(
        model,
        _SMOKE_BATCH_SIZES,
        expected={
            "comm": (None, modulation_order),
            "sensing": (None, sensing_units),
        },
    )
    comm_params = count_trainable_params(comm_head)
    sensing_params = count_trainable_params(sensing_head)
    total_params = count_trainable_params(model)
    logger.info(
        "model '%s' built: backbone=%d params, comm=%d params, "
        "sensing=%d params, TOTAL=%d params",
        _MODEL_MC_DLSK_NAME,
        total_params - comm_params - sensing_params,
        comm_params,
        sensing_params,
        total_params,
    )
    return model

def build_baseline(config: Dict[str, Any], name: str) -> tf.keras.Model:
    
    try:
        name = canonical_model_name(name)
    except (TypeError, ValueError):
        pass

    if not isinstance(name, str) or name not in _VALID_BASELINE_NAMES:
        raise ValueError(
            f"invalid baseline: {name!r} (expected: {list(_VALID_BASELINE_NAMES)})"
        )

    builders: Dict[str, Any] = {
        "lstm": build_lstm_baseline,
        "mc_dlsk": build_mc_dlsk_baseline,
    }
    model = builders[name](config)

    input_shape = tuple(model.input_shape)
    if len(input_shape) != 3 or input_shape[1] is None or input_shape[2] is None:
        raise ValueError(
            f"expected input shape (None, L, F), got: {input_shape!r}"
        )

    if isinstance(model.output, dict):
        output_keys = set(model.output.keys())
    else:
        output_keys = {getattr(model.output, "name", str(model.output))}
    if output_keys != {"comm", "sensing"}:
        raise ValueError(
            f"expected outputs {{'comm', 'sensing'}}, got: {sorted(output_keys)}"
        )

    dummy_input = tf.ones(
        (1,) + tuple(model.input_shape[1:]),
        dtype=model.inputs[0].dtype,
    )
    outputs = model(dummy_input, training=False)
    for output_name in ("comm", "sensing"):
        tf.debugging.assert_all_finite(
            outputs[output_name],
            f"output '{output_name}' not finite on a non-degenerate input (build)",
        )

    logger.info(
        "baseline '%s' built: input=%s, output=%s, params=%d",
        name,
        input_shape,
        sorted(output_keys),
        count_trainable_params(model),
    )
    return model

def main(argv: Optional[Sequence[str]] = None) -> None:
    
    parser = argparse.ArgumentParser(
        description="Smoke test for the DL baselines (LSTM and MC-DLCSK)"
    )
    parser.add_argument(
        "--config", required=True, help="path to the experiment config (YAML)"
    )
    args = parser.parse_args(argv)

    config = load_config(
        config_path=Path(args.config),
        base_config_path=DEFAULT_BASE_CONFIG_PATH,
    )

    general = config.get("general")
    if not isinstance(general, dict):
        raise ValueError("'general' section missing or not a dict in config")
    experiment_name = str(general.get("experiment_name", "ultra_can_isac"))
    log_dir = _REPO_ROOT / "results" / experiment_name / "logs"
    setup_logging(
        log_dir=log_dir,
        level=str(general.get("log_level", "INFO")),
        experiment_name=experiment_name,
    )
    log_config_summary(config, logger)

    lstm = build_baseline(config, "lstm")
    mc_dlsk = build_baseline(config, "mc_dlsk")
    logger.info(
        "baselines ready: lstm=%d params, mc_dlsk=%d params",
        count_trainable_params(lstm),
        count_trainable_params(mc_dlsk),
    )

if __name__ == "__main__":
    main()

