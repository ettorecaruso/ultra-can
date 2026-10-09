
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config_loader import (
    DEFAULT_BASE_CONFIG_PATH,
    load_config,
)
from src.utils.logger import log_config_summary, setup_logging

logger = logging.getLogger(__name__)

_HEAD_INPUT_DIM = 64
_SENSING_OUTPUT_UNITS_ALLOWED: Tuple[int, ...] = (1, 2)
DEFAULT_SENSING_OUTPUT_UNITS = 1
_SENSING_MAXPOOL_KEY = "use_max_pool_feature"
_SENSING_POSITION_KEY = "use_position_feature"
_SENSING_POSITION_EPS = 1e-8
_SENSING_PROFILE_KEY = "use_reference_profile"
_SENSING_PROFILE_LAG_KEY = "profile_max_lag"
_SENSING_DELAY_MODE_KEY = "delay_mode"
_SENSING_DELAY_MODE_DEFAULT = "regression"
_SENSING_DELAY_MODES: Tuple[str, ...] = ("regression", "peak_residual")
_SENSING_PEAK_KEY = "use_delay_peak_feature"
_SENSING_PEAK_SHARPNESS_KEY = "peak_sharpness"
_SENSING_PEAK_SHARPNESS_DEFAULT = 8.0
_MIN_MODULATION_ORDER = 2
_HEAD_NAMES: Tuple[str, ...] = ("communication_head", "sensing_head")
_COMM_REQUIRED_KEYS: Tuple[str, ...] = ("units", "activation", "modulation_order")
_SENSING_REQUIRED_KEYS: Tuple[str, ...] = ("units", "activation", "output_units")

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

def _as_positive_int(value: Any, label: str) -> int:
    
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive integer, got: {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0 or not number.is_integer():
        raise ValueError(f"{label} must be a positive integer, got: {value!r}")
    return int(number)

def _head_input_dim(config: Dict[str, Any]) -> int:
    
    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("'model' section missing or not a dict in config")

    backbone_type = str(model_cfg.get("backbone_type", "conv1d"))
    dim: Optional[int] = None
    if backbone_type == "qkv_attention":
        attention_dim = model_cfg.get("attention_dim")
        if attention_dim is not None:
            dim = _as_positive_int(attention_dim, "model.attention_dim")
    else:
        conv_filters = model_cfg.get("conv_filters")
        if isinstance(conv_filters, (list, tuple)) and len(conv_filters) >= 2:
            dim = _as_positive_int(conv_filters[1], "model.conv_filters[1]")

    if dim is None:
        dim = _HEAD_INPUT_DIM
        logger.debug(
            "feature dim not derivable from config, assuming the contract v in R^%d", dim
        )
    if dim != _HEAD_INPUT_DIM:
        logger.warning(
            "feature dim %d differs from the paper contract v in R^%d: "
            "the expected parameter counts (8578/2146) do not apply",
            dim,
            _HEAD_INPUT_DIM,
        )
    return dim

def _validate_head_config(
    config: Dict[str, Any],
    head_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got: {type(config).__name__}")
    if head_name not in _HEAD_NAMES:
        raise ValueError(
            f"invalid head_name: {head_name!r} (expected: {sorted(_HEAD_NAMES)})"
        )

    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("'model' section missing or not a dict in config")
    head_cfg = model_cfg.get(head_name)
    if not isinstance(head_cfg, dict):
        raise ValueError(
            f"'model.{head_name}' section missing or not a dict in the config"
        )

    _assert_config_finite(head_cfg)

    required = _COMM_REQUIRED_KEYS if head_name == "communication_head" else _SENSING_REQUIRED_KEYS
    missing = [key for key in required if key not in head_cfg]
    if missing:
        raise ValueError(f"missing keys in model.{head_name}: {missing}")

    units = _as_positive_int(head_cfg["units"], f"model.{head_name}.units")
    activation = head_cfg["activation"]
    if not isinstance(activation, str) or not activation.strip():
        raise ValueError(
            f"model.{head_name}.activation must be a non-empty string, "
            f"got: {activation!r}"
        )

    output_units: Optional[int] = None
    if head_name == "communication_head":
        modulation_order = _as_positive_int(
            head_cfg["modulation_order"], "model.communication_head.modulation_order"
        )
        if modulation_order < _MIN_MODULATION_ORDER:
            raise ValueError(
                f"model.communication_head.modulation_order must be >= "
                f"{_MIN_MODULATION_ORDER} (BPSK=2, QPSK=4), got: {modulation_order}"
            )
        dropout = model_cfg.get("dropout_rate")
        if dropout is None:
            raise ValueError("missing key: model.dropout_rate")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(float(dropout))
            or not (0.0 <= float(dropout) < 1.0)
        ):
            raise ValueError(
                f"model.dropout_rate must be in [0, 1), got: {dropout!r}"
            )
    else:
        output_units = _as_positive_int(
            head_cfg["output_units"], "model.sensing_head.output_units"
        )
        if output_units not in _SENSING_OUTPUT_UNITS_ALLOWED:
            raise ValueError(
                f"model.sensing_head.output_units must be one of "
                f"{list(_SENSING_OUTPUT_UNITS_ALLOWED)}, got: {output_units}"
            )

    logger.debug(
        "head config '%s' validated: units=%d, activation=%s, "
        "dropout/out_units=%s",
        head_name,
        units,
        activation,
        model_cfg.get("dropout_rate") if head_name == "communication_head" else output_units,
    )
    return model_cfg, head_cfg

def sensing_output_units(config: Dict[str, Any]) -> int:
    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict) or "sensing_head" not in model_cfg:
        return DEFAULT_SENSING_OUTPUT_UNITS
    _model_cfg, sens_cfg = _validate_head_config(config, "sensing_head")
    return int(sens_cfg["output_units"])

def _smoke_check(
    model: tf.keras.Model,
    expected_output_shape: Sequence[Optional[int]],
    tag: str,
) -> None:
    
    if not tf.executing_eagerly():
        logger.debug("smoke check skipped (non-eager context): %s", tag)
        return

    input_shape = tuple(model.input_shape)
    if len(input_shape) != 2 or input_shape[1] is None:
        raise ValueError(
            f"{tag}: expected input shape (None, D), got: {input_shape!r}"
        )

    dummy = tf.zeros((1, int(input_shape[1])), dtype=model.inputs[0].dtype)
    outputs = model(dummy, training=False)
    actual = outputs.shape.as_list()
    expected = list(expected_output_shape)
    if len(actual) != len(expected) or actual[0] < 1 or actual[1:] != expected[1:]:
        raise ValueError(
            f"{tag}: expected output shape {expected}, "
            f"got: {actual}"
        )

    tf.debugging.assert_all_finite(
        outputs, message=f"{tag}: output contains NaN/Inf"
    )
    logger.debug("smoke check OK: %s output_shape=%s", tag, actual)

def build_communication_head(config: Dict[str, Any]) -> tf.keras.Model:
    
    model_cfg, comm_cfg = _validate_head_config(config, "communication_head")
    dropout_rate = float(model_cfg["dropout_rate"])
    units = int(comm_cfg["units"])
    activation = str(comm_cfg["activation"])
    modulation_order = int(comm_cfg["modulation_order"])
    input_dim = _head_input_dim(config)

    v = tf.keras.layers.Input(shape=(input_dim,), name="comm_features")
    x = tf.keras.layers.Dropout(rate=dropout_rate, name="dropout_comm")(v)
    x = tf.keras.layers.Dense(
        units=units,
        activation=activation,
        kernel_initializer="he_normal",
        name="dense_comm_1",
    )(x)
    logits = tf.keras.layers.Dense(
        units=modulation_order,
        activation="linear",
        name="comm_logits",
    )(x)

    model = tf.keras.Model(inputs=v, outputs=logits, name="communication_head")
    _smoke_check(model, expected_output_shape=[None, modulation_order], tag="comm")
    logger.info(
        "communication_head built: input_dim=%d, units=%d, M=%d, params=%d",
        input_dim,
        units,
        modulation_order,
        num_params(model),
    )
    return model

try:
    from keras.saving import register_keras_serializable as _register_serializable
except ImportError:
    _register_serializable = tf.keras.saving.register_keras_serializable

@_register_serializable(package="src.models.heads")
def _position_feature(t: tf.Tensor) -> tf.Tensor:
    
    length = tf.shape(t)[1]
    t_pos = tf.cast(tf.range(length), tf.float32) / tf.cast(length, tf.float32)
    w = tf.abs(t)
    norm = tf.reduce_sum(w, axis=1, keepdims=True)
    norm = tf.where(norm < _SENSING_POSITION_EPS, tf.ones_like(norm), norm)
    pos = tf.reduce_sum(t_pos[None, :, None] * w, axis=1) / tf.squeeze(norm, axis=1)
    return pos

@_register_serializable(package="src.models.heads")
def _slice_received(inputs: tf.Tensor, num_features: int) -> tf.Tensor:
    
    return inputs[..., :num_features]

@_register_serializable(package="src.models.heads")
def _dca_mf_profile(
    inputs: tf.Tensor, max_delay: int, num_features: int
) -> tf.Tensor:
    
    received = inputs[..., :num_features]
    xref = inputs[..., -1:]
    if num_features == 2:
        z = tf.complex(received[..., 0], received[..., 1])
    else:
        z = tf.complex(received[..., 0], tf.zeros_like(received[..., 0]))
    xr = xref[..., 0]
    xr_c = tf.cast(xr, tf.complex64)

    xr_sq = tf.reduce_sum(tf.square(xr), axis=1)
    xr_sq = tf.where(xr_sq < 1e-12, tf.ones_like(xr_sq), xr_sq)
    h = tf.reduce_sum(z * xr_c, axis=1) / tf.cast(xr_sq, tf.complex64)

    r = z - h[:, None] * xr_c

    abs_list: list = []
    re_list: list = []
    im_list: list = []
    for lag in range(1, max_delay + 1):
        r_slice = r[:, lag:]
        x_slice = xr_c[:, :-lag]
        cc = tf.reduce_sum(tf.math.conj(r_slice) * x_slice, axis=1)
        abs_list.append(tf.abs(cc))
        re_list.append(tf.math.real(cc))
        im_list.append(tf.math.imag(cc))
    profile = tf.stack(abs_list + re_list + im_list, axis=1)
    return profile

def _delay_peak_enabled(sensing_cfg: Any) -> bool:
    
    if not isinstance(sensing_cfg, dict):
        return False
    mode = str(sensing_cfg.get(_SENSING_DELAY_MODE_KEY, _SENSING_DELAY_MODE_DEFAULT))
    if mode == "peak_residual":
        return True
    return bool(sensing_cfg.get(_SENSING_PEAK_KEY, False))

@_register_serializable(package="src.models.heads")
def _last_column(t: tf.Tensor) -> tf.Tensor:
    
    return t[..., -1:]

@_register_serializable(package="src.models.heads")
def _soft_argmax_lag(
    profile: tf.Tensor, max_delay: int, sharpness: float
) -> tf.Tensor:
    
    max_delay = int(max_delay)
    magnitude = tf.abs(profile[:, :max_delay])
    peak = tf.reduce_max(magnitude, axis=-1, keepdims=True)
    normalised = magnitude / (peak + _SENSING_POSITION_EPS)
    weights = tf.nn.softmax(float(sharpness) * normalised, axis=-1)
    lags = tf.cast(tf.range(1, max_delay + 1), profile.dtype) / float(max_delay)
    return tf.reduce_sum(weights * lags[None, :], axis=-1, keepdims=True)

@_register_serializable(package="src.models.heads")
def _attention_pool(alpha_and_h: list) -> tf.Tensor:
    
    alpha, h = alpha_and_h
    return tf.reduce_sum(alpha * h, axis=1)

def build_sensing_features(
    h_att: tf.keras.KerasTensor,
    config: Dict[str, Any],
    received_input: Optional[tf.keras.KerasTensor] = None,
    attention_mode: str = "gap",
    pool_projection: bool = False,
) -> Tuple[tf.keras.KerasTensor, tf.keras.KerasTensor]:
    
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got: {type(config).__name__}")
    if not isinstance(h_att, tf.keras.KerasTensor) or len(h_att.shape) != 3:
        raise ValueError(
            f"h_att must be a 3D KerasTensor (B, att_len, F2), got: {h_att!r}"
        )
    feature_dim = int(h_att.shape[2])
    if feature_dim is None or feature_dim <= 0:
        raise ValueError(f"last dimension of h_att unknown or not positive: {h_att.shape}")

    data_cfg = config.get("data") or {}
    sensing_cfg = config.get("model", {}).get("sensing_head") or {}
    use_max = bool(sensing_cfg.get(_SENSING_MAXPOOL_KEY, True))
    use_position = bool(sensing_cfg.get(_SENSING_POSITION_KEY, False))
    use_profile = bool(sensing_cfg.get(_SENSING_PROFILE_KEY, True))

    if attention_mode == "attention_pool":
        attn_scores = tf.keras.layers.Dense(
            1, name="attn_scores"
        )(h_att)
        alpha = tf.keras.layers.Softmax(axis=1, name="attn_weights")(attn_scores)
        v = tf.keras.layers.Lambda(
            _attention_pool, name="attn_pool"
        )([alpha, h_att])
    else:
        v = tf.keras.layers.GlobalAveragePooling1D(name="shared_gap")(h_att)
    if pool_projection:
        v = tf.keras.layers.Dense(
            64,
            activation="relu",
            kernel_initializer="he_normal",
            name="pool_projection",
        )(v)
    parts: list = [v]
    if use_position:
        v_pos = tf.keras.layers.Lambda(_position_feature, name="sensing_position")(h_att)
        parts.append(v_pos)
    if use_max:
        v_max = tf.keras.layers.GlobalMaxPooling1D(name="sensing_max_pool")(h_att)
        parts.append(v_max)

    profile_max_lag = int(sensing_cfg.get(
        _SENSING_PROFILE_LAG_KEY, int(data_cfg.get("max_delay", 33))
    ))
    delay_profile = None
    if use_profile and received_input is not None:
        num_features = 2 if str(data_cfg.get("feature_mode", "iq")) == "iq" else 1
        delay_profile = tf.keras.layers.Lambda(
            _dca_mf_profile,
            arguments={"max_delay": profile_max_lag, "num_features": num_features},
            name="sensing_delay_profile",
        )(received_input)
        parts.append(delay_profile)

    if _delay_peak_enabled(sensing_cfg) and delay_profile is not None:
        peak_lag = tf.keras.layers.Lambda(
            _soft_argmax_lag,
            arguments={
                "max_delay": profile_max_lag,
                "sharpness": float(sensing_cfg.get(
                    _SENSING_PEAK_SHARPNESS_KEY, _SENSING_PEAK_SHARPNESS_DEFAULT
                )),
            },
            name="sensing_delay_peak",
        )(delay_profile)
        parts.append(peak_lag)

    if len(parts) == 1:
        v_sensing = v
    else:
        v_sensing = tf.keras.layers.Concatenate(name="sensing_features")(parts)
    return v, v_sensing

def build_sensing_head(
    config: Dict[str, Any],
    input_dim: Optional[int] = None,
) -> tf.keras.Model:
    
    _model_cfg, sens_cfg = _validate_head_config(config, "sensing_head")
    units = int(sens_cfg["units"])
    activation = str(sens_cfg["activation"])
    output_units = int(sens_cfg["output_units"])
    if input_dim is None:
        input_dim = _head_input_dim(config)
    else:
        input_dim = int(input_dim)
    if input_dim <= 0:
        raise ValueError(f"input_dim must be > 0, got: {input_dim!r}")

    v = tf.keras.layers.Input(shape=(input_dim,), name="sensing_features")
    x = tf.keras.layers.Dense(
        units=units,
        activation=activation,
        kernel_initializer="he_normal",
        name="dense_sensing_1",
    )(v)
    output_activation = str(sens_cfg.get("output_activation", "sigmoid"))
    if output_activation not in ("sigmoid", "linear"):
        raise ValueError(
            f"model.sensing_head.output_activation must be 'sigmoid' or 'linear', "
            f"got: {output_activation!r}"
        )
    delay_mode = str(sens_cfg.get(_SENSING_DELAY_MODE_KEY, _SENSING_DELAY_MODE_DEFAULT))
    if delay_mode not in _SENSING_DELAY_MODES:
        raise ValueError(
            f"model.sensing_head.{_SENSING_DELAY_MODE_KEY} must be one of "
            f"{sorted(_SENSING_DELAY_MODES)}, got: {delay_mode!r}"
        )
    if delay_mode == "peak_residual":
        if output_units not in _SENSING_OUTPUT_UNITS_ALLOWED:
            raise ValueError(
                f"delay_mode='peak_residual' emits [tau] (or [tau, fD]) and requires "
                f"output_units in {list(_SENSING_OUTPUT_UNITS_ALLOWED)}, got: {output_units}"
            )
        tau_peak = tf.keras.layers.Lambda(
            _last_column, name="sensing_delay_peak_in"
        )(v)
        tau_residual = tf.keras.layers.Dense(
            1,
            activation="linear",
            kernel_initializer="zeros",
            name="sensing_delay_residual",
        )(x)
        tau = tf.keras.layers.Add(name="sensing_delay")([tau_peak, tau_residual])
        if output_units == 1:
            out = tau
        else:
            f_d = tf.keras.layers.Dense(
                1, activation=output_activation, name="sensing_doppler",
            )(x)
            out = tf.keras.layers.Concatenate(name="sensing_out")([tau, f_d])
    else:
        out = tf.keras.layers.Dense(
            units=output_units,
            activation=output_activation,
            name="sensing_out",
        )(x)
    model = tf.keras.Model(inputs=v, outputs=out, name="sensing_head")
    _smoke_check(model, expected_output_shape=[None, output_units], tag="sensing")
    logger.info(
        "sensing_head built: input_dim=%d, units=%d, output_units=%d, params=%d",
        input_dim,
        units,
        output_units,
        num_params(model),
    )
    return model

def num_params(model: tf.keras.Model) -> int:
    
    if not isinstance(model, tf.keras.Model):
        raise TypeError(
            f"model must be a tf.keras.Model, got: {type(model).__name__}"
        )
    try:
        total = int(model.count_params())
    except ValueError:
        input_shape = getattr(model, "input_shape", None)
        if input_shape is None:
            raise ValueError(
                "model not built and no input_shape declared: "
                "cannot count parameters"
            )
        model.build(input_shape=tuple(input_shape))
        total = int(model.count_params())
    logger.debug(
        "num_params=%d (model=%s)",
        total,
        getattr(model, "name", type(model).__name__),
    )
    return total

def main(argv: Optional[Sequence[str]] = None) -> None:
    
    parser = argparse.ArgumentParser(
        description="Smoke test for the Ultra-CAN (ISAC in IoD) heads"
    )
    parser.add_argument(
        "--config", required=True, help="path to the experiment config (YAML)"
    )
    args = parser.parse_args(argv)

    config = load_config(
        config_path=Path(args.config),
        base_config_path=DEFAULT_BASE_CONFIG_PATH,
    )
    _validate_head_config(config, "communication_head")
    _validate_head_config(config, "sensing_head")

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

    comm_head = build_communication_head(config)
    sensing_head = build_sensing_head(config)
    logger.info(
        "heads ready: comm=%d params, sensing=%d params, TOTAL=%d params",
        num_params(comm_head),
        num_params(sensing_head),
        num_params(comm_head) + num_params(sensing_head),
    )

if __name__ == "__main__":
    main()

