
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple

import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.heads import (
    _as_positive_int,
    _assert_config_finite,
    _slice_received,
    build_communication_head,
    build_sensing_features,
    build_sensing_head,
    num_params,
    sensing_output_units,
)
from src.models.ultra_can import (
    _SMOKE_BATCH_SIZES,
    _VALID_FEATURE_MODES,
    _VALID_PADDINGS,
    _VALID_SIZES,
    _expected_att_length,
    _read_conv_dilations,
    _smoke_check_forward,
)
from src.utils.config_loader import (
    DEFAULT_BASE_CONFIG_PATH,
    load_config,
    parse_cli_overrides,
)
from src.utils.logger import log_config_summary, setup_logging

logger = logging.getLogger(__name__)

_BACKBONE_NAME_QKV = "ultra_can_qkv_backbone"
_MODEL_NAME_QKV = "ultra_can_qkv"
_QKV_LAYER_NAME = "qkv_attention"
_SATURATION_MAX_PROB = 0.99
_ENTROPY_LOG_EPS = 1e-12
_BACKBONE_REQUIRED_KEYS: Tuple[str, ...] = (
    "backbone_type",
    "size",
    "conv_filters",
    "conv_kernel",
    "conv_padding",
    "attention_heads",
    "attention_dim",
)
_DATA_REQUIRED_KEYS: Tuple[str, ...] = ("sequence_length", "feature_mode")
_VALID_BACKBONE_TYPES: Tuple[str, ...] = ("qkv_attention",)

class _QkvBackboneConfig(NamedTuple):

    model_cfg: Dict[str, Any]
    seq_len: int
    num_features: int
    conv_filters: Tuple[int, int]
    conv_kernel: int
    conv_padding: str
    conv_dilations: Tuple[int, int]
    attention_dim: int
    attention_heads: int
    att_len: int
    f2: int
    feature_mode: str

class _MultiHeadQKVAttention(tf.keras.layers.Layer):

    def __init__(
        self,
        attention_dim: int,
        num_heads: int,
        temperature: float = 1.0,
        positional_encoding: bool = True,
        qkv_residual: bool = True,
        enable_checks: bool = False,
        store_attention_weights: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        
        super().__init__(name=name, **kwargs)
        self._attention_dim = attention_dim
        self._num_heads = num_heads
        self._temperature = float(temperature)
        self._positional_encoding = bool(positional_encoding)
        self._qkv_residual = bool(qkv_residual)
        self._enable_checks = bool(enable_checks)
        self._store_attention_weights = bool(store_attention_weights)
        if not math.isfinite(self._temperature) or self._temperature <= 0.0:
            raise ValueError(
                f"temperature must be a positive number, got: {self._temperature!r}"
            )
        self._head_dim = attention_dim // num_heads
        self._scale = 1.0 / (self._temperature * math.sqrt(float(self._head_dim)))
        self._dense_q = tf.keras.layers.Dense(
            units=attention_dim,
            kernel_initializer="he_normal",
            name="qkv_query",
        )
        self._dense_k = tf.keras.layers.Dense(
            units=attention_dim,
            kernel_initializer="he_normal",
            name="qkv_key",
        )
        self._dense_v = tf.keras.layers.Dense(
            units=attention_dim,
            kernel_initializer="he_normal",
            name="qkv_value",
        )
        self._out_proj = tf.keras.layers.Dense(
            units=attention_dim,
            kernel_initializer="he_normal",
            name="qkv_out_proj",
        )
        self._last_attention_weights: Optional[tf.Tensor] = None

    @property
    def last_attention_weights(self) -> Optional[tf.Tensor]:
        return self._last_attention_weights

    @staticmethod
    def _positional_table(time_len: tf.Tensor, channels: tf.Tensor) -> tf.Tensor:
        
        half = channels // 2
        width = 2 * half
        freqs = tf.pow(
            10000.0,
            -tf.cast(2 * tf.range(half), tf.float32) / tf.cast(width, tf.float32),
        )
        positions = tf.cast(tf.range(time_len), tf.float32)
        angles = positions[:, None] * freqs[None, :]
        table = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)
        return tf.pad(table, [[0, 0], [0, tf.maximum(channels - width, 0)]])

    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        del training, mask
        batch_size = tf.shape(inputs)[0]
        time_len = tf.shape(inputs)[1]

        h2_inputs = inputs
        if self._positional_encoding:
            pe = self._positional_table(time_len, tf.shape(inputs)[2])
            inputs = inputs + pe[tf.newaxis, :, :]

        q = self._dense_q(inputs)
        k = self._dense_k(inputs)
        v = self._dense_v(inputs)

        def _split_heads(x: tf.Tensor) -> tf.Tensor:
            x = tf.reshape(
                x, (batch_size, time_len, self._num_heads, self._head_dim)
            )
            return tf.transpose(x, perm=(0, 2, 1, 3))

        q_heads = _split_heads(q)
        k_heads = _split_heads(k)
        v_heads = _split_heads(v)

        scores = tf.matmul(q_heads, k_heads, transpose_b=True)
        scores = scores * self._scale
        if self._enable_checks:
            tf.debugging.assert_all_finite(
                scores, message=f"{self.name}: scores contiene NaN/Inf"
            )
        attn_weights = tf.nn.softmax(scores, axis=-1)
        if self._enable_checks:
            tf.debugging.assert_all_finite(
                attn_weights,
                message=f"{self.name}: attention weights contiene NaN/Inf",
            )

        context = tf.matmul(attn_weights, v_heads)
        context = tf.transpose(context, perm=(0, 2, 1, 3))
        context = tf.reshape(
            context, (batch_size, time_len, self._attention_dim)
        )

        output = self._out_proj(context)

        if self._qkv_residual:
            if self._residual_static:
                output = output + h2_inputs
            else:
                in_dim = tf.shape(h2_inputs)[2]
                padded = tf.cond(
                    tf.less(in_dim, self._attention_dim),
                    lambda: tf.pad(
                        h2_inputs,
                        [[0, 0], [0, 0], [0, self._attention_dim - in_dim]],
                    ),
                    lambda: h2_inputs,
                )
                output = tf.cond(
                    tf.less_equal(in_dim, self._attention_dim),
                    lambda: output + padded,
                    lambda: output,
                )

        if self._store_attention_weights:
            self._last_attention_weights = tf.reduce_mean(attn_weights, axis=1)
        return output

    def compute_output_shape(
        self, input_shape: Sequence[Optional[int]]
    ) -> Tuple[Optional[int], ...]:
        return tuple(list(input_shape[:-1]) + [self._attention_dim])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "attention_dim": self._attention_dim,
                "num_heads": self._num_heads,
                "temperature": self._temperature,
                "positional_encoding": self._positional_encoding,
                "qkv_residual": self._qkv_residual,
                "enable_checks": self._enable_checks,
                "store_attention_weights": self._store_attention_weights,
            }
        )
        return config

    def build(self, input_shape: Sequence[Optional[int]]) -> None:
        
        self._build_input_shape = input_shape
        self._dense_q.build(input_shape)
        self._dense_k.build(input_shape)
        self._dense_v.build(input_shape)
        out_shape = (input_shape[0], input_shape[1], self._attention_dim)
        self._out_proj.build(out_shape)
        self._residual_static = bool(input_shape[-1] == self._attention_dim)
        self.built = True

    def get_build_config(self) -> Dict[str, Any]:
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        self.build(config["input_shape"])

def build_qkv_attention_layer(
    attention_dim: int,
    num_heads: int,
    temperature: float = 1.0,
    positional_encoding: bool = True,
    qkv_residual: bool = True,
    enable_checks: bool = False,
    store_attention_weights: bool = False,
) -> tf.keras.layers.Layer:
    
    if isinstance(attention_dim, bool) or not isinstance(attention_dim, int):
        raise TypeError(
            "attention_dim must be int, "
            f"got: {attention_dim!r} ({type(attention_dim).__name__})"
        )
    if isinstance(num_heads, bool) or not isinstance(num_heads, int):
        raise TypeError(
            "num_heads must be int, "
            f"got: {num_heads!r} ({type(num_heads).__name__})"
        )
    if attention_dim <= 0:
        raise ValueError(
            f"attention_dim must be a positive integer, got: "
            f"{attention_dim!r}"
        )
    if num_heads <= 0:
        raise ValueError(
            f"num_heads must be a positive integer, got: {num_heads!r}"
        )
    if attention_dim % num_heads != 0:
        raise ValueError(
            f"attention_dim ({attention_dim}) must be divisible by "
            f"num_heads ({num_heads}): d_k must be >= 1"
        )
    logger.debug(
        "QKV layer: attention_dim=%d, num_heads=%d, d_k=%d, "
        "positional_encoding=%s, qkv_residual=%s, enable_checks=%s, "
        "store_attention_weights=%s",
        attention_dim,
        num_heads,
        attention_dim // num_heads,
        positional_encoding,
        qkv_residual,
        enable_checks,
        store_attention_weights,
    )
    return _MultiHeadQKVAttention(
        attention_dim=attention_dim,
        num_heads=num_heads,
        temperature=temperature,
        positional_encoding=positional_encoding,
        qkv_residual=qkv_residual,
        enable_checks=enable_checks,
        store_attention_weights=store_attention_weights,
        name=_QKV_LAYER_NAME,
    )

def _validate_qkv_backbone_config(
    config: Dict[str, Any],
) -> _QkvBackboneConfig:
    
    if not isinstance(config, dict):
        raise TypeError(
            f"config must be a dict, got: {type(config).__name__}"
        )

    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("'model' section missing or not a dict in config")
    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("'data' section missing or not a dict in config")

    _assert_config_finite(model_cfg)

    missing_model = [
        key for key in _BACKBONE_REQUIRED_KEYS if key not in model_cfg
    ]
    if missing_model:
        raise ValueError(f"missing keys in model: {missing_model}")
    missing_data = [key for key in _DATA_REQUIRED_KEYS if key not in data_cfg]
    if missing_data:
        raise ValueError(f"missing keys in data: {missing_data}")

    backbone_type = str(model_cfg["backbone_type"])
    if backbone_type not in _VALID_BACKBONE_TYPES:
        raise ValueError(
            f"model.backbone_type {backbone_type!r} not supported by this "
            f"module (expected: {list(_VALID_BACKBONE_TYPES)}); for 'conv1d' "
            "use src/models/ultra_can.py"
        )

    size = str(model_cfg["size"])
    if size not in _VALID_SIZES:
        raise ValueError(
            f"model.size must be one of {list(_VALID_SIZES)}, "
            f"got: {size!r}"
        )
    if size == "micro":
        logger.warning(
            "model.size='micro' requires a reduced attention_dim; the expected "
            "head parameter counts (8578/2146) "
            "no longer apply"
        )

    conv_filters = model_cfg["conv_filters"]
    if not isinstance(conv_filters, (list, tuple)) or len(conv_filters) != 2:
        raise ValueError(
            "model.conv_filters must be a list of EXACTLY 2 positive "
            f"integers [F1, F2], got: {conv_filters!r}"
        )
    _as_positive_int(conv_filters[0], "model.conv_filters[0]")
    _as_positive_int(conv_filters[1], "model.conv_filters[1]")

    conv_kernel = _as_positive_int(model_cfg["conv_kernel"], "model.conv_kernel")

    conv_padding = str(model_cfg["conv_padding"])
    if conv_padding not in _VALID_PADDINGS:
        raise ValueError(
            f"model.conv_padding must be one of {list(_VALID_PADDINGS)} "
            f"(got: {conv_padding!r})"
        )
    conv_dilations = _read_conv_dilations(model_cfg, n_layers=2)

    attention_heads = _as_positive_int(
        model_cfg["attention_heads"], "model.attention_heads"
    )
    attention_dim = _as_positive_int(
        model_cfg["attention_dim"], "model.attention_dim"
    )
    if attention_dim % attention_heads != 0:
        raise ValueError(
            f"model.attention_dim ({attention_dim}) must be divisible by "
            f"model.attention_heads ({attention_heads}): d_k must be "
            ">= 1"
        )
    f2 = int(conv_filters[1])
    if attention_dim != f2:
        raise ValueError(
            f"model.attention_dim ({attention_dim}) must match "
            f"model.conv_filters[1] ({f2}): the QKV operates on H^(2) and the heads "
            "receive v in R^(attention_dim) via GAP"
        )

    seq_len = _as_positive_int(
        data_cfg["sequence_length"], "data.sequence_length"
    )
    feature_mode = str(data_cfg["feature_mode"])
    if feature_mode not in _VALID_FEATURE_MODES:
        raise ValueError(
            f"data.feature_mode must be one of {list(_VALID_FEATURE_MODES)}, "
            f"got: {feature_mode!r}"
        )

    if conv_kernel >= seq_len:
        raise ValueError(
            f"model.conv_kernel ({conv_kernel}) must be < "
            f"data.sequence_length ({seq_len}): the 'valid' convolution "
            "would produce a length <= 0"
        )

    att_len = _expected_att_length(
        seq_len,
        conv_kernel,
        padding=conv_padding,
        dilations=conv_dilations,
    )

    num_features = 1 if feature_mode == "real" else 2

    logger.debug(
        "QKV backbone config validated: conv_filters=%s, conv_kernel=%d, "
        "padding=%s, attention_dim=%d, attention_heads=%d, seq_len=%d, "
        "feature_mode=%s, num_features=%d",
        list(conv_filters),
        conv_kernel,
        conv_padding,
        attention_dim,
        attention_heads,
        seq_len,
        feature_mode,
        num_features,
    )
    return _QkvBackboneConfig(
        model_cfg=model_cfg,
        seq_len=seq_len,
        num_features=num_features,
        conv_filters=(int(conv_filters[0]), int(conv_filters[1])),
        conv_kernel=conv_kernel,
        conv_padding=conv_padding,
        conv_dilations=conv_dilations,
        attention_dim=attention_dim,
        attention_heads=attention_heads,
        att_len=att_len,
        f2=f2,
        feature_mode=feature_mode,
    )

def _log_attention_saturation(attn_layer: _MultiHeadQKVAttention) -> None:
    
    if not tf.executing_eagerly():
        return
    attn_weights = attn_layer.last_attention_weights
    if attn_weights is None:
        return
    max_prob = float(tf.reduce_max(attn_weights))
    entropy = float(
        -tf.reduce_mean(
            tf.reduce_sum(
                attn_weights * tf.math.log(attn_weights + _ENTROPY_LOG_EPS),
                axis=-1,
            )
        )
    )
    logger.debug(
        "QKV attention: max prob. = %.4f, mean entropy = %.4f",
        max_prob,
        entropy,
    )
    if max_prob > _SATURATION_MAX_PROB:
        logger.warning(
            "QKV attention saturated: max softmax prob. = %.4f (> %.2f) -> "
            "near-zero gradients on those positions",
            max_prob,
            _SATURATION_MAX_PROB,
        )

def build_qkv_backbone(config: Dict[str, Any]) -> tf.keras.Model:
    
    cfg = _validate_qkv_backbone_config(config)
    return _build_qkv_backbone(cfg)

def _build_qkv_backbone(
    cfg: _QkvBackboneConfig,
    input_tensor: Optional[tf.keras.KerasTensor] = None,
):
    
    seq_len = cfg.seq_len
    num_features = cfg.num_features
    conv_kernel = cfg.conv_kernel
    attention_heads = cfg.attention_heads
    attention_dim = cfg.attention_dim
    att_len = cfg.att_len
    f1, f2 = cfg.conv_filters

    if input_tensor is not None:
        if tuple(input_tensor.shape[1:]) != (seq_len, num_features):
            raise ValueError(
                f"expected input_tensor shape (None, {seq_len}, {num_features}), "
                f"got: {tuple(input_tensor.shape)}"
            )
        r_input = input_tensor
    else:
        r_input = tf.keras.layers.Input(
            shape=(seq_len, num_features), name="r_input"
        )
    h1 = tf.keras.layers.Conv1D(
        filters=f1,
        kernel_size=conv_kernel,
        padding=cfg.conv_padding,
        dilation_rate=cfg.conv_dilations[0],
        activation="relu",
        kernel_initializer="he_normal",
        name="conv1",
    )(r_input)
    h2 = tf.keras.layers.Conv1D(
        filters=f2,
        kernel_size=conv_kernel,
        padding=cfg.conv_padding,
        dilation_rate=cfg.conv_dilations[1],
        activation="relu",
        kernel_initializer="he_normal",
        name="conv2",
    )(h1)
    qkv_temperature = float(cfg.model_cfg.get("attention_temperature", 1.0))
    positional_encoding = bool(cfg.model_cfg.get("positional_encoding", True))
    qkv_residual = bool(cfg.model_cfg.get("qkv_residual", True))
    attention_checks = bool(cfg.model_cfg.get("attention_checks", False))
    store_attention_weights = bool(
        cfg.model_cfg.get("store_attention_weights", False)
    )
    qkv_layer = build_qkv_attention_layer(
        attention_dim,
        attention_heads,
        temperature=qkv_temperature,
        positional_encoding=positional_encoding,
        qkv_residual=qkv_residual,
        enable_checks=attention_checks,
        store_attention_weights=store_attention_weights,
    )
    h_att = qkv_layer(h2)

    if tuple(h_att.shape) != (None, att_len, f2):
        raise ValueError(
            f"expected H^att shape (None, {att_len}, {f2}) (conv 'valid' "
            f"{seq_len} -> {seq_len - (conv_kernel - 1)} -> {att_len}), "
            f"got: {tuple(h_att.shape)}"
        )

    if input_tensor is not None:
        logger.debug(
            "backbone '%s' (shared layers): H^att=(None,%d,%d)",
            _BACKBONE_NAME_QKV, att_len, f2,
        )
        return h_att

    backbone = tf.keras.Model(
        inputs=r_input, outputs=h_att, name=_BACKBONE_NAME_QKV
    )
    _smoke_check_forward(
        backbone,
        _SMOKE_BATCH_SIZES,
        expected={backbone.output.name: (None, att_len, f2)},
    )
    _log_attention_saturation(qkv_layer)
    logger.info(
        "backbone '%s' built: input=(None,%d,%d), H^att=(None,%d,%d), "
        "params=%d",
        _BACKBONE_NAME_QKV,
        seq_len,
        num_features,
        att_len,
        f2,
        count_trainable_params(backbone),
    )
    return backbone

def build_ultra_can_qkv(config: Dict[str, Any]) -> tf.keras.Model:
    
    cfg = _validate_qkv_backbone_config(config)
    comm_head = build_communication_head(config)

    att_len = cfg.att_len
    f2 = cfg.f2
    seq_len = cfg.seq_len
    num_features = cfg.num_features

    model_input = tf.keras.layers.Input(
        shape=(seq_len, num_features + 1), name="isac_input"
    )
    received = tf.keras.layers.Lambda(
        _slice_received,
        arguments={"num_features": num_features},
        name="received_slice",
    )(model_input)

    h_att = _build_qkv_backbone(cfg, input_tensor=received)

    v, v_sensing = build_sensing_features(
        h_att,
        config,
        received_input=model_input,
        pool_projection=bool(cfg.model_cfg.get("pool_projection", True)),
    )
    comm = comm_head(v)
    sensing_head = build_sensing_head(config, input_dim=int(v_sensing.shape[-1]))
    sensing = sensing_head(v_sensing)
    sensing_units = sensing_output_units(config)

    modulation_order = int(comm_head.output_shape[-1])

    if tuple(h_att.shape) != (None, att_len, f2):
        raise ValueError(
            f"expected H^att shape (None, {att_len}, {f2}), "
            f"got: {tuple(h_att.shape)}"
        )
    if tuple(v.shape) != (None, f2):
        raise ValueError(f"expected v shape (None, {f2}), got: {tuple(v.shape)}")
    if tuple(comm.shape) != (None, modulation_order):
        raise ValueError(
            f"expected comm shape (None, {modulation_order}), "
            f"got: {tuple(comm.shape)}"
        )
    if tuple(sensing.shape) != (None, sensing_units):
        raise ValueError(
            f"expected sensing shape (None, {sensing_units}), "
            f"got: {tuple(sensing.shape)}"
        )

    model = tf.keras.Model(
        inputs=model_input,
        outputs={"comm": comm, "sensing": sensing},
        name=_MODEL_NAME_QKV,
    )
    _smoke_check_forward(
        model,
        _SMOKE_BATCH_SIZES,
        expected={
            "comm": (None, modulation_order),
            "sensing": (None, sensing_units),
        },
    )

    training_cfg = config.get("training")
    lambda_mse = (
        training_cfg.get("lambda_mse") if isinstance(training_cfg, dict) else None
    )
    logger.info(
        "model '%s' built: comm=%d params, sensing=%d params, "
        "TOTAL=%d params, lambda_mse=%s",
        _MODEL_NAME_QKV,
        count_trainable_params(comm_head),
        count_trainable_params(sensing_head),
        count_trainable_params(model),
        lambda_mse,
    )
    return model

def count_trainable_params(model: tf.keras.Model) -> int:
    
    if not isinstance(model, tf.keras.Model):
        raise TypeError(
            f"model must be a tf.keras.Model, got: {type(model).__name__}"
        )
    return num_params(model)

def _build_and_smoke_check(config: Dict[str, Any]) -> tf.keras.Model:
    
    model = build_ultra_can_qkv(config)
    logger.info(
        "model '%s' ready: TOTAL=%d params, footprint ~%.0f kB a "
        "4 bytes/param",
        model.name,
        count_trainable_params(model),
        count_trainable_params(model) * 4.0 / 1024.0,
    )
    return model

def main(argv: Optional[Sequence[str]] = None) -> None:
    
    parser = argparse.ArgumentParser(
        description="Smoke test for the Ultra-CAN QKV (ISAC in IoD) model"
        ""
    )
    parser.add_argument(
        "--config", required=True, help="path to the experiment config (YAML)"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="path.to.key=value",
        help="CLI override (can be repeated), e.g. "
        "--set model.backbone_type=qkv_attention",
    )
    args = parser.parse_args(argv)

    overrides = parse_cli_overrides(args.set)
    config = load_config(
        config_path=Path(args.config),
        base_config_path=DEFAULT_BASE_CONFIG_PATH,
        cli_overrides=overrides,
    )

    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict) or (
        str(model_cfg.get("backbone_type", "")) != "qkv_attention"
    ):
        raise ValueError(
            "model.backbone_type must be 'qkv_attention' for this "
            "module (use: --set model.backbone_type=qkv_attention)"
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

    _build_and_smoke_check(config)

if __name__ == "__main__":
    main()

