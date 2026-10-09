
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List

import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.heads import (
    _assert_config_finite,
    _as_positive_int,
    _slice_received,
    build_communication_head,
    build_sensing_features,
    build_sensing_head,
    num_params,
    sensing_output_units,
)
from src.utils.config_loader import (
    DEFAULT_BASE_CONFIG_PATH,
    load_config,
)
from src.utils.logger import log_config_summary, setup_logging

logger = logging.getLogger(__name__)

_BACKBONE_NAME = "ultra_can_backbone"
_MODEL_NAME = "ultra_can"
_SMOKE_BATCH_SIZES: Tuple[int, int] = (1, 1024)
_BACKBONE_REQUIRED_KEYS: Tuple[str, ...] = (
    "backbone_type",
    "size",
    "conv_filters",
    "conv_kernel",
    "conv_padding",
)
_DATA_REQUIRED_KEYS: Tuple[str, ...] = ("sequence_length", "feature_mode")
_VALID_BACKBONE_TYPES: Tuple[str, ...] = ("conv1d",)
_VALID_SIZES: Tuple[str, ...] = ("full", "micro")
_VALID_PADDINGS: Tuple[str, ...] = ("valid", "same")
_VALID_FEATURE_MODES: Tuple[str, ...] = ("real", "iq")
_DEFAULT_CONV_DILATIONS: Tuple[int, int] = (1, 1)

def _read_conv_dilations(
    model_cfg: Dict[str, Any], n_layers: int
) -> Tuple[int, ...]:
    
    raw = model_cfg.get("conv_dilations", list(_DEFAULT_CONV_DILATIONS))
    if not isinstance(raw, (list, tuple)) or len(raw) != n_layers:
        raise ValueError(
            f"model.conv_dilations must be a list of EXACTLY "
            f"{n_layers} integers >= 1, got: {raw!r}"
        )
    dilations: List[int] = []
    for d in raw:
        if isinstance(d, bool) or not isinstance(d, (int,)) or int(d) < 1:
            raise ValueError(
                f"model.conv_dilations must contain only integers >= 1, "
                f"got: {raw!r}"
            )
        dilations.append(int(d))
    return tuple(dilations)

def _expected_att_length(
    seq_len: int,
    kernel: int,
    padding: str = "valid",
    dilations: Sequence[int] = (1, 1),
) -> int:
    
    seq_len_int = _as_positive_int(seq_len, "data.sequence_length")
    kernel_int = _as_positive_int(kernel, "model.conv_kernel")
    dilations_int = tuple(int(d) for d in dilations)
    if len(dilations_int) != 2 or any(d < 1 for d in dilations_int):
        raise ValueError(
            f"dilations must contain EXACTLY 2 integers >= 1, "
            f"got: {dilations!r}"
        )
    if padding == "same":
        return seq_len_int
    if padding != "valid":
        raise ValueError(
            f"padding must be 'valid' or 'same', got: {padding!r}"
        )
    reduction = sum((kernel_int - 1) * d for d in dilations_int)
    att_len = seq_len_int - reduction
    if att_len <= 0:
        raise ValueError(
            f"invalid H^att length ({att_len}): seq_len={seq_len_int}, "
            f"kernel={kernel_int}, dilations={dilations_int} (conv 'valid' "
            f"would produce a length <= 0)"
        )
    return att_len

def _att_len_from_config(model_cfg: Dict[str, Any], seq_len: int) -> int:
    
    conv_padding = str(model_cfg["conv_padding"])
    conv_dilations = _read_conv_dilations(model_cfg, n_layers=2)
    return _expected_att_length(
        seq_len,
        int(model_cfg["conv_kernel"]),
        padding=conv_padding,
        dilations=conv_dilations,
    )

def _validate_backbone_config(
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], int, int]:
    
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got: {type(config).__name__}")

    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("'model' section missing or not a dict in config")
    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("'data' section missing or not a dict in config")

    _assert_config_finite(model_cfg)

    missing_model = [key for key in _BACKBONE_REQUIRED_KEYS if key not in model_cfg]
    if missing_model:
        raise ValueError(f"missing keys in model: {missing_model}")
    missing_data = [key for key in _DATA_REQUIRED_KEYS if key not in data_cfg]
    if missing_data:
        raise ValueError(f"missing keys in data: {missing_data}")

    backbone_type = str(model_cfg["backbone_type"])
    if backbone_type not in _VALID_BACKBONE_TYPES:
        raise ValueError(
            f"model.backbone_type {backbone_type!r} not supported by this "
            f"module (expected: {list(_VALID_BACKBONE_TYPES)}); for "
            "'qkv_attention' use src/models/ultra_can_qkv.py"
        )

    size = str(model_cfg["size"])
    if size not in _VALID_SIZES:
        raise ValueError(
            f"model.size must be one of {list(_VALID_SIZES)}, "
            f"got: {size!r}"
        )
    if size == "micro":
        logger.warning(
            "model.size='micro' is the reduced-size variant, but only the 'full' contract is supported"
            ""
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
    _read_conv_dilations(model_cfg, n_layers=2)

    seq_len = _as_positive_int(data_cfg["sequence_length"], "data.sequence_length")
    feature_mode = str(data_cfg["feature_mode"])
    if feature_mode not in _VALID_FEATURE_MODES:
        raise ValueError(
            f"data.feature_mode must be one of {list(_VALID_FEATURE_MODES)}, "
            f"got: {feature_mode!r}"
        )

    if conv_kernel >= seq_len:
        raise ValueError(
            f"model.conv_kernel ({conv_kernel}) must be < "
            f"data.sequence_length ({seq_len}): conv 'valid' would produce a "
            "length <= 0"
        )

    num_features = 1 if feature_mode == "real" else 2

    logger.debug(
        "backbone config validated: conv_filters=%s, conv_kernel=%d, padding=%s, "
        "seq_len=%d, feature_mode=%s, num_features=%d",
        list(conv_filters),
        conv_kernel,
        conv_padding,
        seq_len,
        feature_mode,
        num_features,
    )
    return model_cfg, seq_len, num_features

def _smoke_check_forward(
    model: tf.keras.Model,
    batch_sizes: Sequence[int],
    expected: Mapping[str, Sequence[Optional[int]]],
) -> None:
    
    if not tf.executing_eagerly():
        logger.debug("smoke check forward skipped (non-eager context)")
        return

    input_shape = tuple(model.input_shape)
    if len(input_shape) != 3 or input_shape[1] is None or input_shape[2] is None:
        raise ValueError(
            f"expected input shape (None, L, F), got: {input_shape!r}"
        )

    for batch_size in batch_sizes:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size must be a positive integer, got: {batch_size!r}"
            )
        dummy = tf.zeros(
            (batch_size,) + tuple(input_shape[1:]),
            dtype=model.inputs[0].dtype,
        )
        outputs = model(dummy, training=False)
        if isinstance(outputs, dict):
            outputs_map: Dict[str, tf.Tensor] = outputs
        else:
            name = model.output.name
            outputs_map = {name: outputs}

        for key, expected_shape in expected.items():
            if key not in outputs_map:
                raise ValueError(
                    f"{key}: output '{key}' not present in the model "
                    f"(available: {sorted(outputs_map.keys())})"
                )
            tensor = outputs_map[key]
            actual = tensor.shape.as_list()
            expected_list = list(expected_shape)
            if not expected_list or expected_list[0] is not None:
                raise ValueError(
                    f"{key}: expected must have None as the batch dimension "
                    f"(e.g. (None, D1, ...)), got: {expected_list}"
                )
            if (
                len(actual) != len(expected_list)
                or actual[0] != batch_size
                or actual[1:] != expected_list[1:]
            ):
                raise ValueError(
                    f"{key}: expected output shape {expected_list}, "
                    f"got: {actual}"
                )
            tf.debugging.assert_all_finite(
                tensor, message=f"{key}: output contains NaN/Inf"
            )
        logger.debug(
            "smoke check OK (batch=%d): %s",
            batch_size,
            {key: outputs_map[key].shape.as_list() for key in expected},
        )

def _build_backbone_graph(
    config: Dict[str, Any],
    r_input: tf.keras.KerasTensor,
    seq_len: int,
    num_features: int,
) -> tf.keras.KerasTensor:
    
    model_cfg, _, _ = _validate_backbone_config(config)
    conv_filters = model_cfg["conv_filters"]
    conv_kernel = int(model_cfg["conv_kernel"])
    conv_padding = str(model_cfg["conv_padding"])
    conv_dilations = _read_conv_dilations(model_cfg, n_layers=2)
    attention_residual = bool(model_cfg.get("attention_residual", True))
    attention_mode = str(model_cfg.get("attention_mode", "attention_pool"))
    pool_projection = bool(model_cfg.get("pool_projection", True))
    att_len = _expected_att_length(
        seq_len, conv_kernel, padding=conv_padding, dilations=conv_dilations
    )
    f1 = int(conv_filters[0])
    f2 = int(conv_filters[1])

    h1 = tf.keras.layers.Conv1D(
        filters=f1,
        kernel_size=conv_kernel,
        padding=conv_padding,
        dilation_rate=conv_dilations[0],
        activation="relu",
        kernel_initializer="he_normal",
        name="conv1",
    )(r_input)
    h2 = tf.keras.layers.Conv1D(
        filters=f2,
        kernel_size=conv_kernel,
        padding=conv_padding,
        dilation_rate=conv_dilations[1],
        activation="relu",
        kernel_initializer="he_normal",
        name="conv2",
    )(h1)
    if pool_projection:
        h2 = tf.keras.layers.Conv1D(
            filters=f2,
            kernel_size=1,
            padding=conv_padding,
            activation="relu",
            kernel_initializer="he_normal",
            name="pool_projection",
        )(h2)
    h_att = h2
    if attention_mode == "gate_residual":
        alpha = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=1,
            activation="sigmoid",
            name="attention_1d",
        )(h2)
        gated = tf.keras.layers.Multiply(name="attention_apply")([alpha, h2])
        if attention_residual:
            h_att = tf.keras.layers.Add(name="attention_residual")([h2, gated])
        else:
            h_att = gated

    if tuple(h_att.shape) != (None, att_len, f2):
        raise ValueError(
            f"expected H^att shape (None, {att_len}, {f2}) (conv "
            f"{seq_len} -> {seq_len - (conv_kernel - 1)} -> {att_len}), "
            f"got: {tuple(h_att.shape)}"
        )
    return h_att

def build_backbone(
    config: Dict[str, Any],
    input_tensor: Optional[tf.keras.KerasTensor] = None,
):
    
    model_cfg, seq_len, num_features = _validate_backbone_config(config)
    if input_tensor is not None:
        if tuple(input_tensor.shape[1:]) != (seq_len, num_features):
            raise ValueError(
                f"expected input_tensor shape (None, {seq_len}, {num_features}), "
                f"got: {tuple(input_tensor.shape)}"
            )
        h_att = _build_backbone_graph(config, input_tensor, seq_len, num_features)
        logger.debug(
            "backbone '%s' (shared layers): H^att=(None,%d,%d)",
            _BACKBONE_NAME,
            _att_len_from_config(model_cfg, seq_len),
            int(model_cfg["conv_filters"][1]),
        )
        return h_att

    r_input = tf.keras.layers.Input(shape=(seq_len, num_features), name="r_input")
    h_att = _build_backbone_graph(config, r_input, seq_len, num_features)
    backbone = tf.keras.Model(inputs=r_input, outputs=h_att, name=_BACKBONE_NAME)
    _smoke_check_forward(
        backbone,
        _SMOKE_BATCH_SIZES,
        expected={
            backbone.output.name: (
                None,
                _att_len_from_config(model_cfg, seq_len),
                int(model_cfg["conv_filters"][1]),
            )
        },
    )
    logger.info(
        "backbone '%s' built: input=(None,%d,%d), H^att=(None,%d,%d), params=%d",
        _BACKBONE_NAME,
        seq_len,
        num_features,
        _att_len_from_config(model_cfg, seq_len),
        int(model_cfg["conv_filters"][1]),
        count_trainable_params(backbone),
    )
    return backbone

def build_ultra_can(config: Dict[str, Any]) -> tf.keras.Model:
    
    comm_head = build_communication_head(config)

    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("'model' section missing or not a dict in config")
    conv_filters = model_cfg["conv_filters"]
    conv_kernel = int(model_cfg["conv_kernel"])
    seq_len = int(config["data"]["sequence_length"])
    att_len = _att_len_from_config(model_cfg, seq_len)
    f2 = int(conv_filters[1])
    feature_mode = str(config.get("data", {}).get("feature_mode", "real"))
    num_features = 1 if feature_mode == "real" else 2
    model_input = tf.keras.layers.Input(
        shape=(seq_len, num_features + 1), name="isac_input"
    )
    received = tf.keras.layers.Lambda(
        _slice_received,
        arguments={"num_features": num_features},
        name="received_slice",
    )(model_input)

    h_att = build_backbone(config, input_tensor=received)

    attn_mode = str(model_cfg.get("attention_mode", "attention_pool"))
    v, v_sensing = build_sensing_features(
        h_att, config, received_input=model_input, attention_mode=attn_mode
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
        name=_MODEL_NAME,
    )
    _smoke_check_forward(
        model,
        _SMOKE_BATCH_SIZES,
        expected={
            "comm": (None, modulation_order),
            "sensing": (None, sensing_units),
        },
    )
    logger.info(
        "model '%s' built: comm=%d params, sensing=%d params, TOTAL=%d params",
        _MODEL_NAME,
        count_trainable_params(comm_head),
        count_trainable_params(sensing_head),
        count_trainable_params(model),
    )
    return model

def count_trainable_params(model: tf.keras.Model) -> int:
    
    if not isinstance(model, tf.keras.Model):
        raise TypeError(
            f"model must be a tf.keras.Model, got: {type(model).__name__}"
        )
    return num_params(model)

def main(argv: Optional[Sequence[str]] = None) -> None:
    
    parser = argparse.ArgumentParser(
        description="Smoke test for the Ultra-CAN (ISAC in IoD) model"
        ""
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

    model = build_ultra_can(config)
    logger.info(
        "Ultra-CAN model ready: %s (TOTAL=%d params, "
        "footprint ~%.0f kB at 4 bytes/param)",
        model.name,
        count_trainable_params(model),
        count_trainable_params(model) * 4.0 / 1024.0,
    )

if __name__ == "__main__":
    main()

