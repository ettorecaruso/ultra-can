from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from src.data.data_loader import (
    _build_feature_matrix,
    build_reference_matrix,
    load_npz_files,
)
from src.experiments import pipeline
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, _deep_merge
from src.utils.dataset_utils import get_dataset_dir
from src.utils.model_io import load_model

SEQ_LEN = 100
PROFILE_LEN = 99
DEPLOY_MODELS = ("conv1d", "qkv")
DEFAULT_SCENARIO = "iod_peers"
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_config(
    experiment: str = "ber_vs_snr",
    variant: str = "full",
    scenario: str = DEFAULT_SCENARIO,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import yaml

    base = yaml.safe_load(DEFAULT_BASE_CONFIG_PATH.read_text())
    exp = yaml.safe_load((REPO_ROOT / "configs" / "experiments.yaml").read_text())
    block = _deep_merge(base, exp[experiment][variant])
    scenarios = block["experiments"][experiment]["scenarios"]
    scenario_cfg = next(s for s in scenarios if s.get("name") == scenario)
    return pipeline._apply_scenario_config(block, scenario_cfg), scenario_cfg


def dataset_dir_for(cfg: Dict[str, Any]) -> Path:
    return get_dataset_dir(cfg)


def load_burst_split(
    cfg: Dict[str, Any],
    snr_values: Sequence[float],
    n_max: int,
    data_dir: Optional[Path] = None,
) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    root = Path(data_dir) if data_dir is not None else dataset_dir_for(cfg)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {root}")
    echoes = [int(k) for k in cfg["data"]["echoes"]]
    mode = str(cfg["data"].get("feature_mode", "iq"))
    norm = cfg["data"].get("feature_norm")
    out: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    for snr in snr_values:
        data = load_npz_files(root, [float(snr)], echoes, "test", cfg)
        n = min(int(n_max), int(np.asarray(data["bit"]).shape[0]))
        bit = np.asarray(data["bit"][:n]).astype(np.int64)
        reference = build_reference_matrix(bit, np.asarray(data["seed"][:n]), cfg)
        features = _build_feature_matrix(
            np.asarray(data["x"][:n]), mode, norm, reference
        )
        out[float(snr)] = (features.astype(np.float32), bit)
    return out


def dca_profile(iq: np.ndarray, reference: np.ndarray, max_delay: int) -> np.ndarray:
    z = (iq[..., 0].astype(np.float32) + 1j * iq[..., 1].astype(np.float32)).astype(
        np.complex64
    )
    xr = reference.astype(np.float32).astype(np.complex64)
    energy = np.sum(np.abs(xr) ** 2, axis=1).astype(np.float32)
    energy = np.where(energy < 1e-12, 1.0, energy).astype(np.float32)
    direct = (np.sum(z * xr, axis=1) / energy).astype(np.complex64)
    residual = z - direct[:, None] * xr
    mag, real, imag = [], [], []
    for lag in range(1, max_delay + 1):
        cc = np.sum(np.conj(residual[:, lag:]) * xr[:, :-lag], axis=1)
        mag.append(np.abs(cc))
        real.append(np.real(cc))
        imag.append(np.imag(cc))
    return np.stack(mag + real + imag, axis=1).astype(np.float32)


def _conv2d_from_conv1d(layer: tf.keras.layers.Layer, x, name: str):
    weights, bias = layer.get_weights()
    conv = tf.keras.layers.Conv2D(
        filters=weights.shape[-1],
        kernel_size=(weights.shape[0], 1),
        padding="same",
        activation="relu",
        name=name,
    )
    y = conv(x)
    conv.set_weights([weights[:, None, :, :], bias])
    return y


def build_deploy_2d(model: tf.keras.Model, arch: str) -> tf.keras.Model:
    iq = tf.keras.Input(shape=(SEQ_LEN, 1, 2), name="iq")
    prof = tf.keras.Input(shape=(PROFILE_LEN,), name="profile")
    x = _conv2d_from_conv1d(model.get_layer("conv1"), iq, "conv1_2d")
    x = _conv2d_from_conv1d(model.get_layer("conv2"), x, "conv2_2d")
    if arch == "conv1d":
        x = _conv2d_from_conv1d(model.get_layer("pool_projection"), x, "pool_2d")
        x = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=2), name="squeeze")(x)
        alpha = tf.keras.layers.Softmax(axis=1, name="deploy_attn_weights")(
            model.get_layer("attn_scores")(x)
        )
        v = tf.keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1], axis=1), name="deploy_attn_pool"
        )([alpha, x])
        v_max = tf.keras.layers.GlobalMaxPooling1D(name="deploy_max")(x)
    else:
        x = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=2), name="squeeze")(x)
        h = model.get_layer("qkv_attention")(x)
        v = model.get_layer("pool_projection")(
            tf.keras.layers.GlobalAveragePooling1D(name="deploy_gap")(h)
        )
        v_max = tf.keras.layers.GlobalMaxPooling1D(name="deploy_max")(h)
    sensing_in = tf.keras.layers.Concatenate(name="deploy_sensing_features")(
        [v, v_max, prof]
    )
    return tf.keras.Model(
        [iq, prof],
        [model.get_layer("communication_head")(v),
         model.get_layer("sensing_head")(sensing_in)],
        name=f"ultra_can_{arch}_deploy2d",
    )


def build_deploy_1d(model: tf.keras.Model, arch: str) -> tf.keras.Model:
    iq = tf.keras.Input(batch_shape=(1, SEQ_LEN, 2), name="iq")
    prof = tf.keras.Input(batch_shape=(1, PROFILE_LEN), name="profile")
    x = model.get_layer("conv1")(iq)
    x = model.get_layer("conv2")(x)
    if arch == "conv1d":
        x = model.get_layer("pool_projection")(x)
        alpha = tf.keras.layers.Softmax(axis=1, name="deploy_attn_weights")(
            model.get_layer("attn_scores")(x)
        )
        v = tf.keras.layers.Lambda(
            lambda z: tf.reduce_sum(z[0] * z[1], axis=1), name="deploy_attn_pool"
        )([alpha, x])
        v_max = tf.keras.layers.GlobalMaxPooling1D(name="deploy_max")(x)
    else:
        h = model.get_layer("qkv_attention")(x)
        v = model.get_layer("pool_projection")(
            tf.keras.layers.GlobalAveragePooling1D(name="deploy_gap")(h)
        )
        v_max = tf.keras.layers.GlobalMaxPooling1D(name="deploy_max")(h)
    sensing_in = tf.keras.layers.Concatenate(name="deploy_sensing_features")(
        [v, v_max, prof]
    )
    return tf.keras.Model(
        [iq, prof],
        [model.get_layer("communication_head")(v),
         model.get_layer("sensing_head")(sensing_in)],
        name=f"ultra_can_{arch}_deploy1d",
    )


def load_checkpoint(arch: str, scenario: str = DEFAULT_SCENARIO) -> tf.keras.Model:
    path = (REPO_ROOT / "results" / "full" / "ber_vs_snr" / scenario / arch
            / "best_model.keras")
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return load_model(path)


def export_int8_model(
    deploy_1d: tf.keras.Model,
    iq: np.ndarray,
    profile: np.ndarray,
    out_path: Path,
    n_cal: int = 400,
    seed: int = 7,
) -> int:
    rng = np.random.default_rng(seed)
    index = rng.choice(len(iq), size=min(int(n_cal), len(iq)), replace=False)
    iq_1d = iq.reshape((-1, SEQ_LEN, 2))

    def make_representative(order: Sequence[str]):
        def representative():
            for i in index:
                sample = {"iq": iq_1d[i:i + 1], "profile": profile[i:i + 1]}
                yield [sample[name] for name in order]

        return representative

    blob = None
    for order in (("profile", "iq"), ("iq", "profile")):
        converter = tf.lite.TFLiteConverter.from_keras_model(deploy_1d)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = make_representative(order)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        try:
            blob = converter.convert()
            break
        except Exception:
            blob = None
    if blob is None:
        raise RuntimeError(f"int8 conversion failed for {deploy_1d.name}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    return len(blob)


def _quantize(array: np.ndarray, details: Dict[str, Any]) -> np.ndarray:
    scale, zero_point = details["quantization"]
    return np.clip(np.round(array / scale) + zero_point, -128, 127).astype(np.int8)


def _dequantize(array: np.ndarray, details: Dict[str, Any]) -> np.ndarray:
    scale, zero_point = details["quantization"]
    return (array.astype(np.float32) - zero_point) * scale


class Int8Model:

    def __init__(self, path: Path):
        self.interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
        self.interpreter.allocate_tensors()
        ins = self.interpreter.get_input_details()
        outs = self.interpreter.get_output_details()
        self.d_iq = next(d for d in ins if int(np.prod(d["shape"])) == 2 * SEQ_LEN)
        self.d_profile = next(d for d in ins if int(np.prod(d["shape"])) == PROFILE_LEN)
        self.d_comm = next(d for d in outs if int(d["shape"][-1]) == 2)
        self.d_sensing = next(d for d in outs if int(d["shape"][-1]) != 2)
        self.iq_shape = tuple(int(s) for s in self.d_iq["shape"])
        self.profile_shape = tuple(int(s) for s in self.d_profile["shape"])

    def __call__(self, features, training: bool = False):
        x = np.asarray(features, dtype=np.float32)
        iq = x[..., :2].reshape((-1, SEQ_LEN, 1, 2))
        profile = dca_profile(x[..., :2], x[..., -1], PROFILE_LEN // 3)
        n = len(x)
        comm = np.empty((n, 2), dtype=np.float32)
        sensing = np.empty((n, int(self.d_sensing["shape"][-1])), dtype=np.float32)
        for i in range(n):
            self.interpreter.set_tensor(
                self.d_iq["index"],
                _quantize(iq[i:i + 1].reshape(self.iq_shape), self.d_iq))
            self.interpreter.set_tensor(
                self.d_profile["index"],
                _quantize(profile[i:i + 1].reshape(self.profile_shape), self.d_profile))
            self.interpreter.invoke()
            comm[i] = _dequantize(
                self.interpreter.get_tensor(self.d_comm["index"])[0], self.d_comm)
            sensing[i] = _dequantize(
                self.interpreter.get_tensor(self.d_sensing["index"])[0], self.d_sensing)
        return {"comm": tf.constant(comm), "sensing": tf.constant(sensing)}


def ber_of(logits: np.ndarray, bits: np.ndarray) -> float:
    return float(np.mean(np.argmax(logits, axis=-1) != bits))


