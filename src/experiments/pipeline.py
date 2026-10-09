

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.dataset_generator import generate_dataset
from src.data.data_loader import (
    build_reference_matrix,
    load_npz_files,
    verify_snr_balance,
)
from src.evaluation.compare_baselines import plot_ber_overlay
from src.evaluation.evaluator import (
    compute_ber_curve,
    evaluate_model,
    evaluate_model_online,
    plot_ber_vs_snr,
)
from src.evaluation.weight_comparison import generate_weight_table as gen_weight_table
from src.models.baselines import build_baseline
from src.models.ultra_can import build_ultra_can
from src.models.ultra_can_qkv import build_ultra_can_qkv
from src.training.trainer import Trainer
from src.utils.config_loader import _deep_merge, save_config_snapshot, validate_config
from src.utils.dataset_utils import get_dataset_dir
from src.utils.logger import get_logger
from src.utils.model_names import canonical_model_name

logger = get_logger(__name__)

_REQUIRED_KEYS: Tuple[str, ...] = (
    "general.experiment_name",
    "general.seed",
    "general.log_level",
    "data.sequence_length",
    "data.feature_mode",
    "data.snr_range",
    "data.snr_step",
    "data.echoes",
    "data.max_delay",
    "data.max_doppler",
    "data.num_symbols_train",
    "data.num_symbols_val",
    "data.num_symbols_test",
    "training.batch_size",
    "training.lambda_mse",
    "training.loss_weights.comm",
    "training.loss_weights.sensing",
    "model.backbone_type",
)

def prepare_dataset(config: dict, no_regen: bool = False) -> Path:
    from src.data.dataset_generator import generate_dataset
    from src.utils.dataset_utils import get_dataset_dir

    data_dir = get_dataset_dir(config)
    if not no_regen:
        splits = ["train", "val", "test"]
        missing = False
        for split in splits:
            if not any(data_dir.glob(f"{split}_*.npz")):
                missing = True
                break
        if missing:
            logger.info("Dataset not found in %s, generating...", data_dir)
            for split in splits:
                generate_dataset(config, split, data_dir)
        else:
            logger.info("Dataset already present in %s", data_dir)
    else:
        logger.info("no_regen=True, reusing existing dataset in %s", data_dir)
    return data_dir

def load_datasets(
    config: Dict[str, Any],
    data_dir: Path,
    include_test: bool = True,
) -> Tuple[Dict[str, np.ndarray], tf.data.Dataset, tf.data.Dataset, Optional[Dict[str, np.ndarray]]]:
    from src.data.dataset_generator import build_snr_grid
    from src.data.data_loader import load_npz_files, verify_snr_balance, build_tf_dataset

    data_cfg = config["data"]
    snr_grid = build_snr_grid(data_cfg["snr_range"], data_cfg["snr_step"])
    echoes = list(data_cfg["echoes"])

    splits = ["train", "val"] + (["test"] if include_test else [])
    data_dicts = {}
    for split in splits:
        data = load_npz_files(data_dir, snr_grid, echoes, split, config)
        verify_snr_balance(data, snr_grid, echoes)
        data["x_ref"] = build_reference_matrix(data["bit"], data["seed"], config)
        data_dicts[split] = data
        logger.info("Split '%s': %d samples", split, data["x"].shape[0])

    batch_size = int(config["training"]["batch_size"])
    seed = int(config["general"]["seed"])
    train_ds = build_tf_dataset(
        data_dicts["train"],
        batch_size=batch_size,
        config=config,
        shuffle=True,
        seed=seed,
    )
    val_ds = build_tf_dataset(
        data_dicts["val"],
        batch_size=batch_size,
        config=config,
        shuffle=False,
        seed=seed,
    )
    return (
        data_dicts["train"],
        train_ds,
        val_ds,
        data_dicts["test"] if include_test else None,
    )

def load_test_data(config: Dict[str, Any], data_dir: Path) -> Dict[str, np.ndarray]:
    from src.data.dataset_generator import build_snr_grid
    from src.data.data_loader import (
        load_npz_files,
        verify_snr_balance,
        build_reference_matrix,
    )

    data_cfg = config["data"]
    snr_grid = build_snr_grid(data_cfg["snr_range"], data_cfg["snr_step"])
    echoes = list(data_cfg["echoes"])
    test_data = load_npz_files(data_dir, snr_grid, echoes, "test", config)
    verify_snr_balance(test_data, snr_grid, echoes)
    test_data["x_ref"] = build_reference_matrix(
        test_data["bit"], test_data["seed"], config
    )
    logger.info("Static test set loaded: %d samples", test_data["x"].shape[0])
    return test_data

def build_model(config: Dict[str, Any], model_type: str) -> tf.keras.Model:

    model_type = canonical_model_name(model_type)

    if model_type == "qkv":
        config["model"]["backbone_type"] = "qkv_attention"
    else:
        config["model"]["backbone_type"] = model_type

    general = config.get("general") or {}
    model_cfg = config.get("model") or {}
    _seed = int(
        (model_cfg.get("seeds") or {}).get(model_type, general.get("seed", 42))
    )
    random.seed(_seed)
    np.random.seed(_seed)
    tf.random.set_seed(_seed)
    logger.info(
        "build_model '%s': model seed applied = %d (from model.seeds.%s, "
        "dataset seed stays %d)",
        model_type, _seed, model_type, int(general.get("seed", 42)),
    )

    builders = {
        "conv1d": build_ultra_can,
        "qkv": build_ultra_can_qkv,
        "lstm": lambda cfg: build_baseline(cfg, "lstm"),
        "mc_dlsk": lambda cfg: build_baseline(cfg, "mc_dlsk"),
    }
    if model_type not in builders:
        raise ValueError(
            f"model_type not supported: {model_type!r} (expected: {list(builders.keys())})"
        )

    model = builders[model_type](config)

    if model.input_shape is None:
        data_cfg = config["data"]
        seq_len = int(data_cfg["sequence_length"])
        feature_mode = str(data_cfg.get("feature_mode", "real"))
        num_features = 2 if feature_mode == "iq" else 1
        model.build(input_shape=(None, seq_len, num_features))
        logger.debug(
            "Model %s built with input_shape=(None,%d,%d)",
            model_type, seq_len, num_features
        )

    dummy = tf.zeros((1,) + tuple(model.input_shape[1:]), dtype=model.inputs[0].dtype)
    outputs = model(dummy, training=False)

    tf.debugging.assert_shapes([
        (outputs["comm"], ("B", "M")),
        (outputs["sensing"], ("B", "S")),
    ])

    for name, tensor in outputs.items():
        tf.debugging.assert_all_finite(tensor, f"Output '{name}' not finite in build.")

    logger.info(
        "Model '%s' built: %d trainable parameters.",
        model.name, model.count_params(),
    )
    return model

def train_and_evaluate(
    config: Dict[str, Any],
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_data: Dict[str, np.ndarray],
    output_dir: Path
) -> Dict[str, Any]:
    
    epochs = int(config["training"]["epochs"])
    lambda_mse = float(config["training"]["lambda_mse"])
    logger.info("Training for %d epochs, lambda_mse=%.4f", epochs, lambda_mse)

    trainer = Trainer(config, model, train_ds, val_ds)
    history = trainer.train()

    trainer.restore_best_weights()

    if config.get("evaluation", {}).get("online_generation", False):
        eval_results = evaluate_model_online(model, config)
    else:
        if test_data is None:
            raise RuntimeError(
                "train_and_evaluate: offline evaluation requires the static test "
                "split, but test_data is None (load_datasets was called with "
                "include_test=False). Use include_test=True when "
                "evaluation.online_generation is disabled."
            )
        eval_results = evaluate_model(model, test_data, config)
    df_ber = compute_ber_curve(eval_results)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    df_ber.to_csv(csv_path, index=False)
    logger.info("Metrics saved to %s", csv_path)

    plot_format = config["visualization"].get("plot_format", "pdf")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    ber_plot_path = plot_ber_vs_snr(df_ber, plot_dir, plot_format, model.name)
    logger.info("BER vs SNR plot saved to %s", ber_plot_path)

    best_model_path = output_dir / "best_model.keras"
    model.save(best_model_path)
    logger.info("Model saved to %s", best_model_path)

    if "ber" in df_ber.columns:
        ber_values = df_ber["ber"].values
        if not np.all(np.isfinite(ber_values)):
            raise ValueError("BER contains NaN or Inf")
        if np.any((ber_values < 0) | (ber_values > 1)):
            raise ValueError("BER out of range [0,1]")

    return {
        "history": history,
        "metrics": df_ber,
        "plots": {"ber_vs_snr": ber_plot_path},
        "model": model
    }

def run_single_experiment(
    config: Dict[str, Any],
    model_type: str,
    output_dir: Path,
    no_regen: bool = False
) -> Dict[str, Any]:
    
    data_dir = prepare_dataset(config, no_regen)

    online_eval = bool(
        (config.get("evaluation") or {}).get("online_generation", False)
    )
    train_data, train_ds, val_ds, test_data = load_datasets(
        config, data_dir, include_test=not online_eval
    )

    model = build_model(config, model_type)

    config.setdefault("general", {})["run_output_dir"] = str(output_dir)

    result = train_and_evaluate(
        config, model, train_ds, val_ds, test_data, output_dir
    )

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, logs_dir)

    return result

def _build_jsr_values(jamming_cfg: Dict[str, Any]) -> List[float]:
    
    jamming_cfg = jamming_cfg or {}
    if jamming_cfg.get("jsr_values"):
        return [float(v) for v in jamming_cfg["jsr_values"]]
    jsr_range = jamming_cfg.get("jsr_range") or [0, 10]
    jsr_step = float(jamming_cfg.get("jsr_step", 2.0))
    jsr_lo, jsr_hi = float(jsr_range[0]), float(jsr_range[1])
    if jsr_step <= 0.0:
        raise ValueError(f"jamming.jsr_step must be > 0, got: {jsr_step}")
    return [round(float(v), 6) for v in np.arange(jsr_lo, jsr_hi + jsr_step / 2.0, jsr_step)]

def evaluate_with_jamming(
    config: Dict[str, Any],
    model: tf.keras.Model,
    test_data: Dict[str, np.ndarray],
    output_dir: Path,
    model_name: str = "model",
) -> Dict[str, Any]:
    
    from src.experiments.run_jamming import evaluate_jamming

    jamming_cfg = config.get("jamming", {}) or {}
    jammer_types = list(jamming_cfg.get("jamming_types") or ["cw", "barrage", "partial_band"])
    jsr_values = _build_jsr_values(jamming_cfg)
    max_symbols = (config.get("experiments", {}).get("jamming") or {}).get("max_symbols")
    per_arch = (config.get("experiments", {}).get("jamming") or {}).get(
        "n_realizations_by_arch"
    ) or {}
    n_realizations = int(
        per_arch.get(model_name, jamming_cfg.get("n_realizations", 1))
    )
    logger.info(
        "Starting jamming evaluation for JSR = %s, types = %s, max_symbols = %s, "
        "realizations = %d",
        jsr_values, jammer_types, max_symbols, n_realizations,
    )

    results = evaluate_jamming(
        model=model,
        test_data=test_data,
        config=config,
        jsr_values=jsr_values,
        jammer_types=jammer_types,
        output_dir=output_dir,
        model_name=model_name,
        n_realizations=n_realizations,
        max_symbols=int(max_symbols) if max_symbols is not None else None,
    )

    logger.info("Jamming evaluation completed. Results saved to %s", output_dir)
    return results

def generate_weight_table(
    config: Dict[str, Any],
    models: Dict[str, Optional[tf.keras.Model]],
    output_path: Path
) -> Path:
    
    table_cfg = config.get("experiments", {}).get("final_report", {})
    include_classical = table_cfg.get("include_classical", True)

    tex_path = gen_weight_table(
        models=models,
        output_path=output_path,
        include_classical=include_classical
    )

    logger.info("Weight table generated: %s", tex_path)
    return tex_path

def collect_and_plot_overlay(
    curves: Dict[str, pd.DataFrame],
    scenario: str,
    output_dir: Path,
    plot_format: str = "pdf"
) -> Path:
    
    plot_path = plot_ber_overlay(
        curves=curves,
        scenario=scenario,
        output_dir=output_dir,
        plot_format=plot_format
    )

    logger.info("Overlay saved to %s", plot_path)
    return plot_path

def plot_layer_activity(
    config: Dict[str, Any],
    model: tf.keras.Model,
    test_data: Dict[str, np.ndarray],
    output_dir: Path
) -> None:
    
    from src.data.data_loader import _build_feature_matrix
    from src.visualization.attention_visualizer import plot_attention_maps
    from src.visualization.layer_activity_visualizer import (
        layer_activation_variance,
        layer_weight_norms,
        plot_layer_activity as _plot_activity_bars,
    )

    vis_cfg = config.get("visualization") or {}
    num_samples = int(vis_cfg.get("num_samples", 5))
    feature_mode = str(config.get("data", {}).get("feature_mode", "real"))
    plot_format = str(vis_cfg.get("plot_format", "pdf"))

    attention_dir = output_dir / "attention"
    try:
        plot_attention_maps(
            model=model,
            test_data=test_data,
            output_dir=attention_dir,
            num_samples=num_samples,
            config=config,
            plot_format=plot_format,
        )
    except ValueError as exc:
        logger.warning(
            "Attention maps cannot be generated (no attention layer in "
            "model '%s'?): %s",
            model.name, exc,
        )
    except Exception as exc:
        logger.error("Attention map generation failed: %s", exc)

    activity_dir = output_dir / "activity"
    activity_dir.mkdir(parents=True, exist_ok=True)
    try:
        weight_norms = layer_weight_norms(model)
        if weight_norms:
            _plot_activity_bars(
                weight_norms,
                activity_dir / f"layer_activity_weights.{plot_format}",
                title_prefix="Per-layer L2 weight norm",
            )
        else:
            logger.warning("No layer with trainable weights: skipping the L2 norms")
    except Exception as exc:
        logger.error("Weight L2 norm computation failed: %s", exc)

    try:
        x_first = np.asarray(test_data["x"])[0:1]
        sample = _build_feature_matrix(x_first, feature_mode)
        activation_variance = layer_activation_variance(model, sample)
        if activation_variance:
            _plot_activity_bars(
                activation_variance,
                activity_dir / f"layer_activity_variance.{plot_format}",
                title_prefix="Per-layer activation variance",
            )
        else:
            logger.warning("No activation variance computed")
    except Exception as exc:
        logger.error("Activation variance computation failed: %s", exc)

    logger.info("Layer activity plots generated in %s", output_dir)

_EXP_BLOCKS: Tuple[str, ...] = ("ber_vs_snr", "classical_receivers", "jamming", "jamming_interpretability", "final_report")
_EXP_MODES: Tuple[str, ...] = ("full", "fast")

def _merge_data_config(base_config: Dict[str, Any], data_params: Dict[str, Any]) -> Dict[str, Any]:
    
    merged = base_config.copy()
    merged["data"] = {**base_config.get("data", {}), **data_params}
    return merged

def _apply_scenario_config(config: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
    
    sc_cfg = config.copy()
    data_overrides: Dict[str, Any] = {
        "echoes": list(scenario["echoes"]),
        "max_doppler": float(scenario["max_doppler"]),
    }
    scenario_data = scenario.get("data")
    if scenario_data is not None:
        if not isinstance(scenario_data, dict):
            raise ValueError(
                f"scenario '{scenario.get('name', '?')}': the 'data' field must "
                f"be a dict of data.* overrides, got: {type(scenario_data).__name__}"
            )
        data_overrides.update(scenario_data)
    sc_cfg["data"] = {**config["data"], **data_overrides}
    scenario_peers = scenario.get("peers")
    if scenario_peers is not None:
        if not isinstance(scenario_peers, dict):
            raise ValueError(
                f"scenario '{scenario.get('name', '?')}': the 'peers' field must be "
                f"a dict of peers.* overrides, got: {type(scenario_peers).__name__}"
            )
        sc_cfg["peers"] = {**(config.get("peers") or {}), **scenario_peers}
    scenario_channel = scenario.get("channel")
    if scenario_channel is not None:
        if not isinstance(scenario_channel, dict):
            raise ValueError(
                f"scenario '{scenario.get('name', '?')}': the 'channel' field must "
                f"be a dict of channel.* overrides, got: {type(scenario_channel).__name__}"
            )
        sc_cfg["channel"] = _deep_merge(config.get("channel") or {}, scenario_channel)
    scenario_model = scenario.get("model")
    if scenario_model is not None:
        if not isinstance(scenario_model, dict):
            raise ValueError(
                f"scenario '{scenario.get('name', '?')}': the 'model' field must "
                f"be a dict of model.* overrides, got: {type(scenario_model).__name__}"
            )
        sc_cfg["model"] = _deep_merge(config.get("model") or {}, scenario_model)
    scenario_training = scenario.get("training")
    if scenario_training is not None:
        if not isinstance(scenario_training, dict):
            raise ValueError(
                f"scenario '{scenario.get('name', '?')}': the 'training' field must "
                f"be a dict of training.* overrides, got: {type(scenario_training).__name__}"
            )
        sc_cfg["training"] = _deep_merge(config.get("training") or {}, scenario_training)
    return sc_cfg

_PREDICT_CHUNK = 512

def predict_in_chunks(
    model: tf.keras.Model,
    features: Any,
    output: str = "comm",
    chunk: int = _PREDICT_CHUNK,
) -> np.ndarray:
    parts = []
    total = int(np.shape(features)[0])
    size = max(1, int(chunk))
    for start in range(0, total, size):
        block = features[start:start + size]
        parts.append(np.asarray(model(block, training=False)[output]))
    return np.concatenate(parts, axis=0)

def resolve_scenario(
    config: Dict[str, Any],
    scenario_name: str,
    experiment: str = "ber_vs_snr",
    profile: str = "full",
) -> Dict[str, Any]:
    section = (config.get("experiments") or {}).get(experiment) or {}
    candidates = []
    campaign = config.get(experiment)
    if isinstance(campaign, dict):
        nested = (campaign.get(profile) or {}).get("experiments") or {}
        nested_scenarios = (nested.get(experiment) or {}).get("scenarios")
        if nested_scenarios:
            candidates.append(list(nested_scenarios))
    if section.get("scenarios"):
        candidates.append(list(section["scenarios"]))
    if not candidates:
        raise ValueError(
            f"no scenarios for '{experiment}' in the config: cannot resolve "
            f"'{scenario_name}'"
        )
    for scenarios in candidates:
        for scenario in scenarios:
            if str(scenario.get("name")) == str(scenario_name):
                return _apply_scenario_config(config, scenario)
    known = sorted({str(item.get("name")) for scenarios in candidates for item in scenarios})
    raise ValueError(
        f"unknown scenario '{scenario_name}' for '{experiment}'; known: "
        + ", ".join(known)
    )

def prepare_all_datasets(
    config_path: Path,
    split: Optional[str] = None,
    no_regen: bool = False
) -> None:
    
    from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_config = load_config(config_path, DEFAULT_BASE_CONFIG_PATH)

    all_configs: List[Dict[str, Any]] = []
    for exp_name in _EXP_BLOCKS:
        exp_cfg = base_config.get(exp_name)
        if not isinstance(exp_cfg, dict):
            continue
        for mode in _EXP_MODES:
            mode_cfg = exp_cfg.get(mode)
            if not isinstance(mode_cfg, dict):
                continue
            data_params = mode_cfg.get("data")
            if not isinstance(data_params, dict):
                continue
            merged = _merge_data_config(base_config, data_params)

            exp_section = mode_cfg.get("experiments")
            if not isinstance(exp_section, dict):
                exp_section = {}

            if exp_name == "ber_vs_snr":
                scenarios = (
                    (exp_section.get("ber_vs_snr") or {}).get("scenarios") or []
                )
                for scenario in scenarios:
                    if not isinstance(scenario, dict) or "name" not in scenario:
                        logger.warning(
                            "ber_vs_snr.%s: malformed scenario in experiments.yaml, skipped: %s",
                            mode, scenario,
                        )
                        continue
                    all_configs.append(_apply_scenario_config(merged, scenario))
                continue

            if exp_name in ("jamming", "jamming_interpretability"):
                all_configs.append(merged)

    if not all_configs:
        logger.warning("No dataset configuration found in %s", config_path)
        return

    seen_hashes: set[str] = set()
    for cfg in all_configs:
        try:
            data_dir = get_dataset_dir(cfg)
        except Exception as e:
            logger.warning("Unable to compute hash for configuration: %s", e)
            continue

        if str(data_dir) in seen_hashes:
            continue
        seen_hashes.add(str(data_dir))

        if split is None:
            logger.info("Generating dataset for %s", data_dir)
            prepare_dataset(cfg, no_regen=no_regen)
        else:
            if split not in ("train", "val", "test"):
                raise ValueError(
                    f"split must be in ('train','val','test'), got: {split!r}"
                )
            logger.info("Generating split '%s' for %s", split, data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            generate_dataset(cfg, split, data_dir)

    logger.info(
        "Dataset preparation completed. %d unique configurations processed.",
        len(seen_hashes)
    )
