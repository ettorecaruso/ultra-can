

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset_generator import DirectPathParams, EchoParams
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config
from src.utils.logger import get_logger, log_config_summary, setup_logging
from src.utils.model_io import load_model

logger = get_logger(__name__)

_ATTENTION_LAYER_NAMES = ("attention_1d", "attn_weights", "qkv_attention")
_SAMPLE_BATCH_DIM = 1
_DEFAULT_SEED = 42

def extract_attention_weights(model: tf.keras.Model, sample: np.ndarray) -> np.ndarray:
    
    if not isinstance(model, tf.keras.Model):
        raise TypeError(
            f"model must be a tf.keras.Model, got: {type(model).__name__}"
        )
    if not isinstance(sample, np.ndarray):
        raise TypeError(
            f"sample must be a np.ndarray, got: {type(sample).__name__}"
        )

    if sample.ndim != 3 or sample.shape[0] != _SAMPLE_BATCH_DIM:
        raise ValueError(
            f"sample must have shape (1, L, F), got: {sample.shape}"
        )
    if sample.shape[1] <= 0 or sample.shape[2] <= 0:
        raise ValueError(
            f"the dimensions of sample must be positive, got: {sample.shape}"
        )
    if not np.all(np.isfinite(sample)):
        raise ValueError("sample contains NaN/Inf")

    attn_layer = None
    for layer in model.layers:
        if layer.name in _ATTENTION_LAYER_NAMES:
            attn_layer = layer
            break

    if attn_layer is None:
        raise ValueError(
            f"No attention layer found in the model. "
            f"Expected: {_ATTENTION_LAYER_NAMES}"
        )

    if attn_layer.name == "qkv_attention":
        attn_layer._store_attention_weights = True
        _ = model(sample, training=False)
        alpha = attn_layer.last_attention_weights
        if alpha is None:
            raise RuntimeError(
                "The QKV layer did not return attention weights "
                "(last_attention_weights is None)"
            )
        alpha = alpha.numpy()
    else:
        intermediate = tf.keras.Model(
            inputs=model.input,
            outputs=attn_layer.output,
            name="attention_extractor",
        )
        alpha = intermediate(sample, training=False).numpy()

    if not np.all(np.isfinite(alpha)):
        raise RuntimeError(
            "Attention weights contain NaN/Inf"
        )

    logger.debug(
        "Attention weights extracted: layer=%s, shape=%s",
        attn_layer.name,
        alpha.shape,
    )
    return alpha

def _normalize_alpha_for_plot(alpha: np.ndarray) -> np.ndarray:
    
    if alpha.ndim == 1:
        return alpha
    if alpha.ndim == 2:
        return alpha
    if alpha.ndim == 3:
        if alpha.shape[0] == 1:
            alpha = alpha[0]
            if alpha.ndim == 2 and alpha.shape[-1] == 1:
                return alpha[..., 0]
            return alpha
        raise ValueError(
            f"alpha with batch > 1 is not supported: {alpha.shape}. "
            "Use a batch of size 1."
        )
    raise ValueError(
        f"alpha must be 1D, 2D or 3D with batch=1, got: {alpha.ndim}D"
    )

def plot_attention_map(
    alpha: np.ndarray,
    output_path: Path,
    title: Optional[str] = None,
) -> Path:
    
    if not isinstance(alpha, np.ndarray):
        raise TypeError(f"alpha must be a np.ndarray, got: {type(alpha).__name__}")

    alpha_plot = _normalize_alpha_for_plot(alpha)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    if alpha_plot.ndim == 1:
        ax.bar(range(len(alpha_plot)), alpha_plot, color="steelblue", alpha=0.8)
        ax.set_xlabel("Time index")
        ax.set_ylabel("Attention weight alpha[i]")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Attention weights (Conv1D)")

    elif alpha_plot.ndim == 2:
        im = ax.imshow(alpha_plot, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax, label="Attention weight")
        ax.set_xlabel("Time position (target)")
        ax.set_ylabel("Time position (query)")
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Attention weights (QKV)")

    else:
        raise ValueError(
            f"Alpha shape not supported for plotting: {alpha_plot.shape}"
        )

    plt.tight_layout()
    plt.savefig(output_path, format=output_path.suffix[1:], bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("Attention plot saved to %s", output_path)
    return output_path

def plot_attention_vs_jamming(
    alpha_clean: np.ndarray,
    alpha_jammed: np.ndarray,
    output_path: Path,
) -> Path:
    
    if not isinstance(alpha_clean, np.ndarray):
        raise TypeError(
            f"alpha_clean must be a np.ndarray, got: {type(alpha_clean).__name__}"
        )
    if not isinstance(alpha_jammed, np.ndarray):
        raise TypeError(
            f"alpha_jammed must be a np.ndarray, got: {type(alpha_jammed).__name__}"
        )

    a_clean = _normalize_alpha_for_plot(alpha_clean)
    a_jammed = _normalize_alpha_for_plot(alpha_jammed)

    if a_clean.shape != a_jammed.shape:
        raise ValueError(
            f"Shape mismatch: clean {a_clean.shape}, jammed {a_jammed.shape}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    titles = ["Clean", "Jammed", "Difference (jammed - clean)"]
    data_list = [a_clean, a_jammed, a_jammed - a_clean]

    is_1d = a_clean.ndim == 1

    for idx, (data, title) in enumerate(zip(data_list, titles)):
        ax = axes[idx]
        if is_1d:
            ax.bar(range(len(data)), data, color="steelblue", alpha=0.8)
            ax.set_ylim(-1.05, 1.05 if idx == 2 else 1.05)
            ax.grid(True, axis="y", linestyle="--", alpha=0.6)
            ax.set_xlabel("Time index")
        else:
            im = ax.imshow(data, cmap="RdBu_r" if idx == 2 else "viridis",
                           aspect="auto", vmin=-1.0 if idx == 2 else 0.0,
                           vmax=1.0 if idx == 2 else 1.0)
            plt.colorbar(im, ax=ax)
            ax.set_xlabel("Target")
            ax.set_ylabel("Query")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, format=output_path.suffix[1:], bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("Attention comparison plot saved to %s", output_path)
    return output_path

def plot_attention_maps(
    model: tf.keras.Model,
    test_data: Dict[str, np.ndarray],
    output_dir: Path,
    num_samples: int = 5,
    config: Optional[Dict[str, Any]] = None,
    plot_format: str = "pdf",
) -> Path:
    
    if not isinstance(model, tf.keras.Model):
        raise TypeError(
            f"model must be a tf.keras.Model, got: {type(model).__name__}"
        )
    if not isinstance(test_data, dict) or "x" not in test_data:
        raise ValueError("test_data must be a dict with the 'x' key")

    x = np.asarray(test_data["x"])
    n_total = int(x.shape[0])
    if n_total == 0:
        raise ValueError("test_data['x'] contains no samples (N=0)")

    feature_mode = "real"
    if isinstance(config, dict):
        data_cfg = config.get("data")
        if isinstance(data_cfg, dict):
            feature_mode = str(data_cfg.get("feature_mode", "real"))

    from src.data.data_loader import _build_feature_matrix, build_reference_matrix

    has_ref = (
        "bit" in test_data
        and "seed" in test_data
        and isinstance(config, dict)
        and "data" in config
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_plot = max(1, min(int(num_samples), n_total))
    for i in range(n_plot):
        reference = None
        if has_ref:
            reference = build_reference_matrix(
                test_data["bit"][i : i + 1], test_data["seed"][i : i + 1], config
            )
        sample = _build_feature_matrix(x[i : i + 1], feature_mode, reference=reference)
        alpha = extract_attention_weights(model, sample)
        out_path = output_dir / f"attention_sample_{i:04d}.{plot_format}"
        plot_attention_map(
            alpha,
            out_path,
            title=f"Attention weights - sample {i + 1}",
        )

    logger.info(
        "Attention maps generated for %d samples in %s", n_plot, output_dir
    )
    return output_dir

def _generate_random_sample(config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    
    from src.data.dataset_generator import (
        apply_aerial_channel,
        generate_chaotic_sequence,
        sample_direct_path,
        sample_echo_parameters,
    )

    if config:
        data_cfg = config.get("data", {})
        seq_len = int(data_cfg.get("sequence_length", 100))
        map_type = str(data_cfg.get("map_type", "logistic"))
        map_param = float(data_cfg.get("map_param", 4.0))
        max_delay = int(data_cfg.get("max_delay", 33))
        max_doppler = float(data_cfg.get("max_doppler", 8e-5))
        feature_mode = str(data_cfg.get("feature_mode", "real"))
        snr_db = 10.0
        seed = int(config.get("general", {}).get("seed", _DEFAULT_SEED))
    else:
        seq_len = 100
        map_type = "logistic"
        map_param = 4.0
        max_delay = 33
        max_doppler = 8e-5
        feature_mode = "real"
        snr_db = 10.0
        seed = _DEFAULT_SEED

    rng = np.random.default_rng(seed)

    seed_seq = int(rng.integers(1, 2**31 - 1))
    x = generate_chaotic_sequence(map_type, map_param, seed_seq, seq_len)

    if config:
        echoes = sample_echo_parameters(1, rng, config)
    else:
        echoes = [EchoParams(tau=10, f_doppler=2e-5, alpha=1e-3)]

    if config:
        direct = sample_direct_path(rng, config)
    else:
        direct = DirectPathParams(h_c=1.0 + 0.0j, f_dc=1e-5)

    y, _ = apply_aerial_channel(x, direct.h_c, direct.f_dc, echoes, snr_db, rng)

    if feature_mode == "iq":
        sample = np.stack([np.real(y), np.imag(y)], axis=-1).astype(np.float32)
    else:
        sample = np.real(y).astype(np.float32)[..., np.newaxis]

    sample = sample.reshape(1, -1, sample.shape[-1])
    sample = np.concatenate(
        [sample, np.zeros_like(sample[..., :1])], axis=-1
    )

    if not np.all(np.isfinite(sample)):
        raise RuntimeError("Generated sample contains NaN/Inf")
    if np.var(sample) < 1e-9:
        logger.warning(
            "The generated sample has very low variance (%.2e): it may be degenerate.",
            np.var(sample),
        )

    logger.debug(
        "Generated sample: shape=%s, feature_mode=%s, variance=%.2e",
        sample.shape,
        feature_mode,
        np.var(sample),
    )
    return sample

def _load_sample_from_npz(sample_path: Path) -> np.ndarray:
    
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    with np.load(sample_path, allow_pickle=False) as npz:
        if "x" not in npz:
            raise ValueError(f"The .npz file does not contain the 'x' key: {sample_path}")
        x = npz["x"]
        if x.shape[0] == 0:
            raise ValueError(f"The .npz file contains no samples: {sample_path}")

        sample = np.real(x[0:1]).astype(np.float32)
        if sample.ndim == 2:
            sample = sample[..., np.newaxis]
        sample = np.concatenate([sample, np.zeros_like(sample[..., :1])], axis=-1)

    if sample.shape[0] != 1 or sample.ndim != 3:
        raise ValueError(
            f"Loaded sample has shape {sample.shape}, expected (1, L, F)"
        )

    if not np.all(np.isfinite(sample)):
        raise ValueError("The loaded sample contains NaN/Inf")

    logger.info("Sample loaded from %s: shape=%s", sample_path, sample.shape)
    return sample

def _load_model(model_path: Path) -> tf.keras.Model:
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        model = load_model(model_path)
    except Exception as e:
        logger.error("Failed to load model %s: %s", model_path, e)
        raise

    if model.input_shape is None:
        raise ValueError("The loaded model has no defined input_shape")

    logger.info("Model loaded from %s: input_shape=%s", model_path, model.input_shape)
    return model

def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(
        description="Extract and visualize the attention weights "
                    "(paper Sec. V-C)"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=Path,
        help="Path to the Keras model (SavedModel or .h5)",
    )
    parser.add_argument(
        "--sample_path",
        type=Path,
        default=None,
        help="Path to a .npz file containing a sample (optional). "
             "If not provided, a random sample is generated.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/attention_maps"),
        help="Plot output directory (default: results/attention_maps)",
    )
    parser.add_argument(
        "--jamming",
        action="store_true",
        help="Apply jamming to the sample for the comparison",
    )
    parser.add_argument(
        "--jsr_db",
        type=float,
        default=5.0,
        help="JSR in dB for the jamming comparison (default: 5.0)",
    )
    parser.add_argument(
        "--jamming_type",
        choices=["cw", "barrage", "partial_band"],
        default="cw",
        help="Jamming type (default: cw)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the YAML config (sample generation and jamming parameters)",
    )
    return parser.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> None:
    
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    config: Dict[str, Any] = {}
    if args.config:
        if not args.config.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        config = load_config(args.config, DEFAULT_BASE_CONFIG_PATH)
        logger.info("Config loaded from %s", args.config)

    log_dir = args.output_dir / "logs"
    log_file = setup_logging(
        log_dir=log_dir,
        level=config.get("general", {}).get("log_level", "INFO"),
        experiment_name="attention_visualizer",
    )
    if config:
        log_config_summary(config, logger)
    logger.info("Log file: %s", log_file)

    model = _load_model(args.model_path)

    if args.sample_path:
        sample = _load_sample_from_npz(args.sample_path)
    else:
        sample = _generate_random_sample(config if config else None)

    if sample.shape[1:] != tuple(model.input_shape[1:]):
        raise ValueError(
            f"Shape mismatch: sample {sample.shape[1:]} vs "
            f"model expected {model.input_shape[1:]}"
        )

    logger.info("Extracting attention weights (clean)...")
    alpha_clean = extract_attention_weights(model, sample)

    alpha_jammed = None
    if args.jamming:
        try:
            from src.experiments.run_jamming import apply_jamming
        except ImportError as e:
            logger.error("run_jamming module not available: %s", e)
            raise RuntimeError(
                "Unable to apply jamming: run_jamming is not importable. "
                "Required modules are incomplete."
            ) from e

        logger.info("Applying jamming: type=%s, JSR=%.1f dB", args.jamming_type, args.jsr_db)
        rng = np.random.default_rng(int(config.get("general", {}).get("seed", _DEFAULT_SEED)))
        sample_jammed = apply_jamming(sample, args.jamming_type, args.jsr_db, rng)

        if not np.all(np.isfinite(sample_jammed)):
            raise RuntimeError("The jammed sample contains NaN/Inf")

        logger.info("Extracting attention weights (jammed)...")
        alpha_jammed = extract_attention_weights(model, sample_jammed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_attention_map(
        alpha_clean,
        output_dir / "attention_clean.pdf",
        title="Attention weights - Clean",
    )

    if alpha_jammed is not None:
        plot_attention_map(
            alpha_jammed,
            output_dir / "attention_jammed.pdf",
            title="Attention weights - Jammed",
        )
        plot_attention_vs_jamming(
            alpha_clean,
            alpha_jammed,
            output_dir / "attention_compare.pdf",
        )

    logger.info("Visualization completed. Output in %s", output_dir)

if __name__ == "__main__":
    main()

