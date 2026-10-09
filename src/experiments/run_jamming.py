

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.data_loader import (
    DataDict,
    build_reference_matrix,
    load_npz_files,
    verify_snr_balance,
)
from src.data.dataset_generator import build_snr_grid
from src.evaluation.evaluator import evaluate_model
from src.models.baselines import build_baseline
from src.models.ultra_can import build_ultra_can
from src.models.ultra_can_qkv import build_ultra_can_qkv
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config, save_config_snapshot, validate_config
from src.utils.dataset_utils import get_dataset_dir
from src.utils.logger import get_logger, log_config_summary, setup_logging
from src.utils.model_io import load_model

logger = get_logger(__name__)

_VALID_JAMMING_TYPES = frozenset({"cw", "barrage", "partial_band"})
_DEFAULT_PARTIAL_BAND_FRACTION = 0.25
_EPS = 1e-12

_REQUIRED_KEYS: Tuple[str, ...] = (
    "general.experiment_name",
    "general.seed",
    "general.log_level",
    "data.sequence_length",
    "data.map_type",
    "data.map_param",
    "data.snr_range",
    "data.snr_step",
    "data.echoes",
    "data.max_delay",
    "data.max_doppler",
    "data.num_symbols_train",
    "data.num_symbols_val",
    "data.num_symbols_test",
    "model.backbone_type",
    "evaluation.snr_test_range",
    "evaluation.bit_error_threshold",
    "evaluation.max_symbols_per_snr",
    "visualization.plot_format",
)

def sample_jammer_waveform(
    shape: Tuple[int, int],
    jamming_type: str,
    rng: np.random.Generator,
    partial_band_fraction: float = _DEFAULT_PARTIAL_BAND_FRACTION,
    realization: Optional[int] = None,
    n_realizations: int = 1,
) -> np.ndarray:
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError(f"shape must be a (N, L) pair, got: {shape!r}")
    n, length = int(shape[0]), int(shape[1])
    if n < 1 or length < 1:
        raise ValueError(f"shape entries must be >= 1, got: {shape!r}")
    if jamming_type not in _VALID_JAMMING_TYPES:
        raise ValueError(
            f"invalid jamming_type: {jamming_type!r} (expected: {sorted(_VALID_JAMMING_TYPES)})"
        )
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a np.random.Generator, got: {type(rng).__name__}")
    if not (0.0 < partial_band_fraction <= 1.0):
        raise ValueError(f"partial_band_fraction must be in (0,1], got: {partial_band_fraction}")
    if isinstance(n_realizations, bool) or not isinstance(n_realizations, (int, np.integer)):
        raise ValueError(f"n_realizations must be an int, got: {n_realizations!r}")
    if int(n_realizations) < 1:
        raise ValueError(f"n_realizations must be >= 1, got: {n_realizations!r}")
    if realization is not None:
        if isinstance(realization, bool) or not isinstance(realization, (int, np.integer)):
            raise ValueError(f"realization must be an int or None, got: {realization!r}")
        if not (0 <= int(realization) < int(n_realizations)):
            raise ValueError(
                f"realization must be in [0, {int(n_realizations) - 1}], got: {realization!r}"
            )

    grid_ratio = None
    if realization is not None:
        grid_ratio = (float(realization) + 0.5) / float(int(n_realizations))

    if jamming_type == "cw":
        if grid_ratio is None:
            f_cw = float(rng.uniform(0.0, 0.5))
        else:
            f_cw = grid_ratio * 0.5
        t = np.arange(length, dtype=np.float64)
        j = np.tile(np.exp(1j * 2.0 * np.pi * f_cw * t), (n, 1))

    elif jamming_type == "barrage":
        n_samples = n * length
        noise_real = rng.normal(0.0, 1.0, size=n_samples)
        noise_imag = rng.normal(0.0, 1.0, size=n_samples)
        j = (noise_real + 1j * noise_imag).reshape(n, length)

    else:
        band_width = max(1, int(length * partial_band_fraction))
        n_positions = max(1, length - band_width + 1)
        if grid_ratio is None:
            start_idx = int(rng.integers(0, n_positions))
        else:
            start_idx = min(int(grid_ratio * n_positions), n_positions - 1)
        mask_freq = np.zeros(length, dtype=bool)
        mask_freq[start_idx:start_idx + band_width] = True

        n_samples = n * length
        noise_real = rng.normal(0.0, 1.0, size=n_samples)
        noise_imag = rng.normal(0.0, 1.0, size=n_samples)
        noise_freq = (noise_real + 1j * noise_imag).reshape(n, length)

        Jf = np.zeros((n, length), dtype=np.complex128)
        Jf[:, mask_freq] = noise_freq[:, mask_freq]
        j = np.fft.ifft(Jf, axis=-1)

    return _normalize_jammer(j)

def _normalize_jammer(j: np.ndarray) -> np.ndarray:
    power = float(np.mean(np.abs(j) ** 2))
    if power <= _EPS:
        return np.zeros_like(j)
    return j * np.sqrt(1.0 / power)

def _add_jammer_at_jsr(y: np.ndarray, j: np.ndarray, jsr_db: float) -> np.ndarray:
    if j.shape != y.shape:
        raise ValueError(f"jammer shape {j.shape} does not match y shape {y.shape}")
    if not np.isfinite(float(jsr_db)):
        raise ValueError(f"jsr_db must be finite, got: {jsr_db!r}")
    signal_power = float(np.mean(np.abs(y) ** 2))
    if signal_power <= _EPS:
        logger.warning("Signal power = %.3e <= eps, jamming not applied", signal_power)
        return y
    jsr_linear = 10.0 ** (float(jsr_db) / 10.0)
    if not np.isfinite(jsr_linear) or jsr_linear < 0.0:
        raise ValueError(f"invalid JSR: {jsr_db} dB -> {jsr_linear}")
    jammer_power = float(np.mean(np.abs(j) ** 2))
    if jammer_power <= _EPS:
        logger.warning("Jamming power = %.3e <= eps, jamming set to zero", jammer_power)
        return y
    scale = np.sqrt(signal_power * jsr_linear / jammer_power)
    y_jammed = y + scale * j
    if not np.all(np.isfinite(y_jammed)):
        raise RuntimeError(f"jammed signal not finite at JSR={jsr_db} dB")
    if logger.isEnabledFor(logging.DEBUG):
        actual_jsr = 10.0 * np.log10(
            np.mean(np.abs(scale * j) ** 2) / signal_power
        ) if signal_power > _EPS else -np.inf
        logger.debug("Jamming target JSR=%.1f dB, actual=%.2f dB", float(jsr_db), actual_jsr)
    return y_jammed

def apply_jamming(
    y: np.ndarray,
    jamming_type: str,
    jsr_db: float,
    rng: np.random.Generator,
    partial_band_fraction: float = _DEFAULT_PARTIAL_BAND_FRACTION,
) -> np.ndarray:
    
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y must be a np.ndarray, got: {type(y).__name__}")
    if jamming_type not in _VALID_JAMMING_TYPES:
        raise ValueError(
            f"invalid jamming_type: {jamming_type!r} (expected: {sorted(_VALID_JAMMING_TYPES)})"
        )
    if not np.isfinite(jsr_db):
        raise ValueError(f"jsr_db must be finite, got: {jsr_db!r}")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN/Inf")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a np.random.Generator, got: {type(rng).__name__}")
    if not (0.0 < partial_band_fraction <= 1.0):
        raise ValueError(f"partial_band_fraction must be in (0,1], got: {partial_band_fraction}")

    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2:
        raise ValueError(f"y must have shape (N, L) or (L,), got: {y.shape}")
    if y.shape[1] == 0:
        raise ValueError("y must not be empty")

    j = sample_jammer_waveform(
        y.shape,
        jamming_type,
        rng,
        partial_band_fraction=partial_band_fraction,
    )
    return _add_jammer_at_jsr(y, j, jsr_db)

def _load_or_build_model(
    model_type: str,
    model_path: Optional[Path],
    config: Dict[str, Any],
) -> tf.keras.Model:
    
    if model_path is not None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        logger.info("Loading model from %s", model_path)
        try:
            model = load_model(model_path)
        except Exception as e:
            raise ValueError(f"Error loading the model: {e}") from e

        if model.input_shape is None:
            raise RuntimeError("The loaded model has no defined input_shape")
        dummy = tf.zeros((1,) + tuple(model.input_shape[1:]), dtype=model.inputs[0].dtype)
        outputs = model(dummy, training=False)
        for name, tensor in outputs.items():
            tf.debugging.assert_all_finite(tensor, f"Output '{name}' not finite when loading")
        return model

    logger.info("Building model %s from scratch", model_type)
    builders = {
        "conv1d": build_ultra_can,
        "qkv": build_ultra_can_qkv,
        "lstm": lambda cfg: build_baseline(cfg, "lstm"),
        "mc_dlsk": lambda cfg: build_baseline(cfg, "mc_dlsk"),
    }
    if model_type not in builders:
        raise ValueError(f"Model not supported: {model_type!r}")
    model = builders[model_type](config)

    checkpoint_path = config.get("training", {}).get("checkpoint_path")
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if not ckpt.is_absolute():
            ckpt = _REPO_ROOT / ckpt
        if ckpt.exists():
            try:
                logger.info("Loading checkpoint from %s", ckpt)
                model = load_model(ckpt)
                logger.info("Checkpoint loaded successfully")
            except Exception as e:
                logger.warning("Failed to load checkpoint: %s, continuing with an untrained model", e)

    if model.input_shape is None:
        raise RuntimeError("The model is not built")
    dummy = tf.zeros((1,) + tuple(model.input_shape[1:]), dtype=model.inputs[0].dtype)
    outputs = model(dummy, training=False)
    for name, tensor in outputs.items():
        tf.debugging.assert_all_finite(tensor, f"Output '{name}' not finite in build")

    return model

def evaluate_jamming(
    model: tf.keras.Model,
    test_data: DataDict,
    config: Dict[str, Any],
    jsr_values: List[float],
    jammer_types: List[str],
    output_dir: Path,
    model_name: str = "model",
    n_realizations: int = 1,
    max_symbols: Optional[int] = None,
) -> Dict[str, Any]:
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running baseline evaluation...")
    baseline_results = evaluate_model(model, test_data, config)
    baseline_ber_mean = float(np.mean(baseline_results["ber"]))
    baseline_df = pd.DataFrame({
        "snr_db": baseline_results["snr_db"],
        "ber": baseline_results["ber"],
        "mse_tau": baseline_results["mse_tau"],
        "mse_fd": baseline_results["mse_fd"],
        "n_errors": baseline_results["n_errors"],
        "n_symbols": baseline_results["n_symbols"],
    })
    baseline_csv = output_dir / "baseline_metrics.csv"
    baseline_df.to_csv(baseline_csv, index=False)
    logger.info("Baseline saved to %s", baseline_csv)
    logger.info("Baseline mean BER = %.6f", baseline_ber_mean)

    invalid_types = [t for t in jammer_types if t not in _VALID_JAMMING_TYPES]
    if invalid_types:
        raise ValueError(f"Invalid jamming types: {invalid_types}. Expected: {sorted(_VALID_JAMMING_TYPES)}")

    partial_band_fraction = config.get("jamming", {}).get("partial_band_fraction", _DEFAULT_PARTIAL_BAND_FRACTION)
    base_seed = int(config["general"].get("seed", 42))
    plot_format = config.get("visualization", {}).get("plot_format", "pdf")

    from src.experiments.jamming_interpretability import _subsample

    sweep_data = _subsample(test_data, max_symbols, tag="jamming")

    all_results: Dict[str, Dict[str, np.ndarray]] = {}
    all_dfs: Dict[str, pd.DataFrame] = {}

    n_realizations = int(n_realizations)
    if n_realizations < 1:
        raise ValueError(f"n_realizations must be >= 1, got: {n_realizations}")

    for jt in jammer_types:
        logger.info("Processing jamming: %s (%d realization(s))", jt, n_realizations)
        per_jsr: Dict[float, List[Dict[str, float]]] = {float(j): [] for j in jsr_values}
        realization_rows: List[Dict[str, Any]] = []
        for realization in range(n_realizations):
            seed = (
                base_seed
                + (sum(ord(ch) for ch in jt) % 10000)
                + realization * 100003
            )
            rng = np.random.default_rng(seed)
            jammer = sample_jammer_waveform(
                sweep_data["x"].shape,
                jt,
                rng,
                partial_band_fraction=partial_band_fraction,
                realization=realization,
                n_realizations=n_realizations,
            )
            for jsr_db in jsr_values:
                try:
                    y_jammed = _add_jammer_at_jsr(
                        sweep_data["x"], jammer, float(jsr_db)
                    )
                    jammed_data = sweep_data.copy()
                    jammed_data["x"] = y_jammed
                    eval_res = evaluate_model(model, jammed_data, config)
                    ber_mean = float(np.mean(eval_res["ber"]))
                    mse_tau = float(np.mean(eval_res["mse_tau"]))
                    mse_fd = float(np.mean(eval_res["mse_fd"]))

                    if not np.isfinite(ber_mean) or not (0.0 <= ber_mean <= 1.0):
                        raise RuntimeError(f"invalid BER for {jt} at JSR={jsr_db}: {ber_mean}")
                    if not np.isfinite(mse_tau) or mse_tau < 0.0:
                        raise RuntimeError(f"invalid MSE_tau: {mse_tau}")
                    if not np.isfinite(mse_fd) or mse_fd < 0.0:
                        raise RuntimeError(f"invalid MSE_fd: {mse_fd}")

                    per_jsr[float(jsr_db)].append({
                        "ber": ber_mean,
                        "mse_tau": mse_tau,
                        "mse_fd": mse_fd,
                    })
                    realization_rows.append({
                        "jsr_db": float(jsr_db),
                        "realization": int(realization),
                        "ber": ber_mean,
                        "mse_tau": mse_tau,
                        "mse_fd": mse_fd,
                    })
                    logger.info(
                        "jamming %s r=%d JSR=%.1f dB | BER=%.5f",
                        jt, realization, float(jsr_db), ber_mean,
                    )
                except Exception as e:
                    logger.error(
                        "Error at JSR %.1f dB (realization %d): %s",
                        jsr_db, realization, e,
                    )

        rows = []
        for jsr_db in jsr_values:
            samples = per_jsr[float(jsr_db)]
            if samples:
                ber_arr = np.asarray([s["ber"] for s in samples], dtype=np.float64)
                mse_tau_arr = np.asarray([s["mse_tau"] for s in samples], dtype=np.float64)
                mse_fd_arr = np.asarray([s["mse_fd"] for s in samples], dtype=np.float64)
                rows.append({
                    "jsr_db": jsr_db,
                    "ber": float(ber_arr.mean()),
                    "ber_std": float(ber_arr.std(ddof=0)),
                    "ber_min": float(ber_arr.min()),
                    "ber_max": float(ber_arr.max()),
                    "mse_tau": float(mse_tau_arr.mean()),
                    "mse_tau_std": float(mse_tau_arr.std(ddof=0)),
                    "mse_fd": float(mse_fd_arr.mean()),
                    "mse_fd_std": float(mse_fd_arr.std(ddof=0)),
                    "n_realizations": int(ber_arr.size),
                })
            else:
                rows.append({
                    "jsr_db": jsr_db,
                    "ber": np.nan,
                    "ber_std": np.nan,
                    "ber_min": np.nan,
                    "ber_max": np.nan,
                    "mse_tau": np.nan,
                    "mse_tau_std": np.nan,
                    "mse_fd": np.nan,
                    "mse_fd_std": np.nan,
                    "n_realizations": 0,
                })

        df = pd.DataFrame(rows)
        csv_path = output_dir / f"jamming_results_{jt}.csv"
        df.to_csv(csv_path, index=False)
        all_dfs[jt] = df
        logger.info("Results for %s saved to %s", jt, csv_path)

        if realization_rows:
            realizations_csv = output_dir / f"jamming_realizations_{jt}.csv"
            pd.DataFrame(realization_rows).to_csv(realizations_csv, index=False)
            logger.info("Per-realization results for %s saved to %s", jt, realizations_csv)

        valid = df["ber"].notna()
        if valid.any():
            all_results[jt] = {
                "jsr": df.loc[valid, "jsr_db"].values,
                "ber": df.loc[valid, "ber"].values,
                "ber_std": df.loc[valid, "ber_std"].values,
                "mse_tau": df.loc[valid, "mse_tau"].values,
                "mse_fd": df.loc[valid, "mse_fd"].values,
            }

    if all_results:
        _plot_jamming_curves(
            all_results,
            output_dir / "plots",
            plot_format,
            model_name=model_name,
            baseline_ber_mean=baseline_ber_mean,
        )
    else:
        logger.warning("No valid results to generate the plots")

    return {
        "results": all_dfs,
        "plots": all_results,
        "baseline": baseline_df,
    }

def _plot_ber_band(
    ax: Any,
    jsr: np.ndarray,
    ber: np.ndarray,
    ber_std: Optional[np.ndarray],
    color: str,
) -> None:
    if ber_std is None:
        return
    std = np.asarray(ber_std, dtype=np.float64)
    if std.size != np.asarray(ber).size or not np.all(np.isfinite(std)):
        return
    lower = np.clip(np.asarray(ber, dtype=np.float64) - std, 1e-6, 1.0)
    upper = np.clip(np.asarray(ber, dtype=np.float64) + std, 1e-6, 1.0)
    ax.fill_between(jsr, lower, upper, color=color, alpha=0.18, linewidth=0)

def _plot_jamming_curves(
    results: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
    plot_format: str,
    model_name: str = "model",
    baseline_ber_mean: Optional[float] = None,
) -> None:
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        return

    colors = {"cw": "blue", "barrage": "red", "partial_band": "green"}
    markers = {"cw": "o", "barrage": "s", "partial_band": "D"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for jt, data in results.items():
        color = colors.get(jt, "black")
        ax.semilogy(
            data["jsr"],
            data["ber"],
            marker=markers.get(jt, "."),
            linestyle="-",
            color=color,
            label=jt.capitalize().replace("_", " "),
            linewidth=2,
            markersize=8,
        )
        _plot_ber_band(ax, data["jsr"], data["ber"], data.get("ber_std"), color)
    ax.set_xlabel("JSR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title("BER vs JSR - comparison of jamming types")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend()
    ax.set_ylim([1e-6, 1.0])

    overlay_path = output_dir / f"ber_vs_jsr_overlay.{plot_format}"
    plt.savefig(overlay_path, format=plot_format, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Overlay plot saved to %s", overlay_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    for jt, data in results.items():
        color = colors.get(jt, "black")
        ax.semilogy(
            data["jsr"],
            data["ber"],
            marker=markers.get(jt, "."),
            linestyle="-",
            color=color,
            label=jt.capitalize().replace("_", " "),
            linewidth=2,
            markersize=8,
        )
        _plot_ber_band(ax, data["jsr"], data["ber"], data.get("ber_std"), color)
    if baseline_ber_mean is not None:
        jsr_span = [min(d["jsr"][0] for d in results.values()),
                    max(d["jsr"][-1] for d in results.values())]
        ax.semilogy(jsr_span, [baseline_ber_mean, baseline_ber_mean],
                    linestyle="--", color="black", linewidth=1.5,
                    label=f"Clean (BER={baseline_ber_mean:.4f})")
    ax.set_xlabel("JSR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title(f"BER vs JSR - {model_name} (jammed vs clean)")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend()
    ax.set_ylim([1e-6, 1.0])

    clean_path = output_dir / f"ber_vs_jsr_{model_name}_vs_clean.{plot_format}"
    plt.savefig(clean_path, format=plot_format, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Jammed vs clean plot saved to %s", clean_path)

    for jt, data in results.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.semilogy(data["jsr"], data["ber"], marker="o", linestyle="-", color="black", linewidth=2, markersize=8)
        _plot_ber_band(ax, data["jsr"], data["ber"], data.get("ber_std"), "black")
        ax.set_xlabel("JSR (dB)")
        ax.set_ylabel("BER")
        ax.set_title(f"BER vs JSR - {jt.capitalize().replace('_', ' ')}")
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.set_ylim([1e-6, 1.0])

        single_path = output_dir / f"ber_vs_jsr_{jt}.{plot_format}"
        plt.savefig(single_path, format=plot_format, bbox_inches="tight", dpi=300)
        plt.close(fig)
        logger.info("Separate plot for %s saved to %s", jt, single_path)

def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BER evaluation under jamming")
    parser.add_argument("--config", required=True, help="Path to the experiment config (YAML).")
    parser.add_argument("--model", choices=["conv1d", "qkv", "lstm", "mc_dlsk"], default="conv1d")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to an already-trained checkpoint.")
    parser.add_argument("--output-dir", default="results/full_experiment/jamming/jamming")
    parser.add_argument("--jsr-range", nargs=2, type=float, default=[0.0, 10.0], help="Min max JSR in dB")
    parser.add_argument("--jsr-step", type=float, default=1.0, help="JSR step in dB")
    parser.add_argument("--plot-format", default=None, help="Plot format (default from config)")
    return parser.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path, DEFAULT_BASE_CONFIG_PATH)
    validate_config(config, _REQUIRED_KEYS)

    output_dir = Path(args.output_dir)
    log_dir = output_dir / "logs"
    log_file = setup_logging(
        log_dir=log_dir,
        level=str(config["general"].get("log_level", "INFO")),
        experiment_name=config["general"].get("experiment_name", "jamming_eval"),
    )
    log_config_summary(config, logger)
    logger.info("Log file: %s", log_file)

    model = _load_or_build_model(args.model, args.model_path, config)

    data_cfg = config["data"]
    raw_dir = get_dataset_dir(config)
    if not raw_dir.is_absolute():
        raw_dir = _REPO_ROOT / raw_dir

    snr_grid = build_snr_grid(data_cfg["snr_range"], data_cfg["snr_step"])
    echoes = list(data_cfg["echoes"])
    test_data = load_npz_files(raw_dir, snr_grid, echoes, "test", config)
    logger.info("Test set loaded: %d samples", test_data["x"].shape[0])
    verify_snr_balance(test_data, snr_grid, echoes)
    test_data["x_ref"] = build_reference_matrix(test_data["bit"], test_data["seed"], config)

    jsr_min, jsr_max = args.jsr_range
    jsr_step = args.jsr_step
    jsr_grid = np.arange(jsr_min, jsr_max + jsr_step / 2.0, jsr_step)
    jsr_grid = np.round(jsr_grid, decimals=6).tolist()
    logger.info("JSR grid: %s", jsr_grid)

    jamming_types = config.get("jamming", {}).get("jamming_types", ["cw", "barrage", "partial_band"])
    jamming_types = [t for t in jamming_types if t in _VALID_JAMMING_TYPES]
    if not jamming_types:
        logger.warning("No valid jamming type in the config, exiting.")
        return

    n_realizations = int(config.get("jamming", {}).get("n_realizations", 1))
    logger.info("Jamming realizations per JSR point: %d", n_realizations)

    result = evaluate_jamming(
        model=model,
        test_data=test_data,
        config=config,
        jsr_values=jsr_grid,
        jammer_types=jamming_types,
        output_dir=output_dir,
        n_realizations=n_realizations,
    )

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tensorflow_version": tf.__version__,
        "model_type": args.model,
        "jsr_grid": jsr_grid,
        "jamming_types": jamming_types,
        "n_realizations": n_realizations,
        "config_snapshot_path": str(log_dir / "config_used.yaml"),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    save_config_snapshot(config, log_dir)
    logger.info("Jamming experiment completed. Output in %s", output_dir)

if __name__ == "__main__":
    main()
