
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.data_loader import _build_feature_matrix
from src.data.dataset_generator import generate_transmitted_batch
from src.data.scene import require_monostatic
from src.evaluation.ranging import scored_mae, scored_median_ae
from src.experiments.pipeline import predict_in_chunks, resolve_scenario
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config
from src.utils.logger import get_logger, log_config_summary, setup_logging
from src.utils.model_io import load_model

logger = get_logger(__name__)

_RESULTS = _REPO_ROOT / "results" / "full"
_DEFAULT_CHECKPOINT_ROOT = _RESULTS / "ber_vs_snr"
_DEFAULT_SYMBOLS = 6000
_DEFAULT_SEED = 20261008
_DEFAULT_RATIO = (0.8, 1.2)
_DEFAULT_SNR = 15.0

def _settings(config: Dict[str, Any]) -> Dict[str, Any]:
    section = (config.get("experiments") or {}).get("ambiguity") or {}
    ratio = list(section.get("amplitude_ratio", list(_DEFAULT_RATIO)))
    return {
        "n_symbols": int(section.get("n_symbols", _DEFAULT_SYMBOLS)),
        "snr_db": float(section.get("snr_db", _DEFAULT_SNR)),
        "amplitude_ratio": (float(ratio[0]), float(ratio[1])),
        "seed": int(section.get("seed", _DEFAULT_SEED)),
        "archs": list(section.get("archs", ["conv1d", "qkv"])),
    }

def _shift(x: np.ndarray, lag: np.ndarray) -> np.ndarray:
    n, length = x.shape
    index = np.arange(length, dtype=np.int64)[None, :] - lag[:, None]
    valid = index >= 0
    return np.where(
        valid, np.take_along_axis(x, np.clip(index, 0, length - 1), axis=1), 0.0
    )

def _pile(
    config: Dict[str, Any], n: int, rng: np.random.Generator, ratio: Tuple[float, float],
    second: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq_len = int(config["data"]["sequence_length"])
    max_delay = int(config["data"]["max_delay"])
    bits = rng.integers(0, 2, n).astype(np.int64)
    seeds = rng.integers(1, 2**31 - 1, n).astype(np.int64)
    x = generate_transmitted_batch(config, bits, seeds)
    tau = rng.integers(3, max(4, min(seq_len // 2, max_delay) - 1), n).astype(np.int64)
    amp = rng.uniform(ratio[0], ratio[1], n)
    y = amp[:, None] * _shift(x, tau) + float(second) * _shift(x, tau + 1)
    power = np.mean(np.abs(y) ** 2, axis=1, keepdims=True)
    noise_var = power * 10.0 ** (-float(config["data"].get("ambiguity_snr_db", 30.0)) / 10.0)
    y = y + np.sqrt(noise_var / 2.0) * rng.standard_normal(y.shape)
    return y.astype(np.float64), x, tau.astype(np.float64)

def _input(config: Dict[str, Any], y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return _build_feature_matrix(
        y.astype(np.complex128),
        str(config["data"].get("feature_mode", "iq")),
        str(config["data"].get("feature_norm", "none")),
        x,
    )

def _profile(y: np.ndarray, x: np.ndarray, max_delay: int) -> np.ndarray:
    energy = np.sum(x * x, axis=1)
    energy = np.where(energy < 1e-12, 1.0, energy)
    direct = np.sum(x * y, axis=1) / energy
    residual = y - direct[:, None] * x
    length = x.shape[1]
    return np.stack(
        [
            np.sum(x[:, : length - lag] * residual[:, lag:], axis=1)
            for lag in range(1, max_delay + 1)
        ],
        axis=1,
    )

def run(model: tf.keras.Model, config: Dict[str, Any], arch: str) -> pd.DataFrame:
    require_monostatic(config)
    settings = _settings(config)
    tau_max = int(config["data"]["max_delay"])
    easy = _pile(
        config, settings["n_symbols"], np.random.default_rng(settings["seed"]),
        (1.0, 1.0), 0.0,
    )
    ambiguous = _pile(
        config, settings["n_symbols"], np.random.default_rng(settings["seed"] + 1),
        settings["amplitude_ratio"], 1.0,
    )
    rows: List[Dict[str, Any]] = []
    for name, (y, x, tau_true) in (("easy", easy), ("ambiguous", ambiguous)):
        features = _input(config, y, x)
        head = (
            np.clip(predict_in_chunks(model, features, "sensing")[:, 0], 0.0, 1.0)
            * tau_max
        )
        peak = (np.argmax(np.abs(_profile(y, x, tau_max)), axis=1) + 1).astype(np.float64)
        mask = np.zeros(tau_true.shape, dtype=bool)
        mae_peak = scored_mae(peak, tau_true, mask)
        mae_head = scored_mae(head, tau_true, mask)
        medae_peak = scored_median_ae(peak, tau_true, mask)
        medae_head = scored_median_ae(head, tau_true, mask)
        rows.append({
            "arch": arch,
            "pile": name,
            "n_symbols": int(tau_true.size),
            "mae_peak_samples": mae_peak,
            "mae_head_samples": mae_head,
            "medae_peak_samples": medae_peak,
            "medae_head_samples": medae_head,
            "gap_samples": mae_peak - mae_head,
            "gap_median_samples": medae_peak - medae_head,
        })
    logger.info("ambiguity %s: %d rows", arch, len(rows))
    return pd.DataFrame(rows)

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_BASE_CONFIG_PATH)
    parser.add_argument("--scenario", default="k3_doppler_full")
    parser.add_argument("--arch", default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=_DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    config = resolve_scenario(
        load_config(args.config, base_config_path=DEFAULT_BASE_CONFIG_PATH), args.scenario
    )
    out_dir = (
        Path(args.output_dir) if args.output_dir else _RESULTS / args.scenario / "ambiguity"
    )
    setup_logging(
        log_dir=out_dir / "logs",
        level=str(config["general"].get("log_level", "INFO")),
        experiment_name=str(config["general"]["experiment_name"]),
    )
    log_config_summary(config, logger)
    archs = [args.arch] if args.arch else _settings(config)["archs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for arch in archs:
        checkpoint = Path(args.checkpoint_root) / args.scenario / arch / "best_model.keras"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        run(load_model(checkpoint), config, arch).to_csv(
            out_dir / f"{arch}.csv", index=False
        )

if __name__ == "__main__":
    main()
