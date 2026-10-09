
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.data_loader import _build_feature_matrix
from src.data.dataset_generator import generate_chaotic_sequence, generate_test_batch
from src.evaluation.metrics import ber_from_logits
from src.evaluation.ranging import gap_resolved, wilson_interval
from src.models.dcsk_correlator import evaluate_classical
from src.experiments.pipeline import predict_in_chunks, resolve_scenario
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config
from src.utils.logger import get_logger, log_config_summary, setup_logging
from src.utils.model_io import load_model

logger = get_logger(__name__)

_RESULTS = _REPO_ROOT / "results" / "full"
_DEFAULT_CHECKPOINT_ROOT = _RESULTS / "ber_vs_snr"
_DEFAULT_SYMBOLS = 20000
_DEFAULT_SEED = 20261008
_DEFAULT_SNR = [-5.0, -1.0, 3.0, 5.0, 9.0, 13.0, 17.0, 21.0]
_DEFAULT_NETS = ["conv1d", "qkv", "lstm", "mc_dlsk"]
_DEFAULT_CLASSICAL = ["dcsk", "matched_filter", "energy_detector"]

def _settings(config: Dict[str, Any]) -> Dict[str, Any]:
    section = (config.get("experiments") or {}).get("unified_benchmark") or {}
    return {
        "n_symbols": int(section.get("n_symbols", _DEFAULT_SYMBOLS)),
        "snr_test_range": [
            float(value) for value in section.get("snr_test_range", _DEFAULT_SNR)
        ],
        "nets": list(section.get("nets", _DEFAULT_NETS)),
        "classical": list(section.get("classical", _DEFAULT_CLASSICAL)),
        "relative_gap": float(section.get("relative_gap", 0.10)),
        "seed": int(section.get("seed", _DEFAULT_SEED)),
    }

def _net_ber(model: tf.keras.Model, config: Dict[str, Any], batch: Dict[str, Any]) -> int:
    features = _build_feature_matrix(
        batch["x"],
        str(config["data"].get("feature_mode", "iq")),
        str(config["data"].get("feature_norm", "none")),
        batch["x_ref"],
    )
    logits = predict_in_chunks(model, features, "comm")
    return int(
        np.count_nonzero(np.argmax(logits, axis=-1) != np.asarray(batch["bit"], dtype=np.int64))
    )

def _classical_errors(
    config: Dict[str, Any], batch: Dict[str, Any], detector: str, template: np.ndarray, beta: int,
) -> int:
    y = np.real(batch["x"])
    bits = np.asarray(batch["bit"])
    if detector == "matched_filter":
        result = evaluate_classical(y[:, beta:], bits, detector, config, template=template)
    else:
        result = evaluate_classical(y, bits, detector, config)
    return int(result["n_errors"])

def run(config: Dict[str, Any], scenario: str, checkpoint_root: Path) -> pd.DataFrame:
    settings = _settings(config)
    beta = int(config["baselines"]["dcsk_correlator"]["correlation_length"])
    template = generate_chaotic_sequence(
        config["data"]["map_type"], float(config["data"]["map_param"]),
        int(config["general"]["seed"]), beta,
    )
    models: Dict[str, tf.keras.Model] = {}
    for net in settings["nets"]:
        checkpoint = Path(checkpoint_root) / scenario / net / "best_model.keras"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        models[net] = load_model(checkpoint)
    k = int(max(config["data"]["echoes"]))
    rng = np.random.default_rng(settings["seed"])
    rows: List[Dict[str, Any]] = []
    for snr in settings["snr_test_range"]:
        batch = generate_test_batch(config, settings["n_symbols"], snr, k, rng)
        total = int(np.asarray(batch["bit"]).size)
        errors: Dict[str, int] = {}
        for net, model in models.items():
            errors[net] = _net_ber(model, config, batch)
        for detector in settings["classical"]:
            errors[detector] = _classical_errors(config, batch, detector, template, beta)
        for method, count in errors.items():
            low, high = wilson_interval(count, total)
            rows.append({
                "method": method,
                "scenario": scenario,
                "snr_db": float(snr),
                "n_symbols": total,
                "n_errors": int(count),
                "ber": float(count) / float(total),
                "ber_low": low,
                "ber_high": high,
            })
    frame = pd.DataFrame(rows)
    frame["gap_below_10pct"] = False
    for snr, group in frame.groupby("snr_db"):
        values = group["ber"].to_numpy()
        for index, value in zip(group.index, values):
            others = np.delete(values, np.where(group.index == index)[0][0])
            resolved = any(
                gap_resolved(value, other, settings["relative_gap"]) for other in others
            )
            frame.at[index, "gap_below_10pct"] = not resolved
    logger.info("unified_benchmark %s: %d rows", scenario, len(frame))
    return frame

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_BASE_CONFIG_PATH)
    parser.add_argument("--scenario", default="k1_doppler_full")
    parser.add_argument("--checkpoint-root", type=Path, default=_DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    config = resolve_scenario(
        load_config(args.config, base_config_path=DEFAULT_BASE_CONFIG_PATH), args.scenario
    )
    out_dir = (
        Path(args.output_dir) if args.output_dir
        else _RESULTS / args.scenario / "unified_benchmark"
    )
    setup_logging(
        log_dir=out_dir / "logs",
        level=str(config["general"].get("log_level", "INFO")),
        experiment_name=str(config["general"]["experiment_name"]),
    )
    log_config_summary(config, logger)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = run(config, args.scenario, Path(args.checkpoint_root))
    frame.to_csv(out_dir / "unified_benchmark.csv", index=False)

if __name__ == "__main__":
    main()

