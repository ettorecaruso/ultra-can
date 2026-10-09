from __future__ import annotations

import argparse
import gc
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments import pipeline
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config
from src.utils.dataset_utils import get_dataset_dir
from src.utils.logger import get_logger, log_config_summary, setup_logging

logger = get_logger(__name__)

_DEFAULT_BATCH = 1
_DEFAULT_BURSTS = 100
_DEFAULT_WARMUP = 20
_BYTES_PER_PARAM_FLOAT32 = 4
_INT8_QUANT_RATIO = 4.0

def _peak_process_memory_kib() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / 1024.0
    except ImportError:
        import resource

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def _settings(config: Dict[str, Any]) -> Dict[str, Any]:
    section = (config.get("experiments") or {}).get("board_footprint") or {}
    return {
        "batch": int(section.get("batch", _DEFAULT_BATCH)),
        "n_bursts": int(section.get("n_bursts", _DEFAULT_BURSTS)),
        "warmup": int(section.get("warmup", _DEFAULT_WARMUP)),
        "archs": list(section.get("archs", ["conv1d"])),
    }

def _input_shape(config: Dict[str, Any]) -> tuple:
    seq_len = int(config["data"]["sequence_length"])
    feature_mode = str(config["data"].get("feature_mode", "iq"))
    num_features = 2 if feature_mode == "iq" else 1
    return (seq_len, num_features + 1)

def _measure(model: tf.keras.Model, shape: tuple, batch: int, n_bursts: int, warmup: int) -> Dict[str, Any]:
    dummy = tf.zeros((batch, *shape), dtype=tf.float32)
    for _ in range(int(warmup)):
        model(dummy, training=False)
    gc.collect()
    latencies_ms: List[float] = []
    for _ in range(int(n_bursts)):
        start = time.perf_counter()
        model(dummy, training=False)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(latencies_ms)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    params = int(model.count_params())
    float32_kib = params * _BYTES_PER_PARAM_FLOAT32 / 1024.0
    return {
        "params": params,
        "float32_kib": float32_kib,
        "int8_kib_estimate": float32_kib / _INT8_QUANT_RATIO,
        "latency_median_ms": float(statistics.median(latencies_ms)),
        "latency_mean_ms": float(statistics.fmean(latencies_ms)),
        "latency_p95_ms": float(ordered[p95_index]),
        "latency_std_ms": float(statistics.pstdev(latencies_ms)),
        "peak_process_memory_kib": _peak_process_memory_kib(),
    }

def run(config: Dict[str, Any], arch: str) -> Dict[str, Any]:
    settings = _settings(config)
    model = pipeline.build_model(config, arch)
    shape = _input_shape(config)
    row = _measure(
        model, shape, settings["batch"], settings["n_bursts"], settings["warmup"]
    )
    row.update({
        "arch": arch,
        "batch": settings["batch"],
        "n_bursts": settings["n_bursts"],
        "input_shape": "x".join(str(dim) for dim in shape),
    })
    logger.info("board_footprint %s: %s", arch, row)
    return row

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_BASE_CONFIG_PATH)
    parser.add_argument("--mode", choices=["fast", "full"], default="full")
    parser.add_argument("--arch", default=None, help="single architecture override")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config, base_config_path=DEFAULT_BASE_CONFIG_PATH)
    experiment_name = str(config["general"]["experiment_name"])
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _REPO_ROOT / "results" / experiment_name / "board_footprint"
    )
    setup_logging(
        log_dir=out_dir / "logs",
        level=str(config["general"].get("log_level", "INFO")),
        experiment_name=experiment_name,
    )
    log_config_summary(config, logger)

    archs = [args.arch] if args.arch else _settings(config)["archs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for arch in archs:
        row = run(config, arch)
        pd.DataFrame([row]).to_csv(out_dir / f"{arch}.csv", index=False)
        if "run_output_dir" not in config["general"]:
            logger.info("dataset dir: %s", get_dataset_dir(config).name)

if __name__ == "__main__":
    main()
