from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.data_loader import build_reference_matrix, load_npz_files
from src.data.dataset_generator import build_snr_grid
from src.experiments.pipeline import _build_jsr_values, prepare_dataset
from src.experiments.run_jamming import evaluate_jamming
from src.experiments.runner import load_experiment_config
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH
from src.utils.logger import get_logger, setup_logging
from src.utils.model_io import load_model

logger = get_logger(__name__)

ARCHS = ("conv1d", "qkv", "lstm", "mc_dlsk")

def _checkpoint(arch: str) -> Path:
    primary = _REPO / "results" / "full" / "jamming" / arch / "best_model.keras"
    if primary.is_file():
        return primary
    return (
        _REPO / "results" / "full" / "ber_vs_snr" / "k3_doppler_full" / arch
        / "best_model.keras"
    )

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archs", default=",".join(ARCHS))
    parser.add_argument("--max-symbols", type=int, default=40000)
    parser.add_argument("--realizations", type=int, default=None)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=_REPO / "results" / "full",
        help=(
            "Root the results and the log are written to; the checkpoints and the "
            "dataset are still read from the repository. Point it at a Drive "
            "directory to keep the output of a Colab session (default: results/full)."
        ),
    )
    return parser.parse_args(argv)

def _subsample(data: Dict[str, Any], max_symbols: int) -> Dict[str, Any]:
    n = int(np.asarray(data["x"]).shape[0])
    if max_symbols >= n:
        return data
    idx = np.linspace(0, n - 1, int(max_symbols)).astype(np.int64)
    out = {}
    for key, value in data.items():
        arr = np.asarray(value)
        out[key] = arr[idx] if arr.shape[:1] == (n,) else value
    return out

def main(argv=None) -> None:
    args = _parse_args(argv)
    setup_logging(log_dir=args.out_root / "jamming" / "logs",
                  level="INFO", experiment_name="rerun_jamming_mc")
    cfg = load_experiment_config(
        experiment_name="jamming",
        mode="full",
        experiments_yaml_path=_REPO / "configs" / "experiments.yaml",
        base_config_path=DEFAULT_BASE_CONFIG_PATH,
        cli_overrides=None,
    )
    jamming_cfg = cfg.get("jamming") or {}
    n_realizations = int(args.realizations or jamming_cfg.get("n_realizations", 1))
    jammer_types = list(jamming_cfg.get("jamming_types") or ["cw", "barrage", "partial_band"])
    jsr_values = _build_jsr_values(jamming_cfg)
    archs = [a.strip() for a in str(args.archs).split(",") if a.strip()]

    raw_dir = prepare_dataset(cfg, no_regen=False)
    snr_grid = build_snr_grid(cfg["data"]["snr_range"], cfg["data"]["snr_step"])
    echoes = [int(value) for value in cfg["data"]["echoes"]]
    test_data = load_npz_files(raw_dir, snr_grid, echoes, "test", cfg)
    test_data["x_ref"] = build_reference_matrix(test_data["bit"], test_data["seed"], cfg)
    test_data = _subsample(test_data, int(args.max_symbols))
    logger.info("test set: %d samples (subsample cap %d), JSR grid %s, realizations %d, archs %s",
                test_data["x"].shape[0], int(args.max_symbols), jsr_values, n_realizations, archs)

    for arch in archs:
        ckpt = _checkpoint(arch)
        if not ckpt.is_file():
            logger.warning("checkpoint not found for %s, skipping", arch)
            continue
        logger.info("=" * 60)
        logger.info("re-evaluating jamming for %s from %s", arch, ckpt)
        model = load_model(ckpt)
        out_dir = args.out_root / "jamming" / arch / "jamming"
        evaluate_jamming(
            model=model,
            test_data=test_data,
            config=cfg,
            jsr_values=jsr_values,
            jammer_types=jammer_types,
            output_dir=out_dir,
            model_name=arch,
            n_realizations=n_realizations,
        )
        del model
        tf.keras.backend.clear_session()

    logger.info("jamming Monte Carlo re-evaluation completed")

if __name__ == "__main__":
    main()
