"""Price of blindness on a short probe: the bit head blind vs with the reference.

Marco asked for a small probe that hands the regenerated reference ``x[n]`` to
the bit decision and shows, with a number, that the blind setting is the harder
one and was chosen deliberately.  The communication head cannot literally take
``x[n]`` as an input (the reference channel is sliced out before the backbone,
see ``src/models/ultra_can.py``), so the reference-aided bound is the same
seed-aware oracle correlator used by ``blind_vs_oracle.py``; the blind column is
the trained head.  A short count (the requested "ten bursts" sanity check) and a
statistically usable count are both reported.
"""

from __future__ import annotations

import argparse
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
from src.data.dataset_generator import generate_test_batch
from src.data.scene import require_monostatic
from src.evaluation.metrics import ber_from_logits
from src.experiments.blind_vs_oracle import _oracle_bits
from src.experiments.pipeline import predict_in_chunks, resolve_scenario
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config
from src.utils.logger import get_logger, log_config_summary, setup_logging
from src.utils.model_io import load_model

logger = get_logger(__name__)

_RESULTS = _REPO_ROOT / "results" / "full"
_DEFAULT_CHECKPOINT_ROOT = _RESULTS / "ber_vs_snr"
_DEFAULT_BURSTS = (10, 2000)
_DEFAULT_SEED = 20261008
_DEFAULT_SNR = (5.0, 11.0, 21.0)

def _settings(config: Dict[str, Any]) -> Dict[str, Any]:
    section = (config.get("experiments") or {}).get("blind_cheat_probe") or {}
    return {
        "n_bursts": [int(value) for value in section.get("n_bursts", _DEFAULT_BURSTS)],
        "snr_db": [float(value) for value in section.get("snr_db", _DEFAULT_SNR)],
        "seed": int(section.get("seed", _DEFAULT_SEED)),
        "archs": list(section.get("archs", ["conv1d"])),
    }

def run(model: tf.keras.Model, config: Dict[str, Any], arch: str) -> pd.DataFrame:
    require_monostatic(config)
    settings = _settings(config)
    k = int(max(config["data"]["echoes"]))
    feature_mode = str(config["data"].get("feature_mode", "iq"))
    feature_norm = str(config["data"].get("feature_norm", "none"))
    rows: List[Dict[str, Any]] = []
    for n_bursts in settings["n_bursts"]:
        rng = np.random.default_rng(settings["seed"])
        for snr in settings["snr_db"]:
            batch = generate_test_batch(config, n_bursts, snr, k, rng)
            features = _build_feature_matrix(
                batch["x"], feature_mode, feature_norm, batch["x_ref"]
            )
            logits = predict_in_chunks(model, features, "comm")
            bits = np.asarray(batch["bit"], dtype=np.int64)
            blind_ber = ber_from_logits(logits, bits)
            oracle = _oracle_bits(batch["x"], batch["seed"], config)
            oracle_ber = float(np.mean(oracle != bits))
            rows.append({
                "arch": arch,
                "snr_db": float(snr),
                "n_bursts": int(n_bursts),
                "ber_blind": float(blind_ber),
                "ber_reference_aided": oracle_ber,
                "ber_gap": float(blind_ber - oracle_ber),
            })
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
        Path(args.output_dir)
        if args.output_dir
        else _RESULTS / args.scenario / "blind_cheat_probe"
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
        frame = run(load_model(checkpoint), config, arch)
        frame.to_csv(out_dir / f"{arch}.csv", index=False)
        logger.info("blind_cheat_probe %s: %d rows", arch, len(frame))

if __name__ == "__main__":
    main()
