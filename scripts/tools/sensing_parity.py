from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.mcu_deployment import (
    DEFAULT_SCENARIO,
    Int8Model,
    load_checkpoint,
)
from src.experiments.peer_estimation import evaluate_peer_estimation
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, _deep_merge

tf.config.threading.set_intra_op_parallelism_threads(1)


def peer_config(n_symbols: int, guards=(1.0, 1.25, 1.5, 2.0)):
    base = yaml.safe_load(DEFAULT_BASE_CONFIG_PATH.read_text())
    exp = yaml.safe_load((REPO_ROOT / "configs" / "experiments.yaml").read_text())
    config = _deep_merge(base, exp["peer_estimation"]["full"])
    section = config["experiments"]["peer_estimation"]
    section["n_symbols"] = int(n_symbols)
    section["guard_factors"] = list(guards)
    section["model_scenario"] = DEFAULT_SCENARIO
    return config


def summarise(samples: pd.DataFrame, precision: str, arch: str) -> pd.DataFrame:
    rows = []
    for guard, group in samples.groupby("guard_factor"):
        answered = ~group["fallback"].to_numpy()
        n_answered = int(answered.sum())
        peer_false_answered = int((group["peer_false"].to_numpy() & answered).sum())
        rows.append({
            "arch": arch,
            "precision": precision,
            "guard": float(guard),
            "n": int(len(group)),
            "target_id_pct": 100.0 * group["target_hit"].mean(),
            "peer_false_pct": 100.0 * group["peer_false"].mean(),
            "coverage_pct": 100.0 * n_answered / len(group),
            "peer_false_answered_pct": (100.0 * peer_false_answered / n_answered)
            if n_answered else float("nan"),
            "mean_abs_err_samples": float(group["abs_err"].mean()),
            "median_abs_err_samples": float(group["abs_err"].median()),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", nargs="+", default=["conv1d", "qkv"])
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--models", type=Path,
                        default=REPO_ROOT / "firmware" / "esp32" / "models")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "results" / "full" / "mcu_bench" / "sensing")
    args = parser.parse_args(argv)

    config = peer_config(args.n)
    print(f"peer_estimation: n_symbols={args.n}, guards=1.0/1.25/1.5/2.0")

    tables = []
    for arch in args.arch:
        model = load_checkpoint(arch, DEFAULT_SCENARIO)
        int8 = Int8Model(args.models / f"ultra_can_{arch}_deploy_int8_recal.tflite")
        frames = {}
        for precision, candidate in (("float", model), ("int8", int8)):
            target = args.output_dir / f"{arch}_{precision}"
            evaluate_peer_estimation(candidate, config, target, arch)
            frames[precision] = pd.read_csv(target / "peer_samples.csv")
        for precision, frame in frames.items():
            tables.append(summarise(frame, precision, arch))
        agreement = np.mean(
            frames["float"]["abs_err"].to_numpy() == frames["int8"]["abs_err"].to_numpy()
        )
        print(f"  {arch}: identical |Delta tau| on {agreement * 100:.2f}% of the bursts")

    frame = pd.concat(tables, ignore_index=True)
    output = REPO_ROOT / "results" / "full" / "mcu_bench" / "sensing_parity.csv"
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"written {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
