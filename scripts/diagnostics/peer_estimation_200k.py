from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.peer_estimation import evaluate_peer_estimation
from src.utils.model_io import load_model

_RESULTS = _REPO_ROOT / "results" / "full"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-symbols", type=int, default=200_000)
    parser.add_argument("--snr-db", type=float, default=21.0)
    parser.add_argument(
        "--guard-factors",
        nargs="+",
        type=float,
        default=[1.0, 1.25, 1.5, 2.0],
        help="guard values evaluated on the same 200k batch",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_RESULTS / "peer_estimation_200k"
    )
    parser.add_argument(
        "--archs", nargs="+", default=["conv1d", "qkv"], choices=["conv1d", "qkv"]
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for arch in args.archs:
        config_path = (
            _RESULTS
            / "ber_vs_snr"
            / "iod_peers"
            / arch
            / "logs"
            / "config_used.yaml"
        )
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config.setdefault("experiments", {}).setdefault("peer_estimation", {})
        config["experiments"]["peer_estimation"].update(
            {
                "n_symbols": int(args.n_symbols),
                "snr_db": float(args.snr_db),
                "echoes": 3,
                "seed": 20260926,
                "guard_factors": [float(v) for v in args.guard_factors],
            }
        )
        checkpoint = (
            _RESULTS / "ber_vs_snr" / "iod_peers" / arch / "best_model.keras"
        )
        print(f"[peer-estimation] {arch}: {checkpoint}", flush=True)
        evaluate_peer_estimation(
            model=load_model(checkpoint),
            config=config,
            output_dir=args.output_dir / arch,
            arch=arch,
        )


if __name__ == "__main__":
    main()
