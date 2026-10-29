from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.mcu_deployment import DEFAULT_SCENARIO, Int8Model, ber_of, load_burst_split, load_checkpoint, resolve_config

tf.config.threading.set_intra_op_parallelism_threads(1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", nargs="+", default=["conv1d", "qkv"])
    parser.add_argument("--snr", type=float, nargs="+", default=[5.0, 9.0, 13.0, 17.0, 21.0])
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--models", type=Path,
                        default=REPO_ROOT / "firmware" / "esp32" / "models")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "results" / "full" / "mcu_bench" / "int8_parity.csv")
    args = parser.parse_args(argv)

    config, scenario = resolve_config(scenario=args.scenario)
    split = load_burst_split(config, args.snr, args.n)
    print(f"scenario={scenario['name']} symbols/snr={args.n}")

    rows = []
    for arch in args.arch:
        model = load_checkpoint(arch, args.scenario)
        int8 = Int8Model(args.models / f"ultra_can_{arch}_deploy_int8_recal.tflite")
        for snr in sorted(split):
            features, bits = split[snr]
            reference = (model(features, training=False)["comm"]).numpy()
            deployed = int8(features)["comm"].numpy()
            rows.append({
                "arch": arch,
                "snr_db": snr,
                "n": int(len(bits)),
                "ber_float": ber_of(reference, bits),
                "ber_int8": ber_of(deployed, bits),
                "agree": float(np.mean(np.argmax(reference, -1) == np.argmax(deployed, -1))),
            })
            print(f"  {arch:6s} snr={snr:5.1f} dB  float={rows[-1]['ber_float']:.5f}"
                  f"  int8={rows[-1]['ber_int8']:.5f}  agree={rows[-1]['agree']*100:.3f}%")

    frame = pd.DataFrame(rows)
    frame["delta"] = frame["ber_int8"] - frame["ber_float"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
