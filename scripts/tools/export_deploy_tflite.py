from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from src.experiments.mcu_deployment import (
    build_deploy_1d,
    deploy_inputs,
    export_int8_model,
    load_burst_split,
    load_checkpoint,
    resolve_config,
    DEFAULT_SCENARIO,
)
from tflite_to_cc import write_c_array

tf.config.threading.set_intra_op_parallelism_threads(1)

CALIBRATION_SNR = (5.0, 13.0)
CALIBRATION_PER_SNR = 200


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", nargs="+", default=["conv1d", "qkv"])
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--snr", type=float, nargs="+", default=list(CALIBRATION_SNR))
    parser.add_argument("--per-snr", type=int, default=CALIBRATION_PER_SNR)
    parser.add_argument("--no-cc", action="store_true",
                        help="do not regenerate the firmware C arrays")
    args = parser.parse_args(argv)

    config, scenario = resolve_config(scenario=args.scenario)
    max_delay = int(config["data"]["max_delay"])
    split = load_burst_split(config, args.snr, args.per_snr)
    iq = np.concatenate([deploy_inputs(split[s][0], max_delay)[0] for s in args.snr])
    profile = np.concatenate([deploy_inputs(split[s][0], max_delay)[1] for s in args.snr])
    print(f"calibration bursts: {len(iq)} from SNR {args.snr} ({scenario['name']})")

    for arch in args.arch:
        model = load_checkpoint(arch, args.scenario)
        deploy = build_deploy_1d(model, arch)
        out = REPO_ROOT / "firmware" / "esp32" / "models" / (
            f"ultra_can_{arch}_deploy_int8_recal.tflite")
        size = export_int8_model(deploy, iq, profile, out)
        print(f"  {arch}: {out.name} ({size} B)")
        if not args.no_cc:
            symbol = "ultra_can_model" if arch == "qkv" else "ultra_can_model_conv1d"
            target = REPO_ROOT / "firmware" / "esp32" / "src" / (
                "model_data.cc" if arch == "qkv" else "model_data_conv1d.cc")
            write_c_array(out, symbol, target)
            print(f"    -> {target.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
