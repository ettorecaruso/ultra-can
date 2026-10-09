
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config, validate_config
from src.utils.logger import get_logger, log_config_summary, setup_logging

logger = get_logger(__name__)

_ARCH_NAMES = ("conv1d", "qkv", "lstm", "mc_dlsk", "dcsk")

_ARCH_DISPLAY_NAMES = {
    "conv1d": "Ultra-CAN (Conv1D+Att.)",
    "qkv": "Ultra-CAN (QKV Attention)",
    "lstm": "LSTM (LSTM-OFDM-DCSK)",
    "mc_dlsk": "MC-DLCSK (BiLSTM)",
    "dcsk": "Classical DCSK correlator",
}

_ARCH_DISPLAY_NAMES.update({
    "dcsk_dcsk": "Classical DCSK",
    "dcsk_matched_filter": "Matched filter",
    "dcsk_energy_detector": "Energy detector",
})

_REDUCED_STYLE = {"marker": "o", "linestyle": "--"}
_ARCH_DISPLAY_NAMES.update({
    "conv1d_reduced": "Ultra-CAN (Conv1D+Att.) reduced",
    "qkv_reduced": "Ultra-CAN (QKV Attention) reduced",
    "lstm_reduced": "LSTM reduced",
    "mc_dlsk_reduced": "MC-DLCSK (BiLSTM) reduced",
})

_PALETTE = {
    "conv1d": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "qkv": {"color": "#ff7f0e", "marker": "s", "linestyle": "-"},
    "lstm": {"color": "#2ca02c", "marker": "^", "linestyle": "-"},
    "mc_dlsk": {"color": "#d62728", "marker": "D", "linestyle": "-"},
    "dcsk": {"color": "#9467bd", "marker": "x", "linestyle": "--"},
}

_PALETTE.update({
    "conv1d_reduced": {"color": "#1f77b4", **_REDUCED_STYLE},
    "qkv_reduced": {"color": "#ff7f0e", **_REDUCED_STYLE},
    "lstm_reduced": {"color": "#2ca02c", **_REDUCED_STYLE},
    "mc_dlsk_reduced": {"color": "#d62728", **_REDUCED_STYLE},
})

_PALETTE.update({
    "dcsk_dcsk": {"color": "#9467bd", "marker": "x", "linestyle": "--"},
    "dcsk_matched_filter": {"color": "#8c564b", "marker": "+", "linestyle": "--"},
    "dcsk_energy_detector": {"color": "#e377c2", "marker": "1", "linestyle": "--"},
})

_TITLE_MAP = {
    "k1_doppler_full": "K=1, full Doppler (fD in [0, 8e-5])",
    "k3_doppler_full": "K=3, full Doppler (fD in [0, 8e-5])",
    "k3_doppler_limited": "K=3, limited Doppler (fD in [0, 4e-5])",
}

_REQUIRED_COLUMNS = frozenset({"snr_db", "ber", "mse_tau", "mse_fd", "n_errors", "n_symbols"})

_Y_LOWER_MIN = 1e-6
_MSE_NEGATIVE_TOL = 1e-9

def _validate_config(config: Dict[str, Any]) -> None:
    
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got: {type(config).__name__}")

    general = config.get("general")
    if not isinstance(general, dict):
        raise ValueError("'general' section missing or not a dict")
    if "experiment_name" not in general:
        raise ValueError("missing key: general.experiment_name")

    ber_vs_snr_cfg = config.get("experiments", {}).get("ber_vs_snr")
    if not isinstance(ber_vs_snr_cfg, dict):
        raise ValueError("'experiments.ber_vs_snr' section missing or not a dict")
    scenarios = ber_vs_snr_cfg.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) == 0:
        raise ValueError("experiments.ber_vs_snr.scenarios must be a non-empty list")
    for idx, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(f"scenario[{idx}] must be a dict")
        if "name" not in scenario:
            raise ValueError(f"scenario[{idx}] is missing the key 'name'")

    vis_cfg = config.get("visualization")
    if vis_cfg is not None and not isinstance(vis_cfg, dict):
        raise ValueError("'visualization' section must be a dict if present")

    logger.debug("Config validated for compare_baselines")

def collect_curves(model_dirs: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    
    if not isinstance(model_dirs, dict):
        raise TypeError(f"model_dirs must be a dict, got: {type(model_dirs).__name__}")

    result: Dict[str, pd.DataFrame] = {}

    for arch_name, dir_path in model_dirs.items():
        csv_path = dir_path / "metrics.csv"

        if not csv_path.exists():
            logger.warning(
                "File %s not found for architecture '%s', skipping",
                csv_path,
                arch_name,
            )
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning(
                "Unable to read %s for architecture '%s': %s, skipping",
                csv_path,
                arch_name,
                e,
            )
            continue

        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            logger.warning(
                "Missing columns in %s for architecture '%s': %s, skipping",
                csv_path,
                arch_name,
                sorted(missing),
            )
            continue

        df = df.sort_values("snr_db").reset_index(drop=True)

        if not np.all(np.isfinite(df["ber"].values)):
            logger.warning(
                "Non-finite BER in %s for architecture '%s', skipping",
                csv_path,
                arch_name,
            )
            continue

        result[arch_name] = df
        logger.debug("Loaded curve for architecture '%s': %d SNR points", arch_name, len(df))

    if not result:
        raise ValueError(
            "No valid curve found. Check that the metrics.csv files exist "
            "and contain the required columns."
        )

    logger.info("Loaded %d curves: %s", len(result), sorted(result.keys()))
    return result

def plot_ber_overlay(
    curves: Dict[str, pd.DataFrame],
    scenario: str,
    output_dir: Path,
    plot_format: str = "pdf",
) -> Path:
    
    if not isinstance(curves, dict):
        raise TypeError(f"curves must be a dict, got: {type(curves).__name__}")
    if not curves:
        raise ValueError("curves is empty: cannot generate the plot")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for arch_name, df in curves.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"curves['{arch_name}'] must be a pd.DataFrame, "
                f"got: {type(df).__name__}"
            )
        if "snr_db" not in df.columns or "ber" not in df.columns:
            raise ValueError(
                f"curves['{arch_name}'] must contain the columns 'snr_db' and 'ber'"
            )

    fig, ax = plt.subplots(figsize=(10, 7))

    min_ber = float("inf")
    for df in curves.values():
        if not df.empty:
            df_min = df["ber"].min()
            if np.isfinite(df_min):
                min_ber = min(min_ber, df_min)

    if np.isfinite(min_ber) and min_ber > 0:
        y_lower = max(_Y_LOWER_MIN, min_ber / 10.0)
    else:
        y_lower = _Y_LOWER_MIN

    for arch_name, df in curves.items():
        if arch_name not in _PALETTE:
            logger.warning(
                "Fallback style for unknown architecture: '%s'",
                arch_name,
            )
            style = {"color": "gray", "marker": ".", "linestyle": "-"}
        else:
            style = _PALETTE[arch_name]

        display_name = _ARCH_DISPLAY_NAMES.get(arch_name, arch_name)

        ax.semilogy(
            df["snr_db"],
            df["ber"],
            marker=style.get("marker", "."),
            linestyle=style.get("linestyle", "-"),
            color=style.get("color", None),
            linewidth=2,
            label=display_name,
            markersize=8,
        )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    title = _TITLE_MAP.get(scenario, scenario)
    ax.set_title(f"BER vs SNR comparison - {title}")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.set_ylim(y_lower, 1.0)
    ax.legend()

    output_path = output_dir / f"ber_vs_snr_{scenario}.{plot_format}"
    plt.savefig(output_path, format=plot_format, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("BER plot saved to %s", output_path)
    return output_path

def plot_rmse_overlay(
    curves: Dict[str, pd.DataFrame],
    scenario: str,
    output_dir: Path,
    plot_format: str = "pdf",
) -> Path:
    
    if not isinstance(curves, dict):
        raise TypeError(f"curves must be a dict, got: {type(curves).__name__}")
    if not curves:
        raise ValueError("curves is empty: cannot generate the plot")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for arch_name, df in curves.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"curves['{arch_name}'] must be a pd.DataFrame, "
                f"got: {type(df).__name__}"
            )
        if "snr_db" not in df.columns:
            raise ValueError(f"curves['{arch_name}'] must contain the column 'snr_db'")
        if "mse_tau" not in df.columns:
            raise ValueError(f"curves['{arch_name}'] must contain the column 'mse_tau'")
        if "mse_fd" not in df.columns:
            raise ValueError(f"curves['{arch_name}'] must contain the column 'mse_fd'")

    curves_rmse: Dict[str, pd.DataFrame] = {}
    for arch_name, df in curves.items():
        df_rmse = df.copy()
        df_rmse["rmse_tau"] = np.sqrt(np.maximum(df["mse_tau"], 0.0))
        df_rmse["rmse_fd"] = np.sqrt(np.maximum(df["mse_fd"], 0.0))
        curves_rmse[arch_name] = df_rmse

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for arch_name, df in curves_rmse.items():
        if arch_name not in _PALETTE:
            logger.warning(
                "Fallback style for unknown architecture: '%s'",
                arch_name,
            )
            style = {"color": "gray", "marker": ".", "linestyle": "-"}
        else:
            style = _PALETTE[arch_name]

        display_name = _ARCH_DISPLAY_NAMES.get(arch_name, arch_name)

        ax1.semilogy(
            df["snr_db"],
            df["rmse_tau"],
            marker=style.get("marker", "."),
            linestyle=style.get("linestyle", "-"),
            color=style.get("color", None),
            linewidth=2,
            label=display_name,
            markersize=8,
        )

    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("RMSE τ (samples)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    for arch_name, df in curves_rmse.items():
        if arch_name not in _PALETTE:
            style = {"color": "gray", "marker": ".", "linestyle": "-"}
        else:
            style = _PALETTE[arch_name]

        display_name = _ARCH_DISPLAY_NAMES.get(arch_name, arch_name)

        ax2.semilogy(
            df["snr_db"],
            df["rmse_fd"],
            marker=style.get("marker", "."),
            linestyle=style.get("linestyle", "-"),
            color=style.get("color", None),
            linewidth=2,
            label=display_name,
            markersize=8,
        )

    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("RMSE fD (cycles/sample)")
    ax2.grid(True, linestyle="--", alpha=0.6)

    title = _TITLE_MAP.get(scenario, scenario)
    fig.suptitle(f"Sensing RMSE vs SNR comparison - {title}")

    output_path = output_dir / f"sensing_rmse_{scenario}.{plot_format}"
    plt.savefig(output_path, format=plot_format, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("RMSE plot saved to %s", output_path)
    return output_path

def main(argv: Optional[Sequence[str]] = None) -> None:
    
    parser = argparse.ArgumentParser(
        description="BER vs SNR and sensing RMSE overlays for all receiver architectures"
    )
    parser.add_argument("--config", required=True, help="path to the experiment config (YAML)")
    parser.add_argument(
        "--scenario",
        default="all",
        help="specific scenario (e.g. k1_doppler_full) or 'all' for all 3 scenarios (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="plot output directory (default: <results-dir>/ber_vs_snr/<scenario>)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "runner results directory (default: results/experiments). "
            "The metrics.csv files are looked up in <results-dir>/ber_vs_snr/<scenario>/<arch>/"
        ),
    )
    parser.add_argument(
        "--plot-format",
        default="pdf",
        help="plot format (default: pdf)",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path, DEFAULT_BASE_CONFIG_PATH)
    _validate_config(config)

    general = config.get("general", {})
    experiment_name = str(general.get("experiment_name", "compare_baselines"))
    log_dir = _REPO_ROOT / "results" / experiment_name / "logs"
    log_file = setup_logging(
        log_dir=log_dir,
        level=str(general.get("log_level", "INFO")),
        experiment_name=f"{experiment_name}_compare",
    )
    log_config_summary(config, logger)
    logger.info("Log file: %s", log_file)

    ber_vs_snr_cfg = config["experiments"]["ber_vs_snr"]
    all_scenarios = ber_vs_snr_cfg["scenarios"]

    if args.scenario == "all":
        scenarios = all_scenarios
    else:
        filtered = [s for s in all_scenarios if s["name"] == args.scenario]
        if not filtered:
            raise ValueError(
                f"Scenario '{args.scenario}' not found. "
                f"Available scenarios: {[s['name'] for s in all_scenarios]}"
            )
        scenarios = filtered

    results_dir = (
        Path(args.results_dir)
        if args.results_dir is not None
        else _REPO_ROOT / "results" / "experiments"
    )

    arch_names = list(_ARCH_NAMES)
    for scenario in scenarios:
        scenario_name = scenario["name"]
        logger.info("Processing scenario: %s", scenario_name)

        model_dirs: Dict[str, Path] = {}
        for arch in arch_names:
            if arch == "dcsk":
                model_dir = results_dir / "ber_vs_snr" / scenario_name / "dcsk_correlator"
            else:
                model_dir = results_dir / "ber_vs_snr" / scenario_name / arch
            model_dirs[arch] = model_dir

        curves = collect_curves(model_dirs)

        if args.output_dir:
            output_dir = Path(args.output_dir) / scenario_name
        else:
            output_dir = results_dir / "ber_vs_snr" / scenario_name

        plot_ber_overlay(curves, scenario_name, output_dir, args.plot_format)
        plot_rmse_overlay(curves, scenario_name, output_dir, args.plot_format)

        logger.info("Scenario %s completed", scenario_name)

    logger.info("Comparison completed for %d scenario(s)", len(scenarios))

if __name__ == "__main__":
    main()

