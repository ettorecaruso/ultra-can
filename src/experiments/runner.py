

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments import pipeline
from src.models.dcsk_correlator import evaluate_classical, dcsk_correlator_demodulate
from src.utils.config_loader import (
    DEFAULT_BASE_CONFIG_PATH,
    _deep_merge,
    load_channels,
    load_config,
    save_config_snapshot,
    validate_config,
    with_channel_variant,
)
from src.utils.dataset_utils import get_dataset_dir
from src.utils.logger import get_logger, log_config_summary, setup_logging
from src.utils.model_io import load_model
from src.utils.model_names import REFERENCE_MODELS, canonical_model_name

logger = get_logger(__name__)

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _psutil = None
    _HAS_PSUTIL = False

def _log_rss(tag: str) -> None:
    
    if _HAS_PSUTIL:
        rss_gb = _psutil.Process().memory_info().rss / 1e9
    else:
        import resource as _res
        rss_gb = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1e6
    logger.info("RSS [%s] = %.2f GB", tag, rss_gb)

_EXPERIMENT_ORDER = ("ber_vs_snr", "classical_receivers", "jamming",
                     "jamming_interpretability", "frequency_agility",
                     "channel_generalization", "peer_estimation",
                     "final_report")

_CHANNEL_HOLD_MODES = ("per_symbol", "per_slot", "per_hop")

_VALID_MODELS = ("conv1d", "qkv", "lstm", "mc_dlsk")
_DEFAULT_MODEL = None
_DEFAULT_MODE = "fast"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "results"

def _validate_common_config(config: Dict[str, Any]) -> None:
    
    required_keys = (
        "general.experiment_name",
        "general.seed",
        "general.log_level",
        "data.sequence_length",
        "data.feature_mode",
        "data.snr_range",
        "data.snr_step",
        "data.echoes",
        "data.max_delay",
        "data.max_doppler",
        "training.batch_size",
        "training.lambda_mse",
        "training.loss_weights.comm",
        "training.loss_weights.sensing",
    )
    validate_config(config, required_keys)

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(
        description="Single entry point for the DH-ISAC experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/experiments/runner.py --experiments ber_vs_snr --mode fast
  python src/experiments/runner.py --experiments ber_vs_snr,jamming --mode full --model qkv
  python src/experiments/runner.py --experiments all --mode full --no-regen
        """,
    )
    parser.add_argument(
        "--experiments",
        required=True,
        help="Experiment identifiers to run, comma separated "
             "(e.g. ber_vs_snr,jamming) or 'all'",
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default=_DEFAULT_MODE,
        help=f"Execution mode (default: {_DEFAULT_MODE})",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=(
            "Architecture to run (single value, e.g. conv1d, or a comma "
            "separated list for jamming_interpretability). "
            f"(choices: {', '.join(_VALID_MODELS)}). When omitted, all "
            "architectures from the experiment config are used."
        ),
    )
    parser.add_argument(
        "--no-regen",
        action="store_true",
        help="If set, do not regenerate the dataset (use existing files)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory; the run root is <output-dir>/<mode>. Default: "
            f"{_DEFAULT_OUTPUT_DIR}."
        ),
    )
    parser.add_argument(
        "--channel",
        default=None,
        help=(
            "Channel variant name(s) from configs/channels.yaml, comma separated. "
            "The first one is applied to the run; channel_generalization evaluates "
            "the whole list. Default: the profile's channel_variants."
        ),
    )
    parser.add_argument(
        "--channel-file",
        type=Path,
        default=None,
        help="Path to channels.yaml (default: configs/channels.yaml)",
    )
    parser.add_argument(
        "--hop-hold-mode",
        choices=list(_CHANNEL_HOLD_MODES),
        default=None,
        help="Override channel.hold_mode (default: the profile's value).",
    )
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=None,
        help="Override the number of jammer realizations of the experiment.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help=(
            "Scenario name(s) of ber_vs_snr, comma separated: filters the scenarios "
            "of the profile, e.g. --scenario k3_doppler_full trains a single channel "
            "configuration instead of the three of the benchmark."
        ),
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help=(
            "Cap the symbols per realization of the jamming and interpretability "
            "experiments. Trading symbols for realizations keeps the wall clock "
            "constant while the dispersion of the mean shrinks."
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help=(
            "Extra run root(s) where the frozen receivers of the other experiments "
            "are looked up, comma separated. <root>/ber_vs_snr/<scenario>/<arch>/"
            "best_model.keras is searched, and <root>/<mode> too, so both the run "
            "root and the directory that contains it can be passed. Needed by an "
            "account that evaluates the checkpoints trained by another account."
        ),
    )
    parser.add_argument(
        "--jammers",
        default=None,
        help=(
            "Jamming types of the sweep, comma separated (cw, barrage, "
            "partial_band). The jamming and jamming_interpretability experiments "
            "then process only those types: the per-type CSV files are written "
            "for the subset, so a long leg can be split across sessions and the "
            "results merged afterwards. Default: the profile's list."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Override training.epochs (smoke tests of reproducibility: the "
            "protocol of the paper stays in the config files)."
        ),
    )
    parser.add_argument(
        "--supersede",
        action="store_true",
        help=(
            "When an experiment directory already contains results, move it to "
            "<output-dir>/archive/<experiment>_<timestamp>/ and write the new ones: "
            "the previous version is never destroyed, and it stays on the storage "
            "of the outputs (Drive included)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the resolved plan (channel, checkpoints, output paths) and exit "
            "without writing anything."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip the experiments whose output directory is already populated.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite a populated experiment directory in place, without archiving "
            "it. Meant for throwaway experiments: without it, and without "
            "--supersede, the runner refuses to write."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs" / "experiments.yaml",
        help="Path to the experiments.yaml file (default: configs/experiments.yaml)",
    )
    return parser.parse_args(argv)

def parse_experiment_list(experiments_arg: str) -> List[str]:
    if experiments_arg.strip().lower() == "all":
        return list(_EXPERIMENT_ORDER)

    parts = [p.strip() for p in experiments_arg.split(",") if p.strip()]
    if not parts:
        raise ValueError("--experiments cannot be empty")

    names: List[str] = []
    for token in parts:
        if token in _EXPERIMENT_ORDER:
            names.append(token)
        else:
            valid = ", ".join(_EXPERIMENT_ORDER)
            raise ValueError(f"Invalid experiment identifier {token!r}. Valid: {valid}")
    return list(dict.fromkeys(names))

def load_experiment_config(
    experiment_name: str,
    mode: str,
    experiments_yaml_path: Path,
    base_config_path: Path = DEFAULT_BASE_CONFIG_PATH,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    
    base = load_config(
        config_path=experiments_yaml_path,
        base_config_path=base_config_path,
        cli_overrides=None,
    )

    with open(experiments_yaml_path, "r", encoding="utf-8") as f:
        experiments_full = yaml.safe_load(f)

    exp_key = experiment_name
    if exp_key not in experiments_full:
        raise ValueError(f"Section '{exp_key}' missing in {experiments_yaml_path}")

    exp_section = experiments_full[exp_key]
    if mode not in exp_section:
        raise ValueError(
            f"Mode '{mode}' missing for {exp_key} in {experiments_yaml_path}"
        )

    exp_config = exp_section[mode]

    merged = _deep_merge(base, exp_config)

    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)

    _validate_common_config(merged)

    logger.debug("Config loaded for %s (%s): %d keys", experiment_name, mode, len(merged))
    return merged

def _evaluate_model(
    model: tf.keras.Model,
    test_data: Dict[str, np.ndarray],
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    
    from src.evaluation.evaluator import (
        compute_ber_curve,
        evaluate_model as eval_model,
        evaluate_model_online,
        plot_ber_vs_snr,
    )

    if config.get("evaluation", {}).get("online_generation", False):
        results = evaluate_model_online(model, config)
    else:
        results = eval_model(model, test_data, config)
    df_ber = compute_ber_curve(results)

    if "corr_tau" in df_ber.columns and len(df_ber) > 0:
        corr_tau_max = float(df_ber["corr_tau"].max())
        mse_tau_high = float(df_ber["mse_tau"].iloc[-1])
        if corr_tau_max < 0.15:
            logger.warning(
                "SENSING COLLAPSED? max corr(tau) = %.3f (threshold 0.15), "
                "MSE_tau high-SNR = %.2f (~variance %s): check "
                "use_reference_profile/feature_mode in model %s",
                corr_tau_max, mse_tau_high, "90.8", model.name,
            )
        else:
            logger.info(
                "Sensing OK: corr(tau) max = %.3f, MSE_tau high-SNR = %.2f",
                corr_tau_max, mse_tau_high,
            )
        if "corr_tau_argmax" in df_ber.columns and "exact_tau" in df_ber.columns:
            corr_argmax_max = float(df_ber["corr_tau_argmax"].max())
            exact_tau_high = float(df_ber["exact_tau"].iloc[-1])
            exact_argmax_high = float(df_ber["exact_tau_argmax"].iloc[-1])
            if corr_argmax_max - corr_tau_max > 0.05:
                logger.warning(
                    "Sensing head below its own peak picker: corr(tau) max = %.3f "
                    "vs argmax of the residual profile %.3f (exact delay rate "
                    "%.3f vs %.3f at the top of the grid): inspect the delay "
                    "head, delay_mode/use_position_feature in model %s",
                    corr_tau_max, corr_argmax_max,
                    exact_tau_high, exact_argmax_high, model.name,
                )
            else:
                logger.info(
                    "Sensing head matches its peak picker: corr(tau) %.3f vs %.3f, "
                    "exact delay rate %.3f vs %.3f",
                    corr_tau_max, corr_argmax_max,
                    exact_tau_high, exact_argmax_high,
                )
        if "corr_fd" in df_ber.columns:
            corr_fd_max = float(df_ber["corr_fd"].max())
            if corr_fd_max < 0.1:
                logger.warning(
                    "fD is not resolvable from the current parameters (max corr(fD) = %.3f): "
                    "the Doppler phase accumulated over N_seq=%d samples with "
                    "max_doppler=%g is ~0.05 rad (Cramer-Rao bound). "
                    "Estimating fD requires longer windows or a larger max_doppler.",
                    corr_fd_max,
                    int(config.get("data", {}).get("sequence_length", 100)),
                    config.get("data", {}).get("max_doppler", 8e-5),
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    df_ber.to_csv(csv_path, index=False)

    plot_format = config["visualization"].get("plot_format", "pdf")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_ber_vs_snr(df_ber, plot_dir, plot_format, model.name)

    return {"metrics": df_ber, "plots": {"ber_vs_snr": plot_path}}

def _evaluate_dcsk_on_test_data(
    test_data: Dict[str, np.ndarray],
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    
    from src.data.dataset_generator import generate_chaotic_sequence

    beta = int(config["baselines"]["dcsk_correlator"]["correlation_length"])
    threshold = float(config["baselines"]["dcsk_correlator"]["threshold"])
    snr_test_range = config["evaluation"]["snr_test_range"]
    batch_size = int(config["training"]["batch_size"])
    max_symbols_per_snr = config["evaluation"]["max_symbols_per_snr"]
    bit_error_threshold = config["evaluation"]["bit_error_threshold"]

    results: List[Dict[str, Any]] = []

    template = generate_chaotic_sequence(
        map_type=config["data"]["map_type"],
        map_param=config["data"]["map_param"],
        seed=int(config["general"]["seed"]),
        sequence_length=beta,
    )

    for snr_target in snr_test_range:
        mask = np.isclose(test_data["snr_db"], snr_target, rtol=0, atol=1e-6)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        y_snr = test_data["x"][idx]
        bit_snr = test_data["bit"][idx]
        n_total = len(bit_snr)

        accum_errors = 0
        accum_symbols = 0

        start = 0
        while (
            start < n_total
            and accum_symbols < max_symbols_per_snr
            and accum_errors < bit_error_threshold
        ):
            end = min(start + batch_size, n_total)
            batch_x = y_snr[start:end]
            batch_bit = bit_snr[start:end]

            bits_pred = dcsk_correlator_demodulate(batch_x, threshold=threshold)

            errors = np.sum(bits_pred != batch_bit)
            accum_errors += int(errors)
            accum_symbols += len(batch_bit)
            start = end

        if accum_symbols > 0:
            ber_val = accum_errors / accum_symbols
            results.append(
                {
                    "snr_db": float(snr_target),
                    "ber": ber_val,
                    "n_errors": accum_errors,
                    "n_symbols": accum_symbols,
                    "mse_tau": 0.0,
                    "mse_fd": 0.0,
                }
            )

    df = pd.DataFrame(results)
    df = df.sort_values("snr_db").reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)

    return df

def _find_scenario_by_name(scenarios: List[Dict], name: str) -> Optional[Dict]:
    
    for s in scenarios:
        if s.get("name") == name:
            return s
    return None

def _compute_winner_summary(
    curves: Dict[str, Dict[str, pd.DataFrame]]
) -> Dict[str, float]:
    
    arch_bers: Dict[str, List[float]] = {}
    for scenario_curves in curves.values():
        for arch, df in scenario_curves.items():
            if "ber" in df.columns and not df.empty:
                arch_bers.setdefault(arch, []).append(float(df["ber"].mean()))

    summary = {arch: float(np.mean(bers)) for arch, bers in arch_bers.items()}
    return summary

def _generate_pdf_from_tex(tex_path: Path) -> Optional[Path]:
    pdf_path = tex_path.with_suffix(".pdf")

    try:
        result = subprocess.run(
            ["pdflatex", "--version"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("pdflatex unavailable, skipping PDF generation")
            return None

        cwd = tex_path.parent
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex_path.name,
        ]
        logger.debug("Running command: %s (cwd=%s)", " ".join(cmd), cwd)

        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if result.returncode != 0:
            logger.error("pdflatex failed (code %d): %s", result.returncode, result.stderr)
            return None

        if pdf_path.exists():
            logger.info("PDF generated: %s", pdf_path)
            return pdf_path
        else:
            logger.warning("PDF not generated although pdflatex ran without errors")
            return None

    except FileNotFoundError:
        logger.warning("pdflatex not found in PATH, skipping PDF generation")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("pdflatex timeout after 30 seconds")
        return None
    except Exception as e:
        logger.warning("Error while generating the PDF: %s", e)
        return None

def run_ber_vs_snr(
    config: Dict[str, Any],
    mode: str,
    model_type: Optional[str],
    no_regen: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    
    logger.info("=" * 60)
    logger.info("ber_vs_snr experiment: BER vs SNR (5 architectures, 3 scenarios)")
    logger.info("=" * 60)

    exp_cfg = config["experiments"]["ber_vs_snr"]
    architectures = list(exp_cfg["architectures"])
    if model_type is not None:
        wanted = {model_type} if isinstance(model_type, str) else set(model_type)
        architectures = [a for a in architectures if a in wanted]
        logger.info(
            "ber_vs_snr: --model=%s filters architectures to: %s", model_type, architectures
        )
    if not architectures:
        raise SystemExit("ber_vs_snr: no architecture left after --model filtering")
    scenarios = exp_cfg["scenarios"]
    plot_format = config["visualization"].get("plot_format", "pdf")

    results: Dict[str, Any] = {
        "scenarios": {},
        "curves": {},
        "plots": {},
    }

    for scenario in scenarios:
        scenario_name = scenario["name"]
        logger.info("--- Scenario: %s ---", scenario_name)

        scenario_config = pipeline._apply_scenario_config(config, scenario)

        data_dir = pipeline.prepare_dataset(scenario_config, no_regen)
        online_eval = bool(
            (scenario_config.get("evaluation") or {}).get("online_generation", False)
        )
        needs_test = (not online_eval) or any(
            a in ("blind_stat", "dcsk") for a in architectures
        )
        train_data, train_ds, val_ds, test_data = pipeline.load_datasets(
            scenario_config, data_dir, include_test=needs_test
        )

        scenario_curves: Dict[str, pd.DataFrame] = {}
        scenario_results = {}

        for arch in architectures:
            logger.info("Architecture: %s", arch)
            if arch in ("conv1d", "qkv", "lstm", "mc_dlsk"):
                arch_output_dir = output_dir / scenario_name / arch
                model = pipeline.build_model(scenario_config, arch)
                scenario_config.setdefault("general", {})["run_output_dir"] = str(
                    arch_output_dir
                )
                res = pipeline.train_and_evaluate(
                    config=scenario_config,
                    model=model,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    test_data=test_data,
                    output_dir=arch_output_dir,
                )
                logs_dir = arch_output_dir / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                save_config_snapshot(scenario_config, logs_dir)
                df_ber = res["metrics"]
                scenario_curves[arch] = df_ber
                scenario_results[arch] = res
                res.pop("model", None)
                del model
                gc.collect()
                tf.keras.backend.clear_session()
                _log_rss(f"ber_vs_snr {scenario_name}/{arch}")
            elif arch == "dcsk":
                arch_output_dir = output_dir / scenario_name / "dcsk_correlator"
                df_dcsk = _evaluate_dcsk_on_test_data(
                    test_data, scenario_config, arch_output_dir
                )
                scenario_curves["dcsk"] = df_dcsk
                scenario_results["dcsk"] = {"metrics": df_dcsk}
            elif arch == "blind_stat":
                from src.models.blind_stat import evaluate_on_test_data as _eval_blind

                arch_output_dir = output_dir / scenario_name / "blind_stat"
                df_blind = _eval_blind(test_data, scenario_config, arch_output_dir)
                scenario_curves["blind_stat"] = df_blind
                scenario_results["blind_stat"] = {"metrics": df_blind}
            else:
                logger.warning("Unsupported architecture: %s, skipping", arch)
                continue

        plot_dir = output_dir / "plots"
        plot_path = pipeline.collect_and_plot_overlay(
            curves=scenario_curves,
            scenario=scenario_name,
            output_dir=plot_dir,
            plot_format=plot_format,
        )
        results["plots"][scenario_name] = plot_path
        results["curves"][scenario_name] = scenario_curves
        results["scenarios"][scenario_name] = scenario_results
        del train_data, train_ds, val_ds, test_data
        gc.collect()
        tf.keras.backend.clear_session()
        _log_rss(f"ber_vs_snr {scenario_name} scenario done")

    logger.info("ber_vs_snr experiment completed.")
    return results

def _find_scenario_trained_model(
    mode: str,
    scenario_name: str,
    arch: str,
    run_root: Optional[Path] = None,
) -> Optional[Path]:
    
    candidates: List[Path] = []
    if run_root is not None:
        candidates.append(
            run_root / "ber_vs_snr" / scenario_name / arch / "best_model.keras"
        )
    candidates.extend(
        [
            _REPO_ROOT / "results" / mode / "ber_vs_snr" / scenario_name / arch / "best_model.keras",
            _REPO_ROOT / "results" / "experiments" / "ber_vs_snr" / scenario_name / arch / "best_model.keras",
        ]
    )
    for ckpt in candidates:
        if ckpt.is_file():
            return ckpt
    return None

def run_classical_receivers(
    config: Dict[str, Any],
    mode: str,
    model_type: Optional[str],
    no_regen: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    
    logger.info("=" * 60)
    logger.info("Experiment classical_receivers: equations only (no DL) - classical receivers")
    logger.info("=" * 60)

    echo_config = copy.deepcopy(config)
    echo_config["channel"] = {
        "add_awgn": False,
        "echo_fading": "none",
        "echo_only_mode": True,
    }

    sweep_k_cfg = config.get("sweep_k", {})
    k_min = sweep_k_cfg.get("min", 1)
    k_max = sweep_k_cfg.get("max", 10)
    k_step = sweep_k_cfg.get("step", 1)
    k_range = range(k_min, k_max + 1, k_step)

    sweep_doppler_cfg = config.get("sweep_doppler", {})
    doppler_min = sweep_doppler_cfg.get("min", 0.0)
    doppler_max = sweep_doppler_cfg.get("max", 8e-5)
    doppler_num = sweep_doppler_cfg.get("num_points", 10)
    doppler_range = np.linspace(doppler_min, doppler_max, doppler_num)

    echo_cfg = config.get("echo_only", {})
    doppler_fixed = echo_cfg.get("doppler_fixed", 4e-5)
    k_fixed = echo_cfg.get("k_fixed", 3)

    num_symbols = config["data"].get("num_symbols_test", 2000)

    results_k = _sweep_k_echo_only(
        echo_config, k_range, doppler_fixed, num_symbols, output_dir, echo_cfg
    )

    results_doppler = _sweep_doppler_echo_only(
        echo_config, k_fixed, doppler_range, num_symbols, output_dir, echo_cfg
    )

    rows_k = [{"k": k, **ber_dict} for k, ber_dict in sorted(results_k.items())]
    pd.DataFrame(rows_k).to_csv(output_dir / "ber_vs_k.csv", index=False)
    rows_d = [{"doppler_max": d, **ber_dict} for d, ber_dict in sorted(results_doppler.items())]
    pd.DataFrame(rows_d).to_csv(output_dir / "ber_vs_doppler.csv", index=False)
    logger.info("classical_receivers results saved to %s/{ber_vs_k,ber_vs_doppler}.csv", output_dir)

    logger.info("classical_receivers experiment completed.")
    return {
        "results_k": results_k,
        "results_doppler": results_doppler,
    }

def _find_clean_ber_curve(
    mode: str,
    arch: str,
    run_root: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    
    bases: List[Path] = []
    if run_root is not None:
        bases.append(run_root / "ber_vs_snr")
    bases.append(_REPO_ROOT / "results" / mode / "ber_vs_snr")

    for base in bases:
        for scenario in ("k1_doppler_full", "k3_doppler_full", "k3_doppler_limited"):
            path = base / scenario / arch / "metrics.csv"
            if not path.is_file():
                continue
            try:
                df = pd.read_csv(path)
                if {"snr_db", "ber"}.issubset(df.columns) and not df.empty:
                    return df
            except Exception as exc:
                logger.warning("ber_vs_snr metrics.csv not readable (%s): %s", path, exc)
    logger.debug("clean ber_vs_snr curve not found for %s (mode=%s)", arch, mode)
    return None

def _plot_jamming_vs_clean_ber(
    model_name: str,
    jamming_dfs: Dict[str, pd.DataFrame],
    clean_curve: Optional[pd.DataFrame],
    output_dir: Path,
    plot_format: str,
) -> Optional[Path]:
    
    if not jamming_dfs:
        logger.warning("No jammed curve for %s: skipping the comparison", model_name)
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_clean, ax_jam) = plt.subplots(1, 2, figsize=(14, 6))

    clean_mean: Optional[float] = None
    if clean_curve is not None and not clean_curve.empty:
        ax_clean.semilogy(
            clean_curve["snr_db"], clean_curve["ber"],
            marker="o", linestyle="-", color="#1f77b4", linewidth=2,
            label="Clean (ber_vs_snr)",
        )
        ax_clean.set_xlabel("SNR (dB)")
        ax_clean.set_ylabel("Bit Error Rate (BER)")
        ax_clean.set_title(f"{model_name} - clean (ber_vs_snr)")
        ax_clean.grid(True, which="both", linestyle="--", alpha=0.6)
        ax_clean.legend()
        clean_mean = float(np.mean(clean_curve["ber"]))
    else:
        ax_clean.text(
            0.5, 0.5, "clean ber_vs_snr curve not available",
            ha="center", va="center", transform=ax_clean.transAxes,
        )
        ax_clean.set_title(f"{model_name} - clean (ber_vs_snr)")

    _JAM_COLORS = {"cw": "blue", "barrage": "red", "partial_band": "green"}
    _JAM_MARKERS = {"cw": "o", "barrage": "s", "partial_band": "D"}
    for jammer_type, df in jamming_dfs.items():
        if df is None or df.empty or "jsr_db" not in df or "ber" not in df:
            continue
        ax_jam.semilogy(
            df["jsr_db"], df["ber"],
            marker=_JAM_MARKERS.get(jammer_type, "o"),
            linestyle="-",
            color=_JAM_COLORS.get(jammer_type),
            linewidth=2,
            label=jammer_type.capitalize().replace("_", " "),
        )
    if clean_mean is not None:
        ax_jam.axhline(
            clean_mean, linestyle="--", color="gray", linewidth=1.5,
            label=f"Clean mean (ber_vs_snr) = {clean_mean:.4f}",
        )
    ax_jam.set_xlabel("JSR (dB)")
    ax_jam.set_ylabel("Bit Error Rate (BER)")
    ax_jam.set_title(f"{model_name} - jammed")
    ax_jam.grid(True, which="both", linestyle="--", alpha=0.6)
    ax_jam.legend()

    fig.suptitle(f"{model_name}: Jammed BER vs clean BER (ber_vs_snr)", y=1.02)
    out_path = output_dir / f"ber_vs_jsr_{model_name}_vs_clean.{plot_format}"
    plt.savefig(out_path, format=plot_format, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Jamming vs clean ber_vs_snr comparison saved to %s", out_path)
    return out_path

def run_jamming(
    config: Dict[str, Any],
    mode: str,
    model_type: Optional[str],
    no_regen: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    
    logger.info("=" * 60)
    logger.info("Experiment jamming: jamming robustness + explainability")
    logger.info("=" * 60)

    jamming_cfg = config.get("experiments", {}).get("jamming", {}) or {}
    models_to_test = list(jamming_cfg.get("models") or ["conv1d", "qkv"])
    if model_type is not None:
        models_to_test = [m for m in models_to_test if m == model_type]
        logger.info(
            "jamming: --model=%s -> models filtered to: %s", model_type, models_to_test
        )
    logger.info("Models to test: %s", models_to_test)

    data_dir = pipeline.prepare_dataset(config, no_regen)
    test_data = pipeline.load_test_data(config, data_dir)

    results: Dict[str, Any] = {"models": {}}
    visualize_layers = bool(jamming_cfg.get("visualize_layers", False))

    for model_name in models_to_test:
        logger.info("=" * 60)
        logger.info("Model: %s", model_name)
        logger.info("=" * 60)

        model_output_dir = output_dir / model_name
        res = pipeline.run_single_experiment(
            config=config,
            model_type=model_name,
            output_dir=model_output_dir,
            no_regen=no_regen,
        )
        model = res.get("model")
        if model is None:
            raise RuntimeError("run_single_experiment did not return a model")

        baseline_results = _evaluate_model(
            model, test_data, config, model_output_dir / "jamming"
        )

        jamming_dir = model_output_dir / "jamming"
        jamming_results = pipeline.evaluate_with_jamming(
            config=config,
            model=model,
            test_data=test_data,
            output_dir=jamming_dir,
            model_name=model_name,
        )

        if visualize_layers:
            layer_dir = model_output_dir / "layer_activity"
            pipeline.plot_layer_activity(
                config=config,
                model=model,
                test_data=test_data,
                output_dir=layer_dir,
            )

        results["models"][model_name] = {
            "baseline": baseline_results,
            "jamming": jamming_results,
            "model_path": model_output_dir / "best_model.keras",
        }

        plot_format = config["visualization"].get("plot_format", "pdf")
        _plot_jamming_vs_clean_ber(
            model_name=model_name,
            jamming_dfs=(jamming_results or {}).get("results", {}),
            clean_curve=_find_clean_ber_curve(
                mode, model_name, run_root=output_dir.parent
            ),
            output_dir=jamming_dir,
            plot_format=plot_format,
        )

        del model, res
        gc.collect()
        tf.keras.backend.clear_session()
        _log_rss(f"jamming {model_name}")

    _plot_jamming_multi_model(results, output_dir, plot_format)

    logger.info("jamming experiment completed.")
    return results

def _plot_jamming_multi_model(
    results: Dict[str, Any],
    output_dir: Path,
    plot_format: str,
) -> Optional[Path]:
    
    models = results.get("models", {})
    if not models:
        logger.warning("No models evaluated: skipping the multi-model plot")
        return None

    first = next(iter(models.values()))
    all_dfs = (first.get("jamming") or {}).get("results", {})
    if not all_dfs:
        logger.warning("No jamming results for the multi-model plot")
        return None

    jammer_types = list(all_dfs.keys())
    _JAM_PALETTE = {
        "conv1d": "#1f77b4", "qkv": "#ff7f0e", "lstm": "#2ca02c", "mc_dlsk": "#d62728",
    }

    fig, axes = plt.subplots(1, len(jammer_types), figsize=(6 * len(jammer_types), 5), squeeze=False)
    for ax, jt in zip(axes[0], jammer_types):
        for name, mod in models.items():
            dfs = (mod.get("jamming") or {}).get("results", {})
            df = dfs.get(jt)
            if df is None or df.empty:
                logger.warning("Model %s without jamming data for %s", name, jt)
                continue
            ax.semilogy(
                df["jsr_db"], df["ber"],
                marker="o", linestyle="-", linewidth=2, markersize=6,
                color=_JAM_PALETTE.get(name, "gray"),
                label=name,
            )
        ax.set_xlabel("JSR (dB)")
        ax.set_ylabel("Bit Error Rate (BER)")
        ax.set_title(jt.capitalize().replace("_", " "))
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.set_ylim([1e-6, 1.0])
        ax.legend()

    fig.suptitle("Jammed BER comparison - all models", y=1.02)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"ber_vs_jsr_all_models.{plot_format}"
    plt.savefig(out_path, format=plot_format, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Multi-model plot saved to %s", out_path)
    return out_path

def run_jamming_interpretability(
    config: Dict[str, Any],
    mode: str,
    model_type: Optional[str],
    no_regen: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    
    from src.data.data_loader import build_reference_matrix, load_npz_files
    from src.experiments.jamming_interpretability import run_jamming_interpretability_probe

    jamming_interpretability_cfg = (config.get("experiments") or {}).get("jamming_interpretability") or {}
    models_to_test = list(jamming_interpretability_cfg.get("models") or ["conv1d", "qkv"])
    if model_type is not None:
        wanted = {model_type} if isinstance(model_type, str) else set(model_type)
        models_to_test = [m for m in models_to_test if m in wanted]
    jsr_values = [float(v) for v in (jamming_interpretability_cfg.get("jsr_values") or [-10.0, -6.0, -2.0, 2.0, 6.0, 10.0])]
    jammer_types = list(jamming_interpretability_cfg.get("jamming_types") or ["cw", "barrage", "partial_band"])
    snr_eval = [float(v) for v in (jamming_interpretability_cfg.get("snr_eval") or [-1.0, 3.0, 7.0, 11.0, 15.0, 21.0])]
    ret_subset = int(jamming_interpretability_cfg.get("ret_subset", 3000))
    max_symbols = jamming_interpretability_cfg.get("max_symbols")
    default_realizations = int(
        jamming_interpretability_cfg.get(
            "n_realizations", (config.get("jamming") or {}).get("n_realizations", 1)
        )
    )
    per_arch_realizations = jamming_interpretability_cfg.get("n_realizations_by_arch") or {}

    logger.info("Experiment jamming_interpretability (models=%s)", models_to_test)
    logger.info("  snr_eval=%s jsr=%s jammer=%s ret_subset=%d realizations=%s/%s max_symbols=%s",
                snr_eval, jsr_values, jammer_types, ret_subset,
                default_realizations, per_arch_realizations, max_symbols)

    data_dir = pipeline.prepare_dataset(config, no_regen)
    echoes = [int(k) for k in config["data"]["echoes"]]
    test_data = load_npz_files(data_dir, snr_eval, echoes, "test", config)
    test_data["x_ref"] = build_reference_matrix(test_data["bit"], test_data["seed"], config)
    logger.info("jamming_interpretability: test set of %d samples (SNR %s)", test_data["x"].shape[0], snr_eval)

    results: Dict[str, Any] = {"models": {}}
    for arch in models_to_test:
        logger.info("=" * 60)
        logger.info("jamming_interpretability architecture: %s", arch)

        arch_out = output_dir / arch
        candidates = [
            output_dir.parent / "jamming" / arch / "best_model.keras",
            _REPO_ROOT / "results" / mode / "jamming" / arch / "best_model.keras",
            _REPO_ROOT / "results" / "full" / "jamming" / arch / "best_model.keras",
        ]
        ckpt_path = next((c for c in candidates if c.is_file()), None)
        if ckpt_path is not None:
            logger.info("jamming_interpretability %s: jamming model loaded from %s", arch, ckpt_path)
            model = load_model(ckpt_path)
        else:
            logger.warning("jamming_interpretability %s: jamming checkpoint not found, training a fallback", arch)
            jamming_dir = output_dir.parent / "jamming" / arch
            jamming_dir.mkdir(parents=True, exist_ok=True)
            res_train = pipeline.run_single_experiment(
                config=config, model_type=arch, output_dir=jamming_dir, no_regen=no_regen,
            )
            model = res_train["model"]

        arch_res = run_jamming_interpretability_probe(
            model=model,
            arch=arch,
            test_data=test_data,
            config=config,
            out_dir=arch_out,
            jsr_values=jsr_values,
            jammer_types=jammer_types,
            ret_subset=ret_subset,
            n_realizations=int(per_arch_realizations.get(arch, default_realizations)),
            max_symbols=int(max_symbols) if max_symbols is not None else None,
        )
        results["models"][arch] = arch_res

    logger.info("jamming_interpretability experiment completed. Output in %s", output_dir)
    return results

_CHECKPOINT_PROVENANCE: Dict[str, Dict[str, Any]] = {}

def _record_checkpoint_provenance(
    exp_name: str,
    arch: str,
    path: Path,
    preferred_scenario: str,
    matched_preferred_scenario: bool,
) -> None:
    _CHECKPOINT_PROVENANCE[f"{exp_name}:{arch}"] = {
        "checkpoint": str(path),
        "preferred_scenario": str(preferred_scenario),
        "matched_preferred_scenario": bool(matched_preferred_scenario),
    }

def _checkpoint_extra_roots(config: Dict[str, Any]) -> List[Path]:
    general = config.get("general") or {}
    raw = general.get("checkpoint_roots") or []
    if isinstance(raw, str):
        raw = [raw]
    roots: List[Path] = []
    for item in raw:
        base = Path(str(item))
        for candidate in (base, base / "full", base / "fast"):
            if candidate not in roots:
                roots.append(candidate)
    return roots

def _find_trained_model(
    config: Dict[str, Any],
    arch: str,
    output_dir: Path,
) -> Optional[Path]:
    
    candidates: List[Path] = []

    ber_vs_snr_cfg = config.get("experiments", {}).get("ber_vs_snr", {})
    scenarios = ber_vs_snr_cfg.get("scenarios", []) if isinstance(ber_vs_snr_cfg, dict) else []

    run_root_candidates = [
        output_dir.parent,
        output_dir.parents[1],
        _REPO_ROOT / "results" / "full",
    ]
    run_root_candidates.extend(_checkpoint_extra_roots(config))
    for scenario in scenarios:
        if isinstance(scenario, dict) and scenario.get("name"):
            for run_root in run_root_candidates:
                candidates.append(
                    run_root / "ber_vs_snr" / str(scenario["name"]) / arch / "best_model.keras"
                )
    candidates.append(_REPO_ROOT / "results" / "experiments" / "ber_vs_snr" / arch / "best_model.keras")

    experiment_name = config["general"]["experiment_name"]
    model_dir = _REPO_ROOT / "results" / experiment_name / arch / "models"
    candidates.extend([
        model_dir / "best_model.keras",
        model_dir / "best_model.h5",
    ])

    for ckpt_path in candidates:
        if ckpt_path.is_file():
            logger.debug("Checkpoint found for '%s': %s", arch, ckpt_path)
            return ckpt_path
    logger.debug("No checkpoint found for '%s' among: %s", arch, candidates)
    return None

def _resolve_arch_checkpoint(
    config: Dict[str, Any],
    arch: str,
    output_dir: Path,
    preferred_scenario: str,
) -> Optional[Path]:
    exp_name = output_dir.name
    candidates = [
        output_dir.parent / "ber_vs_snr" / str(preferred_scenario) / arch / "best_model.keras",
        _REPO_ROOT / "results" / "full" / "ber_vs_snr" / str(preferred_scenario) / arch / "best_model.keras",
    ]
    for root in _checkpoint_extra_roots(config):
        candidates.append(
            root / "ber_vs_snr" / str(preferred_scenario) / arch / "best_model.keras"
        )
    for candidate in candidates:
        if candidate.is_file():
            _record_checkpoint_provenance(exp_name, arch, candidate, preferred_scenario, True)
            return candidate

    fallback = _find_trained_model(config, arch, output_dir)
    if fallback is None:
        return None
    matched = str(preferred_scenario) in fallback.parts
    if not matched:
        logger.warning(
            "%s %s: no checkpoint for scenario '%s'; falling back to %s, which was "
            "trained on another channel: the comparison is not valid. Pass "
            "--checkpoint-root <run-root> or retrain the missing scenario.",
            exp_name, arch, preferred_scenario, fallback,
        )
    _record_checkpoint_provenance(exp_name, arch, fallback, preferred_scenario, matched)
    return fallback

def run_frequency_agility(
    config: Dict[str, Any],
    mode: str,
    model_type: Optional[str],
    no_regen: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    
    from src.experiments.frequency_agility import evaluate_frequency_agility

    fa_cfg = (config.get("experiments") or {}).get("frequency_agility") or {}
    models_to_test = list(fa_cfg.get("models") or ["conv1d", "qkv", "lstm", "mc_dlsk"])
    if model_type is not None:
        wanted = {model_type} if isinstance(model_type, str) else set(model_type)
        models_to_test = [m for m in models_to_test if m in wanted]
    preferred_scenario = str(fa_cfg.get("model_scenario", "k3_doppler_full"))

    if not bool((config.get("frequency_hopping") or {}).get("enable", True)):
        logger.warning("frequency_hopping.enable is false: skipping frequency_agility")
        return {"models": {}}

    logger.info("=" * 60)
    logger.info("frequency_agility experiment (models=%s)", models_to_test)
    logger.info("=" * 60)

    results: Dict[str, Any] = {"models": {}}
    for arch in models_to_test:
        ckpt_path = _resolve_arch_checkpoint(config, arch, output_dir, preferred_scenario)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"frequency_agility {arch}: checkpoint not found for scenario "
                f"{preferred_scenario}; refusing to skip silently"
            )
        logger.info("frequency_agility %s: model loaded from %s", arch, ckpt_path)
        model = load_model(ckpt_path)
        results["models"][arch] = evaluate_frequency_agility(
            model=model,
            config=config,
            output_dir=output_dir / arch,
            arch=arch,
        )
        del model
        gc.collect()
        tf.keras.backend.clear_session()
        _log_rss(f"frequency_agility {arch}")

    logger.info("frequency_agility experiment completed. Output in %s", output_dir)
    return results

def run_channel_generalization(
    config: Dict[str, Any],
    mode: str,
    model_type: Optional[str],
    no_regen: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    
    from src.experiments.channel_generalization import evaluate_channel_generalization

    cg_cfg = (config.get("experiments") or {}).get("channel_generalization") or {}
    models_to_test = list(cg_cfg.get("models") or ["conv1d", "qkv", "lstm", "mc_dlsk"])
    if model_type is not None:
        wanted = {model_type} if isinstance(model_type, str) else set(model_type)
        models_to_test = [m for m in models_to_test if m in wanted]
    preferred_scenario = str(cg_cfg.get("model_scenario", "k3_doppler_full"))

    logger.info("=" * 60)
    logger.info("channel_generalization experiment (models=%s)", models_to_test)
    logger.info("=" * 60)

    results: Dict[str, Any] = {"models": {}}
    for arch in models_to_test:
        ckpt_path = _resolve_arch_checkpoint(config, arch, output_dir, preferred_scenario)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"channel_generalization {arch}: checkpoint not found for scenario "
                f"{preferred_scenario}; refusing to skip silently"
            )
        logger.info("channel_generalization %s: model loaded from %s", arch, ckpt_path)
        model = load_model(ckpt_path)
        results["models"][arch] = evaluate_channel_generalization(
            model=model,
            config=config,
            output_dir=output_dir / arch,
            arch=arch,
        )
        del model
        gc.collect()
        tf.keras.backend.clear_session()
        _log_rss(f"channel_generalization {arch}")

    logger.info("channel_generalization experiment completed. Output in %s", output_dir)
    return results

def run_peer_estimation(
    config: Dict[str, Any],
    mode: str,
    model_type: Optional[str],
    no_regen: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    
    from src.experiments.peer_estimation import evaluate_peer_estimation

    pe_cfg = (config.get("experiments") or {}).get("peer_estimation") or {}
    models_to_test = list(pe_cfg.get("models") or ["conv1d"])
    if model_type is not None:
        wanted = {model_type} if isinstance(model_type, str) else set(model_type)
        models_to_test = [m for m in models_to_test if m in wanted]
    preferred_scenario = str(pe_cfg.get("model_scenario", "iod_peers"))

    logger.info("=" * 60)
    logger.info("peer_estimation experiment (models=%s)", models_to_test)
    logger.info("=" * 60)

    results: Dict[str, Any] = {"models": {}}
    for arch in models_to_test:
        ckpt_path = _resolve_arch_checkpoint(
            config, arch, output_dir, preferred_scenario
        )
        if ckpt_path is None:
            raise FileNotFoundError(
                f"peer_estimation {arch}: checkpoint not found for scenario "
                f"{preferred_scenario}; refusing to skip silently (results would "
                "be missing without notice)"
            )
        logger.info("peer_estimation %s: model loaded from %s", arch, ckpt_path)
        model = load_model(ckpt_path)
        results["models"][arch] = evaluate_peer_estimation(
            model=model,
            config=config,
            output_dir=output_dir / arch,
            arch=arch,
        )
        del model
        gc.collect()
        tf.keras.backend.clear_session()
        _log_rss(f"peer_estimation {arch}")

    logger.info("peer_estimation experiment completed. Output in %s", output_dir)
    return results

def run_final_report(
    config: Dict[str, Any],
    mode: str,
    model_type: Optional[str],
    no_regen: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    
    logger.info("=" * 60)
    logger.info("final_report experiment: final weight table (SWaP-C)")
    logger.info("=" * 60)

    models: Dict[str, Optional[tf.keras.Model]] = {}
    dl_archs = ["conv1d", "qkv", "lstm", "mc_dlsk"]

    for arch in dl_archs:
        ckpt_path = _find_trained_model(config, arch, output_dir)
        if ckpt_path is not None:
            logger.info("Loading model %s from %s", arch, ckpt_path)
            try:
                model = load_model(ckpt_path)
                models[arch] = model
            except Exception as e:
                logger.warning(
                    "Failed to load %s (%s): building from scratch",
                    arch, e,
                )
                try:
                    models[arch] = pipeline.build_model(config, arch)
                except Exception as build_exc:
                    logger.error("Building failed for %s: %s", arch, build_exc)
                    models[arch] = None
        else:
            logger.info("Building model %s from scratch (no checkpoint found)", arch)
            try:
                model = pipeline.build_model(config, arch)
                models[arch] = model
            except Exception as e:
                logger.error("Building failed for %s: %s", arch, e)
                models[arch] = None

    models["classical"] = None

    final_report_cfg = config["experiments"]["final_report"]
    output_rel = final_report_cfg.get("output", "weight_table.tex")
    output_path = Path(output_rel) if Path(output_rel).is_absolute() else output_dir / output_rel
    pipeline.generate_weight_table(
        config=config,
        models=models,
        output_path=output_path,
    )

    pdf_path = _generate_pdf_from_tex(output_path)

    report_path = None
    try:
        from src.experiments.final_report import generate_final_report
        report_path = generate_final_report(output_dir.parent, output_dir / "final_report.md", models)
        logger.info("Final report written to %s", report_path)
    except Exception as exc:
        logger.warning("Could not generate final report: %s", exc)

    logger.info("final_report experiment completed.")
    return {
        "tex_path": output_path,
        "pdf_path": pdf_path,
        "report_path": report_path,
    }

def _sweep_k_echo_only(
    config: Dict[str, Any],
    k_range: range,
    doppler_fixed: float,
    num_symbols: int,
    output_dir: Path,
    echo_cfg: Dict[str, Any],
) -> Dict[int, Dict[str, float]]:
    
    from src.data.dataset_generator import generate_chaotic_sequence

    beta = int(config["baselines"]["dcsk_correlator"]["correlation_length"])
    threshold = float(config["baselines"]["dcsk_correlator"]["threshold"])
    map_type = config["data"]["map_type"]
    map_param = config["data"]["map_param"]
    seed = int(config["general"]["seed"])

    template = generate_chaotic_sequence(map_type, map_param, seed, beta)
    doppler_direct = float(echo_cfg.get("doppler_direct_fixed", 1e-5))

    results: Dict[int, Dict[str, float]] = {}

    for k in k_range:
        logger.debug("K=%d, doppler_direct=%.2e", k, doppler_direct)
        y, bits_true = _build_echo_only_dataset(
            config, k, doppler_direct, num_symbols, echo_cfg
        )

        ber_dict: Dict[str, float] = {}
        for detector in ("dcsk", "matched_filter", "energy_detector"):
            if detector == "matched_filter":
                res = evaluate_classical(
                    y[:, beta:], bits_true, detector, config,
                    ref=None, template=template
                )
            else:
                res = evaluate_classical(
                    y, bits_true, detector, config,
                    ref=None, template=None
                )
            ber = float(res["ber"])
            if not np.isfinite(ber) or ber < 0.0 or ber > 1.0:
                raise RuntimeError(
                    f"BER not finite or out of range for K={k}, detector={detector}: {ber}"
                )
            ber_dict[detector] = ber

        results[k] = ber_dict

    return results

def _sweep_doppler_echo_only(
    config: Dict[str, Any],
    k_fixed: int,
    doppler_range: np.ndarray,
    num_symbols: int,
    output_dir: Path,
    echo_cfg: Dict[str, Any],
) -> Dict[float, Dict[str, float]]:
    
    from src.data.dataset_generator import generate_chaotic_sequence

    beta = int(config["baselines"]["dcsk_correlator"]["correlation_length"])
    threshold = float(config["baselines"]["dcsk_correlator"]["threshold"])
    map_type = config["data"]["map_type"]
    map_param = config["data"]["map_param"]
    seed = int(config["general"]["seed"])

    template = generate_chaotic_sequence(map_type, map_param, seed, beta)

    results: Dict[float, Dict[str, float]] = {}

    for doppler in doppler_range:
        logger.debug("doppler=%.2e, k_fixed=%d", doppler, k_fixed)
        y, bits_true = _build_echo_only_dataset(
            config, k_fixed, doppler, num_symbols, echo_cfg
        )

        ber_dict: Dict[str, float] = {}
        for detector in ("dcsk", "matched_filter", "energy_detector"):
            if detector == "matched_filter":
                res = evaluate_classical(
                    y[:, beta:], bits_true, detector, config,
                    ref=None, template=template
                )
            else:
                res = evaluate_classical(
                    y, bits_true, detector, config,
                    ref=None, template=None
                )
            ber = float(res["ber"])
            if not np.isfinite(ber) or ber < 0.0 or ber > 1.0:
                raise RuntimeError(
                    f"BER not finite or out of range for doppler={doppler}, detector={detector}: {ber}"
                )
            ber_dict[detector] = ber

        results[doppler] = ber_dict

    return results

def _build_echo_only_dataset(
    config: Dict[str, Any],
    k: int,
    doppler_direct: float,
    num_symbols: int,
    echo_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    
    from src.data.dataset_generator import generate_chaotic_sequence

    data_cfg = config["data"]
    sequence_length = int(data_cfg["sequence_length"])
    map_type = str(data_cfg["map_type"])
    map_param = float(data_cfg["map_param"])
    max_delay = int(data_cfg["max_delay"])
    beta = int(config["baselines"]["dcsk_correlator"]["correlation_length"])
    if 2 * beta != sequence_length:
        raise ValueError(
            f"classical_receivers requires 2*correlation_length ({2 * beta}) == sequence_length "
            f"({sequence_length}): the DCSK [ref|data] frame is not representable"
        )

    offset = int(echo_cfg.get("tau_offset", 7))
    alpha_base = float(echo_cfg.get("alpha_base", 0.2))
    alpha_variation = float(echo_cfg.get("alpha_variation", 0.5))
    echo_doppler_max = float(echo_cfg.get("doppler_fixed", 4e-5))
    snr_db = echo_cfg.get("snr_db")

    seed = int(config["general"]["seed"])
    rng = np.random.default_rng(seed + 42)

    bits_true = np.array([0] * (num_symbols // 2) + [1] * (num_symbols // 2), dtype=np.uint8)
    rng.shuffle(bits_true)

    frames = np.zeros((num_symbols, 2 * beta), dtype=np.float64)
    for i in range(num_symbols):
        ref = generate_chaotic_sequence(map_type, map_param, int(rng.integers(1, 2**31 - 1)), beta)
        data = ref if int(bits_true[i]) == 1 else -ref
        frames[i, :beta] = ref
        frames[i, beta:] = data

    guard = int(max_delay)
    stream_len = num_symbols * 2 * beta + guard
    stream = np.zeros(stream_len, dtype=np.float64)
    for i in range(num_symbols):
        stream[i * 2 * beta:(i + 1) * 2 * beta] = frames[i]

    n_idx = np.arange(stream_len, dtype=np.float64)

    alphas = [alpha_base * (1.0 + alpha_variation * (j / max(1, k - 1))) for j in range(k)]
    g = math.sqrt(max(1e-6, 1.0 - sum(a * a for a in alphas)))

    y = (g * stream).astype(np.complex128)

    for j in range(k):
        tau_j = (j * offset + 1) % max_delay + 1
        fd_j = (j / max(1, k - 1)) * echo_doppler_max
        x_delayed = np.zeros(stream_len, dtype=np.float64)
        x_delayed[tau_j:] = stream[: stream_len - tau_j]
        y = y + alphas[j] * x_delayed * np.exp(1j * 2.0 * math.pi * fd_j * n_idx)

    if snr_db is not None:
        snr_db = float(snr_db)
        ps = float(np.mean(np.abs(y) ** 2))
        noise_var = ps * 10.0 ** (-snr_db / 10.0)
        y = y + rng.normal(0.0, math.sqrt(noise_var / 2.0), stream_len) \
              + 1j * rng.normal(0.0, math.sqrt(noise_var / 2.0), stream_len)

    if not np.all(np.isfinite(y)):
        raise RuntimeError("echo-only signal not finite (NaN/Inf)")

    local_n = np.arange(2 * beta, dtype=np.float64)
    direct_phase = np.exp(1j * 2.0 * math.pi * float(doppler_direct) * local_n)
    y_sym = np.stack([
        y[i * 2 * beta:(i + 1) * 2 * beta] * direct_phase
        for i in range(num_symbols)
    ])
    return y_sym, bits_true

def _profile_channel_variants(config: Dict[str, Any], exp_name: str) -> List[str]:
    section = (config.get("experiments") or {}).get(exp_name) or {}
    names = section.get("channel_variants")
    if names is None:
        single = section.get("channel_variant")
        names = [] if single is None else [single]
    if isinstance(names, str):
        names = [names]
    return [str(name) for name in names]

def _cli_channel_variants(args: argparse.Namespace) -> Optional[List[str]]:
    if args.channel is None:
        return None
    names = [name.strip() for name in str(args.channel).split(",") if name.strip()]
    if not names:
        raise SystemExit("--channel does not contain a valid variant name")
    available = sorted(load_channels(args.channel_file))
    unknown = [name for name in names if name not in available]
    if unknown:
        raise SystemExit(f"unknown --channel {unknown}; available: {available}")
    return names

def _apply_channel(
    config: Dict[str, Any],
    exp_name: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    from src.data import channel_models

    selection = _cli_channel_variants(args)
    cli_selection = selection is not None
    if selection is None:
        selection = _profile_channel_variants(config, exp_name)
    if exp_name == "channel_generalization" and cli_selection:
        config.setdefault("experiments", {}).setdefault(exp_name, {})[
            "channel_variants"
        ] = list(selection)
    if selection:
        if exp_name != "channel_generalization" and len(selection) > 1:
            logger.warning(
                "%s: only %s is applied, the other requested channels are ignored",
                exp_name, selection[0],
            )
        config = with_channel_variant(config, selection[0], args.channel_file)
    if args.hop_hold_mode is not None:
        channel = dict(config.get("channel") or {})
        channel["hold_mode"] = str(args.hop_hold_mode)
        config["channel"] = channel
    if args.n_realizations is not None:
        count = int(args.n_realizations)
        if count < 1:
            raise SystemExit("--n-realizations must be >= 1")
        jamming = dict(config.get("jamming") or {})
        jamming["n_realizations"] = count
        config["jamming"] = jamming
        experiments = config.setdefault("experiments", {})
        for name in ("jamming", "jamming_interpretability", "frequency_agility"):
            section = dict(experiments.get(name) or {})
            if "n_realizations" in section:
                section["n_realizations"] = count
                experiments[name] = section
    if args.max_symbols is not None:
        cap = int(args.max_symbols)
        if cap < 1:
            raise SystemExit("--max-symbols must be >= 1")
        experiments = config.setdefault("experiments", {})
        for name in ("jamming", "jamming_interpretability"):
            section = dict(experiments.get(name) or {})
            section["max_symbols"] = cap
            experiments[name] = section
    channel_models.channel_model(config)
    return config

def _cli_path_list(value: Optional[str]) -> List[Path]:
    if value is None:
        return []
    return [Path(part.strip()) for part in str(value).split(",") if part.strip()]

def _apply_cli_protocol_overrides(
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if args.checkpoint_root is not None:
        roots = _cli_path_list(args.checkpoint_root)
        if not roots:
            raise SystemExit("--checkpoint-root does not contain a valid path")
        config.setdefault("general", {})["checkpoint_roots"] = [str(root) for root in roots]
        logger.info("Checkpoint roots (extra): %s", [str(root) for root in roots])
    if args.epochs is not None:
        epochs = int(args.epochs)
        if epochs < 1:
            raise SystemExit("--epochs must be >= 1")
        config.setdefault("training", {})["epochs"] = epochs
        logger.warning("--epochs: training.epochs overridden to %d (smoke run)", epochs)
    if args.jammers is not None:
        wanted = [name.strip() for name in str(args.jammers).split(",") if name.strip()]
        if not wanted:
            raise SystemExit("--jammers does not contain a valid name")
        from src.experiments.run_jamming import _VALID_JAMMING_TYPES
        unknown = [name for name in wanted if name not in _VALID_JAMMING_TYPES]
        if unknown:
            raise SystemExit(
                f"unknown --jammers {unknown}; available: {sorted(_VALID_JAMMING_TYPES)}"
            )
        jamming = dict(config.get("jamming") or {})
        jamming["jamming_types"] = list(wanted)
        config["jamming"] = jamming
        experiments = config.setdefault("experiments", {})
        for name in ("jamming", "jamming_interpretability"):
            section = dict(experiments.get(name) or {})
            section["jamming_types"] = list(wanted)
            experiments[name] = section
        logger.warning(
            "--jammers: the sweep is limited to %s. The per-type CSVs of the other "
            "types are left untouched, so a leg can be completed in another session "
            "on the same output directory; the aggregated plots then show the "
            "types of the current invocation only.",
            wanted,
        )
    return config

def _write_checkpoint_provenance(
    exp_name: str,
    exp_output_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    entries = {
        key: value for key, value in _CHECKPOINT_PROVENANCE.items()
        if key.split(":", 1)[0] == exp_name
    }
    if not entries:
        return {}
    log_dir = exp_output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "checkpoint_provenance.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
    logger.info("checkpoint provenance saved: %s", path)
    return entries

def _apply_scenario_filter(
    config: Dict[str, Any],
    exp_name: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if args.scenario is None:
        return config
    wanted = {name.strip() for name in str(args.scenario).split(",") if name.strip()}
    if not wanted:
        raise SystemExit("--scenario does not contain a valid name")
    if exp_name != "ber_vs_snr":
        logger.warning("--scenario is ignored by the %s experiment", exp_name)
        return config
    experiments = config.setdefault("experiments", {})
    section = dict(experiments.get("ber_vs_snr") or {})
    scenarios = [
        scenario
        for scenario in (section.get("scenarios") or [])
        if str(scenario.get("name")) in wanted
    ]
    if not scenarios:
        raise SystemExit(
            f"--scenario {sorted(wanted)} matches no scenario of the ber_vs_snr profile"
        )
    section["scenarios"] = scenarios
    experiments["ber_vs_snr"] = section
    logger.info("ber_vs_snr: scenarios limited to %s", [s["name"] for s in scenarios])
    return config

def _resolve_run_root(args: argparse.Namespace) -> Path:
    base = Path(args.output_dir) if args.output_dir is not None else _DEFAULT_OUTPUT_DIR
    return base / args.mode

def _archive_root(args: argparse.Namespace) -> Path:
    base = Path(args.output_dir) if args.output_dir is not None else _DEFAULT_OUTPUT_DIR
    return base / "archive"

def _supersede_directory(exp_output_dir: Path, archive_root: Optional[Path] = None) -> Path:
    root = archive_root if archive_root is not None else _REPO_ROOT / "results" / "archive"
    archive = root / f"{exp_output_dir.name}_{time.strftime('%Y-%m-%dT%H%M%S')}"
    if archive.exists():
        raise SystemExit(f"archive target already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exp_output_dir), str(archive))
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("superseded: previous results archived in %s", archive)
    return archive

def _planned_models(config: Dict[str, Any], exp_name: str) -> List[str]:
    section = (config.get("experiments") or {}).get(exp_name) or {}
    models = section.get("models") or section.get("architectures")
    if not models:
        return list(_VALID_MODELS)
    return [str(name) for name in models]

def _log_plan(
    exp_name: str,
    config: Dict[str, Any],
    exp_output_dir: Path,
) -> None:
    from src.data import channel_models

    metadata = channel_models.channel_metadata(config)
    logger.info("[plan] %s", exp_name)
    logger.info(
        "[plan]   channel    : %s (fingerprint %s, hold_mode %s)",
        metadata["model"], metadata["fingerprint"], metadata["hold_mode"],
    )
    logger.info("[plan]   propagation: %s", channel_models.diagnostics(config))
    if exp_name == "channel_generalization":
        variants = _profile_channel_variants(config, exp_name) or ["nominal"]
        logger.info("[plan]   variants   : %s", variants)
    populated = exp_output_dir.exists() and any(exp_output_dir.iterdir())
    logger.info("[plan]   output     : %s (populated: %s)", exp_output_dir, populated)
    if exp_name in ("channel_generalization", "frequency_agility", "peer_estimation"):
        section = (config.get("experiments") or {}).get(exp_name) or {}
        scenario = str(section.get("model_scenario", "k3_doppler_full"))
        for arch in _planned_models(config, exp_name):
            checkpoint = _resolve_arch_checkpoint(config, arch, exp_output_dir, scenario)
            logger.info(
                "[plan]   %-8s checkpoint: %s",
                arch, checkpoint if checkpoint is not None else "MISSING",
            )
    if exp_name == "jamming_interpretability":
        for arch in _planned_models(config, exp_name):
            candidates = [
                exp_output_dir.parent / "jamming" / arch / "best_model.keras",
                _REPO_ROOT / "results" / "full" / "jamming" / arch / "best_model.keras",
            ]
            checkpoint = next((c for c in candidates if c.is_file()), None)
            logger.info(
                "[plan]   %-8s checkpoint: %s",
                arch, checkpoint if checkpoint is not None else "MISSING",
            )

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if isinstance(args.model, str) and args.model.strip():
        args.model = ",".join(
            canonical_model_name(part)
            for part in args.model.split(",")
            if part.strip()
        )

    exp_names = parse_experiment_list(args.experiments)
    logger.info("Experiments to run: %s", exp_names)

    if isinstance(args.model, str) and "," in args.model:
        parts = [m.strip() for m in args.model.split(",") if m.strip()]
        invalid = [m for m in parts if m not in _VALID_MODELS + REFERENCE_MODELS]
        if invalid:
            raise ValueError(f"Invalid models in --model: {invalid}")
        if set(exp_names) - {"ber_vs_snr", "jamming", "jamming_interpretability",
                             "frequency_agility", "channel_generalization"}:
            raise ValueError(
                "--model with multiple architectures is allowed only with "
                "--experiments ber_vs_snr, jamming, jamming_interpretability, "
                "frequency_agility or channel_generalization"
            )
        args.model = parts
        logger.info("Requested models (multi): %s", parts)

    run_root = _resolve_run_root(args)
    if args.dry_run:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
        logger.info("Dry run: no file will be written")
    else:
        log_dir = run_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(log_dir=log_dir, level="INFO", experiment_name="runner")
    logger.info("Mode: %s", args.mode)
    logger.info("Run root: %s", run_root)
    logger.info("Models: %s", args.model if args.model is not None else "ALL (config)")
    logger.info("No dataset regeneration: %s", args.no_regen)

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    runners: Dict[str, Any] = {
        "ber_vs_snr": run_ber_vs_snr,
        "classical_receivers": run_classical_receivers,
        "jamming": run_jamming,
        "jamming_interpretability": run_jamming_interpretability,
        "frequency_agility": run_frequency_agility,
        "channel_generalization": run_channel_generalization,
        "peer_estimation": run_peer_estimation,
        "final_report": run_final_report,
    }

    all_results: Dict[str, Any] = {}
    failed_experiments: List[str] = []
    for exp_name in exp_names:
        logger.info("=" * 80)
        logger.info("EXPERIMENT %s", exp_name)
        logger.info("=" * 80)
        try:
            cli_overrides: Dict[str, Any] = {}
            if isinstance(args.model, str):
                cli_overrides = {"model": {"backbone_type": args.model}}
            config = load_experiment_config(
                experiment_name=exp_name,
                mode=args.mode,
                experiments_yaml_path=args.config,
                base_config_path=DEFAULT_BASE_CONFIG_PATH,
                cli_overrides=cli_overrides,
            )
            config = _apply_channel(config, exp_name, args)
            config = _apply_scenario_filter(config, exp_name, args)
            config = _apply_cli_protocol_overrides(config, args)

            exp_output_dir = run_root / exp_name
            if args.dry_run:
                _log_plan(exp_name, config, exp_output_dir)
                continue
            populated = exp_output_dir.exists() and any(exp_output_dir.iterdir())
            if populated and args.resume:
                logger.info(
                    "--resume: %s is already populated in %s, skipping",
                    exp_name, exp_output_dir,
                )
                all_results[exp_name] = {"skipped": True}
                continue
            if populated:
                if args.supersede:
                    _supersede_directory(exp_output_dir, _archive_root(args))
                elif args.force:
                    logger.warning("--force: overwriting %s in place", exp_output_dir)
                else:
                    message = (
                        f"{exp_output_dir} already contains results: pass --supersede to "
                        "archive them and write the new ones, --resume to keep them, or "
                        "--force to overwrite them in place"
                    )
                    logger.error("%s", message)
                    raise SystemExit(message)
            exp_output_dir.mkdir(parents=True, exist_ok=True)
            log_config_summary(config, logger)

            exp_log_dir = exp_output_dir / "logs"
            exp_log_dir.mkdir(parents=True, exist_ok=True)
            save_config_snapshot(config, exp_log_dir)
            config.setdefault("general", {})["run_output_dir"] = str(exp_output_dir)

            runner_fn = runners[exp_name]
            result = runner_fn(
                config=config,
                mode=args.mode,
                model_type=args.model,
                no_regen=args.no_regen,
                output_dir=exp_output_dir,
            )
            all_results[exp_name] = result
            provenance = _write_checkpoint_provenance(exp_name, exp_output_dir)
            if provenance and isinstance(all_results.get(exp_name), dict):
                all_results[exp_name]["checkpoint_provenance"] = provenance
            logger.info("Experiment %s completed successfully.", exp_name)
        except Exception as exc:
            logger.error("Experiment %s failed: %s", exp_name, exc, exc_info=True)
            all_results[exp_name] = None
            failed_experiments.append(exp_name)
            continue

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for exp_name, result in all_results.items():
        logger.info("%s: %s", exp_name, "OK" if result else "FAILED")
    logger.info("All requested experiments finished. Output in: %s", args.output_dir)
    if failed_experiments:
        raise SystemExit(
            "experiments failed, results incomplete: " + ", ".join(failed_experiments)
        )

if __name__ == "__main__":
    main()
