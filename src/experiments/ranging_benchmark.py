from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.data_loader import _build_feature_matrix
from src.data.dataset_generator import generate_test_batch
from src.data.scene import require_monostatic
from src.evaluation.ranging import (
    abstention_mask,
    oracle_hit_rate,
    samples_to_meters,
    scored_mae,
    scored_median_ae,
    silence_fraction,
)
from src.experiments.peer_estimation import _matched_filter_profile
from src.experiments.pipeline import predict_in_chunks, resolve_scenario
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config
from src.utils.logger import get_logger, log_config_summary, setup_logging
from src.utils.model_io import load_model

logger = get_logger(__name__)

_RESULTS = _REPO_ROOT / "results" / "full"
_DEFAULT_CHECKPOINT_ROOT = _RESULTS / "ber_vs_snr"
_DEFAULT_SYMBOLS = 4000
_DEFAULT_SEED = 20261008
_DEFAULT_SNR = [5.0, 11.0, 21.0]
_DEFAULT_GAMMA = [1.0, 1.25, 1.5, 2.0]

def _settings(config: Dict[str, Any]) -> Dict[str, Any]:
    section = (config.get("experiments") or {}).get("ranging_benchmark") or {}
    return {
        "n_symbols": int(section.get("n_symbols", _DEFAULT_SYMBOLS)),
        "snr_db": [float(value) for value in section.get("snr_db", _DEFAULT_SNR)],
        "gamma": [float(value) for value in section.get("gamma", _DEFAULT_GAMMA)],
        "seed": int(section.get("seed", _DEFAULT_SEED)),
        "archs": list(section.get("archs", ["conv1d", "qkv"])),
    }

def _head_delay(model: tf.keras.Model, features: np.ndarray, tau_max: float) -> np.ndarray:
    prediction = predict_in_chunks(model, features, "sensing")
    return np.clip(prediction[:, 0], 0.0, 1.0) * float(tau_max)

def _pile(
    obstacle_alpha: np.ndarray,
    peer_mask: Optional[np.ndarray] = None,
    peer_alpha: Optional[np.ndarray] = None,
    peer_taus: Optional[np.ndarray] = None,
    tau_true: Optional[np.ndarray] = None,
) -> np.ndarray:
    out = np.full(obstacle_alpha.shape, "obstacle", dtype=object)
    if peer_mask is None or peer_alpha is None:
        return out
    mask = np.asarray(peer_mask, dtype=bool)
    alpha = np.asarray(peer_alpha, dtype=np.float64)
    if mask.ndim != 2 or alpha.ndim != 2 or mask.shape[0] != alpha.shape[0]:
        return out
    width = int(min(mask.shape[1], alpha.shape[1]))
    if width == 0:
        return out
    mask = mask[:, mask.shape[1] - width:]
    alpha = alpha[:, alpha.shape[1] - width:]
    if not np.any(mask):
        return out
    strongest_peer = np.where(mask, alpha, 0.0).max(axis=1)
    out = np.where(strongest_peer > obstacle_alpha, "peer", "obstacle").astype(object)
    if peer_taus is None or tau_true is None:
        return out
    taus = np.asarray(peer_taus, dtype=np.float64)
    if taus.ndim != 2 or taus.shape[0] != mask.shape[0]:
        return out
    width_tau = int(min(taus.shape[1], width))
    if width_tau == 0:
        return out
    taus = taus[:, taus.shape[1] - width_tau:]
    cell_mask = mask[:, width - width_tau:]
    reference = np.rint(np.asarray(tau_true, dtype=np.float64))[:, None]
    same_cell = np.any(
        cell_mask & (np.abs(np.rint(taus) - reference) <= 0.5), axis=1
    )
    return np.where(same_cell, "same_cell", out).astype(object)

def _subset_metrics(
    pred: np.ndarray, true: np.ndarray, keep: np.ndarray
) -> Dict[str, float]:
    pred_arr = np.asarray(pred, dtype=np.float64)
    true_arr = np.asarray(true, dtype=np.float64)
    sel = np.asarray(keep, dtype=bool)
    if pred_arr.shape != true_arr.shape or sel.shape != true_arr.shape:
        raise ValueError("pred/true/keep must share a shape")
    if not np.any(sel):
        return {"n": 0.0, "medae": float("nan"), "mae": float("nan")}
    error = np.abs(pred_arr[sel] - true_arr[sel])
    return {
        "n": float(np.count_nonzero(sel)),
        "medae": float(np.median(error)),
        "mae": float(np.mean(error)),
    }

_PILES: Tuple[str, ...] = ("obstacle", "peer", "same_cell")

_DUMP_PROFILE_CAP = 400

def _dump_per_burst(
    directory: Path,
    arch: str,
    snr: int,
    tau_true: np.ndarray,
    obstacle_alpha: np.ndarray,
    peak_lag: np.ndarray,
    peak_amp: np.ndarray,
    head: np.ndarray,
    pile: np.ndarray,
    flags: Dict[float, np.ndarray],
    profiles: np.ndarray,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "snr_db": float(snr),
        "index": np.arange(tau_true.size, dtype=np.int64),
        "tau_true": tau_true,
        "obstacle_alpha": obstacle_alpha,
        "peak_lag": peak_lag,
        "peak_amplitude": peak_amp,
        "head_tau": head,
        "pile": pile,
    })
    for gamma in sorted(flags):
        frame["silent_g%s" % ("%g" % gamma).replace(".", "p")] = flags[gamma]
    path = directory / ("per_burst_%s.csv" % arch)
    if path.is_file():
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)
    np.savez_compressed(
        directory / ("per_burst_profiles_%s_%d.npz" % (arch, int(snr))),
        profiles=profiles.astype(np.complex64),
        tau_true=tau_true[: profiles.shape[0]],
        peak_lag=peak_lag[: profiles.shape[0]],
    )

def run(model: tf.keras.Model, config: Dict[str, Any], arch: str,
        dump_dir: Optional[Path] = None) -> pd.DataFrame:
    require_monostatic(config)
    settings = _settings(config)
    tau_max = float(config["data"]["max_delay"])
    echoes = [int(k) for k in config["data"]["echoes"]]
    feature_mode = str(config["data"].get("feature_mode", "iq"))
    feature_norm = str(config["data"].get("feature_norm", "none"))
    rng = np.random.default_rng(settings["seed"])
    rows: List[Dict[str, Any]] = []
    for snr in settings["snr_db"]:
        batch = generate_test_batch(config, settings["n_symbols"], snr, echoes[-1], rng)
        features = _build_feature_matrix(
            batch["x"], feature_mode, feature_norm, batch["x_ref"]
        )
        profile_complex = _matched_filter_profile(
            batch["x"], batch["x_ref"], int(tau_max)
        )
        profile = np.abs(profile_complex)
        peak_bin = np.argmax(profile, axis=1)
        peak_lag = (peak_bin + 1).astype(np.float64)
        peak_amp = profile[np.arange(profile.shape[0]), peak_bin]
        tau_true = np.asarray(batch["tau"], dtype=np.float64)
        obstacle_alpha = np.asarray(batch["obstacle_alpha"], dtype=np.float64)
        head = _head_delay(model, features, tau_max)
        pile = _pile(
            obstacle_alpha,
            batch.get("is_peer"),
            batch.get("peer_alphas"),
            batch.get("peer_taus"),
            tau_true,
        )
        meters = float(samples_to_meters(np.array([1.0]), config)[0])
        flags: Dict[float, np.ndarray] = {}
        for gamma in settings["gamma"]:
            mask = abstention_mask(
                peak_lag, peak_amp, obstacle_alpha, int(tau_max), gamma
            )
            flags[float(gamma)] = mask
            mae_peak = scored_mae(peak_lag, tau_true, mask)
            mae_head = scored_mae(head, tau_true, mask)
            medae_peak = scored_median_ae(peak_lag, tau_true, mask)
            medae_head = scored_median_ae(head, tau_true, mask)
            row: Dict[str, Any] = {
                "arch": arch,
                "snr_db": float(snr),
                "gamma": float(gamma),
                "n_symbols": int(tau_true.size),
                "p_oracle": oracle_hit_rate(peak_lag, tau_true),
                "mae_peak_samples": mae_peak,
                "mae_head_samples": mae_head,
                "mae_peak_meters": mae_peak * meters,
                "mae_head_meters": mae_head * meters,
                "medae_peak_samples": medae_peak,
                "medae_head_samples": medae_head,
                "medae_peak_meters": medae_peak * meters,
                "medae_head_meters": medae_head * meters,
                "gap_samples": mae_peak - mae_head,
                "gap_median_samples": medae_peak - medae_head,
                "gap_median_meters": (medae_peak - medae_head) * meters,
                "silence_fraction": silence_fraction(mask),
                "pile_obstacle": float(np.mean(pile == "obstacle")),
                "pile_peer": float(np.mean(pile == "peer")),
                "pile_same_cell": float(np.mean(pile == "same_cell")),
            }
            for name in _PILES:
                keep = (pile == name) & (~mask)
                stats_peak = _subset_metrics(peak_lag, tau_true, keep)
                stats_head = _subset_metrics(head, tau_true, keep)
                row[f"pile3_{name}_n"] = int(stats_peak["n"])
                row[f"pile3_{name}_medae_peak_samples"] = stats_peak["medae"]
                row[f"pile3_{name}_medae_head_samples"] = stats_head["medae"]
                row[f"pile3_{name}_mae_peak_samples"] = stats_peak["mae"]
                row[f"pile3_{name}_mae_head_samples"] = stats_head["mae"]
                row[f"pile3_{name}_medae_peak_meters"] = stats_peak["medae"] * meters
                row[f"pile3_{name}_medae_head_meters"] = stats_head["medae"] * meters
                row[f"pile3_{name}_silence_fraction"] = float(
                    np.mean(pile == name) - np.mean(keep)
                )
            rows.append(row)
        if dump_dir is not None:
            _dump_per_burst(
                Path(dump_dir), arch, int(snr), tau_true, obstacle_alpha, peak_lag,
                peak_amp, head, pile, flags, profile_complex[:_DUMP_PROFILE_CAP],
            )
    logger.info("ranging_benchmark %s: %d rows", arch, len(rows))
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
        Path(args.output_dir) if args.output_dir
        else _RESULTS / args.scenario / "ranging"
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
        frame = run(load_model(checkpoint), config, arch, dump_dir=out_dir)
        frame.to_csv(out_dir / f"{arch}.csv", index=False)

if __name__ == "__main__":
    main()

