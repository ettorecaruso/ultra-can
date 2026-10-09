
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.data_loader import _build_feature_matrix
from src.data.dataset_generator import generate_test_batch, peer_echo_count
from src.data.scene import require_monostatic
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS = _REPO_ROOT / "results" / "full"
_TOL_SAMPLES = 0.5

def _peer_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    
    peers_cfg = config.get("peers")
    if not isinstance(peers_cfg, dict):
        raise ValueError("'peers' section missing or not a dict in config")
    label_mode = str(peers_cfg.get("label_mode", "exclude_with_fallback"))
    if label_mode not in ("exclude", "exclude_with_fallback"):
        raise ValueError(
            "peers.label_mode must be 'exclude' or 'exclude_with_fallback', "
            f"got: {label_mode!r}"
        )
    guard = float(peers_cfg.get("guard_factor", 1.5))
    if not np.isfinite(guard) or guard < 1.0:
        raise ValueError(f"peers.guard_factor must be finite and >= 1, got: {guard!r}")
    return {
        "enable": bool(peers_cfg.get("enable", False)),
        "n_peers": peer_echo_count(config),
        "label_mode": label_mode,
        "guard_factor": guard,
        "alpha_ref": float(peers_cfg.get("peer_alpha_ref", 0.25)),
        "alpha_max": float(peers_cfg.get("peer_alpha_max", 0.6)),
        "range_ref": float(peers_cfg.get("peer_range_ref_m", 30.0)),
    }

def _expected_peer_alpha(
    distances_m: np.ndarray, settings: Dict[str, Any]
) -> np.ndarray:
    
    alpha = settings["alpha_ref"] * (
        settings["range_ref"] / np.maximum(distances_m, 1e-9)
    ) ** 2
    return np.minimum(alpha, settings["alpha_max"])

def _matched_filter_profile(
    y_complex: np.ndarray,
    x_ref: np.ndarray,
    max_delay: int,
) -> np.ndarray:
    
    reference = np.asarray(x_ref, dtype=np.float64)
    received = np.asarray(y_complex)
    length = int(reference.shape[1])
    energy = np.sum(reference * reference, axis=1)
    energy = np.where(energy < 1e-12, 1.0, energy)
    direct = np.sum(reference * received, axis=1) / energy
    residual = received - direct[:, None] * reference
    profile = np.stack(
        [
            np.abs(
                np.sum(reference[:, : length - lag] * residual[:, lag:], axis=1)
            )
            for lag in range(1, int(max_delay) + 1)
        ],
        axis=1,
    )
    return profile / energy[:, None]

def _head_delay(
    model: tf.keras.Model,
    features: np.ndarray,
    tau_max: float,
    predict_batch: int = 1024,
) -> np.ndarray:
    
    parts = []
    for start in range(0, features.shape[0], int(predict_batch)):
        chunk = features[start : start + int(predict_batch)]
        predictions = model(tf.convert_to_tensor(chunk), training=False)
        parts.append(predictions["sensing"].numpy()[:, 0].astype(np.float64))
    return np.concatenate(parts) * float(tau_max)

def _sample_table(
    model: tf.keras.Model,
    config: Dict[str, Any],
    settings: Dict[str, Any],
    arch: str,
) -> pd.DataFrame:
    require_monostatic(config)
    exp_cfg = (config.get("experiments") or {}).get("peer_estimation") or {}
    n_symbols = int(exp_cfg.get("n_symbols", 4000))
    snr_db = float(exp_cfg.get("snr_db", 21.0))
    k_echoes = int(exp_cfg.get("echoes", 3))
    seed = int(exp_cfg.get("seed", 20260926))
    max_delay = int(config["data"]["max_delay"])
    rng = np.random.default_rng(seed)
    batch = generate_test_batch(config, n_symbols, snr_db, k_echoes, rng)
    if "peer_taus" not in batch:
        raise RuntimeError("the peer arrays are missing from the batch")
    peer_taus = np.asarray(batch["peer_taus"], dtype=np.float64)
    if peer_taus.shape[1] == 0:
        raise RuntimeError("no peers in the batch: enable peers before evaluating")
    peer_distances = np.asarray(batch["peer_distances_m"], dtype=np.float64)
    tau_true = np.asarray(batch["tau"], dtype=np.float64)
    profile = _matched_filter_profile(batch["x"], batch["x_ref"], max_delay)
    features = _build_feature_matrix(
        batch["x"],
        str(config["data"].get("feature_mode", "iq")),
        config["data"].get("feature_norm"),
        reference=batch["x_ref"],
    )
    tau_hat = _head_delay(model, features, float(max_delay))

    rows = np.arange(int(n_symbols))
    lags = np.clip(np.rint(peer_taus).astype(np.int64), 1, max_delay)
    nearest = np.argmin(peer_distances, axis=1)
    predicted = _expected_peer_alpha(peer_distances, settings)
    alpha_nearest = predicted[rows, nearest]
    target_lags = np.clip(np.rint(tau_true).astype(np.int64), 1, max_delay)
    ratio = profile[rows, target_lags - 1] / np.maximum(alpha_nearest, 1e-12)

    guard = float(settings["guard_factor"])
    guard_values = [
        float(value)
        for value in exp_cfg.get("guard_factors", [guard])
    ]
    if guard not in guard_values:
        guard_values.append(guard)
    guard_values = sorted(set(guard_values))
    target_hit = np.abs(tau_hat - tau_true) <= _TOL_SAMPLES
    peer_hit = np.zeros(int(n_symbols), dtype=bool)
    for column in range(peer_taus.shape[1]):
        peer_hit |= np.abs(tau_hat - peer_taus[rows, column]) <= _TOL_SAMPLES
    corange = np.abs(tau_true - peer_taus[rows, nearest]) <= _TOL_SAMPLES

    frames = []
    for guard_value in guard_values:
        fallback = np.zeros(int(n_symbols), dtype=bool)
        for column in range(peer_taus.shape[1]):
            measured = profile[rows, lags[rows, column] - 1]
            fallback |= measured > guard_value * predicted[rows, column]
        frames.append(
            pd.DataFrame(
                {
                    "arch": str(arch),
                    "offset": tau_true - peer_taus[rows, nearest],
                    "ratio": ratio,
                    "n_peers": int(peer_taus.shape[1]),
                    "corange": corange,
                    "target_hit": target_hit,
                    "peer_false": peer_hit & ~target_hit,
                    "fallback": fallback,
                    "abs_err": np.abs(tau_hat - tau_true),
                    "label_mode": settings["label_mode"],
                    "guard_factor": guard_value,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)

def _aggregate(
    frame: pd.DataFrame,
    summary: str,
    labels: Optional[pd.Series] = None,
) -> pd.DataFrame:
    
    working = frame if labels is None else frame.assign(key_bin=labels.astype(str))
    if labels is None:
        working = working.assign(key_bin="all")
    out = working.groupby(["arch", "key_bin", "guard_factor"], dropna=False).agg(
        n=("target_hit", "size"),
        target_id_rate=("target_hit", "mean"),
        peer_false_rate=("peer_false", "mean"),
        fallback_rate=("fallback", "mean"),
        mean_abs_err=("abs_err", "mean"),
    ).reset_index()
    out.insert(0, "summary", summary)
    return out

def _summarise(frame: pd.DataFrame) -> pd.DataFrame:
    
    parts = [_aggregate(frame, "all")]
    offset_bin = frame["offset"].round().clip(-4, 4).astype(int).astype(str)
    parts.append(_aggregate(frame, "offset_samples", offset_bin))
    ratio_bin = pd.cut(
        frame["ratio"],
        bins=[0.0, 0.75, 1.25, 2.0, np.inf],
        labels=["peer_stronger", "comparable", "target_gt1.25", "target_gt2"],
        right=False,
    ).astype(str)
    parts.append(_aggregate(frame, "amplitude_ratio", ratio_bin))
    parts.append(
        _aggregate(
            frame,
            "co_range",
            frame["corange"].map({True: "co_range", False: "separated"}),
        )
    )
    return pd.concat(parts, ignore_index=True)

def evaluate_peer_estimation(
    model: tf.keras.Model,
    config: Dict[str, Any],
    output_dir: Path,
    arch: str,
) -> pd.DataFrame:
    
    settings = _peer_settings(config)
    if not settings["enable"] or settings["n_peers"] <= 0:
        raise ValueError(
            "peers.enable is false or peers.n_peers is zero: the peer experiment "
            "needs the cooperative echoes"
        )
    samples = _sample_table(model, config, settings, arch)
    summary = _summarise(samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples.drop(columns=["arch"]).to_csv(output_dir / "peer_samples.csv", index=False)
    summary.to_csv(output_dir / "peer_estimation.csv", index=False)
    logger.info(
        "peer_estimation %s: %d samples, target_id=%.4f, peer_false=%.4f -> %s",
        arch,
        int(samples.shape[0]),
        float(samples["target_hit"].mean()),
        float(samples["peer_false"].mean()),
        output_dir,
    )
    return summary

def main(argv: Optional[Sequence[str]] = None) -> None:
    
    from src.utils.model_io import load_model

    parser = argparse.ArgumentParser(
        description="Peer-aware sensing: label exclusion and conservative fallback"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_BASE_CONFIG_PATH)
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--model", default="conv1d")
    parser.add_argument(
        "--scenario", default="iod_peers",
        help="ber_vs_snr scenario holding the peer-trained receivers",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_RESULTS / "peer_estimation"
    )
    args = parser.parse_args(argv)

    config = load_config(
        config_path=args.config, base_config_path=DEFAULT_BASE_CONFIG_PATH
    )
    checkpoint = (
        _RESULTS / "ber_vs_snr" / args.scenario / args.model / "best_model.keras"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"peer-trained checkpoint not found: {checkpoint}")
    model = load_model(checkpoint)
    evaluate_peer_estimation(
        model=model,
        config=config,
        output_dir=args.output_dir / args.model,
        arch=args.model,
    )

if __name__ == "__main__":
    main()

