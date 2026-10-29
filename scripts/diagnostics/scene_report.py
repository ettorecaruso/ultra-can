from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.dataset_generator import _peer_config, peer_echo_count
from src.data.scene import (
    amplitude_from_distance,
    delay_samples_from_distance,
    doppler_from_radial_speed,
    sample_scene_taps,
)
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config

N_SYMBOLS = 4096
SEED = 20261008
RANGE_POINTS = 9

def _ranges(config: dict) -> np.ndarray:
    scene = config["channel"]["scene"]
    lo, hi = scene["obstacle_distance_m"]
    return np.linspace(float(lo), float(hi), RANGE_POINTS)

def _per_range_table(config: dict) -> pd.DataFrame:
    scene = config["channel"]["scene"]
    distances = _ranges(config)
    speed = 0.5 * sum(float(v) for v in scene["radial_speed_mps"])
    obstacle = np.asarray(
        [amplitude_from_distance(np.array([d]), scene["obstacle_reflectivity"],
                                 scene["reference_range_m"])[0] for d in distances]
    )
    scatterer = np.asarray(
        [amplitude_from_distance(np.array([d]), scene["scatterer_reflectivity"],
                                 scene["reference_range_m"])[0] for d in distances]
    )
    rows = []
    for index, distance in enumerate(distances):
        tau = float(np.rint(delay_samples_from_distance(np.array([distance]), config)[0]))
        doppler = float(doppler_from_radial_speed(np.array([speed]), config)[0])
        rows.append({
            "distance_m": float(distance),
            "delay_samples": tau,
            "delay_microseconds": tau / float(config["data"]["fs_hz"]) * 1e6,
            "obstacle_amplitude": float(obstacle[index]),
            "scatterer_amplitude": float(scatterer[index]),
            "doppler_hz": doppler,
        })
    return pd.DataFrame(rows)

def _peer_table(config: dict) -> pd.DataFrame:
    peers = _peer_config(config)
    if peer_echo_count(config) == 0:
        return pd.DataFrame(columns=["distance_m", "peer_echo_amplitude",
                                     "peer_transmission_amplitude", "power_ratio"])
    lo, hi = peers["peer_range_m"]
    alpha_ref = float(peers["peer_alpha_ref"])
    alpha_max = float(peers["peer_alpha_max"])
    range_ref = float(peers["peer_range_ref_m"])
    gain_ref = float(peers.get("direct_gain_ref", 1.0))
    distances = np.linspace(float(lo), float(hi), RANGE_POINTS)
    rows = []
    for distance in distances:
        echo = min(alpha_ref * (range_ref / float(distance)) ** 2, alpha_max)
        transmission = gain_ref * (range_ref / float(distance))
        rows.append({
            "distance_m": float(distance),
            "peer_echo_amplitude": float(echo),
            "peer_transmission_amplitude": float(transmission),
            "power_ratio": float(transmission / max(echo, 1e-12)),
        })
    return pd.DataFrame(rows)

def _summary(config: dict) -> pd.DataFrame:
    taps = sample_scene_taps(
        N_SYMBOLS, np.random.default_rng(SEED), config,
        int(config["data"]["max_delay"]), max(0, 3 - 1),
    )
    obstacle = taps["alphas"][:, 0]
    strongest = np.argmax(np.abs(taps["alphas"]), axis=1)
    return pd.DataFrame([{
        "n_symbols": N_SYMBOLS,
        "k_geometric": int(taps["k_geometric"]),
        "n_taps": int(taps["taus"].shape[1]),
        "obstacle_amplitude_mean": float(np.mean(obstacle)),
        "obstacle_amplitude_std": float(np.std(obstacle)),
        "delay_samples_min": float(np.min(taps["taus"][:, 0])),
        "delay_samples_max": float(np.max(taps["taus"][:, 0])),
        "doppler_hz_min": float(np.min(taps["dopplers"]) * float(config["data"]["fs_hz"])),
        "doppler_hz_max": float(np.max(taps["dopplers"]) * float(config["data"]["fs_hz"])),
        "obstacle_is_strongest_rate": float(np.mean(strongest == 0)),
    }])

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_BASE_CONFIG_PATH)
    parser.add_argument("--out-root", type=Path, default=_REPO / "results" / "full")
    args = parser.parse_args(argv)

    config = load_config(args.config, base_config_path=DEFAULT_BASE_CONFIG_PATH)
    if str(config["channel"].get("generator")) != "geometry":
        raise ValueError("the scene report needs channel.generator='geometry'")
    out_dir = Path(args.out_root) / "diagnostics" / "scene"
    out_dir.mkdir(parents=True, exist_ok=True)
    geometry = _per_range_table(config)
    summary = _summary(config)
    geometry.to_csv(out_dir / "geometry.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    peers = _peer_table(config)
    if not peers.empty:
        peers.to_csv(out_dir / "peers.csv", index=False)
    print(f"scene report written to {out_dir}")

if __name__ == "__main__":
    main()
