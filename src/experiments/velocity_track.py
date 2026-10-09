
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset_generator import generate_transmitted_batch
from src.data.scene import require_monostatic
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config
from src.utils.logger import get_logger, log_config_summary, setup_logging

logger = get_logger(__name__)

_RESULTS = _REPO_ROOT / "results"
_C_LIGHT_M_S = 299_792_458.0
_DEFAULT_BURSTS = 50
_DEFAULT_TRACKS = 30
_DEFAULT_SNR = 15.0
_DEFAULT_SPEED = (10.0, 30.0)
_DEFAULT_SEED = 20261008
_DEFAULT_BURST_INTERVAL_S = 0.1

def _settings(config: Dict[str, Any]) -> Dict[str, Any]:
    section = (config.get("experiments") or {}).get("velocity_track") or {}
    speed = list(section.get("radial_speed_mps", list(_DEFAULT_SPEED)))
    return {
        "n_bursts": int(section.get("n_bursts", _DEFAULT_BURSTS)),
        "n_tracks": int(section.get("n_tracks", _DEFAULT_TRACKS)),
        "snr_db": float(section.get("snr_db", _DEFAULT_SNR)),
        "radial_speed_mps": (float(speed[0]), float(speed[1])),
        "seed": int(section.get("seed", _DEFAULT_SEED)),
        "burst_interval_s": float(
            section.get("burst_interval_s", _DEFAULT_BURST_INTERVAL_S)
        ),
    }

def _shift(x: np.ndarray, lag: np.ndarray) -> np.ndarray:
    """Ritardo frazionario via rampa di fase (niente wrap: FFT su lunghezza doppia)."""
    n, length = x.shape
    padded = 2 * length
    freq = np.fft.rfftfreq(padded)
    out = np.empty((n, length), dtype=np.float64)
    for index in range(n):
        spectrum = np.fft.rfft(x[index], n=padded)
        phase = np.exp(-2j * np.pi * freq * float(lag[index]))
        out[index] = np.fft.irfft(spectrum * phase, n=padded)[:length]
    return out

def _delay_of_range(distance_m: float, config: Dict[str, Any]) -> float:
    fs_hz = float(config["data"]["fs_hz"])
    return 2.0 * distance_m * fs_hz / _C_LIGHT_M_S

def _samples_per_second(config: Dict[str, Any]) -> float:
    fs_hz = float(config["data"]["fs_hz"])
    return _C_LIGHT_M_S / (2.0 * fs_hz)

def _range_of_delay(delay_samples: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    return np.asarray(delay_samples, dtype=np.float64) * _samples_per_second(config)

def _estimate_delay(y: np.ndarray, x: np.ndarray, max_delay: int) -> np.ndarray:
    energy = np.sum(x * x, axis=1)
    energy = np.where(energy < 1e-12, 1.0, energy)
    residual = y - (np.sum(x * y, axis=1) / energy)[:, None] * x
    length = x.shape[1]
    profile = np.stack(
        [
            np.abs(np.sum(x[:, : length - lag] * residual[:, lag:], axis=1))
            for lag in range(1, max_delay + 1)
        ],
        axis=1,
    )
    peak = np.argmax(profile, axis=1)
    rows = np.arange(profile.shape[0])
    center = profile[rows, peak]
    left = profile[rows, np.clip(peak - 1, 0, max_delay - 1)]
    right = profile[rows, np.clip(peak + 1, 0, max_delay - 1)]
    curvature = left - 2.0 * center + right
    delta = np.where(curvature < -1e-12, 0.5 * (left - right) / curvature, 0.0)
    return (peak + 1).astype(np.float64) + np.clip(delta, -0.5, 0.5)

def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="valid")

def run(config: Dict[str, Any]) -> pd.DataFrame:
    require_monostatic(config)
    settings = _settings(config)
    tau_max = int(config["data"]["max_delay"])
    burst_interval = float(settings["burst_interval_s"])
    if not np.isfinite(burst_interval) or burst_interval <= 0.0:
        raise ValueError(
            f"velocity_track: burst_interval_s must be finite and > 0, "
            f"got: {burst_interval!r}"
        )
    samples_per_second = _samples_per_second(config)
    rng = np.random.default_rng(settings["seed"])
    rows: List[Dict[str, Any]] = []
    for window in (1, 2, 4, 8):
        errors: List[float] = []
        for _ in range(settings["n_tracks"]):
            speed = rng.uniform(*settings["radial_speed_mps"])
            start_range = rng.uniform(120.0, 220.0)
            delays = []
            for burst in range(settings["n_bursts"] + window):
                distance = start_range - speed * burst * burst_interval
                delays.append(_delay_of_range(distance, config))
            delays = np.asarray(delays)
            bits = rng.integers(0, 2, delays.size).astype(np.int64)
            seeds = rng.integers(1, 2**31 - 1, delays.size).astype(np.int64)
            x = generate_transmitted_batch(config, bits, seeds)
            y = _shift(x, delays)
            power = np.mean(np.abs(y) ** 2, axis=1, keepdims=True)
            noise_var = power * 10.0 ** (-settings["snr_db"] / 10.0)
            y = y + np.sqrt(noise_var / 2.0) * rng.standard_normal(y.shape)
            estimated = _estimate_delay(y, x, tau_max)
            smoothed = _moving_average(estimated, window)
            span = (smoothed.size - 1) * burst_interval
            approach = (smoothed[0] - smoothed[-1]) * samples_per_second
            velocity = approach / span
            errors.append(float(velocity - speed))
        rows.append({
            "window_bursts": int(window),
            "n_tracks": settings["n_tracks"],
            "n_bursts": settings["n_bursts"],
            "speed_error_median_mps": float(np.median(errors)),
            "speed_error_abs_median_mps": float(np.median(np.abs(errors))),
        })
    logger.info("velocity_track: %d rows", len(rows))
    return pd.DataFrame(rows)

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_BASE_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config, base_config_path=DEFAULT_BASE_CONFIG_PATH)
    out_dir = (
        Path(args.output_dir) if args.output_dir
        else _RESULTS / str(config["general"]["experiment_name"]) / "velocity_track"
    )
    setup_logging(
        log_dir=out_dir / "logs",
        level=str(config["general"].get("log_level", "INFO")),
        experiment_name=str(config["general"]["experiment_name"]),
    )
    log_config_summary(config, logger)
    out_dir.mkdir(parents=True, exist_ok=True)
    run(config).to_csv(out_dir / "velocity_track.csv", index=False)

if __name__ == "__main__":
    main()

