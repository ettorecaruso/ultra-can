from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FULL = REPO / "results" / "full"
OUT = FULL / "tables"

def _write(frame: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    frame.to_csv(path, index=False)
    print("saved", path.relative_to(REPO), "|", len(frame), "rows")

def scene() -> None:
    src = FULL / "diagnostics" / "scene" / "geometry.csv"
    if src.is_file():
        frame = pd.read_csv(src)
        frame["range_m"] = frame["distance_m"]
        _write(frame, "scene.csv")
    summary = FULL / "diagnostics" / "scene" / "summary.csv"
    if summary.is_file():
        _write(pd.read_csv(summary), "scene_summary.csv")

def piles() -> None:
    rows = []
    for scenario in ("k3_doppler_full", "iod_peers"):
      for arch in ("conv1d", "qkv"):
        path = FULL / scenario / "ranging_benchmark" / ("%s.csv" % arch)
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        for pile in ("obstacle", "peer", "same_cell"):
            for row in frame.itertuples():
                rows.append({
                    "scenario": scenario,
                    "arch": arch,
                    "gamma": float(row.gamma),
                    "pile": pile,
                    "n_answered": int(getattr(row, "pile3_%s_n" % pile)),
                    "median_peak_samples": float(getattr(row, "pile3_%s_medae_peak_samples" % pile)),
                    "median_head_samples": float(getattr(row, "pile3_%s_medae_head_samples" % pile)),
                    "median_peak_m": float(getattr(row, "pile3_%s_medae_peak_meters" % pile)),
                    "median_head_m": float(getattr(row, "pile3_%s_medae_head_meters" % pile)),
                })
    if rows:
        _write(pd.DataFrame(rows), "ranging_piles.csv")

def ambiguity() -> None:
    rows = []
    for arch in ("conv1d", "qkv"):
        path = FULL / "k3_ambiguity_mix" / "ambiguity" / ("%s.csv" % arch)
        if path.is_file():
            rows.append(pd.read_csv(path))
    if rows:
        _write(pd.concat(rows, ignore_index=True), "ambiguity_mixed_sky.csv")

def velocity() -> None:
    path = FULL / "k3_doppler_full" / "velocity_track" / "velocity_track.csv"
    if path.is_file():
        _write(pd.read_csv(path), "velocity_track.csv")

def blindness() -> None:
    rows = []
    for arch in ("conv1d", "qkv"):
        path = FULL / "k3_doppler_full" / "blind_vs_oracle" / ("%s.csv" % arch)
        if path.is_file():
            rows.append(pd.read_csv(path))
    if rows:
        _write(pd.concat(rows, ignore_index=True), "price_of_blindness.csv")

def jamming_tables() -> None:
    rows = []
    for arch in ("conv1d", "qkv", "lstm", "mc_dlsk"):
        base = FULL / "jamming" / arch
        for path in sorted(base.glob("jamming/jamming_results_*.csv")):
            frame = pd.read_csv(path)
            frame["arch"] = arch
            frame["jammer"] = path.stem.replace("jamming_results_", "")
            rows.append(frame)
    if rows:
        _write(pd.concat(rows, ignore_index=True), "jamming_summary.csv")

def hopping_tables() -> None:
    rows = []
    for arch in ("conv1d", "qkv", "lstm", "mc_dlsk"):
        path = FULL / "frequency_agility" / arch / "frequency_agility_vs_jsr.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            frame["arch"] = arch
            rows.append(frame)
    if rows:
        _write(pd.concat(rows, ignore_index=True), "hopping_vs_jsr.csv")

def lpi_table() -> None:
    path = FULL / "diagnostics" / "lpi_detectability" / "lpi_roc.csv"
    if path.is_file():
        frame = pd.read_csv(path)
        _write(frame[frame["snr_db"].isin([0.0, 10.0, 20.0])], "lpi_roc_key_points.csv")

def swarap() -> None:
    rows = []
    for arch in ("conv1d", "qkv"):
        path = FULL / "k3_doppler_full" / "board_footprint" / ("%s.csv" % arch)
        if path.is_file():
            rows.append(pd.read_csv(path))
    if rows:
        _write(pd.concat(rows, ignore_index=True), "onboard_footprint.csv")

def blind_cheat() -> None:
    rows = []
    for arch in ("conv1d", "qkv"):
        path = FULL / "k3_doppler_full" / "blind_cheat_probe" / ("%s.csv" % arch)
        if path.is_file():
            rows.append(pd.read_csv(path))
    if rows:
        _write(pd.concat(rows, ignore_index=True), "blind_cheat_probe.csv")

def observability() -> None:
    base = FULL / "diagnostics" / "channel_label_observability"
    oracle = base / "oracle.csv"
    if not oracle.is_file():
        return
    frame = pd.read_csv(oracle)[["channel_variant", "p_label_is_strongest"]]
    hit = base / "peak_hit.csv"
    if hit.is_file():
        frame = frame.merge(
            pd.read_csv(hit)[["channel_variant", "p_peak_hit"]],
            on="channel_variant", how="left",
        )
    for arch in ("conv1d", "qkv"):
        path = FULL / "channel_generalization" / arch / "summary.csv"
        if path.is_file():
            summary = pd.read_csv(path)[["channel_variant", "corr_tau_top"]]
            summary = summary.rename(columns={"corr_tau_top": "corr_tau_%s" % arch})
            frame = frame.merge(summary, on="channel_variant", how="left")
    _write(frame, "observability.csv")

def latency_us() -> None:
    path = FULL / "final_report" / "latency_results.json"
    if not path.is_file():
        return
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    for receiver, blocks in payload.items():
        batch1 = blocks.get("batch1", {})
        batch32 = blocks.get("batch32_amortized", {})
        rows.append({
            "receiver": receiver,
            "batch1_median_us": batch1.get("median_us"),
            "batch1_mean_us": batch1.get("mean_us"),
            "batch1_std_us": batch1.get("stdev_us"),
            "batch1_min_us": batch1.get("min_us"),
            "batch1_max_us": batch1.get("max_us"),
            "batch32_median_us": batch32.get("median_us"),
        })
    _write(pd.DataFrame(rows), "latency_us.csv")

def main() -> None:
    scene()
    piles()
    ambiguity()
    velocity()
    blindness()
    blind_cheat()
    jamming_tables()
    hopping_tables()
    lpi_table()
    swarap()
    observability()
    latency_us()

if __name__ == "__main__":
    main()
