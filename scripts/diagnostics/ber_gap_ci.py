import csv
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "results" / "full" / "figures_data" / "ber_full_linear"
POOLED = REPO_ROOT / "results" / "full" / "ber_vs_snr" / "operating_region_table.csv"
OUTPUT = REPO_ROOT / "results" / "full" / "tables" / "ber_gap_ci.csv"

SCENARIOS = ("k1_doppler_full", "k3_doppler_full", "k3_doppler_limited")
ARCHS = ("conv1d", "qkv")
SNR_MIN_DB = 5.0
RESAMPLES = 20000
SEED = 0


def per_point_ber(scenario, arch):
    path = RAW / scenario / arch / "metrics.csv"
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            float(row["snr_db"]): float(row["ber"])
            for row in csv.DictReader(handle)
            if float(row["snr_db"]) >= SNR_MIN_DB
        }


def pooled_ber(scenario, arch):
    if not POOLED.is_file():
        return None
    with open(POOLED, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["scenario"] == scenario and row["arch"] == arch:
                return float(row["pooled_ber"])
    return None


def bootstrap_interval(values, resamples=RESAMPLES, seed=SEED):
    rng = random.Random(seed)
    size = len(values)
    means = sorted(
        sum(rng.choice(values) for _ in range(size)) / size for _ in range(resamples)
    )
    return means[int(0.025 * resamples)], means[int(0.975 * resamples)]


def gap_rows(scenario):
    low = per_point_ber(scenario, ARCHS[0])
    high = per_point_ber(scenario, ARCHS[1])
    points = sorted(set(low) & set(high))
    if len(points) < 2:
        return None
    gaps = [low[snr] - high[snr] for snr in points]
    mean = sum(gaps) / len(gaps)
    ci_low, ci_high = bootstrap_interval(gaps)
    ref_low = pooled_ber(scenario, ARCHS[0])
    ref_high = pooled_ber(scenario, ARCHS[1])
    pooled_gap = None if ref_low is None or ref_high is None else ref_low - ref_high
    return {
        "scenario": scenario,
        "arch_better": ARCHS[1],
        "arch_worse": ARCHS[0],
        "n_points": len(gaps),
        "gap_mean": "%.6e" % mean,
        "ci_low": "%.6e" % ci_low,
        "ci_high": "%.6e" % ci_high,
        "gap_min": "%.6e" % min(gaps),
        "gap_max": "%.6e" % max(gaps),
        "n_points_arch_better": sum(1 for gap in gaps if gap > 0),
        "pooled_gap": "" if pooled_gap is None else "%.6e" % pooled_gap,
        "resamples": RESAMPLES,
        "seed": SEED,
        "snr_min_db": SNR_MIN_DB,
    }


def main():
    rows = [row for row in (gap_rows(s) for s in SCENARIOS) if row is not None]
    if not rows:
        raise SystemExit("no per-point BER found under %s" % RAW.relative_to(REPO_ROOT))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("saved", OUTPUT.relative_to(REPO_ROOT))
    for row in rows:
        print(
            "  %-18s n=%2s mean=%s CI95=[%s, %s] max=%s (%s/%s points)"
            % (
                row["scenario"],
                row["n_points"],
                row["gap_mean"],
                row["ci_low"],
                row["ci_high"],
                row["gap_max"],
                row["n_points_arch_better"],
                row["n_points"],
            )
        )


if __name__ == "__main__":
    main()
