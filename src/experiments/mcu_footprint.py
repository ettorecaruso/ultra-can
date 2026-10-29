from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "full" / "tables" / "mcu_footprint.csv"

VOLTAGE = 3.3
CURRENT_MA = (40.0, 60.0)
SYMBOL_AIR_US = 10.0
VELOCITY_WINDOW = 8
PC_LATENCY_MS = {"conv1d": 0.1475, "qkv": 0.5443}

_MODEL = re.compile(r"^MODEL (\S+) .*arena_used=(\d+)")
_START = re.compile(r"^=== (\S+) : (\d+) bytes flatbuffer")
_OP = re.compile(r"^OP (\S+)\s+calls=(\d+) total_us=(\d+)")
_RUN = re.compile(r"^RUN (\S+) i=(\d+) us=(\d+)")
_BENCH = re.compile(
    r"^BENCH (\S+) .*min_us=(\d+) median_us=(\d+) p95_us=(\d+) "
    r"max_us=(\d+) mean_us=([\d.]+).*arena_used=(\d+)")
_ARCH = {"qkv-int8": "qkv", "conv1d-int8": "conv1d"}


def _empty() -> Dict:
    return {"bytes": None, "arena": None, "ops": {}, "runs": [], "bench": None}


def parse_logs(paths: List[Path]) -> Dict[str, Dict]:
    labels: Dict[str, Dict] = {}
    current: Optional[str] = None
    for path in paths:
        if not path.is_file():
            continue
        for raw in Path(path).read_text(errors="replace").splitlines():
            line = raw.strip()
            match = _START.match(line)
            if match:
                current = match.group(1)
                labels.setdefault(current, _empty())["bytes"] = int(match.group(2))
                continue
            match = _MODEL.match(line)
            if match:
                current = match.group(1)
                labels.setdefault(current, _empty())["arena"] = int(match.group(2))
                continue
            match = _OP.match(line)
            if match and current is not None:
                labels.setdefault(current, _empty())["ops"][match.group(1)] = (
                    int(match.group(2)), int(match.group(3)))
                continue
            match = _RUN.match(line)
            if match:
                labels.setdefault(match.group(1), _empty())["runs"].append(
                    int(match.group(3)))
                continue
            match = _BENCH.match(line)
            if match:
                labels.setdefault(match.group(1), _empty())["bench"] = {
                    "min": int(match.group(2)), "median": int(match.group(3)),
                    "p95": int(match.group(4)), "max": int(match.group(5)),
                    "mean": float(match.group(6)), "arena": int(match.group(7)),
                }
    return labels


def _stats(entry: Dict) -> Optional[Dict]:
    bench = entry["bench"]
    if bench:
        return {"n": len(entry["runs"]), **bench}
    runs = entry["runs"]
    if not runs:
        return None
    ordered = sorted(runs)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {"n": len(runs), "min": ordered[0], "median": statistics.median(ordered),
            "p95": p95, "max": ordered[-1], "mean": statistics.fmean(ordered),
            "arena": entry["arena"]}


def _share(ops: Dict[str, tuple], names: tuple) -> float:
    total = sum(value[1] for value in ops.values())
    if not total:
        return 0.0
    return 100.0 * sum(ops[name][1] for name in names if name in ops) / total


def build_rows(paths: List[Path]) -> List[Dict]:
    rows = []
    for label, entry in sorted(parse_logs(paths).items()):
        stats = _stats(entry)
        if stats is None:
            continue
        arch = _ARCH.get(label, label)
        median_ms = stats["median"] / 1000.0
        energy = [VOLTAGE * current * median_ms / 1000.0 for current in CURRENT_MA]
        laptop = PC_LATENCY_MS.get(arch)
        rows.append({
            "model": label,
            "arch": arch,
            "int8_kib": round((entry["bytes"] or 0) / 1024.0, 1),
            "arena_kib": round(stats["arena"] / 1024.0, 1),
            "runs": stats["n"],
            "min_us": round(float(stats["min"]), 1),
            "median_us": round(float(stats["median"]), 1),
            "p95_us": round(float(stats["p95"]), 1),
            "max_us": round(float(stats["max"]), 1),
            "mean_us": round(float(stats["mean"]), 1),
            "bursts_per_s": round(1e6 / stats["median"], 2),
            "dwell_bursts": int(round(stats["median"] / SYMBOL_AIR_US)),
            "payload_bit_s": round(1e6 / stats["median"], 2),
            "velocity_hz": round(1e3 / (VELOCITY_WINDOW * median_ms), 2),
            "energy_mj_low": round(energy[0], 3),
            "energy_mj_high": round(energy[1], 3),
            "mcu_over_pc": round(median_ms / laptop, 1) if laptop else None,
            "conv_share_pct": round(_share(entry["ops"], ("CONV_2D",)), 1),
            "attention_share_pct": round(
                _share(entry["ops"], ("BATCH_MATMUL", "MUL")), 1),
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs", type=Path, nargs="+",
        default=[REPO_ROOT / "results" / "full" / "mcu_bench" / "bench.log"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    rows = build_rows(args.logs)
    if not rows:
        raise SystemExit("no BENCH/RUN lines found in the given logs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['model']:12s} arena={row['arena_kib']:6.1f} KiB  "
              f"median={row['median_us']:9.1f} us  {row['bursts_per_s']:6.2f} burst/s  "
              f"payload={row['payload_bit_s']:6.2f} bit/s  "
              f"E={row['energy_mj_low']:.2f}-{row['energy_mj_high']:.2f} mJ")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

