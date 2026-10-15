"""Baseline ISAC OFDM 5G-NR contro il radar caotico a zero parametri (CPU, numpy, ~2 s).

Perche' esiste
--------------
Il task delle metriche di sensing e' stimare il ritardo del tap **OSTACOLO** (indice 0 nella
scena), non dell'eco piu' forte: il clutter (riflettivita' 0.08) e i peer cooperativi possono
essere piu' forti. Qualsiasi processore che prende il picco (matched filter caotico *o* un
ISAC OFDM) e' quindi limitato al "tetto del peak-picker", che questo modulo misura.

Cosa produce (per scenario e per SNR)
-------------------------------------
* ``chaotic_mf``          : profilo DCA+MF del repo sui tap della scena (zero parametri);
* ``ofdm_<num>_<pattern>``: ISAC OFDM 5G-NR (Sturm/Wiesbeck): divisione per la griglia nota,
                            IFFT, rimozione della direttrice, argmax. Numerologie FR1 con
                            CP esteso (60 kHz, 14 PRB = 168 sottoportanti = 10.08 MHz) e
                            15 kHz (55 PRB = 660), pattern DMRS comb-1/2/4;
* ``peak_picker_ceiling`` : P(ostacolo = tap piu' forte tra i non-peer)  -> tetto teorico;
* ``peak_picker_ceiling_all_taps``: idem includendo i peer (tetto vero con peer);
* ``head_M1`` / ``head_geom``  : testa appresa, letta dai CSV della campagna (se presenti),
                            cosi' il confronto e' in un unico file.

Garanzie di equita': stessa scena, stessa banda, stessa energia per burst (il rumore del
lato OFDM e' riferito alla potenza reale della waveform caotica), stessa verita' (ostacolo).
Limiti dichiarati: sincronizzazione perfetta (niente CFO/timing offset), un simbolo OFDM
(niente integrazione su slot da 14 simboli ne' Doppler), direttrice rimossa analiticamente.

Uso:
  python src/experiments/ofdm_isac.py                       # 4 scenari, SNR di default
  python src/experiments/ofdm_isac.py --scenario k3_doppler_full --snr -5 21 --n-bursts 4000
  python src/experiments/ofdm_isac.py --out-dir results/ofdm_isac --metrics-root results/full
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset_generator import _sample_peer_taps, generate_transmitted_batch_fast  # noqa: E402
from src.data.scene import sample_scene_taps  # noqa: E402
from src.evaluation.metrics import argmax_delay_from_profile  # noqa: E402
from src.experiments.pipeline import resolve_scenario  # noqa: E402
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config  # noqa: E402

logger = logging.getLogger(__name__)

_TOL_SAMPLES = 0.5  # stessa tolleranza di exact_tau dell'evaluator (_EXACT_TOL)
_DEFAULT_SCENARIOS = (
    "k1_doppler_full", "k3_doppler_full", "k3_doppler_limited", "iod_peers",
)
_DEFAULT_SNR = (-5.0, -1.0, 3.0, 7.0, 11.0, 15.0, 21.0)
_DEFAULT_BURSTS = 4000
_DEFAULT_SEED = 20261014

# 5G-NR FR1: (SCS, PRB, CP). Il CP esteso (4.69 us) copre il delay spread della scena
# (33 campioni @10 MHz = 3.3 us); a 120 kHz il CP normale (0.59 us) non lo coprirebbe.
NUMEROLOGIES: Dict[str, Dict[str, float]] = {
    "ofdm_60k_extcp": {"scs": 60e3, "prb": 14, "cp": 4.69e-6, "n_fft": 1024},
    "ofdm_15k": {"scs": 15e3, "prb": 55, "cp": 4.69e-6, "n_fft": 4096},
}
DMRS_PATTERNS = (("all", 1), ("dmrs_comb2", 2), ("dmrs_comb4", 4))
_RICIAN_KAPPA_DB = 10.0


def sample_geometry(config: Dict[str, Any], n: int, rng: np.random.Generator, k: int):
    
    max_delay = int(config["data"]["max_delay"])
    scene = sample_scene_taps(n, rng, config, max_delay, max(0, k - 1))
    taus = np.asarray(scene["taus"], dtype=np.float64)
    alphas = np.asarray(scene["alphas"], dtype=np.float64)
    dopplers = np.asarray(scene["dopplers"], dtype=np.float64)
    is_peer = np.zeros(alphas.shape, dtype=bool)
    peers_cfg = config.get("peers") or {}
    if bool(peers_cfg.get("enable", False)):
        _dist, peer_taus, peer_alphas, peer_dopplers = _sample_peer_taps(
            config, rng, n, max_delay
        )
        taus = np.concatenate([taus, np.asarray(peer_taus, dtype=np.float64)], axis=1)
        alphas = np.concatenate([alphas, np.asarray(peer_alphas, dtype=np.float64)], axis=1)
        dopplers = np.concatenate(
            [dopplers, np.asarray(peer_dopplers, dtype=np.float64)], axis=1
        )
        is_peer = np.concatenate(
            [is_peer, np.ones(np.shape(peer_taus), dtype=bool)], axis=1
        )
        echo_power_fraction = float(scene["echo_power_fraction"])
        total = np.sqrt(np.sum(alphas ** 2, axis=1, keepdims=True))
        alphas = np.sqrt(echo_power_fraction) * alphas / np.maximum(total, 1e-12)
    return taus, alphas, dopplers, is_peer, max_delay


def _label_lag(taus: np.ndarray, max_delay: int) -> np.ndarray:
    """Verita' del task: ritardo del tap OSTACOLO (indice 0), come il generatore reale."""
    return np.clip(np.rint(taus[:, 0]).astype(np.int64), 1, max_delay)


def _chaotic_received(x, taus, alphas, dopplers, config, noise_var, rng):
    """Ricevuto caotico (stesso modello del generatore): direttrice Rician + echi/peer + AWGN."""
    n, seq_len = x.shape
    kappa = 10.0 ** (_RICIAN_KAPPA_DB / 10.0)
    echo_power = np.sum(alphas ** 2, axis=1)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    g = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
    h_c = np.sqrt(kappa / (kappa + 1.0)) * np.exp(1j * theta) + np.sqrt(
        1.0 / (kappa + 1.0)
    ) * g
    h_c_eff = h_c * np.sqrt(np.maximum(1.0 - echo_power, 0.0))
    f_dc = rng.uniform(0.0, float(config["data"]["doppler_direct_max"]), n)
    t = np.arange(seq_len, dtype=np.float64)[None, :]
    y = h_c_eff[:, None] * x * np.exp(1j * 2.0 * np.pi * f_dc[:, None] * t)
    rows_idx = np.arange(n, dtype=np.int64)[:, None]
    cols = np.arange(seq_len, dtype=np.int64)[None, :]
    for j in range(taus.shape[1]):
        lag = np.rint(taus[:, j]).astype(np.int64)[:, None]
        idx = cols - lag
        shifted = np.where(
            idx >= 0, x[rows_idx, np.clip(idx, 0, seq_len - 1)], 0.0
        )
        y += alphas[:, j, None] * shifted * np.exp(
            1j * 2.0 * np.pi * dopplers[:, j, None] * t
        )
    y += np.sqrt(noise_var / 2.0) * (
        rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    )
    return y


def _ofdm_lag(taus, alphas, num, fs, noise_var, rng, max_delay, comb):
    """OFDM radar classico: H=Y/X sulla griglia nota -> IFFT -> argmax (tau in campioni)."""
    n_sc = 12 * int(num["prb"])
    n_fft = int(num["n_fft"])
    k = np.arange(n_sc, dtype=np.float64)
    out = np.zeros(taus.shape[0], dtype=np.int64)
    for i in range(taus.shape[0]):
        h = np.exp(
            -1j * 2.0 * np.pi * np.outer(k, num["scs"] * taus[i] / fs)
        ) @ alphas[i]
        h = h + np.sqrt(noise_var / 2.0) * (
            rng.standard_normal(n_sc) + 1j * rng.standard_normal(n_sc)
        )
        if comb > 1:
            known = np.zeros(n_sc, dtype=bool)
            known[::comb] = True
            h = np.where(known, h - h[known].mean(), 0.0)
        else:
            h = h - h.mean()
        profile = np.abs(np.fft.ifft(h, n=n_fft))
        out[i] = int(np.rint(float(np.argmax(profile)) / (n_fft * num["scs"]) * fs))
    return np.clip(out, 1, max_delay)


def _rates(pred: np.ndarray, label: np.ndarray, taus, is_peer) -> Dict[str, float]:
    hit = np.abs(pred - label) <= _TOL_SAMPLES
    peer_false = np.zeros(pred.size, dtype=bool)
    for j in range(taus.shape[1]):
        if is_peer[:, j].any():
            peer_false |= np.abs(pred - np.rint(taus[:, j])) <= _TOL_SAMPLES
    return {
        "exact_tau": float(np.mean(hit)),
        "medae_samples": float(np.median(np.abs(pred - label))),
        "peer_false": float(np.mean(peer_false & ~hit)) if peer_false.any() else 0.0,
    }


def _head_rows(metrics_root: Path, scenario: str) -> List[Dict[str, Any]]:
    """Righe della testa appresa, lette dai metrics.csv della campagna (se presenti)."""
    rows: List[Dict[str, Any]] = []
    for method, sub in (("head_conv1d", "conv1d"), ("head_qkv", "qkv")):
        path = metrics_root / "ber_vs_snr" / scenario / sub / "metrics.csv"
        if not path.is_file():
            continue
        with open(path) as fh:
            for row in csv.DictReader(fh):
                rows.append({
                    "scenario": scenario, "method": method,
                    "snr_db": float(row["snr_db"]), "n": int(row["n_symbols"]),
                    "exact_tau": float(row["exact_tau"]), "medae_samples": float("nan"),
                    "peer_false": float("nan"),
                    "radar_exact_tau": float(row["exact_tau_argmax"]),
                })
    return rows


def evaluate(config: Dict[str, Any], scenario: str, n_bursts: int, snr_values: Sequence[float],
             seed: int, metrics_root: Path) -> List[Dict[str, Any]]:
    
    k = int(max(config["data"]["echoes"]))
    fs = float(config["data"]["fs_hz"])
    rng = np.random.default_rng(seed)
    taus, alphas, dopplers, is_peer, max_delay = sample_geometry(config, n_bursts, rng, k)
    label = _label_lag(taus, max_delay)
    strongest = np.argmax(np.where(is_peer, -np.inf, alphas), axis=1)
    ceiling_non_peer = float(np.mean(strongest == 0))
    ceiling_all = float(np.mean(np.argmax(alphas, axis=1) == 0))
    bits = rng.integers(0, 2, n_bursts, dtype=np.int64)
    seeds = rng.integers(0, 2 ** 31 - 1, n_bursts, dtype=np.int64)
    x = np.asarray(
        generate_transmitted_batch_fast(config, bits, rng, seeds=seeds), dtype=np.complex128
    )
    signal_power = float(np.mean(np.abs(x) ** 2))
    logger.info(
        "%s: n=%d tap=%d peer=%s | potenza waveform=%.5f | tetto peak-picker %.4f "
        "(tutti i tap %.4f)", scenario, n_bursts, alphas.shape[1], is_peer.any(),
        signal_power, ceiling_non_peer, ceiling_all,
    )
    rows: List[Dict[str, Any]] = []
    for snr_db in snr_values:
        noise_var = signal_power * 10.0 ** (-float(snr_db) / 10.0)
        y = _chaotic_received(x, taus, alphas, dopplers, config, noise_var, rng)
        mf = np.clip(
            np.rint(np.asarray(argmax_delay_from_profile(y, x, max_delay))).astype(np.int64),
            1, max_delay,
        )
        rows.append({"scenario": scenario, "method": "chaotic_mf", "snr_db": float(snr_db),
                     "n": n_bursts, "radar_exact_tau": float("nan"), **_rates(mf, label, taus, is_peer)})
        for tag, num in NUMEROLOGIES.items():
            n_sc = 12 * int(num["prb"])
            # stessa energia per burst della waveform caotica (seq_len campioni a potenza 1)
            noise_var_ofdm = noise_var / (x.shape[1] * signal_power / n_sc)
            for pattern, comb in DMRS_PATTERNS:
                pred = _ofdm_lag(taus, alphas, num, fs, noise_var_ofdm, rng, max_delay, comb)
                rows.append({"scenario": scenario,
                             "method": "%s_%s" % (tag, pattern), "snr_db": float(snr_db),
                             "n": n_bursts, "radar_exact_tau": float("nan"),
                             **_rates(pred, label, taus, is_peer)})
        rows.append({"scenario": scenario, "method": "peak_picker_ceiling",
                     "snr_db": float(snr_db), "n": n_bursts,
                     "exact_tau": ceiling_non_peer, "medae_samples": float("nan"),
                     "peer_false": float("nan"), "radar_exact_tau": float("nan")})
        rows.append({"scenario": scenario, "method": "peak_picker_ceiling_all_taps",
                     "snr_db": float(snr_db), "n": n_bursts,
                     "exact_tau": ceiling_all, "medae_samples": float("nan"),
                     "peer_false": float("nan"), "radar_exact_tau": float("nan")})
    rows.extend(_head_rows(metrics_root, scenario))
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_REPO_ROOT / "configs" / "experiments.yaml")
    parser.add_argument("--scenario", default=",".join(_DEFAULT_SCENARIOS))
    parser.add_argument("--snr", type=float, nargs="*", default=list(_DEFAULT_SNR))
    parser.add_argument("--n-bursts", type=int, default=_DEFAULT_BURSTS)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--metrics-root", type=Path, default=_REPO_ROOT / "results" / "full")
    parser.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "results" / "ofdm_isac")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scenarios = [s.strip() for s in str(args.scenario).split(",") if s.strip()]
    base = load_config(args.config, base_config_path=DEFAULT_BASE_CONFIG_PATH)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        config = resolve_scenario(base, scenario)
        rows.extend(
            evaluate(config, scenario, int(args.n_bursts), args.snr, int(args.seed),
                     args.metrics_root)
        )
    fields = ("scenario", "method", "snr_db", "n", "exact_tau", "medae_samples",
              "peer_false", "radar_exact_tau")
    out_csv = args.out_dir / "ofdm_isac.csv"
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    high = float(max(args.snr))
    summary = [r for r in rows if abs(r["snr_db"] - high) < 1e-6]
    out_summary = args.out_dir / "ofdm_isac_summary.csv"
    with open(out_summary, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    logger.info("scritto %s (%d righe) e %s in %.1f s", out_csv, len(rows), out_summary,
                time.time() - t0)
    print("\n== riepilogo a %.1f dB ==" % high)
    for row in sorted(summary, key=lambda r: (r["scenario"], -r["exact_tau"])):
        print("  %-24s %-28s exact_tau=%.4f" % (row["scenario"], row["method"], row["exact_tau"]))


if __name__ == "__main__":
    main()
