
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def blind_features(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 2 or not np.all(np.isfinite(y)):
        raise ValueError("y must be finite with shape (N, N_seq)")
    re = y.real
    im = y.imag
    a = np.mean(re * re, axis=1)
    b = np.mean(re * im, axis=1)
    c = np.mean(im * im, axis=1)
    tr = a + c
    det = a * c - b * b
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    lam = tr / 2.0 + disc
    ux = b
    uy = lam - a
    norm = np.sqrt(ux * ux + uy * uy)
    nz = norm > 0.0
    ux = np.where(nz, ux / np.where(nz, norm, 1.0), 1.0)
    uy = np.where(nz, uy / np.where(nz, norm, 1.0), 0.0)
    z = re * ux[:, None] + im * uy[:, None]
    std = np.std(z, axis=1)
    z = z / np.where(std > 0.0, std, 1.0)[:, None]

    mabs = np.mean(np.abs(z), axis=1)
    fhi = np.mean(np.abs(z) > 1.5, axis=1)
    fmid = np.mean(np.abs(z) < 0.5, axis=1)
    m2 = np.mean(z * z, axis=1)
    kurt = np.mean(z ** 4, axis=1) / np.maximum(m2 * m2, 1e-12)
    zc = z - np.mean(z, axis=1, keepdims=True)
    skew = np.mean(zc ** 3, axis=1) / np.maximum(std ** 3, 1e-12)
    return np.stack([mabs, fhi, fmid, kurt, skew], axis=1)

def fit_lda(X: np.ndarray, y: np.ndarray):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    m0 = X[y == 0].mean(axis=0)
    m1 = X[y == 1].mean(axis=0)
    mu = np.where((y == 0)[:, None], m0, m1)
    cov = np.cov((X - mu).T) + np.eye(X.shape[1]) * 1e-6
    inv = np.linalg.inv(cov)
    w = inv @ (m1 - m0)
    thr = 0.5 * (m1 + m0) @ w
    return w, thr

def blind_decide(y: np.ndarray, w: np.ndarray, thr: float) -> np.ndarray:
    scores = blind_features(y) @ w
    return (scores > thr).astype(np.int64)

def evaluate_on_test_data(
    test_data: Dict[str, np.ndarray],
    config: Dict,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    x = np.asarray(test_data["x"])
    bit = np.asarray(test_data["bit"]).astype(np.int64)
    snr = np.asarray(test_data["snr_db"]).astype(np.float64)
    if x.ndim != 2 or len(bit) != x.shape[0] or len(snr) != x.shape[0]:
        raise ValueError("test_data inconsistent for blind evaluation")

    snr_range = list(config.get("evaluation", {}).get("snr_test_range", []))
    rows = []
    for snr_db in snr_range:
        mask = np.isclose(snr, snr_db, rtol=0.0, atol=1e-6)
        idx = np.where(mask)[0]
        if idx.size < 4:
            continue
        half = idx.size // 2
        w, thr = fit_lda(blind_features(x[idx[:half]]), bit[idx[:half]])
        pred = blind_decide(x[idx[half:]], w, thr)
        errs = int(np.sum(pred != bit[idx[half:]]))
        rows.append(
            {
                "snr_db": float(snr_db),
                "ber": errs / (idx.size - half),
                "n_errors": errs,
                "n_symbols": idx.size - half,
            }
        )
    df = pd.DataFrame(rows).sort_values("snr_db").reset_index(drop=True)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "metrics.csv", index=False)
    return df
