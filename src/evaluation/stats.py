
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Dict, Optional, Sequence, Tuple

DEFAULT_CONFIDENCE = 0.95
DEFAULT_QUANTILES = (0.05, 0.5, 0.95)

def z_value(confidence: float) -> float:
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")
    return float(NormalDist().inv_cdf(0.5 + float(confidence) / 2.0))

def wilson_interval(
    errors: int,
    symbols: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Tuple[float, float]:
    if isinstance(errors, bool) or not isinstance(errors, (int,)) or errors < 0:
        raise ValueError(f"errors must be a non-negative int, got {errors!r}")
    if isinstance(symbols, bool) or not isinstance(symbols, (int,)) or symbols < 0:
        raise ValueError(f"symbols must be a non-negative int, got {symbols!r}")
    if errors > symbols:
        raise ValueError(f"errors ({errors}) cannot exceed symbols ({symbols})")
    if symbols == 0:
        return (float("nan"), float("nan"))
    z = z_value(confidence)
    n = float(symbols)
    p = float(errors) / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))

def mean_interval(
    values: Sequence[float],
    confidence: float = DEFAULT_CONFIDENCE,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> Dict[str, float]:
    import numpy as np

    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty 1D sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    n = int(array.size)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if n > 1 else 0.0
    half = z_value(confidence) * std / math.sqrt(float(n)) if n > 1 else 0.0
    result: Dict[str, float] = {
        "n": float(n),
        "mean": mean,
        "std": std,
        "sem": float(std / math.sqrt(float(n))) if n > 1 else 0.0,
        "ci_lo": max(0.0, mean - half),
        "ci_hi": mean + half,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
    for q in quantiles:
        result[f"q{int(round(100.0 * float(q))):02d}"] = float(np.quantile(array, float(q)))
    return result

def outage_probability(
    values: Sequence[float],
    target: float,
) -> float:
    import numpy as np

    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty 1D sequence")
    return float(np.mean(array > float(target)))

def describe(
    values: Sequence[float],
    target: Optional[float] = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, float]:
    result = mean_interval(values, confidence=confidence)
    if target is not None:
        result["outage"] = outage_probability(values, float(target))
        result["target"] = float(target)
    return result
