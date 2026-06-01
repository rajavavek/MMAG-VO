from __future__ import annotations

import numpy as np


def compute_depth_metrics(pred, target, min_depth: float = 1e-3, max_depth: float | None = None) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(target) & (target > min_depth) & (pred > min_depth)
    if max_depth is not None:
        mask &= target < max_depth
    pred = pred[mask]
    target = target[mask]
    if pred.size == 0:
        return {k: float("nan") for k in ["absrel", "sqrel", "rms", "rms_log", "delta1", "delta2", "delta3"]}
    diff = pred - target
    absrel = np.mean(np.abs(diff) / target)
    sqrel = np.mean(diff ** 2 / target)
    rms = np.sqrt(np.mean(diff ** 2))
    rms_log = np.sqrt(np.mean((np.log(pred) - np.log(target)) ** 2))
    ratio = np.maximum(pred / target, target / pred)
    return {
        "absrel": float(absrel),
        "sqrel": float(sqrel),
        "rms": float(rms),
        "rms_log": float(rms_log),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25 ** 2)),
        "delta3": float(np.mean(ratio < 1.25 ** 3)),
    }
