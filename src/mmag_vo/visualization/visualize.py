from __future__ import annotations

import cv2
import numpy as np


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    norm = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)


def overlay_segmentation(image_rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rng = np.random.default_rng(12345)
    palette = rng.integers(0, 255, size=(max(int(labels.max()) + 1, 1), 3), dtype=np.uint8)
    color = palette[labels.clip(0, len(palette) - 1)]
    out = (image_rgb.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha).clip(0, 255)
    return out.astype(np.uint8)
