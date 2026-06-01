from __future__ import annotations

import numpy as np


def confusion_matrix(pred, target, num_classes: int, ignore_index: int = 255) -> np.ndarray:
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    pred = pred[valid]
    target = target[valid]
    hist = np.bincount(num_classes * target.astype(int) + pred.astype(int), minlength=num_classes ** 2)
    return hist.reshape(num_classes, num_classes)


def compute_miou(pred, target, num_classes: int, ignore_index: int = 255) -> dict[str, float | list[float]]:
    hist = confusion_matrix(pred, target, num_classes, ignore_index)
    tp = np.diag(hist)
    fp = hist.sum(axis=0) - tp
    fn = hist.sum(axis=1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    return {"miou": float(np.nanmean(iou)), "iou_per_class": iou.astype(float).tolist()}
