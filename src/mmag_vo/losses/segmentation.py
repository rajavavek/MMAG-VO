from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def weighted_cross_entropy(logits: Tensor, target: Tensor, class_weights: Tensor | None = None, ignore_index: int = 255) -> Tensor:
    return F.cross_entropy(logits, target.long(), weight=class_weights, ignore_index=ignore_index)


def lovasz_grad(gt_sorted: Tensor) -> Tensor:
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def flatten_probs(probs: Tensor, labels: Tensor, ignore_index: int = 255) -> tuple[Tensor, Tensor]:
    if probs.ndim != 4:
        raise ValueError("Expected probabilities of shape B,C,H,W")
    b, c, h, w = probs.shape
    probs = probs.permute(0, 2, 3, 1).reshape(-1, c)
    labels = labels.reshape(-1)
    valid = labels != ignore_index
    return probs[valid], labels[valid]


def lovasz_softmax_loss(logits: Tensor, target: Tensor, ignore_index: int = 255) -> Tensor:
    probs = F.softmax(logits, dim=1)
    probs, labels = flatten_probs(probs, target, ignore_index=ignore_index)
    if probs.numel() == 0:
        return logits.new_tensor(0.0)
    c = probs.shape[1]
    losses = []
    for cls in range(c):
        fg = (labels == cls).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probs[:, cls]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def segmentation_total_loss(
    logits: Tensor,
    target: Tensor,
    class_weights: Tensor | None = None,
    lambda_lovasz: float = 0.5,
    ignore_index: int = 255,
) -> tuple[Tensor, dict[str, Tensor]]:
    ce = weighted_cross_entropy(logits, target, class_weights=class_weights, ignore_index=ignore_index)
    lovasz = lovasz_softmax_loss(logits, target, ignore_index=ignore_index)
    total = ce + lambda_lovasz * lovasz
    return total, {"ce": ce.detach(), "lovasz": lovasz.detach()}
