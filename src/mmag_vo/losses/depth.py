from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def valid_depth_mask(target: Tensor, min_depth: float = 1e-3, max_depth: float | None = None) -> Tensor:
    mask = torch.isfinite(target) & (target > min_depth)
    if max_depth is not None:
        mask = mask & (target < max_depth)
    return mask


def silog_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    if mask is None:
        mask = valid_depth_mask(target)
    pred = pred.clamp_min(eps)
    target = target.clamp_min(eps)
    diff = torch.log(pred[mask]) - torch.log(target[mask])
    if diff.numel() == 0:
        return pred.new_tensor(0.0)
    return diff.pow(2).mean() - diff.mean().pow(2)


def _sobel_kernels(device, dtype):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0
    return kx, ky


def gradient_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if target.ndim == 3:
        target = target.unsqueeze(1)
    kx, ky = _sobel_kernels(pred.device, pred.dtype)
    px = F.conv2d(pred, kx, padding=1)
    py = F.conv2d(pred, ky, padding=1)
    tx = F.conv2d(target, kx, padding=1)
    ty = F.conv2d(target, ky, padding=1)
    loss = (px - tx).abs() + (py - ty).abs()
    if mask is not None:
        loss = loss[mask.unsqueeze(1)]
    return loss.mean() if loss.numel() else pred.new_tensor(0.0)


def _simple_ssim(x: Tensor, y: Tensor, window_size: int = 3) -> Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=window_size // 2) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=window_size // 2) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2) + 1e-6)
    return ssim.clamp(0, 1)


def ms_ssim_loss(pred: Tensor, target: Tensor, scales: int = 4) -> Tensor:
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if target.ndim == 3:
        target = target.unsqueeze(1)
    try:
        from pytorch_msssim import ms_ssim

        # Normalize by current max for stable MS-SSIM on metric depth.
        max_val = target.detach().max().clamp_min(1.0)
        return 1.0 - ms_ssim(pred / max_val, target / max_val, data_range=1.0, size_average=True)
    except Exception:
        vals = []
        x, y = pred, target
        for _ in range(scales):
            vals.append(_simple_ssim(x, y).mean())
            if min(x.shape[-2:]) <= 8:
                break
            x = F.avg_pool2d(x, 2, 2)
            y = F.avg_pool2d(y, 2, 2)
        return 1.0 - torch.stack(vals).mean()


def depth_total_loss(
    pred: Tensor,
    target: Tensor,
    lambda_grad: float = 0.5,
    lambda_ssim: float = 0.3,
    max_depth: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    mask = valid_depth_mask(target, max_depth=max_depth)
    l_silog = silog_loss(pred, target, mask)
    l_grad = gradient_loss(pred, target, mask)
    l_ssim = ms_ssim_loss(pred, target)
    total = l_silog + lambda_grad * l_grad + lambda_ssim * l_ssim
    return total, {"silog": l_silog.detach(), "grad": l_grad.detach(), "ssim": l_ssim.detach()}
