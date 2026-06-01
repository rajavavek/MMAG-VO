from __future__ import annotations

import torch
from torch import Tensor


def vo_pose_loss(pred_pose: Tensor, target_pose: Tensor, beta_rotation: float = 100.0) -> tuple[Tensor, dict[str, Tensor]]:
    """Pose loss matching the paper's translation + weighted rotation objective.

    Both tensors are expected as (..., 6): tx, ty, tz, roll/rot-x, pitch/rot-y, yaw/rot-z.
    """
    t_loss = (pred_pose[..., :3] - target_pose[..., :3]).pow(2).sum(dim=-1).mean()
    r_loss = (pred_pose[..., 3:] - target_pose[..., 3:]).pow(2).sum(dim=-1).mean()
    total = t_loss + beta_rotation * r_loss
    return total, {"translation": t_loss.detach(), "rotation": r_loss.detach()}
