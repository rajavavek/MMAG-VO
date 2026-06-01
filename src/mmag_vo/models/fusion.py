from __future__ import annotations

import torch
from torch import Tensor


def skew(v: Tensor) -> Tensor:
    z = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack(
        [
            torch.stack([z, -vz, vy], dim=-1),
            torch.stack([vz, z, -vx], dim=-1),
            torch.stack([-vy, vx, z], dim=-1),
        ],
        dim=-2,
    )


def euler_xyz_to_matrix(euler: Tensor) -> Tensor:
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    zeros = torch.zeros_like(roll)
    ones = torch.ones_like(roll)
    rx = torch.stack(
        [
            torch.stack([ones, zeros, zeros], -1),
            torch.stack([zeros, cr, -sr], -1),
            torch.stack([zeros, sr, cr], -1),
        ],
        -2,
    )
    ry = torch.stack(
        [
            torch.stack([cp, zeros, sp], -1),
            torch.stack([zeros, ones, zeros], -1),
            torch.stack([-sp, zeros, cp], -1),
        ],
        -2,
    )
    rz = torch.stack(
        [
            torch.stack([cy, -sy, zeros], -1),
            torch.stack([sy, cy, zeros], -1),
            torch.stack([zeros, zeros, ones], -1),
        ],
        -2,
    )
    return rz @ ry @ rx


def so3_exp(omega: Tensor) -> Tensor:
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).clamp_min(1e-8)
    unit = omega / theta
    k = skew(unit)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(k.shape)
    st = torch.sin(theta)[..., None]
    ct = torch.cos(theta)[..., None]
    return eye + st * k + (1.0 - ct) * (k @ k)


def so3_log(rotation: Tensor) -> Tensor:
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    denom = (2.0 * torch.sin(theta)).clamp_min(1e-7)
    v = torch.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        dim=-1,
    )
    return v * (theta / denom).unsqueeze(-1)


def pose_vec_to_matrix(pose: Tensor) -> Tensor:
    """Convert (tx,ty,tz,roll,pitch,yaw) vector to homogeneous matrix."""
    translation = pose[..., :3]
    rotation = euler_xyz_to_matrix(pose[..., 3:])
    out_shape = pose.shape[:-1] + (4, 4)
    tmat = torch.zeros(out_shape, device=pose.device, dtype=pose.dtype)
    tmat[..., :3, :3] = rotation
    tmat[..., :3, 3] = translation
    tmat[..., 3, 3] = 1.0
    return tmat


def matrix_to_pose_vec(tmat: Tensor) -> Tensor:
    trans = tmat[..., :3, 3]
    rotvec = so3_log(tmat[..., :3, :3])
    return torch.cat([trans, rotvec], dim=-1)


def se3_log(tmat: Tensor) -> Tensor:
    # For VO fusion, the paper uses twist coordinates. This implementation uses
    # translation + SO(3) logarithm, which is stable and common for small frame-to-frame motion.
    return matrix_to_pose_vec(tmat)


def se3_exp(xi: Tensor) -> Tensor:
    translation = xi[..., :3]
    rotation = so3_exp(xi[..., 3:])
    out_shape = xi.shape[:-1] + (4, 4)
    tmat = torch.zeros(out_shape, device=xi.device, dtype=xi.dtype)
    tmat[..., :3, :3] = rotation
    tmat[..., :3, 3] = translation
    tmat[..., 3, 3] = 1.0
    return tmat


def _as_covariance(cov: Tensor) -> Tensor:
    if cov.shape[-1] == 6 and cov.ndim >= 1 and (cov.ndim < 2 or cov.shape[-2] != 6):
        return torch.diag_embed(cov.clamp_min(1e-8))
    return cov


def covariance_weighted_fusion(xi_orb: Tensor, cov_orb: Tensor, xi_gru: Tensor, cov_gru: Tensor) -> tuple[Tensor, Tensor]:
    """Fuse two 6D pose estimates by Gaussian covariance weighting."""
    cov_orb = _as_covariance(cov_orb)
    cov_gru = _as_covariance(cov_gru)
    eye = torch.eye(6, device=xi_orb.device, dtype=xi_orb.dtype)
    cov_orb = cov_orb + eye * 1e-7
    cov_gru = cov_gru + eye * 1e-7
    inv_orb = torch.linalg.inv(cov_orb)
    inv_gru = torch.linalg.inv(cov_gru)
    cov_fused = torch.linalg.inv(inv_orb + inv_gru)
    rhs = (inv_orb @ xi_orb.unsqueeze(-1)) + (inv_gru @ xi_gru.unsqueeze(-1))
    xi_fused = (cov_fused @ rhs).squeeze(-1)
    return xi_fused, cov_fused


def fuse_pose_matrices(t_orb: Tensor, cov_orb: Tensor, t_gru: Tensor, cov_gru: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    xi_orb = se3_log(t_orb)
    xi_gru = se3_log(t_gru)
    xi_fused, cov = covariance_weighted_fusion(xi_orb, cov_orb, xi_gru, cov_gru)
    return se3_exp(xi_fused), xi_fused, cov
