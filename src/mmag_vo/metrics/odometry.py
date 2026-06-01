from __future__ import annotations

import math
import numpy as np


def read_kitti_poses(path: str) -> np.ndarray:
    mats = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) == 12:
                t = np.eye(4)
                t[:3, :4] = np.array(vals).reshape(3, 4)
                mats.append(t)
            elif len(vals) == 16:
                mats.append(np.array(vals).reshape(4, 4))
    return np.stack(mats, axis=0)


def save_kitti_poses(path: str, poses: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in poses:
            vals = t[:3, :4].reshape(-1)
            f.write(" ".join(f"{v:.9e}" for v in vals) + "\n")


def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[np.ndarray, float, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = dst_c.T @ src_c / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1
    r = u @ s @ vt
    scale = 1.0
    if with_scale:
        scale = np.trace(np.diag(d) @ s) / np.mean(np.sum(src_c ** 2, axis=1))
    t = mu_dst - scale * r @ mu_src
    return r, float(scale), t


def absolute_trajectory_error(pred_poses: np.ndarray, gt_poses: np.ndarray) -> float:
    n = min(len(pred_poses), len(gt_poses))
    pred_xyz = pred_poses[:n, :3, 3]
    gt_xyz = gt_poses[:n, :3, 3]
    r, s, t = umeyama_alignment(pred_xyz, gt_xyz, with_scale=True)
    aligned = (s * (r @ pred_xyz.T)).T + t
    return float(np.sqrt(np.mean(np.sum((aligned - gt_xyz) ** 2, axis=1))))


def rotation_error_deg(r_pred: np.ndarray, r_gt: np.ndarray) -> float:
    r = r_pred.T @ r_gt
    val = (np.trace(r) - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return float(math.degrees(math.acos(val)))


def translation_error_percent(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    denom = np.linalg.norm(t_gt)
    if denom < 1e-12:
        return 0.0
    return float(np.linalg.norm(t_pred - t_gt) / denom * 100.0)


def simple_relative_errors(pred_poses: np.ndarray, gt_poses: np.ndarray) -> dict[str, float]:
    n = min(len(pred_poses), len(gt_poses))
    trel = []
    rrel = []
    for i in range(1, n):
        dp = np.linalg.inv(pred_poses[i - 1]) @ pred_poses[i]
        dg = np.linalg.inv(gt_poses[i - 1]) @ gt_poses[i]
        trel.append(translation_error_percent(dp[:3, 3], dg[:3, 3]))
        rrel.append(rotation_error_deg(dp[:3, :3], dg[:3, :3]))
    return {"trel_percent": float(np.mean(trel)) if trel else float("nan"), "rrel_deg_per_step": float(np.mean(rrel)) if rrel else float("nan")}
