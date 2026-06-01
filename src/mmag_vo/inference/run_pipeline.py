from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import yaml

from mmag_vo.data.transforms import image_to_tensor, normalize_rgb, resize_image_np
from mmag_vo.metrics.odometry import save_kitti_poses
from mmag_vo.models.fusion import fuse_pose_matrices, pose_vec_to_matrix
from mmag_vo.models.mmag import MMAGDepthSegNet
from mmag_vo.models.vo_gru import CNNGRUVO
from mmag_vo.vo.orb_depth import DepthEnhancedORBVO


def load_intrinsics(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return np.array([[data["fx"], 0, data["cx"]], [0, data["fy"], data["cy"]], [0, 0, 1]], dtype=np.float64)


def load_checkpoint_model(model, checkpoint: str, device):
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.to(device).eval()
    return model


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    d = depth.copy()
    valid = np.isfinite(d) & (d > 0)
    if valid.any():
        lo, hi = np.percentile(d[valid], [2, 98])
        d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        d[:] = 0
    return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--mmag-checkpoint", default="")
    parser.add_argument("--vo-checkpoint", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-seg-classes", type=int, default=40)
    parser.add_argument("--image-size", type=int, nargs=2, default=[384, 640])
    parser.add_argument("--mc-dropout-samples", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth_vis").mkdir(parents=True, exist_ok=True)
    (out_dir / "seg").mkdir(parents=True, exist_ok=True)

    k = load_intrinsics(args.intrinsics)
    mmag = load_checkpoint_model(MMAGDepthSegNet(num_seg_classes=args.num_seg_classes), args.mmag_checkpoint, device)
    vo_gru = None
    if args.vo_checkpoint:
        vo_gru = load_checkpoint_model(CNNGRUVO(), args.vo_checkpoint, device)
    orb = DepthEnhancedORBVO(k)

    image_paths = sorted([p for p in Path(args.image_dir).glob("*.png")] + [p for p in Path(args.image_dir).glob("*.jpg")])
    current_pose = np.eye(4, dtype=np.float64)
    poses = [current_pose.copy()]
    feature_buffer = []

    for idx, path in enumerate(tqdm(image_paths, desc="pipeline")):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        resized = resize_image_np(rgb, tuple(args.image_size), is_label=False)
        tensor = normalize_rgb(image_to_tensor(resized)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = mmag(tensor)
            depth_small = out["depth"][0].detach().cpu().numpy()
            seg_small = out["seg_logits"].argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
            feat = out["features_mmag"].detach()
        depth = cv2.resize(depth_small, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        seg = cv2.resize(seg_small, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        np.save(out_dir / "depth" / f"{idx:06d}.npy", depth)
        cv2.imwrite(str(out_dir / "depth_vis" / f"{idx:06d}.png"), colorize_depth(depth))
        cv2.imwrite(str(out_dir / "seg" / f"{idx:06d}.png"), seg)

        orb_res = orb.process(rgb, depth)
        t_orb = torch.from_numpy(orb_res.transform).float().to(device).unsqueeze(0)
        cov_orb = torch.from_numpy(orb_res.covariance).float().to(device).unsqueeze(0)

        if vo_gru is not None:
            feature_buffer.append(feat)
            if len(feature_buffer) > 5:
                feature_buffer.pop(0)
            feats = torch.stack(feature_buffer, dim=1).squeeze(2) if feature_buffer[0].ndim == 5 else torch.cat(feature_buffer, dim=0).unsqueeze(0)
            if feats.ndim == 4:
                feats = feats.unsqueeze(0)
            with torch.no_grad():
                mean_pose, cov_diag = vo_gru.predict_with_mc_dropout(feats, samples=args.mc_dropout_samples)
                pose_vec = mean_pose[:, -1]
                cov_diag = cov_diag[:, -1]
                t_gru = pose_vec_to_matrix(pose_vec)
                t_fused, _, _ = fuse_pose_matrices(t_orb, cov_orb, t_gru, cov_diag)
                rel = t_fused[0].detach().cpu().numpy()
        else:
            rel = orb_res.transform

        current_pose = current_pose @ np.linalg.inv(rel)  # Convert camera motion to world trajectory convention.
        poses.append(current_pose.copy())

    poses_np = np.stack(poses[: len(image_paths)], axis=0)
    np.save(out_dir / "per_frame_pose.npy", poses_np)
    save_kitti_poses(str(out_dir / "trajectory.txt"), poses_np)
    print(f"Saved trajectory and predictions to {out_dir}")


if __name__ == "__main__":
    main()
