from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import image_to_tensor, normalize_rgb, resize_image_np


def _read_split_file(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def read_rgb(path: str | Path, image_size: Tuple[int, int] | None = None) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("RGB"))
    if image_size is not None:
        img = resize_image_np(img, image_size, is_label=False)
    return img


def read_label(path: str | Path, image_size: Tuple[int, int] | None = None) -> np.ndarray:
    lbl = np.asarray(Image.open(path))
    if image_size is not None:
        lbl = resize_image_np(lbl.astype(np.uint8), image_size, is_label=True)
    return lbl.astype(np.int64)


def read_depth(path: str | Path, image_size: Tuple[int, int] | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        depth = np.load(path).astype(np.float32)
    else:
        raw = np.asarray(Image.open(path))
        depth = raw.astype(np.float32)
        # Common encodings: millimeters as uint16 or metric float-like PNG.
        if raw.dtype == np.uint16 or depth.max() > 255:
            depth = depth / 1000.0
    if image_size is not None:
        depth = resize_image_np(depth.astype(np.float32), image_size, is_label=False).astype(np.float32)
    return depth


class JointDepthSegDataset(Dataset):
    """Generic RGB/depth/label dataset used for NYU-Depth-v2 and CamVid.

    Split file rows may be either:
      image_path depth_path label_path
    or relative stem names, in which case `images/`, `depths/`, and `labels/` are used.
    """

    def __init__(self, root: str | Path, split: str = "train.txt", image_size: Tuple[int, int] | None = None) -> None:
        self.root = Path(root)
        self.image_size = image_size
        split_path = self.root / split
        self.rows = _read_split_file(split_path)

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_row(self, row: str) -> tuple[Path, Path, Path]:
        parts = row.split()
        if len(parts) >= 3:
            return tuple((self.root / p if not Path(p).is_absolute() else Path(p)) for p in parts[:3])  # type: ignore[return-value]
        stem = parts[0]
        image = self.root / "images" / stem
        if not image.exists():
            for ext in [".png", ".jpg", ".jpeg"]:
                if (self.root / "images" / f"{stem}{ext}").exists():
                    image = self.root / "images" / f"{stem}{ext}"
                    break
        depth = self.root / "depths" / f"{Path(stem).stem}.npy"
        if not depth.exists():
            depth = self.root / "depths" / f"{Path(stem).stem}.png"
        label = self.root / "labels" / f"{Path(stem).stem}.png"
        return image, depth, label

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_path, depth_path, label_path = self._resolve_row(self.rows[idx])
        image = read_rgb(image_path, self.image_size)
        depth = read_depth(depth_path, self.image_size)
        label = read_label(label_path, self.image_size)
        image_tensor = normalize_rgb(image_to_tensor(image))
        return {
            "image": image_tensor,
            "depth": torch.from_numpy(depth).float(),
            "label": torch.from_numpy(label).long(),
            "image_path": str(image_path),
        }


class NYUDepthV2(JointDepthSegDataset):
    pass


class CamVid(JointDepthSegDataset):
    pass


def load_kitti_poses(path: str | Path) -> np.ndarray:
    mats = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            t = np.eye(4, dtype=np.float32)
            t[:3, :4] = np.array(vals, dtype=np.float32).reshape(3, 4)
            mats.append(t)
    return np.stack(mats, axis=0)


def matrix_to_pose_vec_np(tmat: np.ndarray) -> np.ndarray:
    r = tmat[:3, :3]
    # Rodrigues rotation vector is a compact, stable target for learning.
    rotvec, _ = cv2.Rodrigues(r.astype(np.float64))
    return np.concatenate([tmat[:3, 3], rotvec.reshape(3)]).astype(np.float32)


class KITTIOdometrySequence(Dataset):
    """KITTI sequence snippets for CNN-GRU training."""

    def __init__(
        self,
        root: str | Path,
        sequences: Sequence[str],
        snippet_length: int = 5,
        image_size: Tuple[int, int] | None = None,
        camera: str = "image_2",
    ) -> None:
        self.root = Path(root)
        self.sequences = [str(s).zfill(2) for s in sequences]
        self.snippet_length = int(snippet_length)
        self.image_size = image_size
        self.camera = camera
        self.index: list[tuple[str, int]] = []
        self.images: dict[str, list[Path]] = {}
        self.poses: dict[str, np.ndarray] = {}
        for seq in self.sequences:
            img_dir = self.root / "sequences" / seq / camera
            imgs = sorted([p for p in img_dir.glob("*.png")])
            pose_path = self.root / "poses" / f"{seq}.txt"
            if not imgs or not pose_path.exists():
                continue
            poses = load_kitti_poses(pose_path)
            n = min(len(imgs), len(poses))
            self.images[seq] = imgs[:n]
            self.poses[seq] = poses[:n]
            for start in range(0, n - self.snippet_length):
                self.index.append((seq, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq, start = self.index[idx]
        imgs = []
        for i in range(start, start + self.snippet_length):
            img = read_rgb(self.images[seq][i], self.image_size)
            imgs.append(normalize_rgb(image_to_tensor(img)))
        poses_abs = self.poses[seq][start : start + self.snippet_length]
        rels = []
        for i in range(1, len(poses_abs)):
            rel = np.linalg.inv(poses_abs[i - 1]) @ poses_abs[i]
            rels.append(matrix_to_pose_vec_np(rel))
        # First time step has identity relative motion.
        rels = [np.zeros(6, dtype=np.float32)] + rels
        return {
            "image": torch.stack(imgs, dim=0),
            "pose": torch.from_numpy(np.stack(rels, axis=0)).float(),
            "sequence": seq,
            "start": start,
        }
