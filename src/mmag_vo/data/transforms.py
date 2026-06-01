from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def resize_image_np(image: np.ndarray, size: Tuple[int, int], is_label: bool = False) -> np.ndarray:
    pil = Image.fromarray(image)
    resample = Image.NEAREST if is_label else Image.BILINEAR
    return np.asarray(pil.resize((size[1], size[0]), resample=resample))


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    image = image.astype(np.float32) / 255.0
    if image.ndim == 2:
        image = image[..., None]
    return torch.from_numpy(image.transpose(2, 0, 1))


def normalize_rgb(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def resize_tensor_depth(depth: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(depth[None, None].float(), size=size, mode="nearest").squeeze(0).squeeze(0)
