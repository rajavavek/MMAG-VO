from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class CNNGRUVO(nn.Module):
    """CNN-GRU visual odometry model from MMAG intermediate features.

    Expected input: (B, T, 256, H/4, W/4). The output is a pose vector
    (tx, ty, tz, roll, pitch, yaw) for each time step.
    """

    def __init__(
        self,
        input_channels: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_translation: float = 5.0,
    ) -> None:
        super().__init__()
        self.max_translation = float(max_translation)
        self.cnn = nn.Sequential(
            CNNBlock(input_channels, 64, dropout=dropout),
            CNNBlock(64, 128, dropout=dropout),
            CNNBlock(128, 256, dropout=dropout),
            CNNBlock(256, 512, dropout=dropout),
            CNNBlock(512, 1024, dropout=dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.gru = nn.GRU(
            input_size=1024,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 6)

    def encode_features(self, features: Tensor) -> Tensor:
        # features: (B*T, C, H, W)
        x = self.cnn(features).flatten(1)
        return x

    def forward(self, features: Tensor, hidden: Optional[Tensor] = None) -> Dict[str, Tensor]:
        if features.ndim != 5:
            raise ValueError(f"Expected features of shape (B,T,C,H,W), got {tuple(features.shape)}")
        b, t, c, h, w = features.shape
        x = features.reshape(b * t, c, h, w)
        x = self.encode_features(x).reshape(b, t, 1024)
        y, hidden = self.gru(x, hidden)
        raw = self.head(self.dropout(y))
        scale = torch.tensor(
            [self.max_translation, self.max_translation, self.max_translation, math.pi, math.pi, math.pi],
            dtype=raw.dtype,
            device=raw.device,
        )
        pose = torch.tanh(raw) * scale
        return {"pose": pose, "raw_pose": raw, "hidden": hidden}

    @torch.no_grad()
    def predict_with_mc_dropout(self, features: Tensor, samples: int = 16) -> Tuple[Tensor, Tensor]:
        """Monte-Carlo dropout estimate of mean pose and covariance diagonal."""
        was_training = self.training
        self.train(True)
        preds = []
        for _ in range(samples):
            preds.append(self(features)["pose"])
        if not was_training:
            self.eval()
        stack = torch.stack(preds, dim=0)
        mean = stack.mean(dim=0)
        var = stack.var(dim=0, unbiased=False).clamp_min(1e-6)
        return mean, var
