from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        negative_slope: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if negative_slope is None:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.act(self.bn(self.conv(x))))


class EncoderBlock(nn.Module):
    """Encoder block producing F_k at H/2^k, W/2^k."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = ConvBNAct(in_channels, out_channels, 3, dilation=dilation)
        self.conv2 = ConvBNAct(out_channels, out_channels, 3, dilation=dilation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MMAGEncoder(nn.Module):
    """Modified ResNet-style encoder from the paper.

    It emits multi-scale feature maps with channels [64, 128, 256, 512].
    The final two blocks use dilated convolutions to enlarge receptive field.
    """

    def __init__(self, channels: Tuple[int, int, int, int] = (64, 128, 256, 512)) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.block1 = EncoderBlock(3, c1, dilation=1)
        self.block2 = EncoderBlock(c1, c2, dilation=1)
        self.block3 = EncoderBlock(c2, c3, dilation=2)
        self.block4 = EncoderBlock(c3, c4, dilation=4)

    def forward(self, x: Tensor) -> List[Tensor]:
        f1 = self.block1(x)  # H/2
        f2 = self.block2(f1)  # H/4
        f3 = self.block3(f2)  # H/8
        f4 = self.block4(f3)  # H/16
        return [f1, f2, f3, f4]


class ModifiedMultiAttentionGate(nn.Module):
    """Shared Modified Multi-Attention Gate (MMAG).

    Implements equations 5-9 in the paper:
        q1 = ReLU(Wx * x_dec + bx)
        q2 = ReLU(Wg * x_enc + bg)
        q  = Wpsi * (q1 + q2) + bpsi
        alpha = sigmoid(q)
        x_attended = x_enc * alpha

    One instance is shared by both depth and segmentation decoders at a level.
    """

    def __init__(self, encoder_channels: int, decoder_channels: int, intermediate_channels: int = 64) -> None:
        super().__init__()
        self.wx = nn.Conv2d(decoder_channels, intermediate_channels, kernel_size=1, bias=True)
        self.wg = nn.Conv2d(encoder_channels, intermediate_channels, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(intermediate_channels, 1, kernel_size=1, bias=True)

    def forward(self, x_enc: Tensor, x_dec: Tensor) -> Tuple[Tensor, Tensor]:
        if x_dec.shape[-2:] != x_enc.shape[-2:]:
            x_dec = F.interpolate(x_dec, size=x_enc.shape[-2:], mode="bilinear", align_corners=False)
        q1 = F.relu(self.wx(x_dec), inplace=True)
        q2 = F.relu(self.wg(x_enc), inplace=True)
        alpha = torch.sigmoid(self.psi(q1 + q2))
        return x_enc * alpha, alpha


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = ConvBNAct(in_channels, out_channels, kernel_size=3, negative_slope=0.2)

    def forward(self, x: Tensor, size: Tuple[int, int]) -> Tensor:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.proj(x)


class DecoderConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, out_channels, 3, dilation=dilation, negative_slope=0.2, dropout=dropout),
            ConvBNAct(out_channels, out_channels, 3, dilation=dilation, negative_slope=0.2, dropout=dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MMAGDecoderBranch(nn.Module):
    """One decoder branch. Gates are passed in from the parent and shared."""

    def __init__(
        self,
        out_channels: int,
        gates: nn.ModuleDict,
        dropout: float = 0.1,
        min_depth: float = 0.01,
        is_depth: bool = False,
    ) -> None:
        super().__init__()
        self.gates = gates
        self.is_depth = is_depth
        self.min_depth = min_depth

        self.up3 = UpConv(512, 256)
        self.dec3 = DecoderConvBlock(256 + 256, 256, dilation=4, dropout=dropout)
        self.up2 = UpConv(256, 256)
        self.dec2 = DecoderConvBlock(128 + 256, 256, dilation=2, dropout=dropout)
        self.up1 = UpConv(256, 128)
        self.dec1 = DecoderConvBlock(64 + 128, 128, dilation=1, dropout=dropout)
        self.final = nn.Sequential(
            ConvBNAct(128, 64, 3, negative_slope=0.2, dropout=dropout),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )
        self.depth_activation = nn.Softplus(beta=1.0)

    def forward(self, features: List[Tensor], output_size: Tuple[int, int]) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        f1, f2, f3, f4 = features
        attentions: Dict[str, Tensor] = {}

        x = self.up3(f4, size=f3.shape[-2:])
        f3_att, attentions["level3"] = self.gates["gate3"](f3, x)
        x = self.dec3(torch.cat([x, f3_att], dim=1))

        x = self.up2(x, size=f2.shape[-2:])
        f2_att, attentions["level2"] = self.gates["gate2"](f2, x)
        x = self.dec2(torch.cat([x, f2_att], dim=1))
        f_mmag = x  # H/4 x W/4 x 256, used by CNN-GRU.

        x = self.up1(x, size=f1.shape[-2:])
        f1_att, attentions["level1"] = self.gates["gate1"](f1, x)
        x = self.dec1(torch.cat([x, f1_att], dim=1))

        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        out = self.final(x)
        if self.is_depth:
            out = self.depth_activation(out).squeeze(1) + self.min_depth
        return out, f_mmag, attentions


class MMAGDepthSegNet(nn.Module):
    """Joint depth-estimation and semantic-segmentation network with shared MMAG gates."""

    def __init__(
        self,
        num_seg_classes: int = 40,
        encoder_channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
        attention_intermediate_channels: int = 64,
        dropout: float = 0.1,
        min_depth: float = 0.01,
    ) -> None:
        super().__init__()
        self.encoder = MMAGEncoder(encoder_channels)
        c1, c2, c3, c4 = encoder_channels
        self.shared_gates = nn.ModuleDict(
            {
                "gate3": ModifiedMultiAttentionGate(c3, 256, attention_intermediate_channels),
                "gate2": ModifiedMultiAttentionGate(c2, 256, attention_intermediate_channels),
                "gate1": ModifiedMultiAttentionGate(c1, 128, attention_intermediate_channels),
            }
        )
        self.depth_decoder = MMAGDecoderBranch(1, self.shared_gates, dropout, min_depth, is_depth=True)
        self.seg_decoder = MMAGDecoderBranch(num_seg_classes, self.shared_gates, dropout, min_depth, is_depth=False)

    def forward(self, image: Tensor) -> Dict[str, Tensor | Dict[str, Tensor]]:
        output_size = image.shape[-2:]
        features = self.encoder(image)
        depth, f_depth, att_depth = self.depth_decoder(features, output_size)
        seg_logits, f_seg, att_seg = self.seg_decoder(features, output_size)
        features_mmag = 0.5 * (f_depth + f_seg)
        return {
            "depth": depth,
            "seg_logits": seg_logits,
            "features_mmag": features_mmag,
            "attention_depth": att_depth,
            "attention_seg": att_seg,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
