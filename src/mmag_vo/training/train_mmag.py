from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mmag_vo.config import load_config
from mmag_vo.data.datasets import CamVid, NYUDepthV2
from mmag_vo.losses.depth import depth_total_loss
from mmag_vo.losses.segmentation import segmentation_total_loss
from mmag_vo.metrics.depth import compute_depth_metrics
from mmag_vo.models.mmag import MMAGDepthSegNet
from mmag_vo.utils.seed import seed_everything


def build_dataset(name: str, data_root: Path, split: str, image_size):
    if name.lower() == "nyu":
        return NYUDepthV2(data_root / "nyu_depth_v2", split=split, image_size=tuple(image_size))
    if name.lower() == "camvid":
        return CamVid(data_root / "camvid", split=split, image_size=tuple(image_size))
    raise ValueError(f"Unknown dataset {name}")


def cosine_seg_weight(epoch: int, total_epochs: int, initial: float = 1.0, final: float = 0.0) -> float:
    # Paper: 0.5 * (1 + cos(pi*t/T)); this helper supports optional scaling.
    decay = 0.5 * (1.0 + math.cos(math.pi * epoch / max(total_epochs, 1)))
    return final + (initial - final) * decay


def train_one_epoch(model, loader, optimizer, scaler, cfg, device, epoch):
    model.train()
    lambda_seg = cosine_seg_weight(epoch, cfg.training.epochs, cfg.loss.lambda_seg_initial, cfg.loss.lambda_seg_final)
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"train epoch {epoch}")
    for batch in pbar:
        image = batch["image"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=bool(cfg.training.amp)):
            out = model(image)
            l_depth, depth_parts = depth_total_loss(out["depth"], depth, cfg.loss.lambda_grad, cfg.loss.lambda_ssim)
            l_seg, seg_parts = segmentation_total_loss(
                out["seg_logits"], label, lambda_lovasz=cfg.loss.lambda_lovasz, ignore_index=cfg.loss.ignore_index
            )
            loss = cfg.loss.lambda_depth * l_depth + lambda_seg * l_seg
        scaler.scale(loss).backward()
        if cfg.training.grad_clip_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach())
        pbar.set_postfix(loss=total_loss / (pbar.n + 1), lambda_seg=lambda_seg)
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, cfg, device):
    model.eval()
    losses = []
    metrics = []
    for batch in tqdm(loader, desc="validate"):
        image = batch["image"].to(device)
        depth = batch["depth"].to(device)
        label = batch["label"].to(device)
        out = model(image)
        l_depth, _ = depth_total_loss(out["depth"], depth, cfg.loss.lambda_grad, cfg.loss.lambda_ssim)
        l_seg, _ = segmentation_total_loss(out["seg_logits"], label, lambda_lovasz=cfg.loss.lambda_lovasz, ignore_index=cfg.loss.ignore_index)
        losses.append(float((l_depth + l_seg).detach()))
        pred_np = out["depth"].detach().cpu().numpy()
        depth_np = depth.detach().cpu().numpy()
        for p, t in zip(pred_np, depth_np):
            metrics.append(compute_depth_metrics(p, t))
    rms = sum(m["rms"] for m in metrics) / max(len(metrics), 1)
    absrel = sum(m["absrel"] for m in metrics) / max(len(metrics), 1)
    return {"val_loss": sum(losses) / max(len(losses), 1), "val_depth_rms": rms, "val_depth_absrel": absrel}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb"))

    train_ds = build_dataset(cfg.data.train_dataset, Path(args.data_root), cfg.data.nyu.train_split, cfg.image_size)
    val_ds = build_dataset(cfg.data.val_dataset, Path(args.data_root), cfg.data.nyu.val_split, cfg.image_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, pin_memory=True)

    model = MMAGDepthSegNet(
        num_seg_classes=cfg.num_seg_classes,
        attention_intermediate_channels=cfg.model.attention_intermediate_channels,
        dropout=cfg.model.dropout,
        min_depth=cfg.model.min_depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.training.amp))
    best = float("inf")

    for epoch in range(1, cfg.training.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, cfg, device, epoch)
        val = validate(model, val_loader, cfg, device)
        writer.add_scalar("loss/train", tr_loss, epoch)
        for k, v in val.items():
            writer.add_scalar(k, v, epoch)
        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": dict(cfg)}
        torch.save(ckpt, out_dir / "last.pt")
        monitor = val.get(cfg.checkpoint.monitor, val["val_loss"])
        if monitor < best:
            best = monitor
            torch.save(ckpt, out_dir / "best.pt")
        if epoch % int(cfg.checkpoint.save_every) == 0:
            torch.save(ckpt, out_dir / f"epoch_{epoch:04d}.pt")
        print({"epoch": epoch, "train_loss": tr_loss, **val, "best": best})


if __name__ == "__main__":
    main()
