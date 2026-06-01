from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mmag_vo.config import load_config
from mmag_vo.data.datasets import KITTIOdometrySequence
from mmag_vo.losses.vo import vo_pose_loss
from mmag_vo.models.mmag import MMAGDepthSegNet
from mmag_vo.models.vo_gru import CNNGRUVO
from mmag_vo.utils.seed import seed_everything


def load_mmag(checkpoint: str, device: torch.device, num_classes: int = 40) -> MMAGDepthSegNet:
    model = MMAGDepthSegNet(num_seg_classes=num_classes).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_mmag_sequence(mmag, images):
    # images: B,T,3,H,W
    b, t, c, h, w = images.shape
    feats = []
    for i in range(t):
        out = mmag(images[:, i])
        feats.append(out["features_mmag"])
    return torch.stack(feats, dim=1)


def run_epoch(vo_model, mmag, loader, optimizer, scaler, cfg, device, train: bool):
    vo_model.train(train)
    total = 0.0
    pbar = tqdm(loader, desc="train" if train else "validate")
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["pose"].to(device, non_blocking=True)
        with torch.no_grad():
            feats = extract_mmag_sequence(mmag, images)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=bool(cfg.training.amp)):
            pred = vo_model(feats)["pose"]
            loss, parts = vo_pose_loss(pred, target, beta_rotation=cfg.training.beta_rotation)
        if train:
            scaler.scale(loss).backward()
            if cfg.training.grad_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vo_model.parameters(), cfg.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.detach())
        pbar.set_postfix(loss=total / (pbar.n + 1))
    return total / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--mmag-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-seg-classes", type=int, default=40)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb"))

    train_ds = KITTIOdometrySequence(args.data_root, cfg.data.train_sequences, cfg.sequence_length, tuple(cfg.image_size))
    val_ds = KITTIOdometrySequence(args.data_root, cfg.data.val_sequences, cfg.sequence_length, tuple(cfg.image_size))
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, pin_memory=True)

    mmag = load_mmag(args.mmag_checkpoint, device, args.num_seg_classes)
    vo_model = CNNGRUVO(
        input_channels=cfg.model.input_channels,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        max_translation=cfg.model.max_translation,
    ).to(device)
    optimizer = torch.optim.AdamW(vo_model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.training.amp))
    best = float("inf")

    for epoch in range(1, cfg.training.epochs + 1):
        tr = run_epoch(vo_model, mmag, train_loader, optimizer, scaler, cfg, device, train=True)
        val = run_epoch(vo_model, mmag, val_loader, optimizer, scaler, cfg, device, train=False)
        writer.add_scalar("loss/train", tr, epoch)
        writer.add_scalar("loss/val", val, epoch)
        ckpt = {"epoch": epoch, "model": vo_model.state_dict(), "optimizer": optimizer.state_dict(), "config": dict(cfg)}
        torch.save(ckpt, out_dir / "last.pt")
        if val < best:
            best = val
            torch.save(ckpt, out_dir / "best.pt")
        if epoch % int(cfg.checkpoint.save_every) == 0:
            torch.save(ckpt, out_dir / f"epoch_{epoch:04d}.pt")
        print({"epoch": epoch, "train_loss": tr, "val_loss": val, "best": best})


if __name__ == "__main__":
    main()
