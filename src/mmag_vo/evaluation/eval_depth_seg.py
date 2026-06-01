from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmag_vo.data.datasets import CamVid, NYUDepthV2
from mmag_vo.metrics.depth import compute_depth_metrics
from mmag_vo.metrics.segmentation import compute_miou
from mmag_vo.models.mmag import MMAGDepthSegNet


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["nyu", "camvid"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="test.txt")
    parser.add_argument("--num-classes", type=int, default=40)
    parser.add_argument("--image-size", type=int, nargs=2, default=[384, 640])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_cls = NYUDepthV2 if args.dataset == "nyu" else CamVid
    ds = ds_cls(args.root, split=args.split, image_size=tuple(args.image_size))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = MMAGDepthSegNet(num_seg_classes=args.num_classes).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()

    depth_metrics = []
    miou_metrics = []
    for batch in tqdm(loader):
        image = batch["image"].to(device)
        out = model(image)
        pred_depth = out["depth"].cpu().numpy()
        pred_seg = out["seg_logits"].argmax(1).cpu().numpy()
        gt_depth = batch["depth"].numpy()
        gt_seg = batch["label"].numpy()
        for pd, gd, ps, gs in zip(pred_depth, gt_depth, pred_seg, gt_seg):
            depth_metrics.append(compute_depth_metrics(pd, gd))
            miou_metrics.append(compute_miou(ps, gs, args.num_classes)["miou"])
    keys = depth_metrics[0].keys()
    print({k: sum(m[k] for m in depth_metrics) / len(depth_metrics) for k in keys})
    print({"miou": sum(miou_metrics) / len(miou_metrics)})


if __name__ == "__main__":
    main()
