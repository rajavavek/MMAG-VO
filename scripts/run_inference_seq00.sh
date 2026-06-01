#!/usr/bin/env bash
set -euo pipefail
python -m mmag_vo.inference.run_pipeline \
  --image-dir "${KITTI_ROOT:-/datasets/kitti_odometry}/sequences/00/image_2" \
  --intrinsics configs/kitti_intrinsics_00.yaml \
  --mmag-checkpoint runs/mmag_full/best.pt \
  --vo-checkpoint runs/vo_gru/best.pt \
  --output outputs/seq00
