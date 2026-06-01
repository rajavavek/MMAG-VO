#!/usr/bin/env bash
set -euo pipefail
python -m mmag_vo.training.train_vo_gru \
  --config configs/vo_gru_kitti.yaml \
  --data-root "${KITTI_ROOT:-/datasets/kitti_odometry}" \
  --mmag-checkpoint runs/mmag_full/best.pt \
  --output runs/vo_gru
