#!/usr/bin/env bash
set -euo pipefail
python -m mmag_vo.training.train_mmag \
  --config configs/mmag_nyu_camvid.yaml \
  --data-root "${DATA_ROOT:-/datasets}" \
  --output runs/mmag_full
