# Reproduction Guide

This file maps paper sections to the code files in this package.

## Paper Section 3.1: Joint depth and semantic segmentation with MMAG

| Paper item | Code |
|---|---|
| Shared encoder-decoder backbone | `src/mmag_vo/models/mmag.py::MMAGEncoder`, `MMAGDecoderBranch` |
| Modified Multi-Attention Gate equations 5-9 | `src/mmag_vo/models/mmag.py::ModifiedMultiAttentionGate` |
| Depth output `D` | `src/mmag_vo/models/mmag.py::MMAGDepthSegNet.forward()['depth']` |
| Segmentation output `S` | `src/mmag_vo/models/mmag.py::MMAGDepthSegNet.forward()['seg_logits']` |
| Intermediate `FMMAG` for VO | `src/mmag_vo/models/mmag.py::MMAGDepthSegNet.forward()['features_mmag']` |
| SiLog, gradient, MS-SSIM loss | `src/mmag_vo/losses/depth.py` |
| Cross entropy + Lovasz-Softmax | `src/mmag_vo/losses/segmentation.py` |

## Paper Section 3.1.4-3.1.6: Depth-enhanced feature-based VO

| Paper item | Code |
|---|---|
| ORB keypoint extraction | `src/mmag_vo/vo/orb_depth.py::DepthEnhancedORBVO._extract` |
| Back-projection with depth | `src/mmag_vo/vo/orb_depth.py::DepthEnhancedORBVO._backproject` |
| Descriptor + depth consistency check | `src/mmag_vo/vo/orb_depth.py::DepthEnhancedORBVO._match_with_depth_check` |
| Motion estimate | `src/mmag_vo/vo/orb_depth.py::DepthEnhancedORBVO.process` |
| Optional external ORB-SLAM2 hook | `src/mmag_vo/vo/orb_slam2_adapter.py` |

## Paper Section 3.2: CNN-GRU visual odometry

| Paper item | Code |
|---|---|
| Five CNN blocks | `src/mmag_vo/models/vo_gru.py::CNNGRUVO.cnn` |
| Two stacked GRU layers | `src/mmag_vo/models/vo_gru.py::CNNGRUVO.gru` |
| 6-DoF pose regression and tanh scaling | `src/mmag_vo/models/vo_gru.py::CNNGRUVO.forward` |
| VO loss | `src/mmag_vo/losses/vo.py` |

## Paper Section 3.3: Uncertainty-aware fusion

| Paper item | Code |
|---|---|
| Twist coordinates `xi = log(T)` | `src/mmag_vo/models/fusion.py::se3_log` |
| Covariance-weighted fusion | `src/mmag_vo/models/fusion.py::covariance_weighted_fusion` |
| Fused matrix output | `src/mmag_vo/models/fusion.py::fuse_pose_matrices` |

## Paper Section 4: Evaluation metrics and reported results

| Paper item | Code/data |
|---|---|
| Depth metrics | `src/mmag_vo/metrics/depth.py` |
| Segmentation mIoU | `src/mmag_vo/metrics/segmentation.py` |
| VO ATE and relative errors | `src/mmag_vo/metrics/odometry.py` |
| Paper-reported results | `results/paper_reported_metrics.json` |

## Expected commands

```bash
# 1. Train MMAG
python -m mmag_vo.training.train_mmag --config configs/mmag_nyu_camvid.yaml --data-root /datasets --output runs/mmag_full

# 2. Train VO-GRU using trained MMAG features
python -m mmag_vo.training.train_vo_gru --config configs/vo_gru_kitti.yaml --data-root /datasets/kitti_odometry --mmag-checkpoint runs/mmag_full/best.pt --output runs/vo_gru

# 3. Run full pipeline
python -m mmag_vo.inference.run_pipeline --image-dir /datasets/kitti_odometry/sequences/00/image_2 --intrinsics configs/kitti_intrinsics_00.yaml --mmag-checkpoint runs/mmag_full/best.pt --vo-checkpoint runs/vo_gru/best.pt --output outputs/seq00

# 4. Evaluate outputs
python -m mmag_vo.evaluation.eval_vo --pred outputs/seq00/trajectory.txt --gt /datasets/kitti_odometry/poses/00.txt
```
