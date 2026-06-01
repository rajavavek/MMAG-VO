# MMAG-VO: Monocular Robot Odometry with Depth, Semantics, and Uncertainty Fusion

This repository provides a research implementation of **“Model Generated Spatial and Contextual Information for Accurate Monocular Robot Odometry.”**

The project implements a monocular visual odometry pipeline that combines joint depth estimation, semantic segmentation, feature-based geometric tracking, learning-based pose regression, and uncertainty-aware pose fusion.

## Overview

The pipeline takes monocular RGB image sequences as input and produces:

- Dense depth maps
- Semantic segmentation maps
- Relative camera poses
- Fused visual odometry trajectories

The implementation is organized around four main components:

1. **MMAG depth and segmentation network**  
   A shared encoder-decoder model with Modified Multi-Attention Gates for joint monocular depth estimation and semantic segmentation.

2. **Depth-enhanced ORB visual odometry**  
   A Python/OpenCV RGB-D-style ORB tracker that uses predicted monocular depth for keypoint back-projection, depth consistency filtering, PnP-RANSAC pose estimation, and covariance approximation.

3. **CNN-GRU visual odometry**  
   A learning-based visual odometry module that estimates 6-DoF relative pose from intermediate MMAG feature maps using convolutional feature compression and stacked GRU layers.

4. **Uncertainty-aware pose fusion**  
   A covariance-weighted fusion module that combines geometric ORB-based pose estimates with CNN-GRU pose predictions in Lie algebra space.

## Key Features

- Joint depth estimation and semantic segmentation using shared MMAG attention.
- Depth consistency filtering for ORB feature matching.
- CNN-GRU pose regression from MMAG intermediate representations.
- Covariance-based fusion of geometric and learned visual odometry estimates.
- Dataset loaders for NYU-Depth-v2, CamVid, and KITTI-style odometry data.
- Training, evaluation, and inference entry points.
- Paper-reported reference metrics stored for comparison.

## Reproducibility Notice

The paper describes the model architecture, training objectives, evaluation metrics, and reported results. However, the paper does not provide pretrained model weights, exact preprocessing scripts, train/validation split files, or the full modified C++ ORB-SLAM2 implementation.

This repository therefore provides a structured, executable implementation aligned with the paper methodology. Exact numerical reproduction requires training on the same dataset versions, preprocessing pipeline, camera calibration files, and evaluation protocol used in the original experiments.

Paper-reported reference metrics are available in:

```text
results/paper_reported_metrics.json
```

To display the reported reference metrics:

```bash
python -m mmag_vo.utils.print_paper_results \
  --json results/paper_reported_metrics.json
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/MMAG-VO.git
cd MMAG-VO

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

## Dataset Layout

The default configuration expects the following structure:

```text
DATA_ROOT/
  nyu_depth_v2/
    train.txt
    test.txt
    images/
    depths/
    labels/

  camvid/
    train.txt
    test.txt
    images/
    labels/
    depths/                 # optional, required only when depth supervision is available

  kitti_odometry/
    sequences/
      00/
        image_2/
      01/
        image_2/
      ...
    poses/
      00.txt
      01.txt
      ...
```

Depth files may be stored as `.npy` arrays or depth images, depending on the dataset preprocessing configuration.

## Train MMAG Depth and Segmentation Model

```bash
python -m mmag_vo.training.train_mmag \
  --config configs/mmag_nyu_camvid.yaml \
  --data-root /path/to/DATA_ROOT \
  --output runs/mmag_full
```

The MMAG model jointly optimizes depth estimation and semantic segmentation using the configured depth, segmentation, gradient, structural similarity, and Lovász-style objectives.

## Train CNN-GRU Visual Odometry Model

```bash
python -m mmag_vo.training.train_vo_gru \
  --config configs/vo_gru_kitti.yaml \
  --data-root /path/to/DATA_ROOT/kitti_odometry \
  --mmag-checkpoint runs/mmag_full/best.pt \
  --output runs/vo_gru
```

The CNN-GRU model uses intermediate MMAG features as input and predicts relative 6-DoF camera motion.

## Run Full Inference Pipeline

```bash
python -m mmag_vo.inference.run_pipeline \
  --image-dir /path/to/kitti_odometry/sequences/00/image_2 \
  --intrinsics configs/kitti_intrinsics_00.yaml \
  --mmag-checkpoint runs/mmag_full/best.pt \
  --vo-checkpoint runs/vo_gru/best.pt \
  --output outputs/seq00
```

Expected output structure:

```text
outputs/seq00/
  depth/
  depth_vis/
  seg/
  trajectory.txt
  per_frame_pose.npy
```

## ORB-SLAM2 Integration

The paper uses ORB-SLAM2 in RGB-D mode with predicted monocular depth. This repository includes a Python/OpenCV implementation, `DepthEnhancedORBVO`, for reproducible experimentation without requiring a compiled ORB-SLAM2 dependency.

For users who want to connect an external ORB-SLAM2 build, the repository also provides an adapter interface:

```text
src/mmag_vo/vo/external_orbslam2.py
```

External ORB-SLAM2 code should be placed outside the package source tree, for example:

```text
third_party/ORB_SLAM2/
```

## Repository Structure

```text
configs/                         Configuration files and camera intrinsics templates
results/                         Paper-reported reference metrics
scripts/                         Utility scripts
src/mmag_vo/data/                 Dataset loaders and transforms
src/mmag_vo/evaluation/           Evaluation entry points
src/mmag_vo/inference/            Full pipeline inference
src/mmag_vo/losses/               Depth, segmentation, and VO losses
src/mmag_vo/metrics/              Depth, segmentation, and odometry metrics
src/mmag_vo/models/mmag.py        MMAG depth and segmentation network
src/mmag_vo/models/vo_gru.py      CNN-GRU visual odometry network
src/mmag_vo/models/fusion.py      Pose fusion and Lie algebra utilities
src/mmag_vo/training/             Training entry points
src/mmag_vo/vo/orb_depth.py       Depth-enhanced ORB visual odometry
tests/                            Basic unit and integration tests
```

## Evaluation

Depth, segmentation, and odometry evaluation scripts are provided under:

```text
src/mmag_vo/evaluation/
```

Typical metrics include:

- Depth: AbsRel, SqRel, RMS, RMS-log, and threshold accuracy.
- Segmentation: mean Intersection over Union.
- Odometry: translational error, rotational error, and absolute trajectory error.

## License

This project is released under the MIT License. See `LICENSE` for details.

## Citation

If you use this implementation in academic work, cite the original paper and reference this repository as the implementation source.
