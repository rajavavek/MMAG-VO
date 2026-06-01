# MMAG-VO Paper Code

This repository is a faithful, research-oriented implementation scaffold for the uploaded paper **“Model Generated Spatial and Contextual Information for Accurate Monocular Robot Odometry.”**

Implemented components match the paper pipeline:

1. **MMAG joint depth + semantic segmentation network**
   - Modified encoder/decoder backbone.
   - Shared Modified Multi-Attention Gates for depth and segmentation decoders.
   - Depth head, segmentation head, and intermediate MMAG feature output.
2. **Depth-enhanced ORB visual odometry**
   - Python/OpenCV RGB-D-style ORB tracker using predicted monocular depth.
   - Descriptor threshold and depth consistency filtering as described in the paper.
   - PnP-RANSAC pose estimation and covariance approximation.
3. **CNN-GRU visual odometry**
   - Uses intermediate MMAG features rather than raw RGB.
   - Five CNN blocks + two stacked GRU layers + 6-DoF pose regression head.
4. **Uncertainty-aware fusion**
   - Fuses ORB and GRU twist coordinates with covariance-weighted averaging.
5. **Training/evaluation/inference scripts**
   - NYU-Depth-v2, CamVid, and KITTI-style dataset loaders.
   - Losses and metrics matching the equations in the paper.

## Important reproducibility note

The uploaded PDF describes the architecture and reports results, but it does **not** include trained weights, exact train/val split files, all preprocessing details, or the modified C++ ORB-SLAM2 source. Therefore this zip contains executable code and paper-matched configuration, but it cannot include pretrained models that were not present in the paper. The folder `results/paper_reported_metrics.json` stores the paper-reported numbers as reference targets.

For exact numerical reproduction, train on the same dataset versions and preprocessing pipeline, then compare with:

```bash
python -m mmag_vo.utils.print_paper_results --json results/paper_reported_metrics.json
```

## Installation

```bash
cd mmag_vo_paper_code
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements.txt
```

## Dataset layout

The loaders accept configurable layouts, but the default config expects:

```text
DATA_ROOT/
  nyu_depth_v2/
    train.txt
    test.txt
    images/*.png
    depths/*.npy or depths/*.png
    labels/*.png
  camvid/
    train.txt
    test.txt
    images/*.png
    labels/*.png
    depths/*.npy or depths/*.png   # optional, only if depth supervision exists
  kitti_odometry/
    sequences/00/image_2/*.png
    sequences/01/image_2/*.png
    ...
    poses/00.txt
    poses/01.txt
```

## Train MMAG depth + segmentation

```bash
python -m mmag_vo.training.train_mmag \
  --config configs/mmag_nyu_camvid.yaml \
  --data-root /path/to/DATA_ROOT \
  --output runs/mmag_full
```

## Train CNN-GRU VO

```bash
python -m mmag_vo.training.train_vo_gru \
  --config configs/vo_gru_kitti.yaml \
  --data-root /path/to/DATA_ROOT/kitti_odometry \
  --mmag-checkpoint runs/mmag_full/best.pt \
  --output runs/vo_gru
```

## Run full inference

```bash
python -m mmag_vo.inference.run_pipeline \
  --image-dir /path/to/kitti_odometry/sequences/00/image_2 \
  --intrinsics configs/kitti_intrinsics_00.yaml \
  --mmag-checkpoint runs/mmag_full/best.pt \
  --vo-checkpoint runs/vo_gru/best.pt \
  --output outputs/seq00
```

Outputs:

```text
outputs/seq00/
  depth/*.npy
  depth_vis/*.png
  seg/*.png
  trajectory.txt
  per_frame_pose.npy
```

## External ORB-SLAM2 option

The paper uses a modified ORB-SLAM2 in RGB-D mode. This repository includes a Python/OpenCV replacement (`DepthEnhancedORBVO`) and an adapter stub (`ExternalORBSLAM2Adapter`) for users who have a compiled ORB-SLAM2 binary. Place external code separately under `third_party/ORB_SLAM2` if needed.

## Repository map

```text
configs/                         Hyperparameters and intrinsics templates
src/mmag_vo/models/mmag.py        MMAG model
src/mmag_vo/models/vo_gru.py      CNN-GRU visual odometry model
src/mmag_vo/vo/orb_depth.py       Predicted-depth ORB tracking
src/mmag_vo/models/fusion.py      Lie/twist fusion utilities
src/mmag_vo/losses/               Depth, segmentation, VO losses
src/mmag_vo/metrics/              Depth, segmentation, odometry metrics
src/mmag_vo/data/                 Dataset loaders and transforms
src/mmag_vo/training/             Training entry points
src/mmag_vo/evaluation/           Evaluation entry points
src/mmag_vo/inference/            Full pipeline inference
results/                          Paper-reported reference metrics
```
