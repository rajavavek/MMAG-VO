from __future__ import annotations

import argparse

from mmag_vo.metrics.odometry import absolute_trajectory_error, read_kitti_poses, simple_relative_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    args = parser.parse_args()
    pred = read_kitti_poses(args.pred)
    gt = read_kitti_poses(args.gt)
    print(simple_relative_errors(pred, gt))
    print({"ate_rmse_m": absolute_trajectory_error(pred, gt)})


if __name__ == "__main__":
    main()
