# Third-party ORB-SLAM2

The uploaded paper states that it uses a modified version of ORB-SLAM2 in RGB-D mode with predicted depth maps.

This zip does not vendor ORB-SLAM2. Use one of these options:

1. Use the included Python/OpenCV implementation:
   - `src/mmag_vo/vo/orb_depth.py`
2. Place your own compiled modified ORB-SLAM2 here:
   - `third_party/ORB_SLAM2/`
   - connect it through `src/mmag_vo/vo/orb_slam2_adapter.py`

Suggested external binary interface:

```bash
./rgbd_mmag_vo Vocabulary/ORBvoc.txt config.yaml /path/images /path/depth /path/output_trajectory.txt
```
