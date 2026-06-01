from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class ORBVOResult:
    transform: np.ndarray
    covariance: np.ndarray
    inliers: int
    matches: int
    status: str


class DepthEnhancedORBVO:
    """OpenCV implementation of the paper's depth-enhanced ORB tracking idea.

    This is a Python replacement for the paper's modified ORB-SLAM2 RGB-D mode.
    It extracts ORB features, matches descriptors, applies predicted-depth
    consistency, back-projects previous keypoints, and estimates relative motion
    with PnP-RANSAC.
    """

    def __init__(
        self,
        intrinsics: np.ndarray,
        n_features: int = 2000,
        desc_threshold: int = 50,
        depth_threshold: float = 0.1,
        min_depth: float = 0.1,
        max_depth: float = 80.0,
        ratio_test: float = 0.75,
    ) -> None:
        self.k = np.asarray(intrinsics, dtype=np.float64)
        self.orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.desc_threshold = desc_threshold
        self.depth_threshold = depth_threshold
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.ratio_test = ratio_test
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_depth: Optional[np.ndarray] = None
        self.prev_keypoints = None
        self.prev_desc = None

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_depth = None
        self.prev_keypoints = None
        self.prev_desc = None

    def _extract(self, image_rgb: np.ndarray):
        if image_rgb.ndim == 3:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_rgb
        keypoints, desc = self.orb.detectAndCompute(gray, None)
        return gray, keypoints, desc

    def _depth_at(self, depth: np.ndarray, kp: cv2.KeyPoint) -> float:
        u = int(round(kp.pt[0]))
        v = int(round(kp.pt[1]))
        if v < 0 or v >= depth.shape[0] or u < 0 or u >= depth.shape[1]:
            return float("nan")
        return float(depth[v, u])

    def _backproject(self, kp: cv2.KeyPoint, depth: float) -> np.ndarray:
        u, v = kp.pt
        fx, fy = self.k[0, 0], self.k[1, 1]
        cx, cy = self.k[0, 2], self.k[1, 2]
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        return np.array([x, y, z], dtype=np.float64)

    def _match_with_depth_check(self, keypoints, desc, depth):
        if self.prev_desc is None or desc is None or len(keypoints) < 8 or len(self.prev_keypoints) < 8:
            return [], np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
        raw = self.matcher.knnMatch(self.prev_desc, desc, k=2)
        good = []
        pts3d = []
        pts2d = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance > self.desc_threshold or m.distance >= self.ratio_test * n.distance:
                continue
            kp_prev = self.prev_keypoints[m.queryIdx]
            kp_cur = keypoints[m.trainIdx]
            d_prev = self._depth_at(self.prev_depth, kp_prev)
            d_cur = self._depth_at(depth, kp_cur)
            if not (self.min_depth < d_prev < self.max_depth and self.min_depth < d_cur < self.max_depth):
                continue
            if abs(d_prev - d_cur) > self.depth_threshold:
                continue
            good.append(m)
            pts3d.append(self._backproject(kp_prev, d_prev))
            pts2d.append(np.array(kp_cur.pt, dtype=np.float64))
        return good, np.asarray(pts3d, dtype=np.float64), np.asarray(pts2d, dtype=np.float64)

    def _covariance_from_residuals(self, pts3d, pts2d, rvec, tvec, inlier_mask) -> np.ndarray:
        base = np.diag([0.05, 0.05, 0.05, 0.01, 0.01, 0.01]).astype(np.float64)
        if inlier_mask is None or len(inlier_mask) == 0:
            return base * 100.0
        inliers = inlier_mask.reshape(-1)
        proj, _ = cv2.projectPoints(pts3d[inliers], rvec, tvec, self.k, None)
        proj = proj.reshape(-1, 2)
        residual = np.linalg.norm(proj - pts2d[inliers], axis=1)
        var = float(np.var(residual) + 1e-6)
        n = max(len(inliers), 1)
        cov = base * (var / n + 1e-5)
        return cov

    def process(self, image_rgb: np.ndarray, depth: np.ndarray) -> ORBVOResult:
        gray, keypoints, desc = self._extract(image_rgb)
        identity = np.eye(4, dtype=np.float64)
        if self.prev_gray is None:
            self.prev_gray, self.prev_depth, self.prev_keypoints, self.prev_desc = gray, depth, keypoints, desc
            return ORBVOResult(identity, np.eye(6) * 1e3, 0, 0, "initialized")

        matches, pts3d, pts2d = self._match_with_depth_check(keypoints, desc, depth)
        if len(matches) < 8:
            self.prev_gray, self.prev_depth, self.prev_keypoints, self.prev_desc = gray, depth, keypoints, desc
            return ORBVOResult(identity, np.eye(6) * 1e2, 0, len(matches), "not_enough_matches")

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d,
            pts2d,
            self.k,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=3.0,
            confidence=0.99,
            iterationsCount=100,
        )
        if not ok or inliers is None or len(inliers) < 6:
            self.prev_gray, self.prev_depth, self.prev_keypoints, self.prev_desc = gray, depth, keypoints, desc
            return ORBVOResult(identity, np.eye(6) * 1e2, 0, len(matches), "pnp_failed")

        try:
            cv2.solvePnPRefineLM(pts3d[inliers[:, 0]], pts2d[inliers[:, 0]], self.k, None, rvec, tvec)
        except Exception:
            pass
        rot, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rot
        transform[:3, 3] = tvec.reshape(3)
        cov = self._covariance_from_residuals(pts3d, pts2d, rvec, tvec, inliers[:, 0])

        self.prev_gray, self.prev_depth, self.prev_keypoints, self.prev_desc = gray, depth, keypoints, desc
        return ORBVOResult(transform, cov, int(len(inliers)), len(matches), "ok")
