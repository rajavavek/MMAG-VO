from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ExternalORBSLAM2Config:
    executable: str
    vocabulary: str
    settings_yaml: str
    work_dir: str | None = None


class ExternalORBSLAM2Adapter:
    """Thin adapter for a separately compiled modified ORB-SLAM2 binary.

    The paper uses a modified ORB-SLAM2 RGB-D pipeline. ORB-SLAM2 itself is not
    bundled here. This class is provided so users can connect their own compiled
    binary that accepts RGB image paths and predicted depth paths.
    """

    def __init__(self, config: ExternalORBSLAM2Config) -> None:
        self.config = config

    def run_sequence(self, rgb_dir: str, depth_dir: str, output_trajectory: str) -> None:
        cmd = [
            self.config.executable,
            self.config.vocabulary,
            self.config.settings_yaml,
            rgb_dir,
            depth_dir,
            output_trajectory,
        ]
        subprocess.run(cmd, cwd=self.config.work_dir, check=True)

    @staticmethod
    def load_trajectory(path: str | Path) -> np.ndarray:
        mats = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                vals = [float(v) for v in line.split()]
                if len(vals) == 12:
                    t = np.eye(4)
                    t[:3, :4] = np.array(vals).reshape(3, 4)
                    mats.append(t)
        return np.stack(mats, axis=0)
