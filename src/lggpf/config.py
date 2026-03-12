"""Configuration loader for LGGPF.

Loads pipeline parameters from a YAML file and provides typed access
via a nested dictionary.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from spatialmath import SE3


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML config file. If *None*, falls back to
              ``config/default.yaml`` relative to the project root (two
              levels above this source file).

    Returns:
        Parsed configuration dictionary.
    """
    if path is None:
        # Default: <project_root>/config/default.yaml
        project_root = Path(__file__).resolve().parents[2]
        path = project_root / "config" / "default.yaml"
    else:
        path = Path(path)

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    return cfg


def _matrix_to_se3(matrix: list[list[float]] | np.ndarray) -> SE3:
    """Convert a 4x4 homogeneous matrix to an SE3 object.

    Some ``spatialmath`` versions do not accept a raw 4x4 numpy array
    directly in the ``SE3(...)`` constructor. Building it explicitly from
    rotation and translation is more robust across versions.
    """
    arr = np.array(matrix, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix, got shape {arr.shape}")

    # The calibration matrices in the original code are used directly as SE3
    # transforms, but some spatialmath versions require a strictly valid
    # rotation matrix. Project the 3x3 block to the nearest orthonormal
    # rotation matrix so small calibration noise does not break loading.
    rotation = arr[:3, :3]
    u, _, vh = np.linalg.svd(rotation)
    rotation_ortho = u @ vh
    if np.linalg.det(rotation_ortho) < 0:
        u[:, -1] *= -1
        rotation_ortho = u @ vh

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_ortho
    transform[:3, 3] = arr[:3, 3]
    return SE3(transform, check=False)


def get_calibration_matrices(cfg: dict[str, Any]) -> tuple[SE3, SE3]:
    """Build SE3 calibration matrices from the config dict.

    Returns:
        (T_ET, T_BC) – end-effector-to-tool and base-to-camera transforms.
    """
    T_ET = _matrix_to_se3(cfg["calibration"]["T_ET"])
    T_BC = _matrix_to_se3(cfg["calibration"]["T_BC"])
    return T_ET, T_BC


def get_camera_intrinsics(cfg: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) from the config dict."""
    intr = cfg["camera"]["intrinsics"]
    return intr["fx"], intr["fy"], intr["cx"], intr["cy"]
