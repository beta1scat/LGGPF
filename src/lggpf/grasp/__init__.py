"""Grasp pose generation sub-package.

Generates candidate grasp poses for cuboid, truncated cone, and ellipsoid
primitives, then filters them for feasibility.
"""

from .pose import PickPose

__all__ = ["PickPose"]
