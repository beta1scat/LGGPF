"""Shape fitting sub-package.

Provides point cloud shape classification (PointNet2) and primitive fitting
(cuboid, truncated cone, ellipsoid) algorithms.
"""

from .classifier import PcdClassification
from .fitting import FittingByBGS
from .pointnet2 import get_model
from .pointnet2_utils import PointNetSetAbstraction

__all__ = [
    "PcdClassification",
    "FittingByBGS",
    "get_model",
    "PointNetSetAbstraction",
]
