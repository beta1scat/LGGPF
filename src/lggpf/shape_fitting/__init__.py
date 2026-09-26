"""Shape fitting sub-package.

Provides point cloud shape classification (Mamba3D / PointNet2) and primitive fitting
(cuboid, truncated cone, ellipsoid) algorithms.
"""

from .classifier import ShapeClassifier, PcdClassification
from .fitting import (
    FittingByBGS,
    _oriented_bounding_box,
    align_vector_to_z,
    fit_cuboid_obb2,
    fit_cuboid_obb,
    fit_frustum_cone_adaptive,
    fit_frustum_cone_pca,
    fit_frustum_cone_normal,
    fit_frustum_cone_obb,
    fit_ellipsoid,
    compute_cone_residual,
)
from .pointnet2 import get_model
from .pointnet2_utils import PointNetSetAbstraction

__all__ = [
    "ShapeClassifier",
    "PcdClassification",
    "FittingByBGS",
    "_oriented_bounding_box",
    "align_vector_to_z",
    "fit_cuboid_obb2",
    "fit_cuboid_obb",
    "fit_frustum_cone_adaptive",
    "fit_frustum_cone_pca",
    "fit_frustum_cone_normal",
    "fit_frustum_cone_obb",
    "fit_ellipsoid",
    "compute_cone_residual",
    "get_model",
    "PointNetSetAbstraction",
]
