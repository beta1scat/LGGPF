"""
Basic geometric shape fitting algorithms.

Implements fitting of three primitive types to 3D point clouds:
  - **Cuboid**: via RANSAC plane segmentation + OBB, or OBB-only
  - **Truncated cone (frustum)**: via normal clustering, PCA-based axis, or OBB
  - **Ellipsoid**: via RANSAC + least-squares algebraic fitting

Each fitting function returns shape parameters and a rigid transform (SE3)
that maps the canonical shape frame to the world frame.

Merged from the original ``shape_fitting_bgs.py`` and ``fit_bgspcd_noros.py``.
"""

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares, nnls
from scipy.spatial.transform import Rotation
from spatialmath import SO3, SE3
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from ..utils.pointcloud import (
    CircleLeastSquaresModel,
    ConeAxisLeastSquaresModel,
    EllipsoidLeastSquaresModel,
    NormalLeastSquaresModel,
    ransac,
    fit_circle,
    fit_circle_kasa,
    find_orthogonal_vectors,
    pc_normalize,
    generate_cone_points,
    generate_ellipsoid_points,
)


# =============================================================================
# Helper functions
# =============================================================================


def align_vector_to_z(v):
    """Build a rotation matrix that aligns the Z axis to vector ``v``.

    Args:
        v: Target direction vector (3,).

    Returns:
        (3, 3) rotation matrix.
    """
    v = np.array(v, dtype=np.float64)
    v = v / np.linalg.norm(v)

    # Choose an auxiliary vector not collinear with v
    if np.abs(v[0]) < np.abs(v[1]):
        aux = np.array([1, 0, 0], dtype=np.float64)
    else:
        aux = np.array([0, 1, 0], dtype=np.float64)

    x_axis = np.cross(aux, v)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(v, x_axis)

    return np.column_stack((x_axis, y_axis, v))


def _oriented_bounding_box(pcd):
    """Return the tightest OBB supported by the installed Open3D build.

    Some CUDA-enabled Open3D builds expose ``get_oriented_bounding_box`` but
    not ``get_minimal_oriented_bounding_box``.  The fitting code must support
    both APIs; this is a compatibility choice, not a change of fitting target.
    """
    minimal = getattr(pcd, "get_minimal_oriented_bounding_box", None)
    if minimal is not None:
        return minimal()
    oriented = getattr(pcd, "get_oriented_bounding_box", None)
    if oriented is not None:
        return oriented()
    raise AttributeError("Installed Open3D PointCloud exposes no oriented bounding-box API")


def get_plane_axis(point_cloud, half_dim):
    """Determine which axis is the cone's symmetry (plane) axis.

    Evaluates which pair of bounding-box corners has the closest points,
    indicating the axis with circular cross-sections.

    Args:
        point_cloud: Open3D point cloud.
        half_dim: Half-extents [x, y, z] of the bounding box.

    Returns:
        Axis index (0, 1, or 2) most likely to be the symmetry axis.
    """
    pts = np.asarray(point_cloud.points)
    dist_sum_list = []
    for idx in range(3):
        idx1, idx2 = [i for i in range(3) if i != idx]
        corner = np.array(
            [
                [half_dim[idx1], half_dim[idx2]],
                [-half_dim[idx1], -half_dim[idx2]],
                [-half_dim[idx1], half_dim[idx2]],
                [half_dim[idx1], -half_dim[idx2]],
            ]
        )
        pts2d = np.array([pts[:, idx1], pts[:, idx2]])
        dist_sum = 0
        for i in range(4):
            min_corner = min(
                np.linalg.norm(pts2d - corner[i].reshape(2, 1), ord=2, axis=0)
            )
            # Avoid outlier corners
            if min_corner > half_dim[idx1] or min_corner > half_dim[idx2]:
                min_corner = 0
            dist_sum += min_corner
        dist_sum_list.append(dist_sum)
    return np.argmax(dist_sum_list)


def get_max_num_cluster(pcd, eps=0.1, min_points=10, print_progress=True):
    """Extract the largest DBSCAN cluster from a point cloud.

    Args:
        pcd: Open3D point cloud.
        eps: DBSCAN neighborhood radius.
        min_points: Minimum cluster size.
        print_progress: Show DBSCAN progress.

    Returns:
        Open3D point cloud of the largest cluster.
    """
    labels = np.array(
        pcd.cluster_dbscan(
            eps=eps, min_points=min_points, print_progress=print_progress
        )
    )
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_cluster_label = unique_labels[np.argmax(counts)]
    max_cluster_indices = np.where(labels == max_cluster_label)[0]
    return pcd.select_by_index(max_cluster_indices)


def _get_adjust_transform(pts, num_points, half_dim, plane_axis, pcd):
    """Compute a correction transform for OBB-based cone fitting.

    Fits circles to the top and bottom cross-sections and adjusts the
    coordinate frame to align with the cone axis.

    Args:
        pts: (N, 3) point array.
        num_points: Number of points.
        half_dim: Half-extents of the bounding box.
        plane_axis: Index of the symmetry axis.
        pcd: Open3D point cloud.

    Returns:
        Tuple of (T, top_r, bottom_r).
    """
    idx1, idx2 = [i for i in range(3) if i != plane_axis]

    # Select top and bottom slices
    top_idx = [
        i for i in range(num_points) if pts[i][plane_axis] > 0.9 * half_dim[plane_axis]
    ]
    if len(top_idx) < 500:
        top_idx = np.argsort(pts[:, plane_axis])[-500:]
    bottom_idx = [
        i for i in range(num_points) if pts[i][plane_axis] < -0.9 * half_dim[plane_axis]
    ]
    if len(bottom_idx) < 500:
        bottom_idx = np.argsort(pts[:, plane_axis])[:500]

    top_pcd = pcd.select_by_index(top_idx)
    top_pts = np.asarray(top_pcd.points)
    bottom_pcd = pcd.select_by_index(bottom_idx)
    bottom_pts = np.asarray(bottom_pcd.points)

    # Fit circles on top and bottom slices
    top_cluster_pcd = get_max_num_cluster(top_pcd, 0.1, 10, True)
    top_cluster_pts = np.asarray(top_cluster_pcd.points)
    bottom_cluster_pcd = get_max_num_cluster(bottom_pcd, 0.1, 10, True)
    bottom_cluster_pts = np.asarray(bottom_cluster_pcd.points)

    top_circle, _, _ = fit_circle(top_cluster_pts[:, [idx1, idx2]], 1000, 0.01)
    bottom_circle, _, _ = fit_circle(bottom_cluster_pts[:, [idx1, idx2]], 1000, 0.01)

    # Determine slice size ratios
    top_size_1 = abs(np.max(top_pts[:, idx1]) - np.min(top_pts[:, idx1]))
    top_size_2 = abs(np.max(top_pts[:, idx2]) - np.min(top_pts[:, idx2]))
    top_size_ratio = (
        top_size_1 / top_size_2 if top_size_1 < top_size_2 else top_size_2 / top_size_1
    )
    top_ratio_1 = top_size_1 / (2 * half_dim[idx1])
    top_ratio_2 = top_size_2 / (2 * half_dim[idx2])

    bottom_size_1 = abs(np.max(bottom_pts[:, idx1]) - np.min(bottom_pts[:, idx1]))
    bottom_size_2 = abs(np.max(bottom_pts[:, idx2]) - np.min(bottom_pts[:, idx2]))
    bottom_ratio_1 = bottom_size_1 / (2 * half_dim[idx1])
    bottom_ratio_2 = bottom_size_2 / (2 * half_dim[idx2])

    # Use fitted circle centers
    center_top = top_circle[0]
    center_bottom = bottom_circle[0]

    top_center = [0, 0, 0]
    top_center[plane_axis] = half_dim[plane_axis]
    top_center[idx1] = center_top[0]
    top_center[idx2] = center_top[1]

    bottom_center = [0, 0, 0]
    bottom_center[plane_axis] = -1.0 * half_dim[plane_axis]
    bottom_center[idx1] = center_bottom[0]
    bottom_center[idx2] = center_bottom[1]

    # Build correction transform
    if (
        abs(top_ratio_1 / top_ratio_2 - 1) < 0.01
        and abs(bottom_ratio_1 / bottom_ratio_2 - 1) < 0.01
    ):
        T = SE3.Tx(0)
    else:
        v1 = np.asarray(top_center) - np.asarray(bottom_center)
        v1 = v1 / np.linalg.norm(v1)
        v2, v3 = find_orthogonal_vectors(v1)
        mid = 0.5 * (np.asarray(top_center) + np.asarray(bottom_center))
        if plane_axis == 0:
            T = SE3.Rt(SO3.TwoVectors(x=v1, y=v2), mid)
        elif plane_axis == 1:
            T = SE3.Rt(SO3.TwoVectors(x=v2, y=v1), mid)
        else:
            T = SE3.Rt(SO3.TwoVectors(x=v2, z=v1), mid)

    # Compute radii from circle fits
    top_circle_r = max(
        np.linalg.norm(
            top_pts[:, [idx1, idx2]] - np.array(top_circle[0]), ord=2, axis=1
        )
    )
    bottom_circle_r = max(
        np.linalg.norm(
            bottom_pts[:, [idx1, idx2]] - np.array(bottom_circle[0]), ord=2, axis=1
        )
    )

    max_half = max(half_dim)
    if (
        top_circle_r > max(top_size_1, top_size_2) / 2
        and top_circle[1] < max_half * 1.2
    ):
        top_r = top_circle_r
    else:
        top_r = max(top_size_1, top_size_2) / 2
    if (
        bottom_circle_r > max(bottom_size_1, bottom_size_2) / 2
        and bottom_circle[1] < max_half * 1.2
    ):
        bottom_r = bottom_circle_r
    else:
        bottom_r = max(bottom_size_1, bottom_size_2) / 2

    return T, top_r, bottom_r


# =============================================================================
# Slice-based cone radius estimation
# =============================================================================


def fit_frustum_cone_by_slice_linear(points, num_layers=10, pcd=None, is_debug=False):
    """Estimate cone radii by slicing along the Z axis and fitting circles per layer.

    Uses RANSAC linear regression on per-layer radii to get top/bottom radius.

    Args:
        points: (N, 3) point array (already aligned so Z is the cone axis).
        num_layers: Number of horizontal slices.
        pcd: Optional Open3D point cloud (for debug visualization).
        is_debug: Show debug plots.

    Returns:
        Tuple of (r1, r2, height, center) where r1 is the top radius,
        r2 is the bottom radius, and center is the [x, y, z] center.
        Returns None if fitting fails.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("Frustum fitting requires at least four finite XYZ points")
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 4:
        raise ValueError("Frustum fitting received fewer than four finite XYZ points")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    # Robust quantile bounds (0.5% - 99.5%) prevent outliers from artificially inflating height
    z_min = float(np.quantile(z, 0.005))
    z_max = float(np.quantile(z, 0.995))
    height = z_max - z_min
    if height <= np.finfo(np.float64).eps:
        raise ValueError("Frustum fitting received a degenerate axial extent")

    layer_pts = []
    for i in range(num_layers):
        layer_min = z_min + i * height / num_layers
        layer_max = z_min + (i + 1) * height / num_layers
        # The closed upper bound on the last layer prevents the axial endpoint
        # from being silently discarded.
        mask = (z >= layer_min) & ((z < layer_max) if i + 1 < num_layers else (z <= layer_max))
        layer_pts.append(points[mask])

    # Fit circle to each layer using fast closed-form Kåsa algebraic fit
    layer_radius = {}
    layer_center = {}
    for layer_idx in range(num_layers):
        layer_idx_pts = layer_pts[layer_idx]
        if len(layer_idx_pts) < 4:
            continue
        circle_res = fit_circle_kasa(layer_idx_pts[:, :2])
        if circle_res is not None:
            c, r = circle_res
            if r > 0.001:  # Physical positive radius
                layer_radius[layer_idx] = r
                layer_center[layer_idx] = c

    if len(layer_radius) < max(3, int(np.ceil(0.3 * num_layers))):
        # A partial camera view can leave too few slices for independent
        # circle fits.  Estimate a conservative circular profile rather than
        # returning None and letting the dispatcher count a Python error as a
        # geometric fitting failure.
        center_xy = np.median(points[:, :2], axis=0)
        radius = np.linalg.norm(points[:, :2] - center_xy, axis=1)
        lower = radius[z <= np.quantile(z, 0.25)]
        upper = radius[z >= np.quantile(z, 0.75)]
        valid_radius = radius[np.isfinite(radius) & (radius > 1e-6)]
        if not len(valid_radius):
            raise ValueError("Frustum fitting could not estimate a positive radial profile")
        fallback_radius = float(np.quantile(valid_radius, 0.75))
        r2 = float(np.median(lower)) if len(lower) else fallback_radius
        r1 = float(np.median(upper)) if len(upper) else fallback_radius
        min_physical_r = max(1e-4, fallback_radius * 0.05)
        return max(r1, min_physical_r), max(r2, min_physical_r), height, np.array([
            center_xy[0], center_xy[1], (z_min + z_max) / 2,
        ])

    # RANSAC linear regression on layer radii
    ransac_reg = linear_model.RANSACRegressor(random_state=0)
    X_fit = np.array(list(layer_radius.keys()))[:, np.newaxis]
    y_fit = np.array(list(layer_radius.values()))
    try:
        ransac_reg.fit(X_fit, y_fit)
        inlier_mask = np.flatnonzero(ransac_reg.inlier_mask_)
        line_y = ransac_reg.predict(np.arange(num_layers)[:, np.newaxis])
    except ValueError:
        # Deterministic least-squares is preferable to an exception when the
        # visible portion supplies an underdetermined RANSAC consensus set.
        regressor = LinearRegression().fit(X_fit, y_fit)
        inlier_mask = np.arange(len(y_fit))
        line_y = regressor.predict(np.arange(num_layers)[:, np.newaxis])
    min_physical_r = max(1e-4, float(np.min(y_fit)) * 0.05) if len(y_fit) > 0 else 1e-4
    r2 = max(float(line_y[0]), min_physical_r)  # bottom (min Z)
    r1 = max(float(line_y[-1]), min_physical_r)  # top (max Z)

    center_arr = np.array(list(layer_center.values()), dtype=np.float64)
    if not len(inlier_mask):
        inlier_mask = np.arange(len(center_arr))
    center = np.array(
        [
            np.mean(center_arr[inlier_mask, 0]),
            np.mean(center_arr[inlier_mask, 1]),
            (z_min + z_max) / 2,
        ]
    )

    return r1, r2, height, center


def fit_frustum_cone_by_slice_poly(points, num_layers=10):
    """Estimate cone radii using polynomial RANSAC regression on layer radii.

    Similar to :func:`fit_frustum_cone_by_slice_linear` but uses degree-2
    polynomial regression for better fit on tapered shapes.

    Args:
        points: (N, 3) point array.
        num_layers: Number of horizontal slices.

    Returns:
        Tuple of (r1, r2, height, center).
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("Frustum fitting requires at least four XYZ points")
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    z_min = float(np.quantile(z, 0.005))
    z_max = float(np.quantile(z, 0.995))
    height = z_max - z_min
    if height <= np.finfo(np.float64).eps:
        raise ValueError("Frustum fitting received a degenerate axial extent")

    layer_pts = []
    for i in range(num_layers):
        layer_min = z_min + i * height / num_layers
        layer_max = z_min + (i + 1) * height / num_layers
        mask = (z >= layer_min) & ((z < layer_max) if i + 1 < num_layers else (z <= layer_max))
        layer_pts.append(points[mask])

    layer_radius = {}
    layer_center = {}
    for layer_idx in range(num_layers):
        layer_idx_pts = layer_pts[layer_idx]
        if len(layer_idx_pts) < 4:
            continue
        circle_res = fit_circle_kasa(layer_idx_pts[:, :2])
        if circle_res is not None:
            c, r = circle_res
            if r > 0.001:
                layer_radius[layer_idx] = r
                layer_center[layer_idx] = c

    if len(layer_radius) < max(3, int(np.ceil(0.3 * num_layers))):
        return fit_frustum_cone_by_slice_linear(points, num_layers)

    X_fit = np.array(list(layer_radius.keys()))[:, np.newaxis]
    y_fit = np.array(list(layer_radius.values()))
    model = make_pipeline(
        PolynomialFeatures(degree=2),
        RANSACRegressor(LinearRegression(), random_state=0),
    )
    try:
        model.fit(X_fit, y_fit.ravel())
    except ValueError:
        return fit_frustum_cone_by_slice_linear(points, num_layers)
    ransac_model = model.named_steps["ransacregressor"]
    inlier_mask = ransac_model.inlier_mask_

    line_x = np.array(range(num_layers))[:, np.newaxis]
    line_y = model.predict(line_x)
    min_physical_r = max(1e-4, float(np.min(y_fit)) * 0.05) if len(y_fit) > 0 else 1e-4
    r2 = max(float(line_y[0]), min_physical_r)
    r1 = max(float(line_y[-1]), min_physical_r)

    center_arr = np.array(list(layer_center.values()))
    center = np.array(
        [
            np.mean(center_arr[inlier_mask, 0]),
            np.mean(center_arr[inlier_mask, 1]),
            (z_min + z_max) / 2,
        ]
    )
    return r1, r2, height, center




# =============================================================================
# Core fitting functions
# =============================================================================


def fit_cuboid_obb2(pcd, dist_threshold=None, n=3, num_it=500, is_debug=False):
    """Fit a cuboid using RANSAC plane segmentation + Oriented Bounding Box.

    First segments the dominant plane via RANSAC, then aligns with the plane's
    local frame and calculates robust bounded extents and centers across all points.

    Args:
        pcd: Open3D point cloud.
        dist_threshold: RANSAC plane distance threshold.
        n: Minimum points for plane fit.
        num_it: RANSAC iterations.
        is_debug: Show debug visualization.

    Returns:
        Tuple of (a, b, c, T) where a, b, c are half-extents and
        T is an SE3 transform of the cuboid center and orientation.
    """
    pts = np.asarray(pcd.points, dtype=np.float64)
    diagonal = float(np.linalg.norm(np.ptp(pts, axis=0)))
    if dist_threshold is None:
        dist_threshold = max(diagonal * 0.015, 1e-4)

    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=dist_threshold, ransac_n=n, num_iterations=num_it
        )
    except Exception:
        return fit_cuboid_obb(pcd)

    if len(inliers) < 20:
        return fit_cuboid_obb(pcd)

    plane_cloud = pcd.select_by_index(inliers)
    obb_plane = _oriented_bounding_box(plane_cloud)
    R_plane = np.asarray(obb_plane.R, dtype=np.float64)

    # Project entire point cloud into plane coordinate frame
    pts_local = (pts - np.asarray(obb_plane.center, dtype=np.float64)) @ R_plane

    # Robust quantile filtering (0.5% - 99.5%) to reject sensor depth splatter
    min_b = np.quantile(pts_local, 0.005, axis=0)
    max_b = np.quantile(pts_local, 0.995, axis=0)
    extent = np.maximum(max_b - min_b, max(diagonal * 0.02, 1e-3))
    half_extent = extent / 2.0

    local_center = (min_b + max_b) / 2.0
    cube_center = np.asarray(obb_plane.center, dtype=np.float64) + R_plane @ local_center
    T = SE3.Rt(SO3(R_plane), cube_center)

    if is_debug:
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        coord.transform(T)
        o3d.visualization.draw_geometries([pcd, obb_plane, coord])

    return float(half_extent[0]), float(half_extent[1]), float(half_extent[2]), T


def fit_cuboid_obb(pcd):
    """Fit a cuboid using only Oriented Bounding Box with robust quantile bounds.

    Args:
        pcd: Open3D point cloud.

    Returns:
        Tuple of (a, b, c, T) where a, b, c are half-extents and
        T is an SE3 transform.
    """
    obb = _oriented_bounding_box(pcd)
    R = np.asarray(obb.R, dtype=np.float64)
    center = np.asarray(obb.center, dtype=np.float64)
    pts = np.asarray(pcd.points, dtype=np.float64)

    # Project points to local OBB frame
    pts_local = (pts - center) @ R

    # Robust quantile bounding
    min_b = np.quantile(pts_local, 0.005, axis=0)
    max_b = np.quantile(pts_local, 0.995, axis=0)
    extent = np.maximum(max_b - min_b, 1e-3)
    half_extent = extent / 2.0

    local_center = (min_b + max_b) / 2.0
    cube_center = center + R @ local_center
    T = SE3.Rt(SO3(R), cube_center)

    return float(half_extent[0]), float(half_extent[1]), float(half_extent[2]), T


def fit_frustum_cone_normal(
    pcd,
    use_poly=False,
    plane_t=0.001,
    normal_t=0.02,
    use_plane_normal=True,
    is_debug=False,
):
    """Fit a truncated cone using surface normal clustering to find the axis.

    When ``use_plane_normal=True``, performs K-means clustering on normals,
    then picks the cluster whose points best fit a plane (highest inlier ratio).
    The plane normal becomes the cone axis.

    When ``use_plane_normal=False``, uses RANSAC with ConeAxisLeastSquaresModel
    to find the axis that minimizes angle variance among normals.

    Args:
        pcd: Open3D point cloud (must have normals estimated).
        use_poly: Use polynomial regression for radius estimation.
        plane_t: Plane segmentation distance threshold.
        normal_t: Normal RANSAC threshold.
        use_plane_normal: Whether to use plane-based or RANSAC-based axis.
        is_debug: Show debug visualization.

    Returns:
        Tuple of (r1, r2, height, T) where r1/r2 are top/bottom radii
        (in original scale), height is in original scale, and T is the SE3 transform.
    """
    pcd_normalized = o3d.geometry.PointCloud(pcd)
    pts, m, centroid = pc_normalize(np.asarray(pcd.points))
    pcd_normalized.points = o3d.utility.Vector3dVector(pts)
    points = np.asarray(pcd_normalized.points)
    normals = np.asarray(pcd_normalized.normals) if pcd_normalized.has_normals() else np.empty((0, 3))
    cov = np.cov(points, rowvar=False)
    pca_vals, pca_vecs = np.linalg.eigh(cov)
    axis_candidates = [pca_vecs[:, 2], pca_vecs[:, 0]]

    if not use_plane_normal and len(normals) >= 10:
        try:
            cone_axis_model = ConeAxisLeastSquaresModel()
            best_fit, _ = ransac(
                normals,
                cone_axis_model,
                10,
                200,
                normal_t,
                1,
                inliers_ratio=0.85,
                debug=False,
                return_all=True,
            )
            if best_fit is not None:
                axis_candidates.insert(0, best_fit[0])
        except Exception:
            pass

    def _align_mat_to_z(axis):
        v = axis / np.linalg.norm(axis)
        vx = np.cross(v, [0.0, 0.0, 1.0])
        if np.linalg.norm(vx) < 1e-4:
            vx = np.cross(v, [0.0, 1.0, 0.0])
        vx /= np.linalg.norm(vx)
        vy = np.cross(v, vx)
        return np.column_stack((vx, vy, v))

    best_score = -1.0
    best_cone_normal = axis_candidates[0]
    best_R = _align_mat_to_z(best_cone_normal)

    for cand in axis_candidates:
        R_cand = _align_mat_to_z(cand)
        pts_cand = points @ R_cand
        z = pts_cand[:, 2]
        z_min = float(np.quantile(z, 0.005))
        z_max = float(np.quantile(z, 0.995))
        h = z_max - z_min
        if h <= 1e-4:
            continue
        valid_slices = 0
        for i in range(12):
            mask = (z >= z_min + i * h / 12) & (z <= z_min + (i + 1) * h / 12)
            if np.count_nonzero(mask) >= 5:
                valid_slices += 1
        if valid_slices > best_score:
            best_score = valid_slices
            best_cone_normal = cand
            best_R = R_cand

    pcd_normalized.rotate(best_R.T, center=[0, 0, 0])
    pts_rotated = np.asarray(pcd_normalized.points)

    if use_poly:
        r1, r2, height, center = fit_frustum_cone_by_slice_poly(pts_rotated, 12)
    else:
        r1, r2, height, center = fit_frustum_cone_by_slice_linear(pts_rotated, 12, pcd)

    world_center = centroid + m * (best_R @ np.asarray(center))
    T = SE3.Rt(SO3(best_R), world_center)
    return r1 * m, r2 * m, height * m, T


def fit_frustum_cone_pca(pcd, use_poly=False, is_debug=False, z_dir=0, num_layers=12):
    """Fit a truncated cone using PCA to find the symmetry axis."""
    pcd_normalized = o3d.geometry.PointCloud(pcd)
    pts, m, centroid = pc_normalize(np.asarray(pcd.points))
    pcd_normalized.points = o3d.utility.Vector3dVector(pts)
    points = np.asarray(pcd_normalized.points)

    pca = PCA(n_components=3)
    pca.fit(points)
    cone_normal = pca.components_[z_dir]

    v = cone_normal / np.linalg.norm(cone_normal)
    vx = np.cross(v, [0.0, 0.0, 1.0])
    if np.linalg.norm(vx) < 1e-4:
        vx = np.cross(v, [0.0, 1.0, 0.0])
    vx /= np.linalg.norm(vx)
    vy = np.cross(v, vx)
    R_mat = np.column_stack((vx, vy, v))

    pcd_normalized.rotate(R_mat.T, center=[0, 0, 0])
    pts_rotated = np.asarray(pcd_normalized.points)

    if use_poly:
        r1, r2, height, center = fit_frustum_cone_by_slice_poly(pts_rotated, num_layers)
    else:
        r1, r2, height, center = fit_frustum_cone_by_slice_linear(pts_rotated, num_layers, pcd)

    world_center = centroid + m * (R_mat @ np.asarray(center))
    T = SE3.Rt(SO3(R_mat), world_center)
    return r1 * m, r2 * m, height * m, T


def fit_frustum_cone_obb(pcd):
    """Fit a truncated cone using the Oriented Bounding Box method.

    Uses OBB to establish the initial frame, determines the symmetry axis,
    then fits circles on top/bottom slices.

    Args:
        pcd: Open3D point cloud.

    Returns:
        Tuple of (top_r, bottom_r, height, T).
    """
    pcd_fit = o3d.geometry.PointCloud()
    pcd_fit.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    pts = np.asarray(pcd_fit.points)
    obb = _oriented_bounding_box(pcd_fit)
    T_obb = SE3.Rt(obb.R, obb.center)
    pcd_fit.transform(T_obb.inv())

    half_dim = [max(pts[:, 0]), max(pts[:, 1]), max(pts[:, 2])]
    plane_axis = get_plane_axis(pcd_fit, half_dim)
    if plane_axis == -1:
        print("Error: could not determine plane axis direction")
        return SE3(), 0, 0
    height = half_dim[plane_axis] * 2
    _, top_r, bottom_r = _get_adjust_transform(
        pts, pts.shape[0], half_dim, plane_axis, pcd_fit
    )
    return top_r, bottom_r, height, T_obb



def _rotvec_to_mat(v):
    theta = float(np.linalg.norm(v))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    k = v / theta
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _ellipsoid_geometric_residual(parameters, points):
    center = parameters[:3]
    axes = np.exp(parameters[3:6])
    rotation = _rotvec_to_mat(parameters[6:])
    local = (points - center) @ rotation

    # Taubin metric distance approximation: f(x) / ||grad f(x)||
    val = np.sum((local / axes) ** 2, axis=1) - 1.0
    grad_norm = 2.0 * np.sqrt(np.sum((local / (axes ** 2)) ** 2, axis=1))
    dist = val / np.maximum(grad_norm, 1e-6)

    # Soft regularization on extreme elongation: penalize aspect ratio > 3.0
    aspect = np.max(axes) / np.maximum(np.min(axes), 1e-6)
    reg_aspect = np.maximum(0.0, aspect - 3.0) * 0.05

    return np.append(dist, reg_aspect)


def _fit_ellipsoid_geometric(points):
    """Fast & robust geometric ellipsoid fit for partial, contaminated clouds."""
    pts = np.asarray(points, dtype=np.float64)
    diagonal = float(np.linalg.norm(np.ptp(pts, axis=0)))
    if not np.isfinite(diagonal) or diagonal <= 1e-8:
        return None

    # Fast uniform subsampling to accelerate PCA and optimization
    if len(pts) > 250:
        step = max(1, len(pts) // 250)
        pts_opt = pts[::step][:250]
    else:
        pts_opt = pts

    centroid = np.mean(pts_opt, axis=0)
    cov = np.cov(pts_opt, rowvar=False)
    pca_vals, pca_vecs = np.linalg.eigh(cov)
    if np.linalg.det(pca_vecs) < 0:
        pca_vecs[:, 0] = -pca_vecs[:, 0]

    # Level 1: Closed-form PCA-axial Non-Negative Least Squares (~1-5 ms)
    pts_pca = (pts_opt - centroid) @ pca_vecs
    min_idx = int(np.argmin(pca_vals))
    depth_extent = float(np.ptp(pts_pca[:, min_idx]))
    local_c = np.zeros(3, dtype=np.float64)
    local_c[min_idx] = -0.2 * depth_extent
    local_shifted = pts_pca - local_c

    M = local_shifted ** 2
    y = np.ones(len(pts_opt), dtype=np.float64)
    try:
        coef, _ = nnls(M, y)
    except Exception:
        coef = np.zeros(3)

    if np.all(coef > 1e-8):
        nnls_axes = 1.0 / np.sqrt(coef)
        nnls_aspect = float(np.max(nnls_axes) / np.min(nnls_axes))
        if nnls_aspect <= 3.5 and np.max(nnls_axes) <= 1.5 * diagonal and np.min(nnls_axes) >= diagonal * 0.02:
            world_c = centroid + pca_vecs @ local_c
            return (float(nnls_axes[0]), float(nnls_axes[1]), float(nnls_axes[2]), SE3.Rt(SO3(pca_vecs), world_c))

    # Level 2: Fast bounded Levenberg-Marquardt / TRF on <= 150 points (~30-50 ms)
    if len(pts_opt) > 150:
        pts_trf = pts_opt[::2]
    else:
        pts_trf = pts_opt

    extent = np.ptp(pts_pca, axis=0)
    initial_axes = np.maximum(extent / 2.0, diagonal * 0.05)

    rotvec = Rotation.from_matrix(pca_vecs).as_rotvec()
    pushed_center = centroid + pca_vecs @ local_c
    starts = [
        np.concatenate((centroid, np.log(initial_axes), rotvec)),
        np.concatenate((pushed_center, np.log(initial_axes), rotvec)),
    ]

    start_errors = [
        float(np.median(np.abs(_ellipsoid_geometric_residual(s, pts_trf)[:-1])))
        for s in starts
    ]
    best_initial = starts[int(np.argmin(start_errors))]

    lower = np.concatenate((pts.min(axis=0) - 0.5 * diagonal, np.full(3, np.log(diagonal * 0.02)), np.full(3, -np.pi)))
    upper = np.concatenate((pts.max(axis=0) + 0.5 * diagonal, np.full(3, np.log(diagonal * 1.5)), np.full(3, np.pi)))
    scale = max(diagonal * 0.02, 1e-4)

    try:
        result = least_squares(
            _ellipsoid_geometric_residual,
            best_initial,
            args=(pts_trf,),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=scale,
            max_nfev=40,
            ftol=1e-2,
            xtol=1e-2,
        )
    except Exception:
        return None

    parameters = result.x
    axes = np.exp(parameters[3:6])
    rotation = _rotvec_to_mat(parameters[6:])
    if not np.all(np.isfinite(axes)) or np.any(axes <= 0.0):
        return None

    return (float(axes[0]), float(axes[1]), float(axes[2]), SE3.Rt(SO3(rotation), parameters[:3]))


def fit_ellipsoid(pcd, num_it=100, t=0.015, return_details=False):
    """Fit an ellipsoid using RANSAC + algebraic least-squares with fast geometric fallback.

    Args:
        pcd: Open3D point cloud.
        num_it: RANSAC iterations.
        t: RANSAC inlier threshold (in normalized coordinates).
        return_details: If True, returns (a, b, c, T, is_fallback).

    Returns:
        Tuple of (a, b, c, T) or (a, b, c, T, is_fallback). Returns None if fitting fails.
    """
    raw_points = np.asarray(pcd.points, dtype=np.float64)
    diagonal = float(np.linalg.norm(np.ptp(raw_points, axis=0)))
    points, m, centroid = pc_normalize(raw_points)
    num_points = len(points)
    ellipsoid_model = EllipsoidLeastSquaresModel()
    best_fit, best_inlier_idxs = ransac(
        points,
        ellipsoid_model,
        10,
        num_it,
        t,
        1,
        inliers_ratio=0.8,
        debug=False,
        return_all=True,
    )
    # For single-view data, ~25% inliers of the full ellipsoid is a valid consensus set
    if best_inlier_idxs is not None and len(best_inlier_idxs) >= max(20, int(0.25 * num_points)):
        params = ellipsoid_model.get_ellipsoid_params(best_fit)
        if params is not None:
            x0t, y0t, z0t, a, b, c, R = params
            center = np.array([x0t, y0t, z0t]) * m + centroid
            if np.linalg.norm(center - centroid) <= 0.8 * diagonal and max(a * m, b * m, c * m) <= 1.5 * diagonal:
                result = (a * m, b * m, c * m, SE3.Rt(SO3(np.array(R, dtype=np.float64)), center))
                return (*result, False) if return_details else result

    # Fast geometric fallback
    geom_result = _fit_ellipsoid_geometric(raw_points)
    if geom_result is None:
        return None
    return (*geom_result, True) if return_details else geom_result


# =============================================================================
# Dispatcher class
# =============================================================================


class FittingByBGS:
    """Dispatcher that selects and runs the appropriate fitting algorithm.

    Shape class codes:
      - ``'0'``:  Cuboid via RANSAC plane + OBB
      - ``'01'``: Cuboid via OBB only
      - ``'1'``:  Truncated cone via normal clustering (plane-based)
      - ``'11'``: Truncated cone via OBB
      - ``'12'``: Truncated cone via normal clustering (RANSAC axis)
      - ``'13'``: Truncated cone via PCA (component 0)
      - ``'14'``: Truncated cone via PCA (component 2)
      - ``'2'``:  Ellipsoid via RANSAC + least-squares
    """

    def __init__(self):
        self.last_error = None
        self.last_method = None
        self.last_fallback_triggered = False
        self.last_fallback_type = None

    def fitting(self, pcd, cls="0", visual=False):
        """Fit a primitive shape to the point cloud.

        Args:
            pcd: Open3D point cloud (with normals for cone methods).
            cls: Shape class code string.
            visual: Show debug visualization.

        Returns:
            List of parameters ``[dim1, dim2, dim3, T]`` or empty list on failure.
        """
        self.last_error = None
        self.last_method = None
        self.last_fallback_triggered = False
        self.last_fallback_type = None

        # Downsample if needed
        if len(pcd.points) > 5000:
            pcd_fit = pcd.farthest_point_down_sample(5000)
        else:
            pcd_fit = pcd

        params = []

        if cls == "0":
            try:
                a, b, c, T = fit_cuboid_obb2(pcd_fit, dist_threshold=None)
                self.last_method = "plane_obb"
            except Exception:
                a, b, c, T = fit_cuboid_obb(pcd_fit)
                self.last_method = "obb_fallback"
                self.last_fallback_triggered = True
                self.last_fallback_type = "obb_fallback"
            if visual:
                self._visualize_cuboid(pcd, pcd_fit, a, b, c, T)
            params = [a, b, c, T]

        elif cls == "01":
            a, b, c, T = fit_cuboid_obb(pcd_fit)
            self.last_method = "obb"
            if visual:
                self._visualize_cuboid(pcd, pcd_fit, a, b, c, T)
            params = [a, b, c, T]

        elif cls == "1":
            try:
                r1, r2, height, T = fit_frustum_cone_normal(
                    pcd_fit,
                    plane_t=0.005,
                    normal_t=0.02,
                    use_plane_normal=True,
                )
                self.last_method = "normal_slice"
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                r1, r2, height, T = fit_frustum_cone_pca(pcd_fit, z_dir=0)
                self.last_method = "pca_slice_fallback"
                self.last_fallback_triggered = True
                self.last_fallback_type = "pca_slice_fallback"
                self.last_error = f"normal_slice_fallback: {type(exc).__name__}: {exc}"
            if visual:
                self._visualize_cone(pcd, r1, r2, height, T)
            params = [r1, r2, height, T]

        elif cls == "11":
            r1, r2, height, T = fit_frustum_cone_obb(pcd_fit)
            self.last_method = "cone_obb"
            if visual:
                self._visualize_cone(pcd, r1, r2, height, T)
            params = [r1, r2, height, T]

        elif cls == "12":
            r1, r2, height, T = fit_frustum_cone_normal(
                pcd_fit,
                plane_t=0.01,
                normal_t=0.02,
                use_plane_normal=False,
            )
            self.last_method = "ransac_normal_slice"
            if visual:
                self._visualize_cone(pcd, r1, r2, height, T)
            params = [r1, r2, height, T]

        elif cls == "13":
            r1, r2, height, T = fit_frustum_cone_pca(pcd_fit, z_dir=0)
            self.last_method = "pca_slice_z0"
            if visual:
                self._visualize_cone(pcd, r1, r2, height, T)
            params = [r1, r2, height, T]

        elif cls == "14":
            r1, r2, height, T = fit_frustum_cone_pca(pcd_fit, z_dir=2)
            self.last_method = "pca_slice_z2"
            if visual:
                self._visualize_cone(pcd, r1, r2, height, T)
            params = [r1, r2, height, T]

        elif cls == "2":
            try:
                fit_out = fit_ellipsoid(pcd_fit, num_it=100, t=0.015, return_details=True)
            except Exception as exc:
                self.last_error = f"ellipsoid_exception: {type(exc).__name__}: {exc}"
                return params
            if fit_out is None:
                self.last_error = "ellipsoid_fit_failed"
                return params
            a, b, c, T, is_fallback = fit_out
            self.last_method = "ellipsoid_fallback" if is_fallback else "ellipsoid_ransac"
            self.last_fallback_triggered = is_fallback
            if is_fallback:
                self.last_fallback_type = "ellipsoid_geometric_fallback"
            if visual:
                self._visualize_ellipsoid(pcd, a, b, c, T)
            params = [a, b, c, T]

        return params

    @staticmethod
    def _visualize_cuboid(pcd, pcd_fit, a, b, c, T):
        """Debug visualization for cuboid fitting."""
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        coord.transform(T)
        obb = _oriented_bounding_box(pcd_fit)
        cube = o3d.geometry.TriangleMesh.create_box(
            width=a * 2, height=b * 2, depth=c * 2
        )
        cube.translate(-np.array([a, b, c]))
        cube.transform(T)
        fit_pcd = cube.sample_points_poisson_disk(5000)
        fit_pcd.paint_uniform_color([0, 0, 1])
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        o3d.visualization.draw_geometries([fit_pcd, obb, pcd, coord, origin])

    @staticmethod
    def _visualize_cone(pcd, r1, r2, height, T):
        """Debug visualization for cone fitting."""
        pts = generate_cone_points(
            r_bottom=r2,
            r_top_ratio=r1 / r2,
            height=height,
            delta=0.0,
            points_density=0,
            total_points=5000,
        )
        fit_pcd = o3d.geometry.PointCloud()
        fit_pcd.points = o3d.utility.Vector3dVector(pts)
        fit_pcd.paint_uniform_color([0, 0, 1])
        fit_pcd.transform(T)
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        coord.transform(T)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        o3d.visualization.draw_geometries([fit_pcd, pcd, coord, origin])

    @staticmethod
    def _visualize_ellipsoid(pcd, a, b, c, T):
        """Debug visualization for ellipsoid fitting."""
        pts = generate_ellipsoid_points(a, b, c, total_points=5000)
        fit_pcd = o3d.geometry.PointCloud()
        fit_pcd.points = o3d.utility.Vector3dVector(pts)
        fit_pcd.paint_uniform_color([0, 0, 1])
        fit_pcd.transform(T)
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        coord.transform(T)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        o3d.visualization.draw_geometries([fit_pcd, pcd, coord, origin])
