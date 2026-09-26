"""
Basic geometric shape fitting algorithms.

Implements fitting of three primitive types to 3D point clouds:
  - **Cuboid**: via RANSAC plane segmentation + OBB, or OBB-only
  - **Truncated cone (frustum)**: via normal clustering, PCA-based axis, or OBB
  - **Ellipsoid**: via RANSAC + least-squares algebraic fitting

Each fitting function returns shape parameters and a rigid transform (SE3)
that maps the canonical shape frame to the world frame.

Strictly aligned with the doctoral thesis (Chapters 4 & 5) and code/GPBSF.
"""

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares, nnls
from scipy.spatial.transform import Rotation
from spatialmath import SO3, SE3
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, RANSACRegressor

from ..utils.pointcloud import (
    ConeAxisLeastSquaresModel,
    EllipsoidLeastSquaresModel,
    NormalLeastSquaresModel,
    ransac,
    fit_circle_kasa,
    pc_normalize,
    generate_cone_points,
    generate_ellipsoid_points,
    compute_trimmed_distance,
    segment_plane_with_normals,
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


def _safe_se3(R, t):
    """Construct an SE3 pose guaranteed to be in SO(3) without raising bad argument errors.

    If det(R) < 0 (reflection / left-handed frame from Open3D PCA/OBB), the 3rd column is
    negated to ensure a valid right-handed rotation matrix.
    """
    R_mat = np.asarray(R, dtype=np.float64).copy()
    if R_mat.shape != (3, 3):
        return SE3()
    if np.linalg.det(R_mat) < 0:
        R_mat[:, 2] = -R_mat[:, 2]
    # SVD projection to ensure exact orthonormality and avoid floating point drift
    u, _, vt = np.linalg.svd(R_mat)
    R_clean = u @ vt
    if np.linalg.det(R_clean) < 0:
        u[:, -1] *= -1
        R_clean = u @ vt
    t_vec = np.asarray(t, dtype=np.float64).flatten()[:3]
    return SE3.Rt(SO3(R_clean, check=False), t_vec)


def _oriented_bounding_box(pcd):
    """Return the tightest OBB supported by the installed Open3D build.

    Prefers get_oriented_bounding_box() (covariance-based PCA, ~0.5 ms) over
    get_minimal_oriented_bounding_box() (3D rotating calipers combinatorial search, 100+ ms).
    Falls back gracefully if one is unavailable.
    """
    oriented = getattr(pcd, "get_oriented_bounding_box", None)
    if oriented is not None:
        return oriented()
    minimal = getattr(pcd, "get_minimal_oriented_bounding_box", None)
    if minimal is not None:
        return minimal()
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


def _get_obb_radii(pts, half_dim, plane_axis):
    """Estimate top and bottom cross-sectional radii for OBB cone fitting.

    Uses fast algebraic Kåsa circle fitting on slice points, falling back to
    bounding box half-extents.
    """
    idx1, idx2 = [i for i in range(3) if i != plane_axis]
    num_points = len(pts)

    top_mask = pts[:, plane_axis] > 0.85 * half_dim[plane_axis]
    if np.count_nonzero(top_mask) < 20:
        top_idx = np.argsort(pts[:, plane_axis])[-min(500, num_points):]
    else:
        top_idx = np.flatnonzero(top_mask)

    bottom_mask = pts[:, plane_axis] < -0.85 * half_dim[plane_axis]
    if np.count_nonzero(bottom_mask) < 20:
        bottom_idx = np.argsort(pts[:, plane_axis])[:min(500, num_points)]
    else:
        bottom_idx = np.flatnonzero(bottom_mask)

    top_pts = pts[top_idx][:, [idx1, idx2]]
    bottom_pts = pts[bottom_idx][:, [idx1, idx2]]

    # Top radius
    top_r = None
    if len(top_pts) >= 4:
        kasa_res = fit_circle_kasa(top_pts)
        if kasa_res is not None and kasa_res[1] > 1e-4:
            top_r = kasa_res[1]
    if top_r is None:
        top_size = np.ptp(top_pts, axis=0) if len(top_pts) > 0 else np.array([half_dim[idx1] * 2, half_dim[idx2] * 2])
        top_r = float(np.mean(top_size)) / 2.0

    # Bottom radius
    bottom_r = None
    if len(bottom_pts) >= 4:
        kasa_res = fit_circle_kasa(bottom_pts)
        if kasa_res is not None and kasa_res[1] > 1e-4:
            bottom_r = kasa_res[1]
    if bottom_r is None:
        bottom_size = np.ptp(bottom_pts, axis=0) if len(bottom_pts) > 0 else np.array([half_dim[idx1] * 2, half_dim[idx2] * 2])
        bottom_r = float(np.mean(bottom_size)) / 2.0

    max_allowed_r = max(half_dim) * 2.0
    top_r = min(max(float(top_r), 1e-4), max_allowed_r)
    bottom_r = min(max(float(bottom_r), 1e-4), max_allowed_r)
    return top_r, bottom_r


# =============================================================================
# Slice-based cone radius estimation
# =============================================================================


def fit_frustum_cone_by_slice_linear(points, num_layers=10, pcd=None, is_debug=False):
    """Estimate cone radii by slicing along the Z axis and fitting circles per layer.

    Uses two-point line enumeration and inlier linear least-squares regression strictly
    following thesis Section 4.3.2 Eq. (4.5).

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
        # circle fits. Estimate a conservative circular profile rather than
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

    # Two-point line enumeration and inlier refinement strictly following thesis Section 4.3.2 Eq. (4.5)
    layer_indices = np.array(list(layer_radius.keys()), dtype=np.int32)
    radii = np.array(list(layer_radius.values()), dtype=np.float64)
    n_valid = len(layer_indices)

    r_min_val, r_max_val = float(np.min(radii)), float(np.max(radii))
    res_threshold = max(0.15 * (r_max_val - r_min_val), 0.001)

    best_inlier_mask = None
    best_inlier_count = -1
    best_res_sum = float("inf")

    # Enumerate all pairs of valid slices to generate candidate lines
    for i in range(n_valid):
        for j in range(i + 1, n_valid):
            dx = float(layer_indices[j] - layer_indices[i])
            if abs(dx) < 1e-6:
                continue
            dy = float(radii[j] - radii[i])
            a_cand = dy / dx
            b_cand = radii[i] - a_cand * layer_indices[i]

            pred_r = a_cand * layer_indices + b_cand
            res = np.abs(radii - pred_r)
            inlier_mask = res <= res_threshold
            inlier_count = int(np.count_nonzero(inlier_mask))
            res_sum = float(np.sum(res[inlier_mask])) if inlier_count > 0 else 0.0

            if inlier_count > best_inlier_count or (inlier_count == best_inlier_count and res_sum < best_res_sum):
                best_inlier_count = inlier_count
                best_inlier_mask = inlier_mask
                best_res_sum = res_sum

    if best_inlier_mask is None or best_inlier_count < 2:
        best_inlier_mask = np.ones(n_valid, dtype=bool)

    X_inliers = layer_indices[best_inlier_mask].astype(np.float64)
    y_inliers = radii[best_inlier_mask]

    # Closed-form linear least-squares refinement: r(k) = alpha * k + beta
    if len(X_inliers) >= 2 and (np.max(X_inliers) - np.min(X_inliers)) > 1e-6:
        alpha, beta = np.polyfit(X_inliers, y_inliers, deg=1)
    else:
        alpha = 0.0
        beta = float(np.mean(y_inliers))

    min_physical_r = max(1e-4, r_min_val * 0.05) if r_min_val > 0 else 1e-4
    # r2 = bottom (k = 0, z_min), r1 = top (k = num_layers - 1, z_max)
    r2 = max(float(beta), min_physical_r)
    r1 = max(float(alpha * (num_layers - 1) + beta), min_physical_r)

    # Lateral center: arithmetic mean of inlier layer centers
    center_arr = np.array([layer_center[k] for k in layer_indices[best_inlier_mask]], dtype=np.float64)
    center_xy = np.mean(center_arr, axis=0)
    center = np.array([
        center_xy[0],
        center_xy[1],
        (z_min + z_max) / 2.0,
    ])

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
    T = _safe_se3(R_plane, cube_center)

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
    T = _safe_se3(R, cube_center)

    return float(half_extent[0]), float(half_extent[1]), float(half_extent[2]), T


def fit_frustum_cone_normal(
    pcd,
    plane_t=0.001,
    normal_t=0.02,
    use_plane_normal=True,
    is_debug=False,
):
    """Fit a truncated cone using surface normal clustering or RANSAC to find the axis.

    Args:
        pcd: Open3D point cloud (must have normals estimated).
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

    if use_plane_normal:
        try:
            orig_pts = np.asarray(pcd.points)
            if not pcd.has_normals():
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
            orig_normals = np.asarray(pcd.normals)
            plane_m, inliers = segment_plane_with_normals(
                orig_pts, orig_normals, dist_threshold=plane_t, angle_threshold_deg=15.0
            )
            if plane_m is not None and len(inliers) >= 10:
                p_in = orig_pts[inliers]
                c = np.mean(p_in, axis=0)
                _, _, vh = np.linalg.svd(p_in - c)
                plane_normal = vh[2, :]
                plane_normal /= np.linalg.norm(plane_normal)
                mean_n = np.mean(orig_normals[inliers], axis=0)
                if np.dot(plane_normal, mean_n) < 0:
                    plane_normal = -plane_normal
                axis_candidates.insert(0, plane_normal)
        except Exception:
            pass
    elif len(normals) >= 10:
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

    best_score = -float("inf")
    best_cone_normal = axis_candidates[0]
    best_R = align_vector_to_z(best_cone_normal)

    for cand in axis_candidates:
        R_cand = align_vector_to_z(cand)
        pts_cand = points @ R_cand
        z = pts_cand[:, 2]
        z_min = float(np.quantile(z, 0.005))
        z_max = float(np.quantile(z, 0.995))
        h = z_max - z_min
        if h <= 1e-4:
            continue
        valid_slices = 0
        total_circle_err = 0.0
        for i in range(12):
            mask = (z >= z_min + i * h / 12) & (z <= z_min + (i + 1) * h / 12)
            layer_pts = pts_cand[mask]
            if len(layer_pts) >= 5:
                valid_slices += 1
                c_res = fit_circle_kasa(layer_pts[:, :2])
                if c_res is not None:
                    (cx, cy), cr = c_res
                    dists = np.abs(np.linalg.norm(layer_pts[:, :2] - [cx, cy], axis=1) - cr)
                    total_circle_err += float(np.mean(dists))
        score = valid_slices * 10.0 - (total_circle_err / max(1, valid_slices))
        if score > best_score:
            best_score = score
            best_cone_normal = cand
            best_R = R_cand

    pcd_normalized.rotate(best_R.T, center=[0, 0, 0])
    pts_rotated = np.asarray(pcd_normalized.points)

    r1, r2, height, center = fit_frustum_cone_by_slice_linear(pts_rotated, 12, pcd)

    world_center = centroid + m * (best_R @ np.asarray(center))
    T = _safe_se3(best_R, world_center)
    return r1 * m, r2 * m, height * m, T


def fit_frustum_cone_pca(pcd, is_debug=False, z_dir=0, num_layers=12):
    """Fit a truncated cone using PCA to find the symmetry axis."""
    pcd_normalized = o3d.geometry.PointCloud(pcd)
    pts, m, centroid = pc_normalize(np.asarray(pcd.points))
    pcd_normalized.points = o3d.utility.Vector3dVector(pts)
    points = np.asarray(pcd_normalized.points)

    pca = PCA(n_components=3)
    pca.fit(points)
    cone_normal = pca.components_[z_dir]
    R_mat = align_vector_to_z(cone_normal)

    pcd_normalized.rotate(R_mat.T, center=[0, 0, 0])
    pts_rotated = np.asarray(pcd_normalized.points)

    r1, r2, height, center = fit_frustum_cone_by_slice_linear(pts_rotated, num_layers, pcd)

    world_center = centroid + m * (R_mat @ np.asarray(center))
    T = _safe_se3(R_mat, world_center)
    return r1 * m, r2 * m, height * m, T


def fit_frustum_cone_obb(pcd):
    """Fit a truncated cone using the Oriented Bounding Box method.

    Uses OBB to establish the initial frame, determines the symmetry axis,
    then computes cross-sectional radii on top/bottom slices.

    Args:
        pcd: Open3D point cloud.

    Returns:
        Tuple of (top_r, bottom_r, height, T).
    """
    pcd_fit = o3d.geometry.PointCloud()
    pcd_fit.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    pts = np.asarray(pcd_fit.points)
    obb = _oriented_bounding_box(pcd_fit)
    T_obb = _safe_se3(obb.R, obb.center)
    pcd_fit.transform(T_obb.inv())
    pts_local = np.asarray(pcd_fit.points)

    half_dim = [float(np.max(np.abs(pts_local[:, i]))) for i in range(3)]
    plane_axis = get_plane_axis(pcd_fit, half_dim)
    if plane_axis == -1:
        return 0.0, 0.0, 0.0, SE3()
    height = half_dim[plane_axis] * 2.0
    top_r, bottom_r = _get_obb_radii(pts_local, half_dim, plane_axis)
    return top_r, bottom_r, height, T_obb


def compute_cone_residual(
    pcd, r1, r2, height, T, n_eval_points=2000, inlier_ratio=0.90
) -> float:
    """Compute robust trimmed surface distance residual between observed pcd and fitted cone."""
    if height <= 1e-4 or r1 <= 1e-4 or r2 <= 1e-4:
        return float("inf")
    try:
        pts = generate_cone_points(
            r_bottom=r2,
            r_top_ratio=r1 / r2 if r2 > 1e-6 else 1.0,
            height=height,
            delta=0.0,
            points_density=0,
            total_points=n_eval_points,
        )
        fit_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        T_mat = T.A if hasattr(T, "A") else np.asarray(T, dtype=np.float64)
        fit_pcd.transform(T_mat)

        n_pts = len(pcd.points) if hasattr(pcd, "points") else len(pcd)
        if n_pts > 2000:
            step = max(1, n_pts // 2000)
            eval_pcd = pcd.uniform_down_sample(every_k_points=step) if hasattr(pcd, "uniform_down_sample") else pcd[::step]
        else:
            eval_pcd = pcd

        return compute_trimmed_distance(eval_pcd, fit_pcd, inlier_ratio=inlier_ratio)
    except Exception:
        return float("inf")


def fit_frustum_cone_adaptive(
    pcd,
    tau_cone: float = 2.0,
    is_debug=False,
):
    """Fit a truncated cone using sequential multi-hypothesis threshold early-exit.

    Hypothesis hierarchy sorted in ascending order of computational runtime (shortest first)
    according to Chapter 4 (Algorithm 4.1):
    1. PCA Principal Axis ("pca_z0"): Closed-form eigen-decomposition (~0.1 ms).
    2. PCA Secondary Axis ("pca_z2"): Closed-form secondary eigen-decomposition (~0.1 ms).
    3. Oriented Bounding Box ("obb"): Fast bounding extent projection (~0.8 ms).
    4. Surface Normal Clustering / Plane Segmentation ("normal"): RANSAC plane segmentation (~5 ms).
    5. Surface Normal RANSAC ("normal_ransac"): Non-linear angle variance optimization RANSAC (~30 ms).

    Exits early as soon as a hypothesis achieves robust trimmed distance <= tau_cone (default 2.0 mm).
    If no hypothesis meets the threshold, returns the hypothesis with the minimal residual.

    Returns:
        Tuple of (r1, r2, height, T, method_name, residual)
    """
    # Downsample dense point clouds to ~2048 points for fast, robust geometric axis/parameter fitting
    n_pts = len(pcd.points) if hasattr(pcd, "points") else len(pcd)
    if n_pts > 2048:
        step = max(1, n_pts // 2048)
        pcd_fit = pcd.uniform_down_sample(every_k_points=step) if hasattr(pcd, "uniform_down_sample") else pcd[::step]
    else:
        pcd_fit = pcd

    hypotheses = [
        ("pca_z0", lambda: fit_frustum_cone_pca(pcd_fit, z_dir=0)),
        ("pca_z2", lambda: fit_frustum_cone_pca(pcd_fit, z_dir=2)),
        ("obb", lambda: fit_frustum_cone_obb(pcd_fit)),
        ("normal", lambda: fit_frustum_cone_normal(pcd_fit, plane_t=0.005, normal_t=0.02, use_plane_normal=True)),
        ("normal_ransac", lambda: fit_frustum_cone_normal(pcd_fit, plane_t=0.01, normal_t=0.02, use_plane_normal=False)),
    ]

    best_fit = None
    best_residual = float("inf")
    best_method = "none"

    # Scale-adaptive early exit threshold check:
    # If pointcloud scale is in meters (bbox diagonal < 5.0), convert tau_cone (mm) to meters.
    # If pointcloud scale is in millimeters (bbox diagonal >= 5.0), tau_thresh is directly tau_cone (mm).
    pts_arr = np.asarray(pcd_fit.points) if hasattr(pcd_fit, "points") else np.asarray(pcd_fit)
    diag = float(np.linalg.norm(np.ptp(pts_arr, axis=0))) if len(pts_arr) > 0 else 1.0
    if diag < 5.0:
        tau_thresh = tau_cone * 1e-3 if tau_cone > 0.05 else tau_cone
    else:
        tau_thresh = tau_cone

    for name, solver in hypotheses:
        try:
            r1, r2, h, T = solver()
        except Exception:
            continue

        if h <= 1e-4 or r1 <= 1e-4 or r2 <= 1e-4:
            continue

        residual = compute_cone_residual(pcd, r1, r2, h, T)
        if residual < best_residual:
            best_residual = residual
            best_fit = (r1, r2, h, T)
            best_method = name

        # Threshold Early-Exit condition (门限早停: <= tau_thresh)
        if residual <= tau_thresh:
            return best_fit[0], best_fit[1], best_fit[2], best_fit[3], best_method, best_residual

    if best_fit is not None:
        return best_fit[0], best_fit[1], best_fit[2], best_fit[3], best_method, best_residual

    # Fallback to normal if all failed
    try:
        r1, r2, h, T = fit_frustum_cone_normal(pcd)
        res = compute_cone_residual(pcd, r1, r2, h, T)
        return r1, r2, h, T, "normal_fallback", res
    except Exception:
        pass

    return 0.0, 0.0, 0.0, SE3(), "failed", float("inf")


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
        semi_axes = 1.0 / np.sqrt(coef)
        world_center = centroid + pca_vecs @ local_c
        max_dim = diagonal * 1.5
        min_dim = max(diagonal * 0.02, 1e-3)
        if np.all(semi_axes <= max_dim) and np.all(semi_axes >= min_dim):
            T = _safe_se3(pca_vecs, world_center)
            return (float(semi_axes[0]), float(semi_axes[1]), float(semi_axes[2]), T)

    # Level 2: Robust bounded optimization using Taubin metric + soft_l1 (~20-40 ms)
    init_axes = np.sqrt(np.maximum(pca_vals, 1e-6)) * 2.0
    init_axes = np.clip(init_axes, diagonal * 0.05, diagonal * 0.8)
    initial_params = np.concatenate([centroid, np.log(init_axes), np.zeros(3)])

    lb = np.concatenate([centroid - diagonal * 0.5, np.log(np.full(3, diagonal * 0.02)), -np.full(3, np.pi)])
    ub = np.concatenate([centroid + diagonal * 0.5, np.log(np.full(3, diagonal * 1.5)), np.full(3, np.pi)])

    try:
        res = least_squares(
            _ellipsoid_geometric_residual,
            initial_params,
            bounds=(lb, ub),
            args=(pts_opt,),
            loss="soft_l1",
            f_scale=max(diagonal * 0.02, 1e-3),
            max_nfev=35,
        )
        if not res.success and res.status <= 0:
            return None
        c_opt = res.x[:3]
        axes_opt = np.exp(res.x[3:6])
        R_opt = _rotvec_to_mat(res.x[6:])
        T = _safe_se3(R_opt, c_opt)
        return (float(axes_opt[0]), float(axes_opt[1]), float(axes_opt[2]), T)
    except Exception:
        return None


def fit_ellipsoid(pcd, num_it=100, t=0.015, return_details=False):
    """Fit an ellipsoid to a 3D point cloud using RANSAC + algebraic least squares.

    Falls back to fast, robust geometric fitting if RANSAC fails or yields non-ellipsoidal quadrics.

    Args:
        pcd: Open3D point cloud.
        num_it: RANSAC iterations.
        t: Distance threshold for inlier counting.
        return_details: If True, returns (a, b, c, T, is_fallback).

    Returns:
        Tuple of (a, b, c, T) or None.
    """
    raw_points = np.asarray(pcd.points, dtype=np.float64)
    num_points = len(raw_points)
    if num_points < 9:
        geom_result = _fit_ellipsoid_geometric(raw_points)
        if geom_result is None:
            return None
        return (*geom_result, True) if return_details else geom_result

    diagonal = float(np.linalg.norm(np.ptp(raw_points, axis=0)))
    points, m, centroid = pc_normalize(raw_points)
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
                result = (a * m, b * m, c * m, _safe_se3(R, center))
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
    """Dispatcher that selects and runs the appropriate primitive fitting algorithm.

    Shape class codes:
      - ``'0'``:  Cuboid via RANSAC plane segmentation + OBB (with automatic OBB fallback)
      - ``'1'``:  Truncated cone via sequential adaptive hypothesis fitting (threshold early-exit)
      - ``'2'``:  Ellipsoid via algebraic least-squares + fast geometric fallback
    """

    def __init__(self):
        self.last_error = None
        self.last_method = None
        self.last_fallback_triggered = False
        self.last_fallback_type = None

    def fitting(self, pcd, cls="0", visual=False, tau_cone=2.0):
        """Fit a primitive shape to the point cloud.

        Args:
            pcd: Open3D point cloud (with normals for cone methods).
            cls: Shape class code string ("0", "1", "2").
            visual: Show debug visualization.
            tau_cone: Residual threshold (mm) for cone adaptive early-exit.

        Returns:
            List of parameters ``[dim1, dim2, dim3, T]`` or empty list on failure.
        """
        self.last_error = None
        self.last_method = None
        self.last_fallback_triggered = False
        self.last_fallback_type = None

        # Downsample if needed
        n_pts = len(pcd.points) if hasattr(pcd, "points") else len(pcd)
        if n_pts > 5000:
            step = max(1, n_pts // 5000)
            pcd_fit = pcd.uniform_down_sample(every_k_points=step) if hasattr(pcd, "uniform_down_sample") else pcd[::step]
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

        elif cls == "1":
            try:
                r1, r2, height, T, method, res = fit_frustum_cone_adaptive(
                    pcd_fit, tau_cone=tau_cone
                )
                self.last_method = f"cone_adaptive_{method}"
            except Exception as exc:
                r1, r2, height, T = fit_frustum_cone_pca(pcd_fit, z_dir=0)
                self.last_method = "pca_slice_fallback"
                self.last_fallback_triggered = True
                self.last_fallback_type = "pca_slice_fallback"
                self.last_error = f"cone_adaptive_fallback: {type(exc).__name__}: {exc}"
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

        else:
            self.last_error = f"unsupported_shape_class_{cls}"
            return []

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
