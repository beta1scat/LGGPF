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
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    z_min, z_max = z.min(), z.max()
    height = z_max - z_min

    layer_pts = []
    for i in range(num_layers):
        layer_min = z_min + i * height / num_layers
        layer_max = z_min + (i + 1) * height / num_layers
        mask = (z >= layer_min) & (z < layer_max)
        layer_pts.append(points[mask])

    # Fit circle to each layer
    layer_radius = {}
    layer_center = {}
    for layer_idx in range(num_layers):
        layer_idx_pts = layer_pts[layer_idx]
        if len(layer_idx_pts) < 4:
            continue
        cir_model = CircleLeastSquaresModel()
        bestmodel, inliers = ransac(
            layer_idx_pts[:, :2],
            cir_model,
            3,
            200,
            0.01,
            10,
            debug=False,
            return_all=True,
        )
        if bestmodel is not None:
            layer_radius[layer_idx] = bestmodel[2]
            layer_center[layer_idx] = bestmodel[:2]

    if len(layer_radius) < 0.5 * num_layers:
        return None

    # RANSAC linear regression on layer radii
    ransac_reg = linear_model.RANSACRegressor()
    X_fit = np.array(list(layer_radius.keys()))[:, np.newaxis]
    y_fit = np.array(list(layer_radius.values()))
    ransac_reg.fit(X_fit, y_fit)
    inlier_mask = np.where(ransac_reg.inlier_mask_)[0]
    line_x = np.array(range(num_layers))[:, np.newaxis]
    line_y = ransac_reg.predict(line_x)
    r2 = line_y[0]  # bottom (min Z)
    r1 = line_y[-1]  # top (max Z)

    center_arr = np.array(list(layer_center.values()))
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
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    z_min, z_max = z.min(), z.max()
    height = z_max - z_min

    layer_pts = []
    for i in range(num_layers):
        layer_min = z_min + i * height / num_layers
        layer_max = z_min + (i + 1) * height / num_layers
        mask = (z >= layer_min) & (z < layer_max)
        layer_pts.append(points[mask])

    layer_radius = {}
    layer_center = {}
    for layer_idx in range(num_layers):
        layer_idx_pts = layer_pts[layer_idx]
        if len(layer_idx_pts) < 4:
            continue
        cir_model = CircleLeastSquaresModel()
        bestmodel, inliers = ransac(
            layer_idx_pts[:, :2],
            cir_model,
            3,
            100,
            0.01,
            10,
            debug=False,
            return_all=True,
        )
        if bestmodel is not None:
            layer_radius[layer_idx] = bestmodel[2]
            layer_center[layer_idx] = bestmodel[:2]

    X_fit = np.array(list(layer_radius.keys()))[:, np.newaxis]
    y_fit = np.array(list(layer_radius.values()))
    model = make_pipeline(
        PolynomialFeatures(degree=2),
        RANSACRegressor(LinearRegression()),
    )
    model.fit(X_fit, y_fit.ravel())
    ransac_model = model.named_steps["ransacregressor"]
    inlier_mask = ransac_model.inlier_mask_

    line_x = np.array(range(num_layers))[:, np.newaxis]
    line_y = model.predict(line_x)
    r2 = line_y[0]
    r1 = line_y[-1]

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


def fit_cuboid_obb2(pcd, dist_threshold=0.01, n=3, num_it=500, is_debug=False):
    """Fit a cuboid using RANSAC plane segmentation + Oriented Bounding Box.

    First segments the dominant plane via RANSAC, then uses the OBB of the
    plane points to determine the cuboid orientation, and the full AABB
    (after inverse rotation) for dimensions.

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
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=dist_threshold, ransac_n=n, num_iterations=num_it
    )
    plane_cloud = pcd.select_by_index(inliers)

    obb = plane_cloud.get_minimal_oriented_bounding_box()
    R = SO3(obb.R)
    pcd_rotated = o3d.geometry.PointCloud(pcd)
    pcd_rotated.rotate(R.inv(), [0, 0, 0])
    aabb = pcd_rotated.get_axis_aligned_bounding_box()
    a, b, c = aabb.get_half_extent()
    cube_center = SO3(R) * aabb.get_center()
    T = SE3.Rt(R, cube_center)

    if is_debug:
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        coord.transform(T)
        o3d.visualization.draw_geometries([pcd, obb, coord])

    return a, b, c, T


def fit_cuboid_obb(pcd):
    """Fit a cuboid using only the Oriented Bounding Box (no plane segmentation).

    Args:
        pcd: Open3D point cloud.

    Returns:
        Tuple of (a, b, c, T) where a, b, c are half-extents and
        T is an SE3 transform.
    """
    obb = pcd.get_minimal_oriented_bounding_box()
    T = SE3.Rt(obb.R, obb.center)
    pcd.transform(T.inv())
    aabb = pcd.get_axis_aligned_bounding_box()
    pcd.transform(T)
    a = aabb.max_bound[0]
    b = aabb.max_bound[1]
    c = aabb.max_bound[2]
    return a, b, c, T


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
    normals = np.asarray(pcd_normalized.normals)

    if use_plane_normal:
        num_points = len(points)
        num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, tol=0.01).fit(normals)
        labels = kmeans.labels_

        plane_normals_list = []
        plane_points_ratio_list = []
        pcd_cluster_indices = []
        for i in range(num_clusters):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) < num_points / 10:
                continue
            cluster_pcd = pcd_normalized.select_by_index(cluster_indices)
            plane_model, plane_inliers = cluster_pcd.segment_plane(
                distance_threshold=plane_t, ransac_n=3, num_iterations=1000
            )
            plane_normals_list.append(plane_model[:3])
            ratio = len(plane_inliers) / len(cluster_indices)
            pcd_cluster_indices.append(cluster_indices)
            plane_points_ratio_list.append(ratio)

        # Use the cluster with highest plane inlier ratio
        max_idx = np.argmax(plane_points_ratio_list)
        cone_normal = plane_normals_list[max_idx]
    else:
        # Use RANSAC cone axis estimation from normals
        cone_axis_model = ConeAxisLeastSquaresModel()
        best_fit, _ = ransac(
            normals,
            cone_axis_model,
            10,
            1000,
            normal_t,
            1,
            inliers_ratio=0.9,
            debug=False,
            return_all=True,
        )
        vector, _ = best_fit
        cone_normal = vector

    # Align cone axis to Z
    vec_x = np.cross(cone_normal, [0, 0, 1])
    R = SO3.TwoVectors(x=vec_x, z=cone_normal)
    pcd_normalized.rotate(R.inv(), center=[0, 0, 0])

    # Fit cone radii by slicing
    if use_poly:
        r1, r2, height, center = fit_frustum_cone_by_slice_poly(points, 30)
    else:
        r1, r2, height, center = fit_frustum_cone_by_slice_linear(points, 30, pcd)

    # Scale back to original coordinates
    T = SE3(centroid) * SE3(R) * SE3(np.asarray(center) * m)
    return r1 * m, r2 * m, height * m, T


def fit_frustum_cone_pca(pcd, use_poly=False, is_debug=False, z_dir=0):
    """Fit a truncated cone using PCA to find the symmetry axis.

    Uses PCA on the point positions; the specified principal component
    is used as the cone axis direction.

    Args:
        pcd: Open3D point cloud.
        use_poly: Use polynomial regression for radius estimation.
        is_debug: Show debug visualization.
        z_dir: Index of the PCA component to use as Z axis (0, 1, or 2).

    Returns:
        Tuple of (r1, r2, height, T).
    """
    pcd_normalized = o3d.geometry.PointCloud(pcd)
    pts, m, centroid = pc_normalize(np.asarray(pcd.points))
    pcd_normalized.points = o3d.utility.Vector3dVector(pts)
    points = np.asarray(pcd_normalized.points)

    pca = PCA(n_components=3)
    pca.fit(points)
    cone_normal = pca.components_[z_dir]

    vec_x = np.cross(cone_normal, [0, 0, 1])
    R = SO3.TwoVectors(x=vec_x, z=cone_normal)
    pcd_normalized.rotate(R.inv(), center=[0, 0, 0])

    if use_poly:
        r1, r2, height, center = fit_frustum_cone_by_slice_poly(points, 30)
    else:
        r1, r2, height, center = fit_frustum_cone_by_slice_linear(points, 30, pcd)

    T = SE3(centroid) * SE3(R) * SE3(np.asarray(center) * m)
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
    obb = pcd_fit.get_minimal_oriented_bounding_box()
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


def fit_ellipsoid(pcd, num_it=100, t=0.01):
    """Fit an ellipsoid using RANSAC + algebraic least-squares.

    Uses the :class:`EllipsoidLeastSquaresModel` with sympy-based
    eigenvalue decomposition to recover the ellipsoid parameters.

    Args:
        pcd: Open3D point cloud.
        num_it: RANSAC iterations.
        t: RANSAC inlier threshold (in normalized coordinates).

    Returns:
        Tuple of (a, b, c, T) where a, b, c are semi-axis lengths
        (in original scale) and T is the SE3 transform. Returns None
        if fitting fails.
    """
    points, m, centroid = pc_normalize(np.asarray(pcd.points))
    num_points = len(points)
    ellipsoid_model = EllipsoidLeastSquaresModel()
    best_fit, best_inlier_idxs = ransac(
        points,
        ellipsoid_model,
        10,
        num_it,
        t,
        1,
        inliers_ratio=0.9,
        debug=False,
        return_all=True,
    )
    if best_inlier_idxs is None or len(best_inlier_idxs) < 0.6 * num_points:
        return None
    params = ellipsoid_model.get_ellipsoid_params(best_fit)
    if params is None:
        return None
    x0t, y0t, z0t, a, b, c, R = params
    center = np.array([x0t, y0t, z0t]) * m + centroid
    T = SE3.Rt(SO3(np.array(R, dtype=np.float64)), center)
    return a * m, b * m, c * m, T


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

    def fitting(self, pcd, cls="0", visual=False):
        """Fit a primitive shape to the point cloud.

        Args:
            pcd: Open3D point cloud (with normals for cone methods).
            cls: Shape class code string.
            visual: Show debug visualization.

        Returns:
            List of parameters ``[dim1, dim2, dim3, T]`` or empty list on failure.
        """
        # Downsample if needed
        if len(pcd.points) > 5000:
            pcd_fit = pcd.farthest_point_down_sample(5000)
        else:
            pcd_fit = pcd

        params = []

        if cls == "0":
            a, b, c, T = fit_cuboid_obb2(pcd_fit, dist_threshold=2)
            if visual:
                self._visualize_cuboid(pcd, pcd_fit, a, b, c, T)
            params = [a, b, c, T]

        elif cls == "01":
            a, b, c, T = fit_cuboid_obb(pcd_fit)
            if visual:
                self._visualize_cuboid(pcd, pcd_fit, a, b, c, T)
            params = [a, b, c, T]

        elif cls == "1":
            r1, r2, height, T = fit_frustum_cone_normal(
                pcd_fit,
                plane_t=0.005,
                normal_t=0.02,
                use_plane_normal=True,
            )
            if visual:
                self._visualize_cone(pcd, r1, r2, height, T)
            params = [r1, r2, height, T]

        elif cls == "11":
            r1, r2, height, T = fit_frustum_cone_obb(pcd_fit)
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
            if visual:
                self._visualize_cone(pcd, r1, r2, height, T)
            params = [r1, r2, height, T]

        elif cls == "13":
            r1, r2, height, T = fit_frustum_cone_pca(pcd_fit, z_dir=0)
            if visual:
                self._visualize_cone(pcd, r1, r2, height, T)
            params = [r1, r2, height, T]

        elif cls == "14":
            r1, r2, height, T = fit_frustum_cone_pca(pcd_fit, z_dir=2)
            if visual:
                self._visualize_cone(pcd, r1, r2, height, T)
            params = [r1, r2, height, T]

        elif cls == "2":
            try:
                result = fit_ellipsoid(pcd_fit, num_it=500, t=0.01)
            except Exception:
                return params
            if result is None:
                return params
            a, b, c, T = result
            if visual:
                self._visualize_ellipsoid(pcd_fit, a, b, c, T)
            params = [a, b, c, T]

        else:
            print(f"Unknown shape class: {cls}")

        return params

    @staticmethod
    def _visualize_cuboid(pcd, pcd_fit, a, b, c, T):
        """Debug visualization for cuboid fitting."""
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        coord.transform(T)
        obb = pcd_fit.get_minimal_oriented_bounding_box()
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
