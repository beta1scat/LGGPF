"""
Point cloud utilities, RANSAC fitting models, shape generators, and pose filtering.

Consolidated from the original modules/utils/utils.py and duplicate functions
from modules/grasp_generation/pick_pose.py.
"""

import numpy as np
import open3d as o3d
import scipy
import scipy.optimize
import sympy as sp
from spatialmath import SE3, SO3

np.set_printoptions(suppress=True)


# =============================================================================
# Point cloud normalization
# =============================================================================


def pc_normalize(pc):
    """Normalize a point cloud to unit sphere.

    Args:
        pc: (N, 3) array of points.

    Returns:
        Tuple of (normalized_points, scale_factor, centroid).
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, m, centroid


# =============================================================================
# RANSAC
# =============================================================================


def random_partition(n, n_data):
    """Return n random indices and the remaining indices.

    Args:
        n: Number of indices to sample.
        n_data: Total number of data points.

    Returns:
        Tuple of (sampled_indices, remaining_indices).
    """
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    return all_idxs[:n], all_idxs[n:]


def ransac(data, model, n, k, t, d, inliers_ratio=0.5, debug=False, return_all=False):
    """Fit model parameters to data using the RANSAC algorithm.

    Reference: http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

    Args:
        data: (N, D) array of observed data points.
        model: Object implementing ``fit(data)`` and ``get_error(data, params)``.
        n: Minimum number of data points required to fit the model.
        k: Maximum number of iterations.
        t: Threshold for determining when a point fits the model.
        d: Minimum number of inliers to assert a good fit.
        inliers_ratio: Early termination when inlier fraction exceeds this.
        debug: Print debug information per iteration.
        return_all: If True, also return inlier indices.

    Returns:
        bestfit: Best model parameters, or None if fitting failed.
        best_inlier_idxs: (only if return_all) Indices of inliers.
    """
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    data_size = data.shape[0]
    inliers_condition = inliers_ratio * data_size

    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data_size)
        maybeinliers = data[maybe_idxs, :]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error(test_points, maybemodel)
        also_idxs = test_idxs[test_err < t]
        alsoinliers = data[also_idxs, :]
        alsoinliers_num = len(alsoinliers)

        if debug:
            print(f"test_err.min() {test_err.min()}")
            print(f"test_err.max() {test_err.max()}")
            print(f"np.mean(test_err) {np.mean(test_err)}")
            print(f"iteration {iterations}: len(alsoinliers) = {alsoinliers_num}")

        if alsoinliers_num > d:
            betterdata = np.concatenate((maybeinliers, alsoinliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))

        iterations += 1
        if alsoinliers_num + n > inliers_condition:
            break

    if bestfit is None:
        print("Fit failed in RANSAC")

    if return_all:
        return bestfit, best_inlier_idxs
    else:
        return bestfit


# =============================================================================
# Least-squares model classes for RANSAC
# =============================================================================


def fit_circle_kasa(points):
    """Direct closed-form algebraic circle fitting in 2D (Kåsa method).

    Solves (x - xc)^2 + (y - yc)^2 = r^2 via linear least squares:
      2*xc*x + 2*yc*y + (r^2 - xc^2 - yc^2) = x^2 + y^2

    Args:
        points: (N, 2) array of 2D points.

    Returns:
        Tuple of (center, radius) where center is (2,) np.ndarray, or None.
    """
    pts = np.asarray(points)
    if len(pts) < 4:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        xc = float(sol[0] / 2.0)
        yc = float(sol[1] / 2.0)
        rad_sq = float(sol[2] + xc**2 + yc**2)
        if rad_sq <= 0 or np.isnan(rad_sq):
            return None
        r = float(np.sqrt(rad_sq))
        if np.isnan(r) or np.isinf(r) or r <= 0:
            return None
        return np.array([xc, yc]), r
    except Exception:
        return None


class CircleLeastSquaresModel:
    """2D circle fitting model. Data shape: (N, 2)."""

    def fit(self, data):
        kasa_res = fit_circle_kasa(data)
        if kasa_res is not None:
            c, r = kasa_res
            return (c[0], c[1], r)
        return None

    def get_error(self, data, model):
        if model is None:
            return np.full(data.shape[0], np.inf)
        x0, y0, r = model
        return np.abs(np.linalg.norm(data - np.array([x0, y0]), axis=1) - r)


class EllipsoidLeastSquaresModel:
    """3D ellipsoid fitting model. Data shape: (N, 3).

    Analytical algebraic least-squares model with closed-form eigendecomposition
    to recover ellipsoid center, semi-axes, and rotation matrix without SymPy.
    """

    @staticmethod
    def get_design_matrix(pts: np.ndarray) -> np.ndarray:
        """Vectorized construction of quadratic design matrix (N, 10)."""
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        return np.column_stack([
            x**2, y**2, z**2,
            x * y, x * z, y * z,
            x, y, z,
            np.ones_like(x),
        ])

    def fit(self, data):
        data_size = len(data)
        if data_size > 100:
            data = data[np.random.choice(data_size, 100, replace=False)]
        A = self.get_design_matrix(data)
        try:
            _, _, V = scipy.linalg.svd(A, full_matrices=False)
            model = V[-1]
            if self.get_ellipsoid_params(model) is None:
                return None
            return model
        except Exception:
            return None

    def get_error(self, data, model):
        if model is None:
            return np.full(data.shape[0], np.inf)
        A = self.get_design_matrix(data)
        return np.abs(A @ model)

    def get_ellipsoid_params(self, model):
        """Extract ellipsoid parameters from implicit equation coefficients.

        Returns:
            Tuple (x0, y0, z0, a, b, c, R) where (x0,y0,z0) is the center,
            (a,b,c) are the semi-axes, and R is the 3x3 rotation matrix.
            Returns None if the model does not represent a valid ellipsoid.
        """
        if model is None:
            return None
        a, b, c, d, e, f, g, h, i, j = model

        # 3x3 Quadric coefficient matrix and 3x1 linear vector
        A0 = np.array([
            [a, d / 2.0, e / 2.0],
            [d / 2.0, b, f / 2.0],
            [e / 2.0, f / 2.0, c],
        ], dtype=np.float64)
        b0 = np.array([g, h, i], dtype=np.float64)

        # 1. Analytic center: x0 = -0.5 * A0^{-1} * b0
        try:
            center = -0.5 * np.linalg.solve(A0, b0)
        except np.linalg.LinAlgError:
            return None

        # 2. Translated constant: k = j + 0.5 * b0^T * center
        k = j + 0.5 * np.dot(b0, center)
        if np.isclose(k, 0.0):
            return None

        # 3. Normalized center quadric: x'^T * (A0 / -k) * x' = 1
        M = A0 / (-k)

        # 4. Eigendecomposition of real symmetric matrix M
        try:
            eigenvals, R = np.linalg.eigh(M)
        except np.linalg.LinAlgError:
            return None

        # Ellipsoid requires strictly positive eigenvalues
        if np.any(eigenvals <= 1e-7):
            return None

        # 5. Semi-axes lengths
        semi_axes = 1.0 / np.sqrt(eigenvals)

        # Ensure right-handed coordinate frame
        if np.linalg.det(R) < 0:
            R[:, 0] = -R[:, 0]

        return (
            float(center[0]),
            float(center[1]),
            float(center[2]),
            float(semi_axes[0]),
            float(semi_axes[1]),
            float(semi_axes[2]),
            R,
        )


class NormalLeastSquaresModel:
    """Fits a cone-like surface normal distribution. Data shape: (N, 3) unit normals.

    Finds the axis vector that minimizes pairwise cosine angle differences,
    i.e. all normals make approximately the same angle with the axis.
    """

    def fit(self, data):
        init_guess = np.array([0.57735027, 0.57735027, 0.57735027])
        data_size = len(data)
        if data_size > 100:
            data = data[np.random.choice(data_size, 100, replace=False)]
        # Minimize pairwise cosine angle differences
        result = scipy.optimize.minimize(self._angle_diff, init_guess, args=(data,))
        vector = result.x / np.linalg.norm(result.x)
        angle = np.mean(np.arccos(np.dot(data, vector)))
        return vector, angle

    def get_error(self, data, model):
        vector, angle = model
        angles = np.arccos(np.dot(data, vector))
        return np.abs(angles - angle)

    @staticmethod
    def _angle_diff(X, normals):
        X = X / np.linalg.norm(X)
        size = len(normals)
        cos_theta = np.dot(normals, X)
        diff_matrix = cos_theta[:, np.newaxis] - cos_theta[np.newaxis, :]
        return np.sum(diff_matrix[np.triu_indices(size, k=1)] ** 2)


class ConeAxisLeastSquaresModel:
    """Fits a cone axis from surface normals by minimizing angle variance.

    Data shape: (N, 3) unit normals.
    """

    def fit(self, data):
        init_guess = np.array(
            [0.57735027, 0.57735027, 0.57735027]
        )  # [1,1,1] normalized
        # Minimize variance of angles between normals and candidate axis
        result = scipy.optimize.minimize(
            self._angle_diff_variance, init_guess, args=(data,)
        )
        vector = result.x / np.linalg.norm(result.x)
        angle = np.mean(np.arccos(np.dot(data, vector)))
        return vector, angle

    def get_error(self, data, model):
        vector, angle = model
        angles = np.arccos(np.clip(np.dot(data, vector), -1, 1))
        return np.abs(angles - angle)

    @staticmethod
    def _angle_diff_variance(X, normals):
        X = X / np.linalg.norm(X)
        angles = np.arccos(np.clip(np.dot(normals, X), -1, 1))
        return np.var(angles)


# =============================================================================
# Synthetic shape point cloud generators
# =============================================================================


def generate_cube_points(
    size=(10, 10, 10), delta=0.0, points_density=1, total_points=10000
):
    """Generate surface points on a cuboid.

    Args:
        size: (x, y, z) dimensions of the cuboid.
        delta: Random noise magnitude added to each coordinate.
        points_density: Points per unit area. If 0, use total_points instead.
        total_points: Total number of points (used when points_density == 0).

    Returns:
        List of [x, y, z] points on the cuboid surface.
    """
    assert min(size) > 0, "cube(x, y, z) should > 0"
    assert points_density >= 0, "number of points density should >= 0"
    assert total_points > 0, "number of points should > 0"

    half_size = np.array(size) / 2
    points = []

    area1 = size[0] * size[1]  # top/bottom
    area2 = size[1] * size[2]  # left/right (yz)
    area3 = size[0] * size[2]  # front/back (xz)
    total_area = 2 * (area1 + area2 + area3)

    def _noise():
        return np.random.uniform(-1, 1) * delta

    # Top and bottom surfaces
    if points_density != 0:
        num_points_tb = int(size[0] * size[1] * points_density)
    else:
        num_points_tb = int(total_points * (area1 / total_area))
    for _ in range(num_points_tb):
        x = np.random.uniform(-1, 1) * half_size[0]
        y = np.random.uniform(-1, 1) * half_size[1]
        points.append([x + _noise(), y + _noise(), half_size[2] + _noise()])
    for _ in range(num_points_tb):
        x = np.random.uniform(-1, 1) * half_size[0]
        y = np.random.uniform(-1, 1) * half_size[1]
        points.append([x + _noise(), y + _noise(), -half_size[2] + _noise()])

    # Left/right surfaces (yz)
    if points_density != 0:
        num_point_yz = int(size[1] * size[2] * points_density)
    else:
        num_point_yz = int(total_points * (area2 / total_area))
    for _ in range(num_point_yz):
        y = np.random.uniform(-1, 1) * half_size[1]
        z = np.random.uniform(-1, 1) * half_size[2]
        points.append([half_size[0] + _noise(), y + _noise(), z + _noise()])
    for _ in range(num_point_yz):
        y = np.random.uniform(-1, 1) * half_size[1]
        z = np.random.uniform(-1, 1) * half_size[2]
        points.append([-half_size[0] + _noise(), y + _noise(), z + _noise()])

    # Front/back surfaces (xz)
    if points_density != 0:
        num_point_xz = int(size[0] * size[2] * points_density)
    else:
        num_point_xz = int(total_points * (area3 / total_area))
    for _ in range(num_point_xz):
        x = np.random.uniform(-1, 1) * half_size[0]
        z = np.random.uniform(-1, 1) * half_size[2]
        points.append([x + _noise(), half_size[1] + _noise(), z + _noise()])
    for _ in range(num_point_xz):
        x = np.random.uniform(-1, 1) * half_size[0]
        z = np.random.uniform(-1, 1) * half_size[2]
        points.append([x + _noise(), -half_size[1] + _noise(), z + _noise()])

    return points


def generate_cone_points(
    r_bottom=10,
    r_top_ratio=0.5,
    height=20,
    delta=0.0,
    points_density=1,
    total_points=10000,
):
    """Generate surface points on a truncated cone (frustum).

    Args:
        r_bottom: Bottom radius.
        r_top_ratio: Ratio of top radius to bottom radius.
        height: Height of the frustum.
        delta: Random noise magnitude.
        points_density: Points per unit area. If 0, use total_points instead.
        total_points: Total number of points (used when points_density == 0).

    Returns:
        (M, 3) numpy array of points on the frustum surface.
    """
    assert r_bottom > 0, "cone r_bottom should > 0"
    assert height > 0, "cone height should > 0"
    assert points_density >= 0, "number of points density should >= 0"
    assert total_points > 0, "number of points should > 0"

    r_top = r_bottom * r_top_ratio
    half_height = height / 2
    points = []

    area_top = np.pi * r_top * r_top
    area_bottom = np.pi * r_bottom * r_bottom
    slant = np.sqrt((r_bottom - r_top) ** 2 + height**2)
    area_lateral = np.pi * (r_top + r_bottom) * slant
    total_area = area_top + area_bottom + area_lateral

    def _noise():
        return np.random.uniform(-1, 1) * delta

    # Top cap
    if points_density != 0:
        num_top = int(np.pi * r_top * r_top * points_density)
    else:
        num_top = int(total_points * (area_top / total_area))
    for _ in range(num_top):
        r = np.random.uniform() * r_top
        phi = 2 * np.pi * np.random.rand()
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x + _noise(), y + _noise(), half_height + _noise()])

    # Bottom cap
    if points_density != 0:
        num_bottom = int(np.pi * r_bottom * r_bottom * points_density)
    else:
        num_bottom = int(total_points * (area_bottom / total_area))
    for _ in range(num_bottom):
        r = np.random.uniform() * r_bottom
        phi = 2 * np.pi * np.random.rand()
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x + _noise(), y + _noise(), -half_height + _noise()])

    # Lateral surface
    if points_density != 0:
        num_lateral = int(np.pi * (r_top + r_bottom) * slant * points_density)
    else:
        num_lateral = int(total_points * (area_lateral / total_area))
    for _ in range(num_lateral):
        ratio = np.random.uniform(-1, 1)
        ratio_0_1 = (ratio + 1) / 2
        z = ratio * half_height
        r = ratio_0_1 * (r_top - r_bottom) + r_bottom
        phi = 2 * np.pi * np.random.rand()
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x + _noise(), y + _noise(), z + _noise()])

    return np.array(points)


def generate_ellipsoid_points(a=10, b=10, c=10, total_points=10000):
    """Generate surface points on an ellipsoid.

    Parametric equations:
        x = a * sin(theta) * cos(phi)
        y = b * sin(theta) * sin(phi)
        z = c * cos(theta)

    Args:
        a, b, c: Semi-axis lengths.
        total_points: Number of points to generate.

    Returns:
        (total_points, 3) numpy array of surface points.
    """
    theta = np.pi * np.random.rand(total_points)
    phi = 2 * np.pi * np.random.rand(total_points)
    x = a * np.sin(theta) * np.cos(phi)
    y = b * np.sin(theta) * np.sin(phi)
    z = c * np.cos(theta)
    return np.column_stack((x, y, z))


# =============================================================================
# Geometric utilities
# =============================================================================


def points_to_point_distance(points, point):
    """Compute Euclidean distance from each point in an array to a single point."""
    return np.linalg.norm(points - point, ord=2, axis=1)


def fit_circle(points, num_iterations=100, threshold=0.01):
    """Circle fitting for 2D points using fast algebraic Kåsa initialization with RANSAC fallback.

    Args:
        points: (N, 2) array of 2D points.
        num_iterations: Number of RANSAC iterations.
        threshold: Inlier distance threshold.

    Returns:
        Tuple of (best_circle, best_num_inliers, outlier_indices) where
        best_circle is (center, radius) or None.
    """
    points = np.asarray(points)
    num_total_points = len(points)
    if num_total_points < 3:
        return None, 0, None

    # Fast direct algebraic fit shortcut (0.01 ms)
    kasa_fit = fit_circle_kasa(points)
    if kasa_fit is not None:
        c_kasa, r_kasa = kasa_fit
        dists = np.abs(points_to_point_distance(points, c_kasa) - r_kasa)
        inliers_idx = np.where(dists <= threshold)[0]
        if len(inliers_idx) >= 0.7 * num_total_points:
            outliers_idx = np.where(dists > threshold)[0]
            return (c_kasa, r_kasa), len(inliers_idx), outliers_idx

    best_circle = None
    best_num_inliers = 0
    remained_points_indices = None
    num_iterations = min(num_iterations, 100)

    for _ in range(num_iterations):
        random_indices = np.random.choice(num_total_points, 3, replace=False)
        circle_points = points[random_indices]
        x1, x2, x3 = circle_points[0][0], circle_points[1][0], circle_points[2][0]
        y1, y2, y3 = circle_points[0][1], circle_points[1][1], circle_points[2][1]
        A = np.array(
            [
                [2 * (x1 - x2), 2 * (y1 - y2)],
                [2 * (x1 - x3), 2 * (y1 - y3)],
                [2 * (x2 - x3), 2 * (y2 - y3)],
            ]
        )
        B = np.array(
            [
                [x1**2 + y1**2 - x2**2 - y2**2],
                [x1**2 + y1**2 - x3**2 - y3**2],
                [x2**2 + y2**2 - x3**2 - y3**2],
            ]
        )
        try:
            X = np.linalg.lstsq(A, B, rcond=None)[0]
            center = np.array([X[0][0], X[1][0]])
            r = np.linalg.norm(circle_points[0] - center)
        except Exception:
            continue
        outliers = np.where(
            abs(points_to_point_distance(points, center) - r) > threshold
        )[0]
        num_inliers = num_total_points - len(outliers)
        if num_inliers > best_num_inliers:
            best_circle = (center, r)
            best_num_inliers = num_inliers
            remained_points_indices = outliers

    return best_circle, best_num_inliers, remained_points_indices


def find_orthogonal_vectors(normal_vector):
    """Find two mutually orthogonal vectors perpendicular to a given normal.

    Args:
        normal_vector: (3,) array representing the surface normal.

    Returns:
        Tuple of two (3,) orthogonal vectors lying in the plane.
    """
    v1 = np.random.rand(3)
    v1_proj = np.dot(v1, normal_vector) / np.linalg.norm(normal_vector) * normal_vector
    v1_ortho = v1 - v1_proj
    v2_ortho = np.cross(normal_vector, v1_ortho)
    return v1_ortho, v2_ortho


def depth_to_pointcloud(depth_image, fx, fy, cx, cy):
    """Convert a depth image to a 3D point cloud using pinhole camera intrinsics.

    Args:
        depth_image: (H, W) depth map (values in camera units, e.g. mm).
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in pixels.

    Returns:
        (M, 3) array of 3D points (zero-depth points removed).
    """
    height, width = depth_image.shape
    u, v = np.meshgrid(np.arange(1, width + 1), np.arange(1, height + 1))
    z = depth_image
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pointcloud = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    nonzero_indices = np.all(pointcloud != [0, 0, 0], axis=1)
    return pointcloud[nonzero_indices]


# =============================================================================
# Visualization
# =============================================================================


def view_coordinate(poses, pcd, length=100):
    """Visualize poses as coordinate frames on a point cloud using Open3D.

    Args:
        poses: List of SE3 or 4x4 homogeneous transformation matrices.
        pcd: Open3D point cloud to display.
        length: Size of coordinate frame axes.
    """
    scene = [pcd]
    for pose in poses:
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=length / 2, origin=[0, 0, 0]
        )
        coord.transform(pose)
        scene.append(coord)
    o3d.visualization.draw_geometries(scene)


# =============================================================================
# Pose filtering and sorting utilities
# =============================================================================


def _angle_between_axes(pose_axis, ref_axis):
    """Compute the angle (radians) between two 3D vectors."""
    pose_axis = np.array(pose_axis)
    ref_axis = np.array(ref_axis)
    cos_angle = np.dot(pose_axis, ref_axis) / (
        np.linalg.norm(pose_axis) * np.linalg.norm(ref_axis)
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def filter_pose_by_axis_diff(
    poses, axis=2, ref_axis=(0, 0, 1), t=np.pi / 2, sorted=False
):
    """Filter poses by the angle between a specified axis and a reference axis.

    Args:
        poses: List of SE3 objects or 4x4 arrays.
        axis: Column index of the rotation matrix (0=x, 1=y, 2=z).
        ref_axis: Reference direction vector.
        t: Angle threshold (radians); poses exceeding this are discarded.
        sorted: If True, sort the remaining poses by angle (ascending).

    Returns:
        List of poses whose specified axis is within ``t`` radians of ``ref_axis``.
    """
    valid_poses = []
    angles = []
    for pose in poses:
        if isinstance(pose, SE3):
            pose_axis = pose.A[:3, axis]
        else:
            pose_axis = pose[:3, axis]
        angle = _angle_between_axes(pose_axis, ref_axis)
        if angle <= t:
            valid_poses.append(pose)
            angles.append(angle)
    if sorted:
        sorted_indices = np.argsort(angles)
        valid_poses = [valid_poses[i] for i in sorted_indices]
    return valid_poses


def sort_pose_by_rot_diff(poses, ref_pose):
    """Sort poses by rotation difference from a reference pose.

    Uses the angular distance metric (metric=6) from spatialmath.

    Args:
        poses: List of SE3 poses.
        ref_pose: Reference SE3 pose.

    Returns:
        List of SE3 poses sorted by ascending rotation difference.
    """
    differences = [pose.angdist(ref_pose, metric=6) for pose in poses]
    sorted_indices = np.argsort(differences)
    return [poses[i] for i in sorted_indices]


def filter_pose_by_bin_side(poses, bin_size, ignore_size, bin_pose, threshold):
    """Filter poses based on their alignment with the nearest bin wall.

    Poses near the center of the bin (within ``ignore_size``) are always kept.
    Poses closer to a wall are kept only if their x-axis (approach direction)
    aligns with the wall normal within ``threshold`` radians.

    Args:
        poses: List of SE3 poses.
        bin_size: (x, y) or (x, y, z) dimensions of the bin.
        ignore_size: (x, y) central region where all poses are accepted.
        bin_pose: SE3 pose of the bin center.
        threshold: Angle threshold (radians) for wall alignment.

    Returns:
        Filtered list of SE3 poses.
    """
    half_size = np.asarray(bin_size) / 2
    half_ignore_size = np.asarray(ignore_size) / 2
    pose_filtered = []

    for pose in poses:
        pose_in_bin = bin_pose.inv() * pose
        # Poses in the central region pass unconditionally
        if (
            abs(pose_in_bin.t[0]) < half_ignore_size[0]
            and abs(pose_in_bin.t[1]) < half_ignore_size[1]
        ):
            pose_filtered.append(pose)
            continue
        # Determine which wall the pose is closest to
        if half_size[0] - abs(pose_in_bin.t[0]) < half_size[1] - abs(pose_in_bin.t[1]):
            if pose_in_bin.t[0] > 0:
                angle_between_X = np.arccos(np.dot(bin_pose.n, pose_in_bin.n))
            else:
                angle_between_X = np.arccos(np.dot(-1 * bin_pose.n, pose_in_bin.n))
        else:
            if pose_in_bin.t[1] > 0:
                angle_between_X = np.arccos(np.dot(bin_pose.o, pose_in_bin.n))
            else:
                angle_between_X = np.arccos(np.dot(-1 * bin_pose.o, pose_in_bin.n))
        if angle_between_X < threshold:
            pose_filtered.append(pose)

    return pose_filtered


# =============================================================================
# Gripper feasibility checks
# =============================================================================


def check_pick_pose_for_2finger_gripper(pcd, poses, finger_range, t=10):
    """Filter pick poses by checking for sufficient contact points in the gripper OBB.

    Args:
        pcd: Open3D point cloud.
        poses: List of SE3 candidate grasp poses.
        finger_range: [w, h, d] contact range in x, y, z at the pick pose frame.
        t: Minimum number of points required inside the finger OBB.

    Returns:
        List of SE3 poses that have at least ``t`` points in the contact region.
    """
    filtered = []
    for pose in poses:
        finger_obb = o3d.geometry.OrientedBoundingBox(
            pose.t, pose.R, np.array(finger_range)
        )
        idxs = finger_obb.get_point_indices_within_bounding_box(pcd.points)
        if len(idxs) > t:
            filtered.append(pose)
    return filtered


def check_pick_pose_for_2finger_gripper_range(pcd, poses, finger_range):
    """Filter pick poses by checking if the object fits within the gripper opening.

    Args:
        pcd: Open3D point cloud.
        poses: List of SE3 candidate grasp poses.
        finger_range: Maximum opening width of the gripper (along y-axis in pose frame).

    Returns:
        List of SE3 poses where the object extent fits within the gripper.
    """
    filtered = []
    for pose in poses:
        pcd_trans = o3d.geometry.PointCloud(pcd)
        pcd_trans.transform(pose.inv())
        points = np.asarray(pcd_trans.points)
        range_y = np.max(points[:, 1]) - np.min(points[:, 1])
        if range_y < finger_range:
            filtered.append(pose)
        else:
            print(f"range_y: {range_y}, finger_range: {finger_range}")
    return filtered
