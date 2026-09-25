"""Grasp pose generation for basic geometric shapes.

Generates candidate SE3 grasp poses for three primitive types:
  - **Cuboid**: side, center, and end grasps
  - **Truncated cone (frustum)**: side, center, and end grasps
  - **Ellipsoid**: side and center grasps

All methods are static; the class serves as a namespace.

Pose filtering and gripper-feasibility checks are in
:mod:`lggpf.utils.pointcloud`.
"""

import numpy as np
import open3d as o3d
from math import pi, cos, sin, atan
from spatialmath import SE3, SO3


class PickPose:
    """Static methods for generating candidate grasp poses on shape primitives.

    All poses are expressed in the shape's local (canonical) frame.
    The caller is responsible for transforming them to the world/base frame.
    """

    # =========================================================================
    # Cuboid grasp poses
    # =========================================================================

    @staticmethod
    def gen_cube_side_pick_poses(size, num_each_side):
        """Generate grasp poses on the side faces of a cuboid.

        Args:
            size: (x, y, z) full dimensions of the cuboid.
            num_each_side: Number of grasp positions along each edge.

        Returns:
            List of SE3 grasp poses.
        """
        pick_poses = []
        half_size = np.asarray(size) / 2
        for idx in range(3):
            idx1, idx2 = [i for i in range(3) if i != idx]
            tL = np.array(
                [[half_size[0], half_size[1], half_size[2]]] * 4 * num_each_side
            )
            combinations = [(i, j) for i in range(2) for j in range(2)]
            for idx_t in range(4):
                step = 2 * half_size[idx] / (num_each_side + 1)
                for idx_n in range(num_each_side):
                    tL[idx_n + idx_t * num_each_side, idx] = (
                        half_size[idx] - (idx_n + 1) * step
                    )
                    tL[idx_n + idx_t * num_each_side, idx1] *= (-1) ** combinations[
                        idx_t
                    ][0]
                    tL[idx_n + idx_t * num_each_side, idx2] *= (-1) ** combinations[
                        idx_t
                    ][1]
                    xL = np.array([[0, 0, 0]] * 4)
                    zL = np.array([[0, 0, 0]] * 2)
                    xL[0, idx1] = 1
                    xL[1, idx1] = -1
                    xL[2, idx2] = 1
                    xL[3, idx2] = -1
                    zL[0, idx2] = (-1) ** (combinations[idx_t][1] + 1)
                    zL[1, idx1] = (-1) ** (combinations[idx_t][0] + 1)
                    t_vec = tL[idx_n + idx_t * num_each_side]
                    pick_poses.append(
                        SE3.Rt(SO3.TwoVectors(x=xL[0], z=zL[0]), t_vec)
                        * SE3.Rz(np.pi / 2)
                    )
                    pick_poses.append(
                        SE3.Rt(SO3.TwoVectors(x=xL[1], z=zL[0]), t_vec)
                        * SE3.Rz(np.pi / 2)
                    )
                    pick_poses.append(
                        SE3.Rt(SO3.TwoVectors(x=xL[2], z=zL[1]), t_vec)
                        * SE3.Rz(np.pi / 2)
                    )
                    pick_poses.append(
                        SE3.Rt(SO3.TwoVectors(x=xL[3], z=zL[1]), t_vec)
                        * SE3.Rz(np.pi / 2)
                    )
        return pick_poses

    @staticmethod
    def gen_cube_center_pick_poses(size, center=None, gripper_depth=0.1):
        """Generate grasp poses through the center of a cuboid along each axis.

        Args:
            size: (x, y, z) full dimensions.
            center: (x, y, z) offset from shape origin. Defaults to [0, 0, 0].
            gripper_depth: How deep the gripper penetrates from the surface.

        Returns:
            List of SE3 grasp poses.
        """
        if center is None:
            center = [0.0, 0.0, 0.0]
        pick_poses = []
        half_size = np.array(size) / 2
        for idx_z in range(3):
            idx1, idx2 = [i for i in range(3) if i != idx_z]
            xL = np.array([[0.0, 0.0, 0.0]] * 4)
            zL = np.array([[0.0, 0.0, 0.0]] * 2)
            center_top = np.array([0.0, 0.0, 0.0])
            center_bottom = np.array([0.0, 0.0, 0.0])
            center_top[idx_z] = half_size[idx_z] - gripper_depth
            center_bottom[idx_z] = -half_size[idx_z] + gripper_depth
            zL[0, idx_z] = 1
            zL[1, idx_z] = -1
            xL[0, idx1] = 1
            xL[1, idx1] = -1
            xL[2, idx2] = 1
            xL[3, idx2] = -1
            c = np.asarray(center)
            # Poses at bottom offset
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[0], z=zL[0]), c + center_bottom)
            )
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[1], z=zL[0]), c + center_bottom)
            )
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[2], z=zL[0]), c + center_bottom)
            )
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[3], z=zL[0]), c + center_bottom)
            )
            # Poses at top offset
            pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[0], z=zL[1]), c + center_top))
            pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[1], z=zL[1]), c + center_top))
            pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[2], z=zL[1]), c + center_top))
            pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[3], z=zL[1]), c + center_top))
            # Poses at geometric center
            pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[0], z=zL[0]), c))
            pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[1], z=zL[0]), c))
            pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[2], z=zL[0]), c))
            pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[3], z=zL[0]), c))
            pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[0], z=zL[1]), c))
            pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[1], z=zL[1]), c))
            pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[2], z=zL[1]), c))
            pick_poses.append(SE3.Rt(SO3.TwoVectors(x=xL[3], z=zL[1]), c))
        return pick_poses

    @staticmethod
    def gen_cube_end_pick_poses(size, center=None, gripper_depth=0.1):
        """Generate grasp poses at the edges/corners of a cuboid.

        Args:
            size: (x, y, z) full dimensions.
            center: (x, y, z) offset from shape origin. Defaults to [0, 0, 0].
            gripper_depth: How deep the gripper penetrates from the surface.

        Returns:
            List of SE3 grasp poses.
        """
        if center is None:
            center = [0.0, 0.0, 0.0]
        pick_poses = []
        half_size = np.array(size) / 2
        for idx_z in range(3):
            idx1, idx2 = [i for i in range(3) if i != idx_z]
            xL = np.array([[0.0, 0.0, 0.0]] * 4)
            zL = np.array([[0.0, 0.0, 0.0]] * 2)
            center_top = np.array([0.0, 0.0, 0.0])
            center_bottom = np.array([0.0, 0.0, 0.0])
            center_top[idx_z] = half_size[idx_z]
            center_bottom[idx_z] = -half_size[idx_z]
            center_x = np.array([0.0, 0.0, 0.0])
            center_y = np.array([0.0, 0.0, 0.0])
            center_x[idx1] = half_size[idx1] - gripper_depth
            center_y[idx2] = -half_size[idx2] + gripper_depth
            zL[0, idx_z] = 1
            zL[1, idx_z] = -1
            xL[0, idx1] = 1
            xL[1, idx1] = -1
            xL[2, idx2] = 1
            xL[3, idx2] = -1
            c = np.asarray(center)
            # Bottom face, positive/negative offsets
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[0], z=zL[0]), c + center_bottom + center_x)
            )
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[1], z=zL[0]), c + center_bottom + center_x)
            )
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[2], z=zL[0]), c + center_bottom + center_y)
            )
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[3], z=zL[0]), c + center_bottom + center_y)
            )
            # Top face, positive/negative offsets
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[0], z=zL[1]), c + center_top + center_x)
            )
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[1], z=zL[1]), c + center_top + center_x)
            )
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[2], z=zL[1]), c + center_top + center_y)
            )
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[3], z=zL[1]), c + center_top + center_y)
            )
            # Mirror offsets
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[0], z=zL[0]), c + center_bottom - center_x)
            )
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[1], z=zL[0]), c + center_bottom - center_x)
            )
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[2], z=zL[0]), c + center_bottom - center_y)
            )
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[3], z=zL[0]), c + center_bottom - center_y)
            )
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[0], z=zL[1]), c + center_top - center_x)
            )
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[1], z=zL[1]), c + center_top - center_x)
            )
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[2], z=zL[1]), c + center_top - center_y)
            )
            pick_poses.append(
                SE3.Rt(SO3.TwoVectors(x=xL[3], z=zL[1]), c + center_top - center_y)
            )
        return pick_poses

    # =========================================================================
    # Truncated cone (frustum) grasp poses
    # =========================================================================

    @staticmethod
    def gen_cone_side_pick_poses(
        height, top_r, bottom_r, num_each_side, gripper_depth=0.1
    ):
        """Generate grasp poses on the lateral surface of a truncated cone.

        Args:
            height: Height of the frustum.
            top_r: Top circle radius.
            bottom_r: Bottom circle radius.
            num_each_side: Number of angular samples around the circumference.
            gripper_depth: How far from the cap edge the gripper is placed.

        Returns:
            List of SE3 grasp poses.
        """
        pick_poses = []
        step = 0 if num_each_side == 1 else 2 * pi / num_each_side
        alpha = atan(abs(top_r - bottom_r) / height)
        for idx in range(num_each_side):
            t = idx * step
            top_x = top_r * cos(t)
            top_y = top_r * sin(t)
            top_z = height / 2 - gripper_depth
            bottom_x = bottom_r * cos(t)
            bottom_y = bottom_r * sin(t)
            bottom_z = -1 * height / 2 + gripper_depth
            top_xL = np.array([[top_x, top_y, 0]])
            top_zL = np.array([[0, 0, -1]])
            bottom_xL = np.array([[bottom_x, bottom_y, 0]])
            bottom_zL = np.array([[0, 0, 1]])
            if top_r > bottom_r:
                top_T = SE3.Rt(
                    SO3.TwoVectors(x=top_xL, z=top_zL) * SO3.Ry(-alpha),
                    [top_x, top_y, top_z],
                )
                top_T2 = SE3.Rt(
                    SO3.TwoVectors(x=top_xL, z=top_zL), [top_x, top_y, top_z]
                )
                bottom_T = SE3.Rt(
                    SO3.TwoVectors(x=bottom_xL, z=bottom_zL) * SO3.Ry(-alpha),
                    [bottom_x, bottom_y, bottom_z],
                )
                bottom_T2 = SE3.Rt(
                    SO3.TwoVectors(x=bottom_xL, z=bottom_zL),
                    [bottom_x, bottom_y, bottom_z],
                )
            else:
                top_T = SE3.Rt(
                    SO3.TwoVectors(x=top_xL, z=top_zL) * SO3.Ry(alpha),
                    [top_x, top_y, top_z],
                )
                top_T2 = SE3.Rt(
                    SO3.TwoVectors(x=top_xL, z=top_zL), [top_x, top_y, top_z]
                )
                bottom_T = SE3.Rt(
                    SO3.TwoVectors(x=bottom_xL, z=bottom_zL) * SO3.Ry(-alpha),
                    [bottom_x, bottom_y, bottom_z],
                )
                bottom_T2 = SE3.Rt(
                    SO3.TwoVectors(x=bottom_xL, z=bottom_zL),
                    [bottom_x, bottom_y, bottom_z],
                )
            pick_poses.append(top_T * SE3.Rz(np.pi / 2))
            pick_poses.append(top_T2 * SE3.Rz(np.pi / 2))
            pick_poses.append(bottom_T * SE3.Rz(np.pi / 2))
            pick_poses.append(bottom_T2 * SE3.Rz(np.pi / 2))

        return pick_poses

    @staticmethod
    def gen_cone_center_pick_poses(
        height, num_each_position, center=None, gripper_depth=0.1
    ):
        """Generate grasp poses through the center/axis of a truncated cone.

        Produces top-down, bottom-up, and lateral center grasps at various
        rotations around the cone axis.

        Args:
            height: Height of the frustum.
            num_each_position: Number of angular samples.
            center: Center offset. Defaults to [0, 0, 0].
            gripper_depth: Depth offset from cap surfaces.

        Returns:
            List of SE3 grasp poses.
        """
        if center is None:
            center = [0, 0, 0]
        pick_poses = []
        step = 0 if num_each_position == 1 else 2 * pi / num_each_position
        # Top-down and bottom-up grasps
        for idx in range(num_each_position):
            top_z = height / 2 - gripper_depth
            bottom_z = -1 * height / 2 + gripper_depth
            t = step * idx
            top_T = SE3.Rt(SO3.Rz(t) * SO3.Rx(pi), [0, 0, top_z])
            bottom_T = SE3.Rt(SO3.Rz(-t), [0, 0, bottom_z])
            pick_poses.append(top_T)
            pick_poses.append(bottom_T)
        # Lateral center grasps
        for idx in range(num_each_position):
            t = step * idx
            center_T1 = SE3.Rt(SO3.Ry(pi / 2) * SO3.Rx(t), center)
            center_T2 = SE3.Rt(SO3.Ry(-pi / 2) * SO3.Rx(t), center)
            pick_poses.append(center_T1)
            pick_poses.append(center_T2)

        return pick_poses

    @staticmethod
    def gen_cone_end_pick_poses(
        height, num_each_position, center=None, gripper_depth=0.1
    ):
        """Generate grasp poses at the ends of a truncated cone.

        Produces lateral grasps near the top and bottom caps.

        Args:
            height: Height of the frustum.
            num_each_position: Number of angular samples.
            center: Center offset. Defaults to [0, 0, 0].
            gripper_depth: Depth offset from cap surfaces.

        Returns:
            List of SE3 grasp poses.
        """
        if center is None:
            center = [0, 0, 0]
        pick_poses = []
        step = 0 if num_each_position == 1 else 2 * pi / num_each_position
        for idx in range(num_each_position):
            top_z = height / 2 - gripper_depth
            bottom_z = -1 * height / 2 + gripper_depth
            t = step * idx
            top_T1 = SE3.Rt(SO3.Ry(pi / 2) * SO3.Rx(t), [0, 0, top_z])
            top_T2 = SE3.Rt(SO3.Ry(-pi / 2) * SO3.Rx(t), [0, 0, top_z])
            bottom_T1 = SE3.Rt(SO3.Ry(pi / 2) * SO3.Rx(t), [0, 0, bottom_z])
            bottom_T2 = SE3.Rt(SO3.Ry(-pi / 2) * SO3.Rx(t), [0, 0, bottom_z])
            pick_poses.append(top_T1)
            pick_poses.append(top_T2)
            pick_poses.append(bottom_T1)
            pick_poses.append(bottom_T2)
        return pick_poses

    # =========================================================================
    # Ellipsoid grasp poses
    # =========================================================================

    @staticmethod
    def gen_ellipsoid_side_pick_poses(num, a, b, c, T, pcd):
        """Generate grasp poses on the side of an ellipsoid.

        Identifies the main axis of the point cloud (axis with largest
        discrepancy between AABB extent and fitted semi-axis), then
        places grasp poses around the ellipsoid cross-section.

        Args:
            num: Number of angular samples around the cross-section.
            a, b, c: Fitted ellipsoid semi-axes.
            T: SE3 transform from ellipsoid local frame to camera frame.
            pcd: Open3D point cloud (in camera frame).

        Returns:
            List of SE3 grasp poses (in camera frame).
        """
        pcdCp = o3d.geometry.PointCloud(pcd)
        pcdCp.transform(T.inv())
        aabb = pcdCp.get_axis_aligned_bounding_box()
        pts = np.asarray(pcdCp.points)
        pick_poses = []
        step = 0 if num == 1 else 2 * pi / num
        ABC = np.array([a, b, c])
        aabbExtent = aabb.get_extent()
        aabbCenter = aabb.get_center()
        diff = np.abs(aabbExtent - 2 * ABC)
        mainIdx = np.argmax(diff)

        # Find the extremal point along the main axis
        minPtIdxInMainDir = np.argmax(pts[:, mainIdx])
        minPt = pts[minPtIdxInMainDir]
        maxPtRef = -1 * minPt
        distance2maxPtRef = np.linalg.norm(pts - maxPtRef)
        maxPt = pts[np.argmin(distance2maxPtRef)]

        # Compute corrective rotation to align the partial ellipsoid
        correctAngle = np.arccos(np.dot(-minPt, maxPt - minPt))
        unitVec = np.array([0.0, 0.0, 0.0])
        unitVec[mainIdx] = 1.0
        rotDir = np.cross(minPt, unitVec)
        T_correct = SE3.AngVec(correctAngle, rotDir)

        # Recompute AABB after correction
        pcdCp.transform(T_correct.inv())
        aabb = pcdCp.get_axis_aligned_bounding_box()
        aabbExtent = aabb.get_extent()
        aabbCenter = aabb.get_center()
        ABC = aabbExtent / 2

        idx1, idx2 = [i for i in range(3) if i != mainIdx]
        XYZ = np.array([0.0, 0.0, 0.0])
        top = aabbCenter[mainIdx] + aabbExtent[mainIdx] / 2
        bottom = aabbCenter[mainIdx] - aabbExtent[mainIdx] / 2
        XYZ[mainIdx] = top if np.abs(top) < np.abs(bottom) else bottom

        z_dir = np.array([0.0, 0.0, 0.0])
        z_dir[mainIdx] = aabbCenter[mainIdx]
        z_dir = z_dir / np.linalg.norm(z_dir)

        for idx in range(num):
            t = step * idx
            xyz = [0, 0, 0]
            xyz[idx1] = XYZ[idx1] + ABC[idx1] * np.cos(t)
            xyz[idx2] = XYZ[idx2] + ABC[idx2] * np.sin(t)
            xyz[mainIdx] = XYZ[mainIdx]
            x_dir = np.array([0.0, 0.0, 0.0])
            x_dir[idx1] = np.cos(t)
            x_dir[idx2] = np.sin(t)
            T_pick1 = SE3.Rt(SO3.TwoVectors(x=x_dir, z=z_dir), xyz)
            T_pick2 = SE3.Rt(SO3.TwoVectors(x=-1 * x_dir, z=z_dir), xyz)
            pick_poses.append(T * T_correct * T_pick1 * SE3.Rz(np.pi / 2))
            pick_poses.append(T * T_correct * T_pick2 * SE3.Rz(np.pi / 2))
        return pick_poses

    @staticmethod
    def gen_ellipsoid_center_pick_poses(num_each_direction, center=None):
        """Generate grasp poses through the center of an ellipsoid.

        Produces grasps along all three principal axes at various rotations.

        Args:
            num_each_direction: Number of angular samples per axis.
            center: Center position. Defaults to [0, 0, 0].

        Returns:
            List of SE3 grasp poses.
        """
        if center is None:
            center = [0, 0, 0]
        pick_poses = []
        step = 0 if num_each_direction == 1 else 2 * pi / num_each_direction
        for idx in range(num_each_direction):
            t = step * idx
            T1 = SE3.Rt(SO3.Rz(t), center)
            T2 = SE3.Rt(SO3.Rx(pi) * SO3.Rz(t), center)
            T3 = SE3.Rt(SO3.Ry(pi / 2) * SO3.Rz(t), center)
            T4 = SE3.Rt(SO3.Ry(pi / 2) * SO3.Rx(pi) * SO3.Rz(t), center)
            T5 = SE3.Rt(SO3.Rx(pi / 2) * SO3.Rz(t), center)
            T6 = SE3.Rt(SO3.Rx(pi / 2) * SO3.Rx(pi) * SO3.Rz(t), center)
            pick_poses.extend([T1, T2, T3, T4, T5, T6])
        return pick_poses
