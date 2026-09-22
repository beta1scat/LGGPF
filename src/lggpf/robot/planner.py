"""Robot model, collision detection, and trajectory planning.

Uses Pinocchio for forward kinematics and Coal for collision checking.
Both are optional dependencies (``pip install pin coal``).

Cleaned from the original ``modules/robot/robot.py``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

try:
    import pinocchio as pin
    from pinocchio.visualize import MeshcatVisualizer
    import coal

    _HAS_PINOCCHIO = True
except ImportError:
    _HAS_PINOCCHIO = False

if TYPE_CHECKING:
    import pinocchio as pin
    import coal


def _require_pinocchio():
    if not _HAS_PINOCCHIO:
        raise ImportError(
            "pinocchio, coal, and meshcat are required for robot planning. "
            "Install with: pip install pin coal meshcat"
        )


# =============================================================================
# RM65 Robot Dynamic and Kinematic Specifications
# =============================================================================

# RealMan RM65-B 6-DOF Robot kinematic limits (radians)
# Limits: [-178°, +178°], [-130°, +130°], [-135°, +135°], [-178°, +178°], [-128°, +128°], [-360°, +360°]
RM65_JOINT_LIMITS_LOWER = np.array(
    [-3.106686, -2.268928, -2.356194, -3.106686, -2.234021, -6.283185],
    dtype=np.float64,
)
RM65_JOINT_LIMITS_UPPER = np.array(
    [3.106686, 2.268928, 2.356194, 3.106686, 2.234021, 6.283185],
    dtype=np.float64,
)

# Maximum joint velocities (rad/s): [180°/s, 180°/s, 225°/s, 225°/s, 225°/s, 360°/s]
RM65_VELOCITY_LIMITS = np.array(
    [np.pi, np.pi, 1.25 * np.pi, 1.25 * np.pi, 1.25 * np.pi, 2.0 * np.pi],
    dtype=np.float64,
)

# Maximum joint accelerations (rad/s^2)
RM65_ACCELERATION_LIMITS = np.array(
    [10.0, 10.0, 15.0, 15.0, 15.0, 20.0],
    dtype=np.float64,
)


# =============================================================================
# Collision detection
# =============================================================================


class CollisionResult:
    """Container for collision check results."""

    def __init__(self):
        self.is_collision: bool = False
        self.contact_points = None
        self.min_distance: float = float("inf")
        self.colliding_pair: tuple = None


class CollisionDetector:
    """Checks collisions between robot links and environment objects with clearance margin.

    Args:
        robot: A :class:`Robot` instance.
        safety_margin: Safety clearance distance margin in meters (default: 0.02 m).
    """

    def __init__(self, robot: Robot, safety_margin: float = 0.02):
        self.robot = robot
        self.safety_margin = float(safety_margin)

    def check_collision(
        self,
        q: np.ndarray,
        stop_at_first_collision: bool = True,
        safety_margin: float | None = None,
    ) -> CollisionResult:
        """Check whether the robot at configuration *q* collides with the environment.

        Args:
            q: Joint angles (radians).
            stop_at_first_collision: Return immediately on first collision.
            safety_margin: Optional override for obstacle clearance margin (meters).

        Returns:
            CollisionResult with ``is_collision`` flag and clearance distance.
        """
        self.robot.update_state(q)
        result = CollisionResult()
        margin = self.safety_margin if safety_margin is None else float(safety_margin)

        for idx_r, robot_obj in enumerate(self.robot.robot_collision_objects):
            if idx_r == len(self.robot.robot_collision_objects) - 1:
                T1 = coal.Transform3s()
                T1.setTranslation(self.robot.data.oMi[-1].translation)
                T1.setRotation(self.robot.data.oMi[-1].rotation)
            else:
                T1 = coal.Transform3s()
                T1.setTranslation(self.robot.data.oMi[idx_r].translation)
                T1.setRotation(self.robot.data.oMi[idx_r].rotation)

            for env_obj in self.robot.env_collision_objects:
                request = coal.CollisionRequest()
                collision_result = coal.CollisionResult()
                if hasattr(request, "security_margin"):
                    request.security_margin = float(margin)

                coal.collide(
                    robot_obj.collisionGeometry(),
                    T1,
                    env_obj.collisionGeometry(),
                    env_obj.getTransform(),
                    request,
                    collision_result,
                )
                if collision_result.isCollision():
                    result.is_collision = True
                    result.contact_points = collision_result.getContacts()
                    result.colliding_pair = (
                        getattr(robot_obj, "name", f"link_{idx_r}"),
                        getattr(env_obj, "name", "environment"),
                    )
                    if stop_at_first_collision:
                        return result

                # In addition, check Euclidean clearance if distance query is supported
                if margin > 0.0 and hasattr(coal, "distance"):
                    dist_req = coal.DistanceRequest()
                    dist_res = coal.DistanceResult()
                    try:
                        coal.distance(
                            robot_obj.collisionGeometry(),
                            T1,
                            env_obj.collisionGeometry(),
                            env_obj.getTransform(),
                            dist_req,
                            dist_res,
                        )
                        if dist_res.min_distance < result.min_distance:
                            result.min_distance = dist_res.min_distance
                        if dist_res.min_distance < margin:
                            result.is_collision = True
                            result.colliding_pair = (
                                getattr(robot_obj, "name", f"link_{idx_r}"),
                                getattr(env_obj, "name", "environment"),
                            )
                            if stop_at_first_collision:
                                return result
                    except Exception:
                        pass

        return result

    def get_collision_distance(self, q: np.ndarray):
        """Compute minimum distance between robot links and environment objects.

        Args:
            q: Joint angles (radians).

        Returns:
            Tuple of (min_distance, closest_pair) or (inf, None).
        """
        self.robot.update_state(q)
        min_distance = float("inf")
        closest_pair = None

        for idx_r, robot_obj in enumerate(self.robot.robot_collision_objects):
            if idx_r == len(self.robot.robot_collision_objects) - 1:
                T1 = coal.Transform3s()
                T1.setTranslation(self.robot.data.oMi[-1].translation)
                T1.setRotation(self.robot.data.oMi[-1].rotation)
            else:
                T1 = coal.Transform3s()
                T1.setTranslation(self.robot.data.oMi[idx_r].translation)
                T1.setRotation(self.robot.data.oMi[idx_r].rotation)

            for env_obj in self.robot.env_collision_objects:
                request = coal.DistanceRequest()
                distance_result = coal.DistanceResult()
                try:
                    coal.distance(
                        robot_obj.collisionGeometry(),
                        T1,
                        env_obj.collisionGeometry(),
                        env_obj.getTransform(),
                        request,
                        distance_result,
                    )
                    if distance_result.min_distance < min_distance:
                        min_distance = distance_result.min_distance
                        closest_pair = (robot_obj, env_obj)
                except Exception:
                    pass

        return min_distance, closest_pair


# =============================================================================
# Trajectory planner
# =============================================================================


class JointSpacePlanner:
    """Quintic polynomial joint-space trajectory planner with dynamic limits and collision checking.

    Features:
      - Validates RM65 joint angle, velocity, and angular acceleration bounds.
      - Automatically scales execution time T if dynamics limits would be exceeded.
      - Enforces Coal collision safety margin (clearance).
      - Waypoint adaptive subdivision to prevent discrete tunneling through obstacles.

    Args:
        robot: A :class:`Robot` instance.
        safety_margin: Safety clearance distance margin (meters, default: 0.02).
        velocity_limits: Per-joint velocity limits (rad/s). Defaults to RM65 specs.
        acceleration_limits: Per-joint acceleration limits (rad/s^2). Defaults to RM65 specs.
        max_joint_step: Max allowed joint displacement (rad) between consecutive waypoints (default: 0.05).
    """

    def __init__(
        self,
        robot: Robot,
        safety_margin: float = 0.02,
        velocity_limits: np.ndarray | None = None,
        acceleration_limits: np.ndarray | None = None,
        max_joint_step: float = 0.05,
    ):
        self.robot = robot
        self.safety_margin = float(safety_margin)
        self.collision_detector = CollisionDetector(robot, safety_margin=self.safety_margin)
        self.velocity_limits = (
            np.asarray(velocity_limits, dtype=np.float64)
            if velocity_limits is not None
            else RM65_VELOCITY_LIMITS.copy()
        )
        self.acceleration_limits = (
            np.asarray(acceleration_limits, dtype=np.float64)
            if acceleration_limits is not None
            else RM65_ACCELERATION_LIMITS.copy()
        )
        self.max_joint_step = float(max_joint_step)

    def add_environment_object(self, obj_type: str, **kwargs):
        """Add an environment obstacle for collision checking.

        Args:
            obj_type: One of ``'point_cloud'``, ``'mesh'``, ``'box'``,
                      ``'sphere'``, ``'cylinder'``, ``'capsule'``.
            **kwargs: Parameters forwarded to the corresponding Robot method.
        """
        if obj_type == "point_cloud":
            self.robot.add_point_cloud(**kwargs)
        elif obj_type == "mesh":
            self.robot.add_mesh(**kwargs)
        else:
            self.robot.add_geometry(obj_type, **kwargs)

    def quintic_trajectory(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        n_points: int = 100,
        T: float = 1.0,
        check_collision: bool = True,
        safety_margin: float | None = None,
        max_joint_step: float | None = None,
        auto_time_scaling: bool = True,
        return_derivatives: bool = False,
    ):
        """Plan a quintic polynomial trajectory from *q_start* to *q_goal*.

        The trajectory has zero velocity and acceleration at both endpoints.
        Dynamically adapts duration T to ensure joint velocity and acceleration bounds.
        Subdivides path if waypoint-to-waypoint displacement exceeds max_joint_step.

        Args:
            q_start: Start joint angles (radians).
            q_goal: Goal joint angles (radians).
            n_points: Minimum number of waypoints.
            T: Desired trajectory duration (seconds).
            check_collision: Whether to check each waypoint for collisions.
            safety_margin: Optional override for obstacle clearance margin (meters).
            max_joint_step: Max angular change (rad) between adjacent waypoints.
            auto_time_scaling: Automatically increase T if velocity/acceleration limits exceeded.
            return_derivatives: If True, returns (q_traj, collision_idx, qd_traj, qdd_traj, actual_T).

        Returns:
            Tuple of (trajectory, collision_indices) by default, matching existing pipeline.
        """
        q_start = np.asarray(q_start, dtype=np.float64)
        q_goal = np.asarray(q_goal, dtype=np.float64)
        delta_q = np.abs(q_goal - q_start)

        # 1. Kinematic joint limit check
        num_joints = len(q_start)
        lower_limits = RM65_JOINT_LIMITS_LOWER[:num_joints]
        upper_limits = RM65_JOINT_LIMITS_UPPER[:num_joints]
        if np.any(q_start < lower_limits) or np.any(q_start > upper_limits):
            print("Warning: q_start exceeds RM65 joint position limits!")
        if np.any(q_goal < lower_limits) or np.any(q_goal > upper_limits):
            print("Warning: q_goal exceeds RM65 joint position limits!")

        # 2. Dynamic limits verification & time scaling
        # For quintic polynomial with zero endpoint derivatives:
        # peak velocity = 15/8 * |delta_q| / T = 1.875 * |delta_q| / T
        # peak acceleration = 10*sqrt(3)/3 * |delta_q| / T^2 = 5.7735 * |delta_q| / T^2
        v_lim = self.velocity_limits[:num_joints]
        a_lim = self.acceleration_limits[:num_joints]

        req_T_vel = float(np.max(1.875 * delta_q / v_lim)) if np.any(v_lim > 0) else 0.0
        req_T_acc = float(np.max(np.sqrt(5.7735 * delta_q / a_lim))) if np.any(a_lim > 0) else 0.0
        min_feasible_T = max(req_T_vel, req_T_acc)

        actual_T = float(T)
        if auto_time_scaling and min_feasible_T > actual_T:
            actual_T = float(min_feasible_T * 1.05)  # 5% safety margin

        # 3. Solve analytical quintic polynomial
        t = np.linspace(0, actual_T, n_points)
        v_start = np.zeros_like(q_start)
        v_goal = np.zeros_like(q_goal)
        a_start = np.zeros_like(q_start)
        a_goal = np.zeros_like(q_goal)

        A = np.array(
            [
                [0, 0, 0, 0, 0, 1],
                [actual_T**5, actual_T**4, actual_T**3, actual_T**2, actual_T, 1],
                [0, 0, 0, 0, 1, 0],
                [5 * actual_T**4, 4 * actual_T**3, 3 * actual_T**2, 2 * actual_T, 1, 0],
                [0, 0, 0, 2, 0, 0],
                [20 * actual_T**3, 12 * actual_T**2, 6 * actual_T, 2, 0, 0],
            ],
            dtype=np.float64,
        )

        q_list, qd_list, qdd_list = [], [], []
        for i in range(num_joints):
            b = np.array(
                [q_start[i], q_goal[i], v_start[i], v_goal[i], a_start[i], a_goal[i]],
                dtype=np.float64,
            )
            x = np.linalg.solve(A, b)
            pos = x[0] * t**5 + x[1] * t**4 + x[2] * t**3 + x[3] * t**2 + x[4] * t + x[5]
            vel = 5 * x[0] * t**4 + 4 * x[1] * t**3 + 3 * x[2] * t**2 + 2 * x[3] * t + x[4]
            acc = 20 * x[0] * t**3 + 12 * x[1] * t**2 + 6 * x[2] * t + 2 * x[3]
            q_list.append(pos)
            qd_list.append(vel)
            qdd_list.append(acc)

        q_traj = np.array(q_list).T
        qd_traj = np.array(qd_list).T
        qdd_traj = np.array(qdd_list).T

        # 4. Waypoint subdivision for dense collision checking
        step_threshold = self.max_joint_step if max_joint_step is None else float(max_joint_step)
        subdivided_traj = [q_traj[0]]
        for k in range(len(q_traj) - 1):
            curr_q = q_traj[k]
            next_q = q_traj[k + 1]
            max_disp = float(np.max(np.abs(next_q - curr_q)))
            if max_disp > step_threshold:
                sub_steps = int(np.ceil(max_disp / step_threshold))
                for s in range(1, sub_steps):
                    alpha = s / sub_steps
                    subdivided_traj.append((1.0 - alpha) * curr_q + alpha * next_q)
            subdivided_traj.append(next_q)

        final_traj = np.array(subdivided_traj)

        # 5. Collision checking with safety margin
        collision_idx = []
        margin = self.safety_margin if safety_margin is None else float(safety_margin)
        if check_collision:
            for i, q in enumerate(final_traj):
                res = self.collision_detector.check_collision(q, safety_margin=margin)
                if res.is_collision:
                    collision_idx.append(i)

        if return_derivatives:
            return final_traj, collision_idx, qd_traj, qdd_traj, actual_T
        return final_traj, collision_idx

    def get_collision_distance(self, q: np.ndarray):
        """Compute minimum distance from robot to environment at configuration *q*."""
        return self.collision_detector.get_collision_distance(q)

    def check_collision(
        self, q: np.ndarray, safety_margin: float | None = None
    ) -> CollisionResult:
        """Check collision at configuration *q*."""
        return self.collision_detector.check_collision(q, safety_margin=safety_margin)


# =============================================================================
# Robot model
# =============================================================================


class Robot:
    """Pinocchio-based robot model with collision geometry and visualization.

    Args:
        urdf_path: Path to the URDF file.
        mesh_path: Directory containing link mesh files.
    """

    def __init__(self, urdf_path: str, mesh_path: str):
        _require_pinocchio()

        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, mesh_path
        )
        self.data = self.model.createData()
        self.collision_data = self.collision_model.createData()
        self.visual_data = self.visual_model.createData()
        self.viz = MeshcatVisualizer(self.model, visual_model=self.visual_model)

        # Robot link collision objects
        self.robot_collision_objects = []
        # Environment collision objects
        self.env_collision_objects = []

        for geom in self.collision_model.geometryObjects:
            if geom.meshPath:
                loader = coal.MeshLoader()
                mesh = loader.load(geom.meshPath)
                collision_obj = coal.CollisionObject(mesh)
            else:
                if isinstance(geom.geometry, pin.GeometryObject):
                    gtype = geom.geometry.type
                    if gtype == pin.GeometryType.BOX:
                        dims = geom.geometry.dimensions
                        collision_obj = coal.CollisionObject(
                            coal.Box(dims[0], dims[1], dims[2])
                        )
                    elif gtype == pin.GeometryType.SPHERE:
                        collision_obj = coal.CollisionObject(
                            coal.Sphere(geom.geometry.radius)
                        )
                    elif gtype == pin.GeometryType.CYLINDER:
                        collision_obj = coal.CollisionObject(
                            coal.Cylinder(geom.geometry.radius, geom.geometry.length)
                        )
                    elif gtype == pin.GeometryType.CAPSULE:
                        collision_obj = coal.CollisionObject(
                            coal.Capsule(geom.geometry.radius, geom.geometry.length)
                        )
                    else:
                        print(
                            f"Warning: unsupported geometry type {gtype} for {geom.name}"
                        )
                        continue
                else:
                    print(f"Warning: unknown geometry type for {geom.name}")
                    continue

            collision_obj.name = geom.name
            self.robot_collision_objects.append(collision_obj)

        self.environment_objects = []
        self.environment_object_names = {}

    def init_visualizer(self, open_viewer: bool = False):
        """Initialize the Meshcat visualizer.

        Args:
            open_viewer: Whether to open the browser viewer.
        """
        self.viz.initViewer(open=open_viewer)
        self.viz.loadViewerModel()
        q0 = pin.neutral(self.model)
        self.viz.display(q0)
        self.viz.displayVisuals(True)
        self._setup_camera()

    def _setup_camera(self):
        """Set the default camera viewpoint."""
        self.viz.viewer["/Cameras/default"].set_transform(np.eye(4))

    def add_point_cloud(
        self,
        points: np.ndarray,
        name: str | None = None,
        placement=None,
        color=None,
        resolution: float = 0.01,
    ):
        """Add a point cloud obstacle to the environment.

        Args:
            points: (N, 3) point array.
            name: Display name.
            placement: pin.SE3 placement (default: identity).
            color: RGBA color list.
            resolution: Octree voxel resolution in meters.
        """
        if placement is None:
            placement = pin.SE3.Identity()
        if color is None:
            color = [0.5, 0.5, 0.5, 1.0]
        if name is None:
            name = f"point_cloud_{len(self.env_collision_objects)}"

        octree_obj = coal.makeOctree(points, resolution)
        collision_obj = coal.CollisionObject(octree_obj)
        collision_obj.setTransform(placement)
        collision_obj.name = name
        self.env_collision_objects.append(collision_obj)

        point_cloud = coal.BVHModelOBBRSS()
        point_cloud.beginModel(0, points.shape[0])
        point_cloud.addVertices(points)
        point_cloud.endModel()

        go_point_cloud = pin.GeometryObject(name, 0, placement, point_cloud)
        go_point_cloud.meshColor = np.array(color)
        self.environment_objects.append(go_point_cloud)
        self.environment_object_names[name] = go_point_cloud
        self.viz.addGeometryObject(go_point_cloud)

    def add_mesh(
        self,
        mesh_path: str,
        name: str | None = None,
        placement=None,
        scale: tuple = (1.0, 1.0, 1.0),
        color=None,
    ):
        """Add a mesh obstacle to the environment.

        Args:
            mesh_path: Path to mesh file.
            name: Display name.
            placement: pin.SE3 placement.
            scale: Mesh scale factors.
            color: RGBA color list.
        """
        if placement is None:
            placement = pin.SE3.Identity()
        if color is None:
            color = [0.5, 0.5, 0.5, 1.0]
        if name is None:
            name = f"mesh_{len(self.env_collision_objects)}"

        mesh = coal.BVHModelOBBRSS()
        mesh.loadMesh(mesh_path, scale)

        collision_obj = coal.CollisionObject(mesh)
        collision_obj.setTransform(placement)
        self.env_collision_objects.append(collision_obj)

        go_mesh = pin.GeometryObject(name, 0, placement, mesh)
        go_mesh.meshColor = np.array(color)
        self.environment_objects.append(go_mesh)
        self.environment_object_names[name] = go_mesh
        self.viz.addGeometryObject(go_mesh)

    def add_geometry(
        self,
        geometry_type: str,
        params: dict,
        name: str | None = None,
        placement=None,
        color=None,
    ):
        """Add a primitive geometry obstacle to the environment.

        Args:
            geometry_type: One of ``'box'``, ``'sphere'``, ``'cylinder'``, ``'capsule'``.
            params: Shape parameters dict (e.g. ``{'dimensions': [1, 1, 1]}``).
            name: Display name.
            placement: pin.SE3 placement.
            color: RGBA color list.
        """
        if placement is None:
            placement = pin.SE3.Identity()
        if color is None:
            color = [0.5, 0.5, 0.5, 1.0]
        if name is None:
            name = f"{geometry_type}_{len(self.env_collision_objects)}"

        if geometry_type == "box":
            geometry = coal.Box(*params["dimensions"])
        elif geometry_type == "sphere":
            geometry = coal.Sphere(params["radius"])
        elif geometry_type == "cylinder":
            geometry = coal.Cylinder(params["radius"], params["length"])
        elif geometry_type == "capsule":
            geometry = coal.Capsule(params["radius"], params["length"])
        else:
            raise ValueError(f"Unsupported geometry type: {geometry_type}")

        collision_obj = coal.CollisionObject(geometry)
        collision_obj.setTransform(placement)
        collision_obj.name = name
        self.env_collision_objects.append(collision_obj)

        go_geometry = pin.GeometryObject(name, 0, placement, geometry)
        go_geometry.meshColor = np.array(color)
        self.environment_objects.append(go_geometry)
        self.environment_object_names[name] = go_geometry
        self.viz.addGeometryObject(go_geometry)

    def update_state(self, q: np.ndarray):
        """Update robot joint configuration and refresh visualizer.

        Args:
            q: Joint angles (radians).
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.updateGeometryPlacements(
            self.model, self.data, self.collision_model, self.collision_data, q
        )
        self.viz.display(q)

    def get_frame_id(self, frame_name: str) -> int:
        """Get the Pinocchio frame ID for a named frame.

        Args:
            frame_name: Name of the frame in the URDF.

        Returns:
            Frame ID.
        """
        return self.model.getFrameId(frame_name)

    def create_video(self, output_path: str, qs, dt: float, callback=None):
        """Record a trajectory animation to video.

        Args:
            output_path: Output video file path.
            qs: List of joint configurations.
            dt: Time step between frames.
            callback: Optional callback per frame.
        """
        with self.viz.create_video_ctx(output_path):
            self.viz.play(qs, dt, callback=callback)
