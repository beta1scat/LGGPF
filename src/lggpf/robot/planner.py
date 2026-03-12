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
# Collision detection
# =============================================================================


class CollisionResult:
    """Container for collision check results."""

    def __init__(self):
        self.is_collision: bool = False
        self.contact_points = None


class CollisionDetector:
    """Checks collisions between robot links and environment objects.

    Args:
        robot: A :class:`Robot` instance.
    """

    def __init__(self, robot: Robot):
        self.robot = robot

    def check_collision(
        self, q: np.ndarray, stop_at_first_collision: bool = True
    ) -> CollisionResult:
        """Check whether the robot at configuration *q* collides with the environment.

        Args:
            q: Joint angles (radians).
            stop_at_first_collision: Return immediately on first collision.

        Returns:
            CollisionResult with ``is_collision`` flag.
        """
        self.robot.update_state(q)
        result = CollisionResult()

        for idx_r, robot_obj in enumerate(self.robot.robot_collision_objects):
            for env_obj in self.robot.env_collision_objects:
                request = coal.CollisionRequest()
                collision_result = coal.CollisionResult()

                if idx_r == len(self.robot.robot_collision_objects) - 1:
                    T1 = coal.Transform3s()
                    T1.setTranslation(self.robot.data.oMi[-1].translation)
                    T1.setRotation(self.robot.data.oMi[-1].rotation)
                else:
                    T1 = coal.Transform3s()
                    T1.setTranslation(self.robot.data.oMi[idx_r].translation)
                    T1.setRotation(self.robot.data.oMi[idx_r].rotation)

                coal.collide(
                    robot_obj.collisionGeometry(),
                    T1,
                    env_obj.collisionGeometry(),
                    env_obj.getTransform(),
                    request,
                    collision_result,
                )
                if collision_result.isCollision():
                    print(
                        f"Collision detected: {idx_r}, {robot_obj.name}-{env_obj.name}"
                    )
                    result.is_collision = True
                    result.contact_points = collision_result.getContacts()
                    if stop_at_first_collision:
                        return result

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

        for robot_obj in self.robot.robot_collision_objects:
            for env_obj in self.robot.env_collision_objects:
                request = coal.DistanceRequest()
                distance_result = coal.DistanceResult()
                coal.distance(robot_obj, env_obj, request, distance_result)
                if distance_result.min_distance < min_distance:
                    min_distance = distance_result.min_distance
                    closest_pair = (robot_obj, env_obj)

        return min_distance, closest_pair


# =============================================================================
# Trajectory planner
# =============================================================================


class JointSpacePlanner:
    """Quintic polynomial joint-space trajectory planner with collision checking.

    Args:
        robot: A :class:`Robot` instance.
    """

    def __init__(self, robot: Robot):
        self.robot = robot
        self.collision_detector = CollisionDetector(robot)

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
        n_points: int,
        T: float = 1.0,
        check_collision: bool = True,
    ):
        """Plan a quintic polynomial trajectory from *q_start* to *q_goal*.

        The trajectory has zero velocity and acceleration at both endpoints.

        Args:
            q_start: Start joint angles (radians).
            q_goal: Goal joint angles (radians).
            n_points: Number of waypoints.
            T: Total trajectory duration (seconds).
            check_collision: Whether to check each waypoint for collisions.

        Returns:
            Tuple of (trajectory, collision_indices) where trajectory is
            (n_points, n_joints) array and collision_indices lists the
            indices of waypoints in collision.
        """
        t = np.linspace(0, T, n_points)
        v_start = np.zeros_like(q_start)
        v_goal = np.zeros_like(q_goal)
        a_start = np.zeros_like(q_start)
        a_goal = np.zeros_like(q_goal)

        # Quintic polynomial coefficient matrix
        A = np.array(
            [
                [0, 0, 0, 0, 0, 1],
                [T**5, T**4, T**3, T**2, T, 1],
                [0, 0, 0, 0, 1, 0],
                [5 * T**4, 4 * T**3, 3 * T**2, 2 * T, 1, 0],
                [0, 0, 0, 2, 0, 0],
                [20 * T**3, 12 * T**2, 6 * T, 2, 0, 0],
            ]
        )

        q_traj = []
        for i in range(len(q_start)):
            b = np.array(
                [q_start[i], q_goal[i], v_start[i], v_goal[i], a_start[i], a_goal[i]]
            )
            x = np.linalg.solve(A, b)
            q_joint = np.array(
                [
                    x[0] * tt**5
                    + x[1] * tt**4
                    + x[2] * tt**3
                    + x[3] * tt**2
                    + x[4] * tt
                    + x[5]
                    for tt in t
                ]
            )
            q_traj.append(q_joint)

        q_traj = np.array(q_traj).T
        collision_idx = []

        if check_collision:
            for i, q in enumerate(q_traj):
                if self.collision_detector.check_collision(q).is_collision:
                    print(f"Waypoint {i} has collision")
                    collision_idx.append(i)

        return q_traj, collision_idx

    def get_collision_distance(self, q: np.ndarray):
        """Compute minimum distance from robot to environment at configuration *q*."""
        return self.collision_detector.get_collision_distance(q)

    def check_collision(self, q: np.ndarray) -> CollisionResult:
        """Check collision at configuration *q*."""
        return self.collision_detector.check_collision(q)


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
