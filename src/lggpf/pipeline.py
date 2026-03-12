"""LGGPF pipeline orchestrator.

Encapsulates the full language-guided grasping pipeline:
    1. Load models (camera, OWLv2, SAM, PointNet2, fitting, robot)
    2. Capture image (RGB + depth)
    3. Detect object (OWLv2 open-vocabulary detection)
    4. Select bounding box
    5. Segment object (SAM)
    6. Classify + fit primitive shape (language-guided brute-force)
    7. Generate grasp poses
    8. Plan trajectory (collision-aware quintic polynomial)
    9. Execute on robot

All hard-coded values are read from the configuration dictionary
(see :func:`lggpf.config.load_config`).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from spatialmath import SE3

from .config import load_config, get_calibration_matrices, get_camera_intrinsics
from .camera import CameraController
from .detection import VisionLanguageOwlVit
from .segmentation import SegmentAnythingModel
from .shape_fitting import FittingByBGS, PcdClassification
from .grasp import PickPose
from .robot import Robot, JointSpacePlanner
from .robot.communicator import RobotArmController
from .utils.pointcloud import (
    depth_to_pointcloud,
    generate_cone_points,
    generate_cube_points,
    generate_ellipsoid_points,
    view_coordinate,
    filter_pose_by_axis_diff,
    check_pick_pose_for_2finger_gripper_range,
)

logger = logging.getLogger(__name__)

DEGREE_TO_RADIAN = np.pi / 180
RADIAN_TO_DEGREE = 180 / np.pi


class GraspingPipeline:
    """Full LGGPF grasping pipeline.

    Args:
        config: Configuration dictionary. If *None*, loads the default
            ``config/default.yaml``.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        if config is None:
            config = load_config()
        self.cfg = config

        # Calibration matrices
        self.T_ET_Cali, self.T_BC_Cali = get_calibration_matrices(config)
        self.fx, self.fy, self.cx, self.cy = get_camera_intrinsics(config)

        # Pipeline configuration
        pipe_cfg = config.get("pipeline", {})
        self.detection_resize_factor = pipe_cfg.get("detection_resize_factor", 0.5)
        self.detection_threshold = pipe_cfg.get("detection_threshold", 0.1)
        self.segmentation_max_points = pipe_cfg.get("segmentation_max_points", 10000)
        self.fitting_synthetic_points = pipe_cfg.get("fitting_synthetic_points", 5000)

        grasp_cfg = pipe_cfg.get("grasp", {})
        self.z_axis_filter_threshold = grasp_cfg.get(
            "z_axis_filter_threshold", np.pi / 4
        )
        self.finger_range = config.get("gripper", {}).get("finger_range", 65)
        self.gripper_depth_center = grasp_cfg.get("gripper_depth_center", 10)
        self.gripper_depth_side = grasp_cfg.get("gripper_depth_side", 20)
        self.gripper_depth_end = grasp_cfg.get("gripper_depth_end", 25)
        self.cone_num_positions = grasp_cfg.get("cone_num_positions", 20)
        self.ellipsoid_num_directions = grasp_cfg.get("ellipsoid_num_directions", 20)
        self.approach_offset = grasp_cfg.get("approach_offset", 25)

        traj_cfg = pipe_cfg.get("trajectory", {})
        self.num_path_joints = traj_cfg.get("num_path_joints", 100)
        self.path_time = traj_cfg.get("path_time", 3.0)

        # Language type-selection keywords
        lang_cfg = pipe_cfg.get("language_type_map", {})
        self.cone_keywords = lang_cfg.get("cone_keywords", ["cup", "bowl", "tube"])
        self.ellipsoid_keywords = lang_cfg.get("ellipsoid_keywords", ["ball"])
        self.center_keywords = lang_cfg.get("center_keywords", ["center"])
        self.side_keywords = lang_cfg.get("side_keywords", ["side"])

        # Camera ROI
        self.roi = config.get("camera", {}).get("roi", [230, 210, 1500, 800])
        self.normal_orientation_location = config.get("camera", {}).get(
            "normal_orientation_location", [0, 0, 800]
        )

        # Logging config
        log_cfg = config.get("logging", {})
        self.save_results = log_cfg.get("save_results", True)
        self.debug_visualization = log_cfg.get("debug_visualization", False)
        self.log_directory = log_cfg.get("log_directory", "data/logs")

        # Model holder
        self.models: dict[str, Any] = {}
        # Pipeline data (intermediate results)
        self.data: dict[str, Any] = {}
        # Current log path
        self.current_path: str = ""

    # =========================================================================
    # Logging helpers
    # =========================================================================

    def _init_log_session(self) -> str:
        """Create a timestamped log directory for this session.

        Returns:
            Path to the session log directory.
        """
        if not self.save_results:
            return ""
        now = datetime.now()
        date_folder = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        session_path = os.path.join(self.log_directory, date_folder, time_str)
        os.makedirs(session_path, exist_ok=True)
        self.current_path = session_path
        self.data["time_str"] = time_str
        return session_path

    def _save_artifact(self, filename: str, data: Any) -> None:
        """Save an artifact to the current log directory."""
        if not self.save_results or not self.current_path:
            return
        filepath = os.path.join(self.current_path, filename)
        if isinstance(data, np.ndarray) and data.ndim >= 2:
            # Image
            cv2.imwrite(filepath, data)
        elif isinstance(data, o3d.geometry.PointCloud):
            o3d.io.write_point_cloud(filepath, data)
        elif isinstance(data, str):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(data)
        else:
            np.save(filepath, data)

    # =========================================================================
    # 1. Load models
    # =========================================================================

    def load_models(self) -> None:
        """Load all models and initialize connections.

        Initializes camera, OWLv2, SAM, PointNet2, shape fitting,
        robot model, robot arm controller, and trajectory planner.
        """
        cam_cfg = self.cfg.get("camera", {})
        model_cfg = self.cfg.get("models", {})
        robot_cfg = self.cfg.get("robot", {})

        # Camera
        camera = CameraController()
        camera.connect(cam_cfg.get("ip", "192.168.1.200"))
        self.models["camera"] = camera

        # Vision-language detection
        self.models["vlm"] = VisionLanguageOwlVit(model_cfg.get("owlv2", ""))

        # Segmentation
        self.models["segmentation"] = SegmentAnythingModel(model_cfg.get("sam", ""))

        # Point cloud classification (PointNet2)
        self.models["classification"] = PcdClassification(
            model_cfg.get("pointnet2", "")
        )

        # Shape fitting
        self.models["fitting"] = FittingByBGS()

        # Robot model (Pinocchio)
        # Reuse the existing robot visualizer if it was already initialized
        # by the web index route. Otherwise the browser iframe stays connected
        # to the old MeshCat instance and planned motion will not appear.
        if "robot" in self.models:
            robot = self.models["robot"]
        else:
            urdf_path = robot_cfg.get("urdf_path", "config/rm65/rm65.urdf")
            mesh_path = robot_cfg.get("mesh_path", "config/rm65/")
            robot = Robot(urdf_path, mesh_path)
            robot.init_visualizer()
            self.models["robot"] = robot

        # Robot arm controller
        self.models["robot_ctrl"] = RobotArmController(
            robot_cfg.get("ip", "192.168.1.18"),
            robot_cfg.get("port", 8080),
            robot_cfg.get("connection_level", 3),
            gripper_cfg=self.cfg.get("gripper", {}),
        )

        # Trajectory planner
        self.models["planner"] = JointSpacePlanner(robot)

        logger.info("All models loaded successfully.")

    def init_robot_only(self) -> None:
        """Initialize only the robot model (for visualization without hardware)."""
        robot_cfg = self.cfg.get("robot", {})
        urdf_path = robot_cfg.get("urdf_path", "config/rm65/rm65.urdf")
        mesh_path = robot_cfg.get("mesh_path", "config/rm65/")
        robot = Robot(urdf_path, mesh_path)
        robot.init_visualizer()
        self.models["robot"] = robot

    # =========================================================================
    # 2. Capture image
    # =========================================================================

    def capture_image(self) -> tuple[np.ndarray, str]:
        """Capture RGB and depth images from the camera.

        Returns:
            Tuple of (base64-encoded JPEG string, time_str identifier).
        """
        import base64

        self._init_log_session()
        cam_ctrl = self.models["camera"]
        self.data["img_rgb"] = cam_ctrl.capture_2d_image()
        self.data["img_dep"] = cam_ctrl.capture_depth_map()

        roi = self.roi
        if self.save_results and self.current_path:
            self._save_artifact("image.jpg", self.data["img_rgb"])
            self._save_artifact("depth.tiff", self.data["img_dep"])
            self._save_artifact("depth.png", self.data["img_dep"])
            self._save_artifact(
                "image_roi.jpg",
                self.data["img_rgb"][
                    roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]
                ],
            )
            self._save_artifact(
                "depth_roi.tiff",
                self.data["img_dep"][
                    roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]
                ],
            )
            self._save_artifact(
                "depth_roi.png",
                self.data["img_dep"][
                    roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]
                ],
            )

        _, img_encoded = cv2.imencode(".jpg", self.data["img_rgb"])
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")
        return img_base64, self.data.get("time_str", "")

    # =========================================================================
    # 3. Detect objects
    # =========================================================================

    def detect_objects(self, text: str) -> int:
        """Run OWLv2 open-vocabulary detection.

        Args:
            text: Comma-separated text query. The first term is used for
                detection; the full string is stored for language-guided
                type selection later.

        Returns:
            Number of detected bounding boxes.

        Raises:
            ValueError: If no objects are detected.
        """
        self.data["text"] = text
        if self.save_results and self.current_path:
            self._save_artifact("instruction.txt", text)

        vlm = self.models["vlm"]
        factor = self.detection_resize_factor
        resized_img = cv2.resize(
            self.data["img_rgb"],
            (0, 0),
            fx=factor,
            fy=factor,
            interpolation=cv2.INTER_AREA,
        )
        boxes, scores = vlm.get_boxes_by_text(
            resized_img, text.split(",")[0], threshold=self.detection_threshold
        )
        if len(boxes) < 1:
            raise ValueError("No objects detected.")

        # Scale boxes back to original image resolution
        scale = int(round(1.0 / factor))
        self.data["boxes"] = boxes.numpy().astype(int) * scale
        logger.info("Detected %d bounding boxes.", len(self.data["boxes"]))
        return len(self.data["boxes"])

    # =========================================================================
    # 4. Select bounding box
    # =========================================================================

    def select_box(self, box_index: int) -> tuple[list[int], str]:
        """Select a bounding box by index.

        Args:
            box_index: 1-based index into the detected boxes.

        Returns:
            Tuple of (box as [x1,y1,x2,y2], base64-encoded JPEG with box overlay).
        """
        import base64

        idx = box_index - 1
        self.data["box"] = self.data["boxes"][idx]
        box = self.data["box"]
        img_with_box = cv2.rectangle(
            self.data["img_rgb"].copy(),
            (box[0], box[1]),
            (box[2], box[3]),
            (0, 255, 0),
            2,
        )

        roi = self.roi
        if self.save_results and self.current_path:
            np.savetxt(os.path.join(self.current_path, "box.txt"), box)
            self._save_artifact("image_with_box.jpg", img_with_box)
            self._save_artifact(
                "image_with_box_roi.jpg",
                img_with_box[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]],
            )

        _, img_encoded = cv2.imencode(".jpg", img_with_box)
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")
        return box.tolist(), img_base64

    # =========================================================================
    # 5. Segment object
    # =========================================================================

    def segment_object(self) -> int:
        """Segment the object within the selected bounding box.

        Creates a masked depth image and converts it to a point cloud.

        Returns:
            Number of points in the segmented point cloud.
        """
        seg_model = self.models["segmentation"]
        mask = seg_model.segment(self.data["img_rgb"], self.data["box"])

        seg_depth = np.copy(self.data["img_dep"])
        seg_depth[~mask] = 0
        self.data["seg_img_dep"] = seg_depth

        pointcloud = depth_to_pointcloud(seg_depth, self.fx, self.fy, self.cx, self.cy)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pcd.remove_non_finite_points()
        pcd.estimate_normals()

        if len(pcd.points) > self.segmentation_max_points:
            pcd = pcd.farthest_point_down_sample(self.segmentation_max_points)

        logger.info("Segmented point cloud: %d points.", len(pcd.points))
        pcd.orient_normals_towards_camera_location(self.normal_orientation_location)
        self.data["pcd"] = o3d.geometry.PointCloud(pcd)

        if self.save_results and self.current_path:
            overlay = self.data["img_rgb"].copy()
            colored_mask = np.ones_like(self.data["img_rgb"], dtype=np.uint8) * 255
            colored_mask[mask] = np.array([0, 0, 0])
            cv2.addWeighted(colored_mask, 0.8, overlay, 1 - 0.8, 0, overlay)
            self._save_artifact("image_with_mask.jpg", overlay)
            roi = self.roi
            self._save_artifact(
                "image_with_mask_roi.jpg",
                overlay[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]],
            )
            self._save_artifact("pcd.ply", pcd)

        if self.debug_visualization:
            o3d.visualization.draw_geometries([pcd], point_show_normal=True)

        return len(pcd.points)

    # =========================================================================
    # 6. Classify and fit shape
    # =========================================================================

    def _determine_type_list(self) -> list[str]:
        """Determine candidate fitting types from the instruction text.

        Uses language-guided keyword matching:
        - "cup"/"bowl"/"tube" -> cone types ["01", "11", "12", "13", "14"]
        - "ball" -> ellipsoid ["2"]
        - else -> cuboid ["0"]
        """
        text = self.data.get("text", "").lower()
        for kw in self.cone_keywords:
            if kw in text:
                return ["01", "11", "12", "13", "14"]
        for kw in self.ellipsoid_keywords:
            if kw in text:
                return ["2"]
        return ["0"]

    def classify_and_fit(self) -> dict[str, Any]:
        """Run brute-force shape fitting and select the best match.

        Tries all candidate fitting types (determined by language keywords),
        generates synthetic point clouds for each, and selects the one with
        the lowest mean bidirectional point cloud distance.

        Returns:
            Dict with keys: ``category``, ``params`` (doubled sizes), and
            raw ``category_code``.
        """
        fbg = self.models["fitting"]
        pcd = self.data["pcd"]
        type_list = self._determine_type_list()
        total_points = self.fitting_synthetic_points

        params_list: list[Any] = []
        pcd_fit_list: list[o3d.geometry.PointCloud | None] = []
        min_dist_list: list[float] = []

        for tp in type_list:
            logger.info("Trying fitting type: %s", tp)
            try:
                params = fbg.fitting(pcd, tp)
            except Exception:
                logger.warning("Fitting type %s failed, skipping.", tp)
                params_list.append(None)
                pcd_fit_list.append(None)
                min_dist_list.append(np.inf)
                continue

            if params == []:
                logger.warning("Fitting type %s returned empty params, skipping.", tp)
                params_list.append(None)
                pcd_fit_list.append(None)
                min_dist_list.append(np.inf)
                continue

            params_list.append(params)

            # Generate synthetic point cloud for comparison
            if tp in ("0", "01"):
                points = generate_cube_points(
                    np.array(params[:3]) * 2, total_points=total_points
                )
            elif tp in ("1", "11", "12", "13", "14"):
                r1, r2, height, _ = params
                points = generate_cone_points(
                    r_bottom=r2,
                    r_top_ratio=r1 / r2,
                    height=height,
                    total_points=total_points,
                )
            elif tp == "2":
                points = generate_ellipsoid_points(
                    *params[:3], total_points=total_points
                )
            else:
                logger.warning("Unsupported fitting type: %s", tp)
                pcd_fit_list.append(None)
                min_dist_list.append(np.inf)
                continue

            pcd_fit = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            pcd_fit_list.append(pcd_fit)

            # Compute bidirectional mean distance as quality metric
            dist_pcd_fit = o3d.geometry.PointCloud(pcd_fit)
            dist_pcd_fit.transform(params[-1])
            min_dist1 = pcd.compute_point_cloud_distance(dist_pcd_fit)
            min_dist2 = dist_pcd_fit.compute_point_cloud_distance(pcd)
            dist_score = np.mean(min_dist1) + np.mean(min_dist2)
            logger.info("Type %s distance score: %.4f", tp, dist_score)
            min_dist_list.append(dist_score)

        # Select best fit
        min_idx = int(np.argmin(min_dist_list))
        cls = type_list[min_idx]
        pcd_fit = pcd_fit_list[min_idx]

        self.data["category"] = cls
        self.data["params"] = params_list[min_idx]
        self.data["pcd_fit"] = pcd_fit

        # Visualization
        if pcd_fit is not None:
            view_pcd_fit = o3d.geometry.PointCloud(pcd_fit)
            view_pcd_fit.transform(self.data["params"][-1])
            if self.save_results and self.current_path:
                self._save_artifact("fit_pcd.ply", view_pcd_fit)
            if self.debug_visualization:
                o3d.visualization.draw_geometries(
                    [self.data["pcd"], view_pcd_fit], point_show_normal=True
                )

        params_display = [round(x * 2, 2) for x in self.data["params"][:-1]]
        logger.info("Best fit: type=%s, params=%s", cls, params_display)
        return {
            "category": cls[0],
            "category_code": cls,
            "params": params_display,
        }

    # =========================================================================
    # 7. Generate grasp poses
    # =========================================================================

    def generate_pick_poses(self) -> int:
        """Generate and filter candidate grasp poses.

        Uses the fitted shape parameters and language-guided strategy
        (center/side/end) to generate SE3 grasp poses, then:
        1. Transforms them to the robot base frame via T_BC * t_OC * pose
        2. Filters by z-axis alignment (pi/4 threshold)
        3. Checks 2-finger gripper feasibility

        Returns:
            Number of feasible grasp poses.
        """
        category = self.data["category"]
        params = self.data["params"]
        t_OC = params[-1]  # Object-to-camera transform
        text = self.data.get("text", "").lower()

        # Determine grasp strategy from language
        is_center = any(kw in text for kw in self.center_keywords)
        is_side = any(kw in text for kw in self.side_keywords)

        # Generate candidate poses based on shape type
        if category in ("0", "01"):
            if is_center:
                ppose = PickPose.gen_cube_center_pick_poses(
                    [x * 2 for x in params[:3]],
                    gripper_depth=self.gripper_depth_center,
                )
            else:
                ppose = PickPose.gen_cube_end_pick_poses(
                    [x * 2 for x in params[:3]],
                    gripper_depth=self.gripper_depth_center,
                )
        elif category in ("1", "11", "12", "13", "14"):
            if is_center:
                ppose = PickPose.gen_cone_center_pick_poses(
                    params[2],
                    self.cone_num_positions,
                    gripper_depth=self.gripper_depth_center,
                )
            elif is_side:
                ppose = PickPose.gen_cone_side_pick_poses(
                    params[2],
                    params[0],
                    params[1],
                    num_each_side=self.cone_num_positions,
                    gripper_depth=self.gripper_depth_side,
                )
            else:
                ppose = PickPose.gen_cone_end_pick_poses(
                    params[2],
                    self.cone_num_positions,
                    gripper_depth=self.gripper_depth_end,
                )
        elif category == "2":
            ppose = PickPose.gen_ellipsoid_center_pick_poses(
                self.ellipsoid_num_directions
            )
        else:
            raise ValueError(f"Unsupported shape category: {category}")

        # Transform poses to robot base frame
        for i, pose in enumerate(ppose):
            if category == "2":
                # Ellipsoid: use only translation, not rotation from t_OC
                ppose[i] = self.T_BC_Cali * SE3.Rt(np.eye(3), t_OC.t) * pose
            else:
                ppose[i] = self.T_BC_Cali * t_OC * pose

        # Filter poses by z-axis alignment (approach direction should be downward)
        ppose = filter_pose_by_axis_diff(
            ppose,
            axis=2,
            ref_axis=[0, 0, -1],
            t=self.z_axis_filter_threshold,
            sorted=True,
        )

        # Check 2-finger gripper feasibility (skip for side grasps)
        if not is_side:
            pcd_model = o3d.geometry.PointCloud(self.data["pcd_fit"])
            pcd_model.transform(self.T_BC_Cali * t_OC)
            ppose = check_pick_pose_for_2finger_gripper_range(
                pcd_model, ppose, self.finger_range
            )

        self.data["pick_poses"] = ppose

        if self.debug_visualization:
            pcd_view = o3d.geometry.PointCloud(self.data["pcd"])
            pcd_view.transform(self.T_BC_Cali)
            view_coordinate(ppose, pcd_view, 100)

        if self.save_results and self.current_path:
            np.save(os.path.join(self.current_path, "poses.npy"), ppose)

        logger.info("Generated %d feasible grasp poses.", len(ppose))
        return len(ppose)

    # =========================================================================
    # 8. Plan trajectory
    # =========================================================================

    def plan_trajectory(self, pick_pose_index: int) -> bool:
        """Plan a collision-free trajectory to the selected grasp pose.

        Args:
            pick_pose_index: 1-based index into the filtered grasp poses.

        Returns:
            True if a collision-free trajectory was found, False otherwise.
        """
        try:
            import pinocchio as pin
        except ImportError:
            raise ImportError(
                "Pinocchio is required for trajectory planning. "
                "Install with: pip install pin coal"
            )

        idx = pick_pose_index - 1
        planner: JointSpacePlanner = self.models["planner"]
        robot_ctrl: RobotArmController = self.models["robot_ctrl"]
        robot: Robot = self.models["robot"]
        pcd = self.data["pcd"]
        t_OC = self.data["params"][-1]
        ppose = self.data["pick_poses"]

        # Add environment objects for collision checking
        pcd_view = o3d.geometry.PointCloud(pcd)
        pcd_view.transform(self.T_BC_Cali)
        pin_identity4 = pin.SE3.Identity()
        planner.add_environment_object(
            "point_cloud",
            points=np.asarray(pcd_view.points) / 1000,
            name="capture_cloud",
            placement=pin_identity4,
            color=[0.8, 0.2, 0.2, 1.0],
            resolution=0.001,
        )

        # Add table as collision object
        table_cfg = self.cfg.get("robot", {}).get("table", {})
        table_placement = pin.SE3.Identity()
        table_pos = table_cfg.get("placement", [-0.5, 0.0, -0.51])
        table_placement.translation = np.array(table_pos)
        table_dims = table_cfg.get("dimensions", [1, 1, 1])
        planner.add_environment_object(
            "box",
            name="table",
            params={"dimensions": table_dims},
            placement=table_placement,
            color=[255 / 255.0, 211 / 255.0, 155 / 255.0, 0.8],
        )

        # Set start configuration
        q_start = np.array(
            self.cfg.get("robot", {}).get(
                "start_pose", [0, 0, np.pi / 2, 0, np.pi / 2, 0]
            )
        )
        robot.update_state(q_start)

        # Select grasp pose and compute IK
        pick = ppose[idx]
        self.data["pick_pose_current"] = pick
        pre_pick_pose = pick * self.T_ET_Cali.inv()
        q_pose = [*pre_pick_pose.t / 1000, *pre_pick_pose.rpy()]

        q_goal = robot_ctrl.inverse_kinematics(q_start, q_pose)
        if not isinstance(q_goal, np.ndarray):
            logger.warning("Inverse kinematics failed.")
            return False
        logger.info("Inverse kinematics succeeded.")
        q_goal = q_goal * DEGREE_TO_RADIAN

        # Plan quintic polynomial trajectory
        trajectory, collision_idx = planner.quintic_trajectory(
            q_start, q_goal, n_points=self.num_path_joints, T=self.path_time
        )
        logger.info("Trajectory planning completed.")

        if len(collision_idx) == 0:
            self.data["trajectory_safe"] = trajectory
            logger.info("Trajectory is collision-free. Visualizing...")
            for q in trajectory:
                robot.update_state(q)
                time.sleep(self.path_time / self.num_path_joints)
            logger.info("Safe trajectory visualization complete.")
            return True
        else:
            logger.warning("Collision detected at %d waypoints.", len(collision_idx))
            for idx_c in collision_idx:
                robot.update_state(trajectory[idx_c])
                time.sleep(0.5)
            logger.info("Collision trajectory visualization complete.")
            return False

    # =========================================================================
    # 9. Execute on robot
    # =========================================================================

    def execute(self) -> None:
        """Execute the planned trajectory on the real robot.

        Sequence:
        1. Open gripper
        2. Move to start (home) position
        3. Execute smooth trajectory via ``movej_canfd``
        4. Move to pick pose (with approach offset)
        5. Close gripper
        6. Retract to pre-pick pose
        7. Return to start position
        8. Move to place-ready position
        9. Open gripper
        10. Return to start position
        """
        robot_ctrl: RobotArmController = self.models["robot_ctrl"]
        trajectory_safe = self.data["trajectory_safe"]
        pick = self.data["pick_pose_current"]

        q_start = np.array(
            self.cfg.get("robot", {}).get(
                "start_pose", [0, 0, np.pi / 2, 0, np.pi / 2, 0]
            )
        )
        q_pic = np.array([np.pi / 2, 0, np.pi / 2, 0, np.pi / 2, 0])

        robot_ctrl.open_gripper()
        robot_ctrl.movej(q_start * RADIAN_TO_DEGREE)

        pre_pick_pose = pick * self.T_ET_Cali.inv()
        pick_pose = pre_pick_pose * SE3(0, 0, self.approach_offset)
        pre_pick_pose_goal = [*pre_pick_pose.t / 1000, *pre_pick_pose.rpy()]
        pick_pose_goal = [*pick_pose.t / 1000, *pick_pose.rpy()]

        # Execute smooth trajectory
        for q in trajectory_safe:
            robot_ctrl.movej_canfd(q * RADIAN_TO_DEGREE)
            time.sleep(self.path_time / self.num_path_joints)

        # Move to pick pose, grasp, retract
        robot_ctrl.movej_p(pick_pose_goal)
        robot_ctrl.close_gripper()
        time.sleep(1)

        robot_ctrl.movej_p(pre_pick_pose_goal)
        robot_ctrl.movej(q_start * RADIAN_TO_DEGREE)
        robot_ctrl.movej(q_pic * RADIAN_TO_DEGREE)

        # Open gripper to release (place position is user-defined)
        robot_ctrl.open_gripper()
        time.sleep(1)

        robot_ctrl.movej(q_start * RADIAN_TO_DEGREE)
        logger.info("Execution complete.")
