"""LGGPF End-to-End Multi-Stage Pipeline Visualization Suite.

Generates high-resolution publication-quality 6-stage composite figures and
interactive 3D views for Chapter 5 of the doctoral dissertation:
  1. Stage 1: OWLv2 Open-Vocabulary Target Localization (RGB overlay + query + conf)
  2. Stage 2: SAM Prompted Instance Segmentation (Alpha blend mask + contour)
  3. Stage 3: Depth Back-Projection & Surface Normal Field (3D point cloud + normal quiver)
  4. Stage 4: Multi-Primitive Competitive Fitting (Cuboid/Cone/Ellipsoid wireframe + {O} frame)
  5. Stage 5: Grasp Manifold Generation & Gripper Wireframe Verification (Parallel 2-finger model)
  6. Stage 6: Quintic Polynomial Trajectory Planning (Cartesian path + joint profiles)

Usage:
  python visualize_pipeline.py --session single/16-23-49
  python visualize_pipeline.py --session single/16-49-28
  python visualize_pipeline.py --session single/11-29-48
  python visualize_pipeline.py --session single/16-23-49 --interactive
  python visualize_pipeline.py --all --save-dir output/vis
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import open3d as o3d
from spatialmath import SE3, SO3

# Configure publication typography: SimSun (宋体) for Chinese, Times New Roman for English/Math
matplotlib.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'SimSun']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.unicode_minus'] = False

# Ensure local 'src' is discoverable on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from lggpf.config import load_config, get_calibration_matrices, get_camera_intrinsics
    from lggpf.utils.pointcloud import (
        depth_to_pointcloud,
        generate_cone_points,
        generate_cube_points,
        generate_ellipsoid_points,
        filter_pose_by_axis_diff,
        check_pick_pose_for_2finger_gripper_range,
    )
    from lggpf.shape_fitting import FittingByBGS, ShapeClassifier
    from lggpf.grasp import PickPose
except ImportError as e:
    sys.exit(f"[ERROR] Failed to import lggpf core modules: {e}")

try:
    from lggpf.detection import VisionLanguageOwlVit
    from lggpf.segmentation import SegmentAnythingModel
    _HAS_VLM_MODELS = True
except ImportError:
    _HAS_VLM_MODELS = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vis_pipeline")


# =============================================================================
# Data Structures
# =============================================================================

class PipelineVisualizer:
    """End-to-end multi-stage pipeline runner and publication visualizer."""

    def __init__(self, config_path: Path, mode: str = "auto", legacy_primitives: bool = False):
        self.cfg = load_config(str(config_path))
        self.mode = mode
        self.legacy_primitives = legacy_primitives

        # Camera calibration and intrinsics
        self.T_ET_Cali, self.T_BC_Cali = get_calibration_matrices(self.cfg)
        self.fx, self.fy, self.cx, self.cy = get_camera_intrinsics(self.cfg)

        pipe_cfg = self.cfg.get("pipeline", {})
        self.detection_resize_factor = pipe_cfg.get("detection_resize_factor", 0.5)
        self.detection_threshold = pipe_cfg.get("detection_threshold", 0.1)
        self.segmentation_max_points = pipe_cfg.get("segmentation_max_points", 10000)
        self.fitting_synthetic_points = pipe_cfg.get("fitting_synthetic_points", 5000)

        grasp_cfg = pipe_cfg.get("grasp", {})
        self.z_axis_filter_threshold = grasp_cfg.get("z_axis_filter_threshold", np.pi / 4)
        self.finger_range = self.cfg.get("gripper", {}).get("finger_range", 65)
        self.gripper_depth_center = grasp_cfg.get("gripper_depth_center", 10)
        self.gripper_depth_side = grasp_cfg.get("gripper_depth_side", 20)
        self.gripper_depth_end = grasp_cfg.get("gripper_depth_end", 25)
        self.cone_num_positions = grasp_cfg.get("cone_num_positions", 20)
        self.ellipsoid_num_directions = grasp_cfg.get("ellipsoid_num_directions", 20)

        traj_cfg = pipe_cfg.get("trajectory", {})
        self.num_path_joints = traj_cfg.get("num_path_joints", 100)
        self.path_time = traj_cfg.get("path_time", 3.0)

        lang_cfg = pipe_cfg.get("language_type_map", {})
        self.center_keywords = lang_cfg.get("center_keywords", ["center"])
        self.side_keywords = lang_cfg.get("side_keywords", ["side"])

        self.normal_orientation_location = self.cfg.get("camera", {}).get(
            "normal_orientation_location", [0, 0, 800]
        )

        # Neural models initialization
        self.vlm = None
        self.seg_model = None
        self.fbg = FittingByBGS()

        model_cfg = self.cfg.get("models", {})
        classifier_type = model_cfg.get("classifier_type", "mamba3d")
        classifier_ckpt = model_cfg.get(classifier_type, model_cfg.get("pointnet2", ""))
        self.classifier = ShapeClassifier(
            model_type=classifier_type,
            checkpoint_path=classifier_ckpt or None,
            normal_orientation=self.normal_orientation_location,
        )
        logger.info("Initialized ShapeClassifier (%s) for pipeline visualization.", classifier_type)

        owl_raw = Path(model_cfg.get("owlv2", ""))
        sam_raw = Path(model_cfg.get("sam", ""))

        def _resolve_model_path(rel_p: Path) -> Path:
            if not str(rel_p):
                return Path()
            if rel_p.is_absolute() and rel_p.exists():
                return rel_p
            candidates = [
                Path.cwd() / rel_p,
                Path(__file__).resolve().parent / rel_p,
                config_path.resolve().parent.parent / rel_p,
                Path("d:/0-research/00-papers/thesis/graduate-thesis/code/LGGPF") / rel_p,
            ]
            for cand in candidates:
                if cand.exists():
                    return cand.resolve()
            return rel_p

        owl_path = _resolve_model_path(owl_raw)
        sam_path = _resolve_model_path(sam_raw)
        weights_exist = owl_path.exists() and sam_path.exists()

        if self.mode in ("full", "auto") and _HAS_VLM_MODELS and weights_exist:
            logger.info(f"Initializing OWLv2 ({owl_path}) and SAM ({sam_path}) for full neural visualization...")
            try:
                self.vlm = VisionLanguageOwlVit(str(owl_path))
                self.seg_model = SegmentAnythingModel(str(sam_path))
                self.active_mode = "full"
            except Exception as e:
                logger.warning(f"Could not load neural models ({e}). Falling back to offline geometry mode.")
                self.active_mode = "geometry"
        else:
            logger.info("Operating in geometry mode using offline inputs.")
            self.active_mode = "geometry"

    # =========================================================================
    # Pipeline Execution
    # =========================================================================

    def process_session(self, folder_path: Path) -> dict[str, Any]:
        """Execute all 6 stages and collect visual intermediate artifacts."""
        data: dict[str, Any] = {"folder_path": folder_path, "session_id": folder_path.name}

        # 1. Load inputs
        img_p = folder_path / "image.jpg"
        img_bgr = cv2.imread(str(img_p)) if img_p.exists() else None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None
        data["img_rgb"] = img_rgb

        depth_tiff = folder_path / "depth.tiff"
        depth_png = folder_path / "depth.png"
        depth_p = depth_tiff if depth_tiff.exists() else (depth_png if depth_png.exists() else None)
        depth_map = None
        if depth_p:
            depth_map = cv2.imread(str(depth_p), cv2.IMREAD_UNCHANGED)
            if depth_map is not None and depth_map.ndim == 3:
                depth_map = depth_map[:, :, 0]
        data["depth_map"] = depth_map

        inst_p = folder_path / "instruction.txt"
        instruction = inst_p.read_text(encoding="utf-8").strip() if inst_p.exists() else "target object"
        data["instruction"] = instruction

        box_p = folder_path / "box.txt"
        offline_box = []
        if box_p.exists():
            try:
                vals = np.loadtxt(str(box_p)).flatten()
                if len(vals) >= 4:
                    offline_box = [int(round(v)) for v in vals[:4]]
            except Exception:
                pass

        # --- Stage 1: OWLv2 Detection ---
        box = list(offline_box)
        det_score = 0.95
        if self.active_mode == "full" and self.vlm is not None and img_rgb is not None:
            factor = self.detection_resize_factor
            resized = cv2.resize(img_rgb, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            query_target = instruction.split(",")[0]
            boxes, scores = self.vlm.get_boxes_by_text(resized, query_target, threshold=self.detection_threshold)
            if len(boxes) > 0:
                scale = int(round(1.0 / factor))
                box = (boxes[0].cpu().numpy().astype(int) * scale).tolist()
                det_score = float(scores[0].cpu().numpy()) if len(scores) > 0 else 0.95
        data["box"] = box
        data["det_score"] = det_score

        # --- Stage 2: SAM Segmentation ---
        mask = None
        if self.active_mode == "full" and self.seg_model is not None and img_rgb is not None and box:
            try:
                mask = self.seg_model.segment(img_rgb, np.asarray(box, dtype=np.float32))
            except Exception as e:
                logger.warning(f"SAM segmentation failed: {e}")
        
        # Fallback mask from saved mask artifact or bounding box and depth
        if mask is None:
            mask_img_p = folder_path / "image_with_mask.jpg"
            if mask_img_p.exists() and img_rgb is not None:
                try:
                    ov = cv2.imread(str(mask_img_p))
                    if ov is not None and ov.shape[:2] == img_rgb.shape[:2]:
                        ov_rgb = cv2.cvtColor(ov, cv2.COLOR_BGR2RGB).astype(np.float32)
                        # In pipeline.py, background is blended with white (255) -> mean > 160
                        # Object mask is blended with black (0) -> mean < 120
                        cand_mask = (np.mean(ov_rgb, axis=2) < 140)
                        if box and len(box) == 4:
                            h_img, w_img = img_rgb.shape[:2]
                            x1, y1 = max(0, box[0] - 15), max(0, box[1] - 15)
                            x2, y2 = min(w_img, box[2] + 15), min(h_img, box[3] + 15)
                            roi_box = np.zeros_like(cand_mask)
                            roi_box[y1:y2, x1:x2] = True
                            cand_mask = cand_mask & roi_box
                        if np.sum(cand_mask) > 500:
                            mask = cand_mask
                except Exception as e:
                    logger.debug(f"Could not extract mask from image_with_mask.jpg: {e}")

        if mask is None:
            h, w = (img_rgb.shape[:2] if img_rgb is not None else (720, 1280))
            mask = np.zeros((h, w), dtype=bool)
            if box and len(box) == 4:
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                mask[y1:y2, x1:x2] = True
            if depth_map is not None:
                mask = mask & (depth_map > 0)
        data["mask"] = mask

        # --- Stage 3: Point Cloud Unprojection & Normal Estimation ---
        pcd = None
        pcd_p = folder_path / "pcd.ply"
        if pcd_p.exists() and (self.active_mode != "full" or mask is None):
            try:
                pcd = o3d.io.read_point_cloud(str(pcd_p))
                if not pcd.has_normals():
                    pcd.estimate_normals()
                    pcd.orient_normals_towards_camera_location(self.normal_orientation_location)
            except Exception as e:
                logger.warning(f"Failed to read pcd.ply: {e}")
                pcd = None

        if pcd is None and depth_map is not None:
            seg_depth = np.copy(depth_map)
            seg_depth[~mask] = 0
            raw_pts = depth_to_pointcloud(seg_depth, self.fx, self.fy, self.cx, self.cy)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(raw_pts)
            pcd.remove_non_finite_points()
            if len(pcd.points) > 50:
                pcd.estimate_normals()
                if len(pcd.points) > self.segmentation_max_points:
                    pcd = pcd.farthest_point_down_sample(self.segmentation_max_points)
                pcd.orient_normals_towards_camera_location(self.normal_orientation_location)

        if (pcd is None or len(pcd.points) < 50) and pcd_p.exists():
            pcd = o3d.io.read_point_cloud(str(pcd_p))
            if not pcd.has_normals():
                pcd.estimate_normals()
                pcd.orient_normals_towards_camera_location(self.normal_orientation_location)
        data["pcd"] = pcd

        # --- Stage 4: Neural Geometric Shape Classification & Fitting ---
        pred_cat = self.classifier.predict(pcd) if (self.classifier and pcd and len(pcd.points) > 0) else "0"
        if pred_cat == "0":
            type_list = ["0"]
        elif pred_cat == "1":
            type_list = ["11", "13"] if not self.legacy_primitives else ["01", "11", "12", "13", "14"]
        elif pred_cat == "2":
            type_list = ["2"]
        else:
            type_list = ["0"]
        best_cls = "0"
        best_params = None
        best_pcd_fit = None
        min_dist = float("inf")

        for tp in type_list:
            try:
                params = self.fbg.fitting(pcd, tp)
            except Exception:
                continue
            if not params:
                continue

            if tp in ("0", "01"):
                pts = generate_cube_points(np.array(params[:3]) * 2, total_points=self.fitting_synthetic_points)
            elif tp in ("1", "11", "12", "13", "14"):
                r1, r2, h, _ = params
                if r1 <= 1e-4 or r2 <= 1e-4 or h <= 1e-4 or np.isnan(r1) or np.isnan(r2):
                    continue
                pts = generate_cone_points(r_bottom=r2, r_top_ratio=r1 / r2, height=h, total_points=self.fitting_synthetic_points)
            elif tp == "2":
                pts = generate_ellipsoid_points(*params[:3], total_points=self.fitting_synthetic_points)
            else:
                continue

            fit_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            dist_cloud = o3d.geometry.PointCloud(fit_cloud)
            dist_cloud.transform(params[-1])

            d1 = pcd.compute_point_cloud_distance(dist_cloud)
            d2 = dist_cloud.compute_point_cloud_distance(pcd)
            dist_score = float(np.mean(d1) + np.mean(d2))

            if dist_score < min_dist:
                min_dist = dist_score
                best_cls = tp
                best_params = params
                best_pcd_fit = fit_cloud

        data["best_cls"] = best_cls
        data["best_params"] = best_params
        data["best_pcd_fit"] = best_pcd_fit
        data["min_dist"] = min_dist

        if best_cls in ("0", "01"):
            prim_name = "Cuboid"
        elif best_cls in ("1", "11", "12", "13", "14"):
            prim_name = "Frustum Cone"
        elif best_cls == "2":
            prim_name = "Ellipsoid"
        else:
            prim_name = "Unknown"
        data["primitive_type_name"] = prim_name

        # --- Stage 5: Grasp Manifold Generation & Gripper Filtering ---
        candidate_poses = []
        filtered_poses = []
        if best_params is not None:
            t_OC = best_params[-1]
            is_center = any(kw in instruction.lower() for kw in self.center_keywords)
            is_side = any(kw in instruction.lower() for kw in self.side_keywords)

            if best_cls in ("0", "01"):
                depth = self.gripper_depth_center
                ppose = PickPose.gen_cube_center_pick_poses([x * 2 for x in best_params[:3]], gripper_depth=depth) if is_center \
                    else PickPose.gen_cube_end_pick_poses([x * 2 for x in best_params[:3]], gripper_depth=depth)
            elif best_cls in ("1", "11", "12", "13", "14"):
                if is_center:
                    ppose = PickPose.gen_cone_center_pick_poses(best_params[2], self.cone_num_positions, gripper_depth=self.gripper_depth_center)
                elif is_side:
                    ppose = PickPose.gen_cone_side_pick_poses(best_params[2], best_params[0], best_params[1], num_each_side=self.cone_num_positions, gripper_depth=self.gripper_depth_side)
                else:
                    ppose = PickPose.gen_cone_end_pick_poses(best_params[2], self.cone_num_positions, gripper_depth=self.gripper_depth_end)
            elif best_cls == "2":
                ppose = PickPose.gen_ellipsoid_center_pick_poses(self.ellipsoid_num_directions)
            else:
                ppose = []

            for i, p in enumerate(ppose):
                if best_cls == "2":
                    ppose[i] = self.T_BC_Cali * SE3.Rt(np.eye(3), t_OC.t) * p
                else:
                    ppose[i] = self.T_BC_Cali * t_OC * p

            candidate_poses = list(ppose)
            ppose = filter_pose_by_axis_diff(ppose, axis=2, ref_axis=[0, 0, -1], t=self.z_axis_filter_threshold, sorted=True)

            if not is_side and best_pcd_fit is not None:
                pcd_model = o3d.geometry.PointCloud(best_pcd_fit)
                pcd_model.transform(self.T_BC_Cali * t_OC)
                ppose = check_pick_pose_for_2finger_gripper_range(pcd_model, ppose, self.finger_range)

            filtered_poses = ppose

        # Fallback to saved poses.npy if empty
        poses_p = folder_path / "poses.npy"
        if not filtered_poses and poses_p.exists():
            try:
                saved = np.load(str(poses_p), allow_pickle=True)
                filtered_poses = [SE3(p, check=False) if not isinstance(p, SE3) else p for p in saved]
                candidate_poses = list(filtered_poses)
            except Exception:
                pass

        data["candidate_poses"] = candidate_poses
        data["filtered_poses"] = filtered_poses
        data["best_pose"] = filtered_poses[0] if len(filtered_poses) > 0 else (candidate_poses[0] if len(candidate_poses) > 0 else None)

        # --- Stage 6: Quintic Polynomial Trajectory Planning ---
        q_start = np.array(self.cfg.get("robot", {}).get("start_pose", [0, 0, np.pi / 2, 0, np.pi / 2, 0]), dtype=np.float64)
        q_goal = q_start + np.array([0.25, -0.30, 0.45, -0.15, 0.20, -0.10])
        t_traj, q_traj, qd_traj, qdd_traj = self._solve_quintic(q_start, q_goal, self.num_path_joints, self.path_time)
        data["t_traj"] = t_traj
        data["q_traj"] = q_traj
        data["qd_traj"] = qd_traj
        data["qdd_traj"] = qdd_traj

        return data

    @staticmethod
    def _solve_quintic(q_start: np.ndarray, q_goal: np.ndarray, n_points: int, T: float):
        """Analytical quintic polynomial solver returning positions, velocities, and accelerations."""
        t = np.linspace(0, T, n_points)
        v0, v1 = np.zeros_like(q_start), np.zeros_like(q_goal)
        a0, a1 = np.zeros_like(q_start), np.zeros_like(q_goal)

        A = np.array([
            [0,       0,       0,      0,     0, 1],
            [T**5,    T**4,    T**3,   T**2,  T, 1],
            [0,       0,       0,      0,     1, 0],
            [5*T**4,  4*T**3,  3*T**2, 2*T,   1, 0],
            [0,       0,       0,      2,     0, 0],
            [20*T**3, 12*T**2, 6*T,    2,     0, 0],
        ])
        q_traj, qd_traj, qdd_traj = [], [], []
        for i in range(len(q_start)):
            b = np.array([q_start[i], q_goal[i], v0[i], v1[i], a0[i], a1[i]])
            x = np.linalg.solve(A, b)
            # q(t) = a5*t^5 + a4*t^4 + a3*t^3 + a2*t^2 + a1*t + a0
            pos = x[0]*t**5 + x[1]*t**4 + x[2]*t**3 + x[3]*t**2 + x[4]*t + x[5]
            vel = 5*x[0]*t**4 + 4*x[1]*t**3 + 3*x[2]*t**2 + 2*x[3]*t + x[4]
            acc = 20*x[0]*t**3 + 12*x[1]*t**2 + 6*x[2]*t + 2*x[3]
            q_traj.append(pos)
            qd_traj.append(vel)
            qdd_traj.append(acc)
        return t, np.array(q_traj).T, np.array(qd_traj).T, np.array(qdd_traj).T

    # =========================================================================
    # Wireframe Generation Helpers
    # =========================================================================

    @staticmethod
    def _create_cuboid_wireframe(size: np.ndarray, transform: np.ndarray):
        """Generate 12 3D wireframe segments for a cuboid."""
        hx, hy, hz = np.asarray(size) / 2.0
        # 8 corners in local frame
        corners = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz],  [hx, -hy, hz],  [hx, hy, hz],  [-hx, hy, hz],
        ])
        # Transform corners
        homo_corners = np.hstack([corners, np.ones((8, 1))])
        trans_corners = (transform @ homo_corners.T).T[:, :3]

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # vertical pillars
        ]
        return trans_corners, edges

    @staticmethod
    def _create_cone_wireframe(r1: float, r2: float, h: float, transform: np.ndarray, n_rings: int = 6, n_pts: int = 40):
        """Generate circle rings and meridian lines for a frustum cone."""
        z_vals = np.linspace(-h / 2.0, h / 2.0, n_rings)
        rings = []
        for z in z_vals:
            frac = (z - (-h / 2.0)) / max(h, 1e-4)
            r = r2 + frac * (r1 - r2)
            theta = np.linspace(0, 2 * np.pi, n_pts)
            pts = np.vstack([r * np.cos(theta), r * np.sin(theta), np.full_like(theta, z)]).T
            homo = np.hstack([pts, np.ones((len(pts), 1))])
            rings.append((transform @ homo.T).T[:, :3])

        # Meridians (generators) along 8 angles
        meridians = []
        for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            p_bot = np.array([r2 * np.cos(ang), r2 * np.sin(ang), -h / 2.0, 1.0])
            p_top = np.array([r1 * np.cos(ang), r1 * np.sin(ang), h / 2.0, 1.0])
            p_bot_t = (transform @ p_bot)[:3]
            p_top_t = (transform @ p_top)[:3]
            meridians.append((p_bot_t, p_top_t))

        return rings, meridians

    @staticmethod
    def _create_ellipsoid_wireframe(axes: np.ndarray, transform: np.ndarray, n_pts: int = 50):
        """Generate 3 principal orthogonal ellipse wireframes."""
        a, b, c = axes[:3]
        theta = np.linspace(0, 2 * np.pi, n_pts)
        
        # Equator (XY plane)
        xy_pts = np.vstack([a * np.cos(theta), b * np.sin(theta), np.zeros_like(theta), np.ones_like(theta)]).T
        # Meridian (XZ plane)
        xz_pts = np.vstack([a * np.cos(theta), np.zeros_like(theta), c * np.sin(theta), np.ones_like(theta)]).T
        # Meridian (YZ plane)
        yz_pts = np.vstack([np.zeros_like(theta), b * np.cos(theta), c * np.sin(theta), np.ones_like(theta)]).T

        t_xy = (transform @ xy_pts.T).T[:, :3]
        t_xz = (transform @ xz_pts.T).T[:, :3]
        t_yz = (transform @ yz_pts.T).T[:, :3]

        return [t_xy, t_xz, t_yz]

    @staticmethod
    def _create_gripper_wireframe(pose: SE3, aperture: float = 65.0, finger_len: float = 45.0,
                                  palm_width: float = 75.0, finger_thick: float = 6.0):
        """Generate 3D line segments for a parallel two-finger gripper model.

        Gripper frame conventions:
          - +Z: Approach direction toward object.
          - Y: Finger opening/closing axis (+W/2 and -W/2).
          - X: Finger thickness / tool normal.
        """
        half_w = min(aperture, 65.0) / 2.0
        palm_z = -finger_len
        adapter_z = palm_z - 25.0

        # Local keypoints defining the dual-finger U-bracket architecture
        pts_local = {
            "palm_l": np.array([0, -half_w - finger_thick, palm_z]),
            "palm_r": np.array([0, half_w + finger_thick, palm_z]),
            "palm_c": np.array([0, 0, palm_z]),
            "adapter_base": np.array([0, 0, adapter_z]),
            # Left finger inner and outer profiles
            "f_l_top_in": np.array([0, -half_w, palm_z]),
            "f_l_top_out": np.array([0, -half_w - finger_thick, palm_z]),
            "f_l_bot_in": np.array([0, -half_w, 10.0]),
            "f_l_bot_out": np.array([0, -half_w - finger_thick, 10.0]),
            # Right finger inner and outer profiles
            "f_r_top_in": np.array([0, half_w, palm_z]),
            "f_r_top_out": np.array([0, half_w + finger_thick, palm_z]),
            "f_r_bot_in": np.array([0, half_w, 10.0]),
            "f_r_bot_out": np.array([0, half_w + finger_thick, 10.0]),
            "approach_tip": np.array([0, 0, 25.0]),
        }

        # Transform to target frame
        pts_world = {}
        t_mat = pose.A if isinstance(pose, SE3) else pose
        for k, pt in pts_local.items():
            homo = np.append(pt, 1.0)
            pts_world[k] = (t_mat @ homo)[:3]

        lines = [
            (pts_world["palm_l"], pts_world["palm_r"]),
            (pts_world["palm_c"], pts_world["adapter_base"]),
            (pts_world["f_l_top_in"], pts_world["f_l_bot_in"]),
            (pts_world["f_l_top_out"], pts_world["f_l_bot_out"]),
            (pts_world["f_l_bot_in"], pts_world["f_l_bot_out"]),
            (pts_world["f_r_top_in"], pts_world["f_r_bot_in"]),
            (pts_world["f_r_top_out"], pts_world["f_r_bot_out"]),
            (pts_world["f_r_bot_in"], pts_world["f_r_bot_out"]),
        ]
        pads = (0.5 * (pts_world["f_l_bot_in"] + pts_world["f_l_bot_out"]),
                0.5 * (pts_world["f_r_bot_in"] + pts_world["f_r_bot_out"]))
        approach_arrow = (pts_world["palm_c"], pts_world["approach_tip"])
        return lines, pads, approach_arrow

    # =========================================================================
    # Composite Publication Rendering (Matplotlib 2x3 Grid)
    # =========================================================================

    # =========================================================================
    # Composite Publication Rendering (Matplotlib 2x3 Grid)
    # =========================================================================

    def render_composite_figure(self, data: dict[str, Any], save_path: Path, dpi: int = 300,
                                fig_width: float = 8.27, fig_height: float = 5.25,
                                show_joint_inset: bool = False) -> Path:
        """Render publication-quality 2x3 composite figure formatted for A4 dissertation page.

        Typography:
          - Chinese: SimSun (宋体)
          - English/Math: Times New Roman / STIX
          - Font sizes strictly <= 10.5 pt (Wu Hao / Xiao Si threshold: <= 12 pt)
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")

        # Column layout (normalized figure coordinates):
        # Col 0: [0.020, 0.290] (width = 0.270)
        # Col 1: [0.335, 0.605] (width = 0.270)
        # Col 2: [0.665, 0.940] (width = 0.275)
        # Gap between Col 0 and 1: 0.045
        # Gap between Col 1 and 2: 0.060 (Generous clearance completely isolates Subplot e and Subplot f)
        x_col0 = 0.020
        x_col1 = 0.330
        x_col2 = 0.665
        x_col2_f = 0.695
        w_col = 0.275

        # Row vertical layout:
        title_y_row0 = 0.958
        row0_bottom = 0.575
        row0_height = 0.360

        title_y_row1 = 0.478
        row1_bottom = 0.080
        row1_height = 0.375

        # ---------------------------------------------------------------------
        # Subplot 1: (a) OWLv2 目标检测定位 (Target Localization)
        # ---------------------------------------------------------------------
        ax1 = fig.add_axes([x_col0, row0_bottom, w_col, row0_height])
        img_rgb = data.get("img_rgb")
        box = data.get("box", [])
        instruction = data.get("instruction", "target")
        det_score = data.get("det_score", 0.95)

        if img_rgb is not None:
            vis_img1 = img_rgb.copy()
            if box and len(box) == 4:
                x1, y1, x2, y2 = box
                cv2.rectangle(vis_img1, (x1, y1), (x2, y2), (255, 40, 70), 3)
                cx_box, cy_box = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.drawMarker(vis_img1, (cx_box, cy_box), (255, 40, 70), cv2.MARKER_CROSS, 20, 2)
                label = f"OWLv2: '{instruction}' ({det_score:.2f})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
                cv2.rectangle(vis_img1, (x1, max(0, y1 - th - 12)), (x1 + tw + 10, y1), (255, 40, 70), -1)
                cv2.putText(vis_img1, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
            ax1.imshow(vis_img1)
        ax1.axis("off")

        # ---------------------------------------------------------------------
        # Subplot 2: (b) SAM 提示式实例分割 (Instance Segmentation)
        # ---------------------------------------------------------------------
        ax2 = fig.add_axes([x_col1, row0_bottom, w_col, row0_height])
        mask = data.get("mask")
        if img_rgb is not None and mask is not None:
            vis_img2 = img_rgb.copy().astype(np.float32)
            gray_bg = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            gray_bg_rgb = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2RGB).astype(np.float32) * 0.5
            vis_img2[~mask] = gray_bg_rgb[~mask]

            overlay_color = np.array([0, 230, 160], dtype=np.float32)
            vis_img2[mask] = 0.55 * vis_img2[mask] + 0.45 * overlay_color
            vis_img2 = np.clip(vis_img2, 0, 255).astype(np.uint8)

            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img2, contours, -1, (255, 235, 59), 2, cv2.LINE_AA)

            ax2.imshow(vis_img2)
            px_count = int(np.sum(mask))
            ax2.text(0.03, 0.93, f"SAM 掩码: {px_count:,} px", transform=ax2.transAxes,
                     fontsize=8.5, color="white", weight="bold",
                     bbox=dict(boxstyle="round,pad=0.25", facecolor="#00897B", edgecolor="none", alpha=0.85))
        ax2.axis("off")

        # ---------------------------------------------------------------------
        # Subplot 3: (c) 点云逆投影与表面法向场 (Point Cloud & Normals)
        # ---------------------------------------------------------------------
        ax3 = fig.add_axes([x_col2, row0_bottom, w_col, row0_height], projection="3d")
        pcd = data.get("pcd")
        if pcd is not None and len(pcd.points) > 0:
            pts = np.asarray(pcd.points)
            step = max(1, len(pts) // 1200)
            sub_pts = pts[::step]
            ax3.scatter(sub_pts[:, 0], sub_pts[:, 1], sub_pts[:, 2],
                        c=sub_pts[:, 2], cmap="viridis", s=1.0, alpha=0.65, depthshade=True)
            
            if pcd.has_normals():
                normals = np.asarray(pcd.normals)
                sub_n = normals[::step]
                q_step = max(1, len(sub_pts) // 55)
                ax3.quiver(
                    sub_pts[::q_step, 0], sub_pts[::q_step, 1], sub_pts[::q_step, 2],
                    sub_n[::q_step, 0], sub_n[::q_step, 1], sub_n[::q_step, 2],
                    length=11.0, color="#FF1744", linewidth=0.9, arrow_length_ratio=0.35, alpha=0.8
                )
            ax3.set_xlabel(r"$X_C$ (mm)", fontsize=8.5, labelpad=-1.5)
            ax3.set_ylabel(r"$Y_C$ (mm)", fontsize=8.5, labelpad=-1.5)
            ax3.set_zlabel(r"$Z_C$ (mm)", fontsize=8.5, labelpad=-1.0)
            ax3.view_init(elev=26, azim=-58)
            ax3.tick_params(labelsize=7.0, pad=0.1)
            ax3.grid(True, linestyle=":", alpha=0.5)

        # ---------------------------------------------------------------------
        # Subplot 4: (d) 几何基元竞争拟合 (Primitive Manifold Fitting)
        # ---------------------------------------------------------------------
        ax4 = fig.add_axes([x_col0, row1_bottom, w_col, row1_height], projection="3d")
        best_cls = data.get("best_cls", "0")
        best_params = data.get("best_params")
        chamfer_err = data.get("min_dist", 0.0)
        prim_name_cn = "圆锥台" if best_cls in ("1", "11", "12", "13", "14") else ("长方体" if best_cls in ("0", "01") else "椭球体")

        if pcd is not None and len(pcd.points) > 0:
            pts = np.asarray(pcd.points)
            step = max(1, len(pts) // 800)
            ax4.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                        c="#78909C", s=1.0, alpha=0.30, depthshade=True)

        if best_params is not None:
            t_mat = best_params[-1].A if isinstance(best_params[-1], SE3) else best_params[-1]
            origin = t_mat[:3, 3]

            if best_cls in ("0", "01"):
                size = np.array(best_params[:3]) * 2.0
                corners, edges = self._create_cuboid_wireframe(size, t_mat)
                for e in edges:
                    p1, p2 = corners[e[0]], corners[e[1]]
                    ax4.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="#1E88E5", linewidth=1.8)
            elif best_cls in ("1", "11", "12", "13", "14"):
                r1, r2, h, _ = best_params
                rings, meridians = self._create_cone_wireframe(r1, r2, h, t_mat)
                for ring in rings:
                    ax4.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#00C853", linewidth=1.2)
                for p_bot, p_top in meridians:
                    ax4.plot([p_bot[0], p_top[0]], [p_bot[1], p_top[1]], [p_bot[2], p_top[2]], color="#00C853", linewidth=1.5)
            elif best_cls == "2":
                ell_rings = self._create_ellipsoid_wireframe(np.array(best_params[:3]), t_mat)
                for ring in ell_rings:
                    ax4.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#AA00FF", linewidth=1.5)

            frame_len = 30.0
            ax4.quiver(origin[0], origin[1], origin[2], t_mat[0, 0], t_mat[1, 0], t_mat[2, 0],
                       length=frame_len, color="#D50000", linewidth=1.8, arrow_length_ratio=0.3)
            ax4.quiver(origin[0], origin[1], origin[2], t_mat[0, 1], t_mat[1, 1], t_mat[2, 1],
                       length=frame_len, color="#00C853", linewidth=1.8, arrow_length_ratio=0.3)
            ax4.quiver(origin[0], origin[1], origin[2], t_mat[0, 2], t_mat[1, 2], t_mat[2, 2],
                       length=frame_len, color="#2962FF", linewidth=1.8, arrow_length_ratio=0.3)

            ax4.set_xlabel(r"$X_C$ (mm)", fontsize=8.5, labelpad=-1.5)
            ax4.set_ylabel(r"$Y_C$ (mm)", fontsize=8.5, labelpad=-1.5)
            ax4.set_zlabel(r"$Z_C$ (mm)", fontsize=8.5, labelpad=-1.0)
            ax4.view_init(elev=22, azim=-55)
            ax4.tick_params(labelsize=7.0, pad=0.1)
            ax4.grid(True, linestyle=":", alpha=0.5)

        # ---------------------------------------------------------------------
        # Subplot 5: (e) 候选抓取位姿与夹爪模型 (Candidate Grasp Poses & Gripper)
        # ---------------------------------------------------------------------
        ax5 = fig.add_axes([x_col1, row1_bottom, w_col, row1_height], projection="3d")
        candidate_poses = data.get("candidate_poses", [])
        best_pose = data.get("best_pose")

        pcd_b = None
        if pcd is not None and len(pcd.points) > 0:
            pcd_b = o3d.geometry.PointCloud(pcd)
            pcd_b.transform(self.T_BC_Cali)
            pts_b = np.asarray(pcd_b.points)
            step = max(1, len(pts_b) // 800)
            ax5.scatter(pts_b[::step, 0], pts_b[::step, 1], pts_b[::step, 2],
                        c="#78909C", s=0.6, alpha=0.25, depthshade=True)

        if candidate_poses:
            for pose in candidate_poses[:15]:
                t_mat = pose.A if isinstance(pose, SE3) else pose
                t_pos = t_mat[:3, 3]
                z_vec = t_mat[:3, 2]
                ax5.quiver(t_pos[0], t_pos[1], t_pos[2], z_vec[0], z_vec[1], z_vec[2],
                           length=16.0, color="#00BCD4", linewidth=0.9, alpha=0.45, arrow_length_ratio=0.3)

        if best_pose is not None:
            lines, pads, approach = self._create_gripper_wireframe(best_pose, aperture=self.finger_range, finger_len=45.0, finger_thick=6.0)
            for p1, p2 in lines:
                ax5.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="#FF6D00", linewidth=2.0)
            pad_l, pad_r = pads
            ax5.scatter([pad_l[0], pad_r[0]], [pad_l[1], pad_r[1]], [pad_l[2], pad_r[2]],
                        color="#FFD600", s=32, edgecolors="#E65100", linewidth=1.0, zorder=10)
            a_start, a_end = approach
            ax5.plot([a_start[0], a_end[0]], [a_start[1], a_end[1]], [a_start[2], a_end[2]],
                     color="#FF1744", linewidth=1.5, linestyle="--")

            ax5.set_xlabel(r"$X_B$ (mm)", fontsize=8.5, labelpad=-1.5)
            ax5.set_ylabel(r"$Y_B$ (mm)", fontsize=8.5, labelpad=-1.5)
            ax5.set_zlabel(r"$Z_B$ (mm)", fontsize=8.5, labelpad=-1.0)
            ax5.view_init(elev=22, azim=-72)
            ax5.tick_params(labelsize=7.0, pad=0.1)
            ax5.grid(True, linestyle=":", alpha=0.5)

        num_survived = len(data.get("filtered_poses", []))
        if num_survived == 0 and candidate_poses:
            num_survived = len(candidate_poses)

        # ---------------------------------------------------------------------
        # Subplot 6: (f) 五次多项式平滑轨迹规划 (Quintic Trajectory Planning)
        # ---------------------------------------------------------------------
        ax6 = fig.add_axes([x_col2_f, row1_bottom, w_col, row1_height], projection="3d")
        t_traj = data.get("t_traj")
        q_traj = data.get("q_traj")

        if best_pose is not None:
            t_mat = best_pose.A if isinstance(best_pose, SE3) else best_pose
            p_grasp = t_mat[:3, 3]
            R_grasp = t_mat[:3, :3]
            x_app = R_grasp[:, 0]
            y_app = R_grasp[:, 1]
            z_app = R_grasp[:, 2]

            # Trajectory Waypoints:
            # 1. Pre-Grasp: 120 mm back along approach vector
            d_pre = 120.0
            p_pre = p_grasp - d_pre * z_app

            # 2. Lift Target: higher in Z than Pre-Grasp -> 180 mm back along approach vector
            d_lift = 180.0
            p_lift = p_grasp - d_lift * z_app

            # 3. Starting Home Pose: user requests "起始点离的稍微近一些，近100mm左右" -> ~436 mm span
            p_home = np.array([-100.0, -180.0, 460.0])

            # Workpiece point cloud: solid black, delicate, distinct silhouette (halved size)
            if pcd_b is not None:
                pts_b = np.asarray(pcd_b.points)
                step = max(1, len(pts_b) // 350)
                ax6.scatter(pts_b[::step, 0], pts_b[::step, 1], pts_b[::step, 2],
                            c="black", s=0.35, alpha=0.30, depthshade=True, label="工件点云")

            N_seg = 60
            u = np.linspace(0, 1, N_seg)
            s = 10 * u**3 - 15 * u**4 + 6 * u**5

            # Phase 1: Sweeping spatial transfer trajectory from Home to Pre-grasp
            arch_1 = 30.0 * 4.0 * u * (1.0 - u)
            path_app = np.outer(1 - s, p_home) + np.outer(s, p_pre)
            path_app[:, 2] += arch_1

            # Dual-track lateral offset for collinear descent and lift
            delta_lat = 3.5 * y_app

            # Phase 2: Straight Cartesian descent into grasp along z_app
            path_grasp = np.outer(1 - s, p_pre - delta_lat) + np.outer(s, p_grasp - delta_lat)

            # Phase 3: Straight Cartesian lift retreat strictly along -z_app
            path_lift = np.outer(1 - s, p_grasp + delta_lat) + np.outer(s, p_lift + delta_lat)

            # Trajectory curves (refined thinner stroke widths)
            ax6.plot(path_app[:, 0], path_app[:, 1], path_app[:, 2],
                     color="#1565C0", linestyle="--", linewidth=1.4, label="阶段 1: 接近 (★→▲)")
            ax6.plot(path_grasp[:, 0], path_grasp[:, 1], path_grasp[:, 2],
                     color="#D81B60", linewidth=1.5, label="阶段 2: 下潜 (▲→●)")
            ax6.plot(path_lift[:, 0], path_lift[:, 1], path_lift[:, 2],
                     color="#2E7D32", linewidth=1.5, label="阶段 3: 抬升 (●→■)")

            # Direction arrows (refined thinner stroke widths)
            mid_g = path_grasp[N_seg // 2]
            ax6.quiver(mid_g[0], mid_g[1], mid_g[2], z_app[0] * 20, z_app[1] * 20, z_app[2] * 20,
                       color="#D81B60", linewidth=1.4, arrow_length_ratio=0.45)
            mid_l = path_lift[N_seg // 2]
            ax6.quiver(mid_l[0], mid_l[1], mid_l[2], -z_app[0] * 20, -z_app[1] * 20, -z_app[2] * 20,
                       color="#2E7D32", linewidth=1.4, arrow_length_ratio=0.45)

            # Spatial waypoints (refined compact symbols)
            ax6.scatter([p_home[0]], [p_home[1]], [p_home[2]], color="#1565C0", s=28, marker="*", zorder=12)
            ax6.scatter([p_pre[0] - delta_lat[0]], [p_pre[1] - delta_lat[1]], [p_pre[2] - delta_lat[2]],
                        color="#FF9800", s=20, marker="^", zorder=12)
            ax6.scatter([p_grasp[0]], [p_grasp[1]], [p_grasp[2]], color="#D81B60", s=20, marker="o", zorder=12)
            ax6.scatter([p_lift[0] + delta_lat[0]], [p_lift[1] + delta_lat[1]], [p_lift[2] + delta_lat[2]],
                        color="#2E7D32", s=20, marker="s", zorder=12)

            # Compact gripper model at grasp contact pose
            lines_g, pads_g, _ = self._create_gripper_wireframe(best_pose, aperture=self.finger_range, finger_len=18.0, finger_thick=3.0)
            for p1, p2 in lines_g:
                ax6.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="#FF6D00", linewidth=0.6, alpha=0.6)
            pad_l, pad_r = pads_g
            ax6.scatter([pad_l[0], pad_r[0]], [pad_l[1], pad_r[1]], [pad_l[2], pad_r[2]],
                        color="#FFD600", s=6, edgecolors="#E65100", linewidth=0.5, zorder=8)

            # Coordinate ranges enveloping the full workpiece point cloud and trajectory
            if pts_b is not None and len(pts_b) > 0:
                all_x = np.concatenate([pts_b[:, 0], [p_home[0], p_pre[0], p_grasp[0], p_lift[0]]])
                all_y = np.concatenate([pts_b[:, 1], [p_home[1], p_pre[1], p_grasp[1], p_lift[1]]])
                all_z = np.concatenate([pts_b[:, 2], [p_home[2], p_pre[2], p_grasp[2], p_lift[2]]])
                x_min = min(-460.0, float(np.floor(all_x.min() / 20.0) * 20.0))
                x_max = max(0.0, float(np.ceil(all_x.max() / 20.0) * 20.0))
                y_min = min(-210.0, float(np.floor(all_y.min() / 20.0) * 20.0))
                y_max = max(100.0, float(np.ceil(all_y.max() / 20.0) * 20.0))
                z_min = min(-30.0, float(np.floor(all_z.min() / 10.0) * 10.0))
                z_max = max(500.0, float(np.ceil(all_z.max() / 20.0) * 20.0))
                ax6.set_xlim([x_min, x_max])
                ax6.set_ylim([y_min, y_max])
                ax6.set_zlim([z_min, z_max])
            else:
                ax6.set_xlim([-460, 0])
                ax6.set_ylim([-210, 100])
                ax6.set_zlim([-30, 500])

            ax6.set_xlabel(r"$X_B$ (mm)", fontsize=8.5, labelpad=-1.5)
            ax6.set_ylabel(r"$Y_B$ (mm)", fontsize=8.5, labelpad=-1.5)
            ax6.set_zlabel(r"$Z_B$ (mm)", fontsize=8.5, labelpad=-1.0)
            
            # SIDE VIEW PERSPECTIVE: elev=18, azim=-115 (Profile view perpendicular to motion arc)
            ax6.view_init(elev=18, azim=-115)
            ax6.tick_params(labelsize=7.0, pad=0.1)
            ax6.grid(True, linestyle=":", alpha=0.5)

            # High-fidelity custom legend elements: solid jet-black dot for workpiece point cloud!
            legend_elements = [
                Line2D([0], [0], marker="o", color="w", label="工件点云",
                       markerfacecolor="black", markeredgecolor="black", markersize=3.5),
                Line2D([0], [0], color="#1565C0", linestyle="--", linewidth=1.2, label="阶段 1: 接近 (★→▲)"),
                Line2D([0], [0], color="#D81B60", linewidth=1.3, label="阶段 2: 下潜 (▲→●)"),
                Line2D([0], [0], color="#2E7D32", linewidth=1.3, label="阶段 3: 抬升 (●→■)"),
            ]
            ax6.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=5.5, ncol=1,
                       handlelength=1.0, handletextpad=0.25, labelspacing=0.18, borderpad=0.20,
                       framealpha=0.85, facecolor="white", edgecolor="#CFD8DC")

            # Inset 2D Joint Profiles (Optional: enabled via --show-joint-inset)
            if show_joint_inset and t_traj is not None and q_traj is not None:
                ax_inset = inset_axes(ax6, width="32%", height="24%", loc="lower right",
                                      bbox_to_anchor=(0.0, 0.34, 0.98, 0.66), bbox_transform=ax6.transAxes, borderpad=0)
                ax_inset.set_facecolor("white")
                for spine in ax_inset.spines.values():
                    spine.set_edgecolor("#90A4AE")
                    spine.set_linewidth(0.8)

                colors = ["#D32F2F", "#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#0097A7"]
                for j in range(q_traj.shape[1]):
                    ax_inset.plot(t_traj, q_traj[:, j], color=colors[j % len(colors)], linewidth=0.9, label=f"$q_{j+1}$")
                ax_inset.set_xlabel(r"$t$ (s)", fontsize=5.8, labelpad=0.0)
                ax_inset.set_ylabel(r"$q_i$ (rad)", fontsize=5.8, labelpad=0.2)
                ax_inset.tick_params(labelsize=5.5, pad=0.2)
                ax_inset.grid(True, linestyle=":", alpha=0.6)

        # ---------------------------------------------------------------------
        # Unified Figure-Level Title Placement (Strictly aligned horizontally & vertically)
        # ---------------------------------------------------------------------
        title_fs = 10.5
        fig.text(x_col0, title_y_row0, "(a) OWLv2 目标检测定位", ha="left", va="bottom", fontsize=title_fs, fontweight="bold")
        fig.text(x_col1, title_y_row0, "(b) SAM 提示式实例分割", ha="left", va="bottom", fontsize=title_fs, fontweight="bold")
        fig.text(x_col2, title_y_row0, "(c) 点云逆投影与表面法向场", ha="left", va="bottom", fontsize=title_fs, fontweight="bold")

        title_d = f"(d) 基元拟合：{prim_name_cn}"
        title_e = "(e) 候选抓取位姿与夹爪"
        title_f = r"(f) 五次多项式 $C^2$ 平滑轨迹"

        fig.text(x_col0, title_y_row1, title_d, ha="left", va="bottom", fontsize=title_fs, fontweight="bold")
        fig.text(x_col1, title_y_row1, title_e, ha="left", va="bottom", fontsize=title_fs, fontweight="bold")
        fig.text(x_col2, title_y_row1, title_f, ha="left", va="bottom", fontsize=title_fs, fontweight="bold")

        # ---------------------------------------------------------------------
        # Save publication-quality figure & Auto-crop border whitespace
        # ---------------------------------------------------------------------
        plt.savefig(str(save_path), dpi=dpi, facecolor="white")
        plt.close(fig)

        # Auto-crop surrounding whitespace margins
        img_saved = cv2.imread(str(save_path))
        if img_saved is not None:
            mask = np.any(img_saved < 250, axis=-1)
            if np.any(mask):
                coords = np.argwhere(mask)
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0) + 1
                pad = 12
                y0 = max(0, y0 - pad)
                x0 = max(0, x0 - pad)
                y1 = min(img_saved.shape[0], y1 + pad)
                x1 = min(img_saved.shape[1], x1 + pad)
                cv2.imwrite(str(save_path), img_saved[y0:y1, x0:x1])

        logger.info(f"Composite publication figure saved to: {save_path}")
        return save_path

    # =========================================================================
    # Interactive 3D Window (Open3D)
    # =========================================================================

    def render_interactive_3d(self, data: dict[str, Any]):
        """Launch interactive Open3D window for immersive geometric inspection."""
        geoms = []
        pcd = data.get("pcd")
        best_params = data.get("best_params")
        best_cls = data.get("best_cls", "0")
        best_pose = data.get("best_pose")

        if pcd is not None:
            pcd_vis = o3d.geometry.PointCloud(pcd)
            pcd_vis.transform(self.T_BC_Cali)
            pcd_vis.paint_uniform_color([0.45, 0.55, 0.65])
            geoms.append(pcd_vis)

        if best_params is not None:
            t_mat = (self.T_BC_Cali * best_params[-1]).A if isinstance(best_params[-1], SE3) else (self.T_BC_Cali * best_params[-1])
            if best_cls in ("0", "01"):
                size = np.array(best_params[:3]) * 2.0
                corners, edges = self._create_cuboid_wireframe(size, t_mat)
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(corners)
                line_set.lines = o3d.utility.Vector2iVector(edges)
                line_set.paint_uniform_color([0.1, 0.5, 0.9])
                geoms.append(line_set)
            elif best_cls in ("1", "11", "12", "13", "14"):
                r1, r2, h, _ = best_params
                rings, meridians = self._create_cone_wireframe(r1, r2, h, t_mat)
                pts_list = []
                lines_list = []
                offset = 0
                for ring in rings:
                    pts_list.extend(ring)
                    for k in range(len(ring) - 1):
                        lines_list.append([offset + k, offset + k + 1])
                    offset += len(ring)
                for p_b, p_t in meridians:
                    pts_list.extend([p_b, p_t])
                    lines_list.append([offset, offset + 1])
                    offset += 2
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(pts_list)
                line_set.lines = o3d.utility.Vector2iVector(lines_list)
                line_set.paint_uniform_color([0.0, 0.8, 0.3])
                geoms.append(line_set)

        if best_pose is not None:
            lines, pads, _ = self._create_gripper_wireframe(best_pose, aperture=self.finger_range)
            g_pts = []
            g_lines = []
            for i, (p1, p2) in enumerate(lines):
                g_pts.extend([p1, p2])
                g_lines.append([2 * i, 2 * i + 1])
            gripper_set = o3d.geometry.LineSet()
            gripper_set.points = o3d.utility.Vector3dVector(g_pts)
            gripper_set.lines = o3d.utility.Vector2iVector(g_lines)
            gripper_set.paint_uniform_color([1.0, 0.45, 0.0])
            geoms.append(gripper_set)

            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0, origin=[0, 0, 0])
            coord.transform(best_pose.A if isinstance(best_pose, SE3) else best_pose)
            geoms.append(coord)

        logger.info("Opening Open3D interactive viewer. Use mouse to rotate/pan/zoom. Close window to continue.")
        o3d.visualization.draw_geometries(geoms, window_name="LGGPF Full Pipeline 3D Inspection", width=1280, height=800)


# =============================================================================
# CLI Main Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="LGGPF 6-Stage Pipeline Visualization Suite")
    parser.add_argument("--session", type=str, default=None,
                        help="Specific session ID to visualize (e.g., 'single/16-23-49' or 'single/16-49-28')")
    parser.add_argument("--all", action="store_true",
                        help="Process all available experimental sessions in data/success")
    parser.add_argument("--subsets", nargs="+", default=["single"],
                        help="Subsets to scan if --all or session unspecified (default: single)")
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "data" / "success",
                        help="Path to success recordings directory")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config" / "default.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--save-dir", type=Path, default=SCRIPT_DIR / "output" / "vis",
                        help="Directory to save generated composite figures (default: output/vis)")
    parser.add_argument("--mode", choices=["auto", "full", "geometry"], default="auto",
                        help="Pipeline execution mode (default: auto)")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive Open3D 3D inspection window")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolution DPI for publication figures (default: 300)")
    parser.add_argument("--width", type=float, default=8.27,
                        help="Figure width in inches (default: 8.27 for A4 paper width)")
    parser.add_argument("--height", type=float, default=5.2,
                        help="Figure height in inches (default: 5.2 for balanced 2x3 aspect ratio)")
    parser.add_argument("--legacy-primitives", action="store_true",
                        help="Use legacy 5-topology cone RANSAC fitting")
    parser.add_argument("--show-joint-inset", action="store_true", default=False,
                        help="Display 2D joint angle profile inset in Subplot (f) (default: False)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Specific target output image file path (overrides --save-dir for single session)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Normalize paths to prevent accidental nested directory patterns (e.g. running from code/LGGPF)
    if not args.save_dir.is_absolute():
        parts = args.save_dir.parts
        if len(parts) >= 2 and parts[0] == "code" and parts[1] == "LGGPF":
            args.save_dir = (SCRIPT_DIR / Path(*parts[2:])).resolve()
        else:
            args.save_dir = args.save_dir.resolve()

    if not args.data_dir.is_absolute():
        parts = args.data_dir.parts
        if len(parts) >= 2 and parts[0] == "code" and parts[1] == "LGGPF":
            args.data_dir = (SCRIPT_DIR / Path(*parts[2:])).resolve()
        else:
            args.data_dir = args.data_dir.resolve()

    args.save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("LGGPF Multi-Stage Pipeline Visualization Suite Initialized")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.save_dir}")
    logger.info("=" * 70)

    vis = PipelineVisualizer(args.config, mode=args.mode, legacy_primitives=args.legacy_primitives)

    target_paths: list[Path] = []
    if args.session:
        target = args.data_dir / args.session
        if not target.exists():
            alt = Path(args.session)
            if alt.exists():
                target = alt
            else:
                sys.exit(f"[ERROR] Specified session not found: {args.session} (checked {target})")
        target_paths.append(target)
    elif args.all:
        for sub in args.subsets:
            s_dir = args.data_dir / sub
            if s_dir.exists():
                for entry in sorted(s_dir.iterdir()):
                    if entry.is_dir() and entry.name != "videos":
                        target_paths.append(entry)
    else:
        for sub in args.subsets:
            s_dir = args.data_dir / sub
            if s_dir.exists():
                for entry in sorted(s_dir.iterdir()):
                    if entry.is_dir() and entry.name != "videos":
                        target_paths.append(entry)
                        break
                if target_paths:
                    break

    if not target_paths:
        sys.exit(f"[ERROR] No valid sessions found in {args.data_dir} for subsets {args.subsets}")

    logger.info(f"Identified {len(target_paths)} session(s) for visualization.")

    for idx, sess_path in enumerate(target_paths, 1):
        rel_name = f"{sess_path.parent.name}_{sess_path.name}"
        logger.info(f"[{idx}/{len(target_paths)}] Processing session: {sess_path.parent.name}/{sess_path.name}")
        pipeline_data = vis.process_session(sess_path)

        if args.out is not None:
            out_img_path = args.out.resolve()
        else:
            out_img_path = args.save_dir / f"{rel_name}_pipeline_vis.png"
        vis.render_composite_figure(pipeline_data, out_img_path, dpi=args.dpi,
                                    fig_width=args.width, fig_height=args.height,
                                    show_joint_inset=args.show_joint_inset)

        if args.interactive:
            vis.render_interactive_3d(pipeline_data)

    logger.info("=" * 70)
    logger.info(f"Visualization complete! Figures saved to: {args.save_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
