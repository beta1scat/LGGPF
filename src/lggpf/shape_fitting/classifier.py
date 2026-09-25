"""Point cloud shape classifier using Mamba3D and PointNet2.

Classifies point clouds into canonical geometric shape categories:
  - '0': Cuboid
  - '1': Truncated cone / Frustum
  - '2': Ellipsoid

Supports Mamba3D state-space model, PointNet2 SSG, and a pass-through/none mode.
Gracefully falls back to PointNet2 if Mamba3D encounters an unrecoverable runtime error.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from ..utils.pointcloud import pc_normalize
from .mamba3d import Mamba3D
from .pointnet2 import get_model

logger = logging.getLogger(__name__)

DEFAULT_MODEL_CONFIG = {
    "NAME": "Mamba3D",
    "trans_dim": 384,
    "depth": 12,
    "drop_path_rate": 0.2,
    "cls_dim": 3,
    "num_heads": 6,
    "group_size": 32,
    "num_group": 128,
    "encoder_dims": 384,
    "bimamba_type": "v4",
    "center_local_k": 4,
    "ordering": False,
    "label_smooth": 0.0,
    "lr_ratio_cls": 1.0,
    "lr_ratio_lfa": 1.0,
}

CLASS_MAP = {
    0: "0",  # cuboid
    1: "1",  # cone / frustum
    2: "2",  # ellipsoid
}


class ShapeClassifier:
    """Unified 3D point cloud geometric shape classifier.

    Supports Mamba3D (default), PointNet2, and 'none' modes.
    Classifies segmented point clouds into one of three primitive types:
    '0' (cuboid), '1' (truncated cone), or '2' (ellipsoid).

    Args:
        model_type: Classifier architecture: 'mamba3d', 'pointnet2', or 'none'.
            If a path string is provided (legacy usage), it is treated as checkpoint_path.
        checkpoint_path: Path to the trained model checkpoint (.pt or .pth).
        num_class: Number of target shape classes (default 3).
        normal_orientation: Camera orientation for normal alignment [x, y, z].
        device: Device for inference ('cuda' or 'cpu'). Defaults to CUDA if available.
    """

    def __init__(
        self,
        model_type: str = "mamba3d",
        checkpoint_path: str | Path | None = None,
        num_class: int = 3,
        normal_orientation: list[float] | None = None,
        device: str | torch.device | None = None,
    ):
        self.num_class = num_class
        self.normal_orientation = normal_orientation or [0.0, 0.0, 800.0]

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Backward compatibility: PcdClassification(path) passed checkpoint path as first arg
        if model_type not in ("mamba3d", "pointnet2", "none"):
            checkpoint_path = model_type
            if checkpoint_path and "mamba" in str(checkpoint_path).lower():
                model_type = "mamba3d"
            else:
                model_type = "pointnet2"

        self.model_type = model_type
        self.checkpoint_path: str | None = None
        self.model: torch.nn.Module | None = None

        if self.model_type == "mamba3d":
            success = self._init_mamba3d(checkpoint_path)
            if not success:
                logger.warning("Mamba3D initialization failed; falling back to PointNet2.")
                self._init_pointnet2(None)
        elif self.model_type == "pointnet2":
            self._init_pointnet2(checkpoint_path)
        elif self.model_type == "none":
            self.model = None
            logger.info("ShapeClassifier initialized in 'none' mode (neural classification disabled).")
        else:
            raise ValueError(
                f"Unknown classifier model_type: {model_type}. Expected 'mamba3d', 'pointnet2', or 'none'."
            )

    def _init_mamba3d(self, checkpoint_path: str | Path | None = None) -> bool:
        """Initialize Mamba3D classification model. Returns True on success, False on failure."""
        lggpf_root = Path(__file__).resolve().parents[3]

        default_candidates = [
            lggpf_root / "data" / "models" / "mamba3d_best.pt",
            lggpf_root / "data" / "models" / "best.pt",
            lggpf_root / "data" / "models" / "mamba3d" / "best.pt",
        ]

        ckpt_path: Path | None = None
        if checkpoint_path is not None:
            raw_p = Path(checkpoint_path)
            if raw_p.is_absolute() and raw_p.is_file():
                ckpt_path = raw_p
            else:
                for base in [Path.cwd(), lggpf_root]:
                    cand = (base / raw_p).resolve()
                    if cand.is_file():
                        ckpt_path = cand
                        break
                if ckpt_path is None:
                    ckpt_path = raw_p

        if ckpt_path is None or not ckpt_path.is_file():
            for cand in default_candidates:
                if cand.is_file():
                    ckpt_path = cand.resolve()
                    break

        if ckpt_path is None:
            ckpt_path = default_candidates[0]

        try:
            # Adapt causal_conv1d_cuda signature differences (5 args in v1.1 vs 7 args in v1.4+)
            try:
                import causal_conv1d_cuda
                if hasattr(causal_conv1d_cuda, "causal_conv1d_fwd"):
                    _orig_fwd = causal_conv1d_cuda.causal_conv1d_fwd

                    def _adapted_causal_conv1d_fwd(*args, **kwargs):
                        if len(args) == 5:
                            x, weight, bias, seq_idx, silu = args
                            try:
                                return _orig_fwd(x, weight, bias, seq_idx, None, None, silu)
                            except TypeError:
                                pass
                            try:
                                from causal_conv1d import causal_conv1d_fn
                                return causal_conv1d_fn(
                                    x, weight, bias=bias, seq_idx=seq_idx, activation="silu" if silu else None
                                )
                            except Exception:
                                pass
                            w = weight.shape[-1]
                            x_padded = torch.nn.functional.pad(x, (w - 1, 0))
                            out = torch.nn.functional.conv1d(
                                x_padded, weight.unsqueeze(1), bias=bias, groups=x.shape[1]
                            )
                            return torch.nn.functional.silu(out) if silu else out
                        return _orig_fwd(*args, **kwargs)

                    causal_conv1d_cuda.causal_conv1d_fwd = _adapted_causal_conv1d_fwd
            except Exception as _c1d_err:
                logger.debug("causal_conv1d_cuda adaptation skipped: %s", _c1d_err)

            config = dict(DEFAULT_MODEL_CONFIG)
            config["cls_dim"] = self.num_class

            model = Mamba3D(config).to(self.device)

            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Mamba3D checkpoint missing: {ckpt_path}")

            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict):
                if "model_state" in checkpoint:
                    raw_state = checkpoint["model_state"]
                elif "base_model" in checkpoint:
                    raw_state = checkpoint["base_model"]
                elif "model" in checkpoint:
                    raw_state = checkpoint["model"]
                else:
                    raw_state = checkpoint
            else:
                raw_state = checkpoint

            model_state = {k.replace("module.", ""): v for k, v in raw_state.items()}
            model.load_state_dict(model_state, strict=True)
            model.eval()

            self.model = model
            self.model_type = "mamba3d"
            self.checkpoint_path = str(ckpt_path)
            logger.info("Loaded Mamba3D checkpoint from %s", ckpt_path)
            return True

        except Exception as exc:
            warnings.warn(
                f"Mamba3D initialization failed ({type(exc).__name__}: {exc}). "
                "Safely falling back to PointNet2 classifier.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

    def _init_pointnet2(self, checkpoint_path: str | Path | None = None) -> None:
        """Initialize PointNet2 SSG classification model."""
        if checkpoint_path is None:
            lggpf_root = Path(__file__).resolve().parents[3]
            checkpoint_path = lggpf_root / "data" / "models" / "best_model_5000.pth"

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_absolute():
            candidates = [
                Path.cwd() / ckpt_path,
                Path(__file__).resolve().parents[3] / ckpt_path,
            ]
            for cand in candidates:
                if cand.is_file():
                    ckpt_path = cand.resolve()
                    break

        model = get_model(self.num_class, normal_channel=True).to(self.device)
        if ckpt_path.is_file():
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("model_state", checkpoint))
            model.load_state_dict(state_dict)
            logger.info("Loaded PointNet2 checkpoint from %s", ckpt_path)
        else:
            warnings.warn(
                f"PointNet2 checkpoint not found at {ckpt_path}. Model weights remain uninitialized.",
                RuntimeWarning,
                stacklevel=2,
            )

        model.eval()
        self.model = model
        self.model_type = "pointnet2"
        self.checkpoint_path = str(ckpt_path)

    def _preprocess_mamba3d(self, pcd: o3d.geometry.PointCloud) -> torch.Tensor:
        """Preprocess Open3D point cloud for Mamba3D:
        1. Mean-centering
        2. Normalize to unit sphere (divide by max radius)
        3. Fixed sampling to 2048 points
        4. Output shape (1, 2048, 3) FloatTensor
        """
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0:
            raise ValueError("Input point cloud is empty.")

        # 1. Mean-centering
        centered = pts - np.mean(pts, axis=0, keepdims=True)

        # 2. Divide by maximum radius to scale into unit sphere
        radius = float(np.linalg.norm(centered, axis=1).max())
        if np.isfinite(radius) and radius > 0.0:
            normalized = (centered / radius).astype(np.float32)
        else:
            normalized = centered.astype(np.float32)

        # 3. Fixed to 2048 points
        target_n = 2048
        n = normalized.shape[0]
        if n == target_n:
            sampled = normalized
        elif n > target_n:
            rng = np.random.default_rng(3407)
            indices = rng.choice(n, size=target_n, replace=False)
            sampled = normalized[indices]
        else:
            rng = np.random.default_rng(3407)
            indices = rng.choice(n, size=target_n, replace=True)
            sampled = normalized[indices]

        # 4. Shape (1, 2048, 3)
        return torch.from_numpy(sampled).unsqueeze(0).float().to(self.device)

    def _preprocess_pointnet2(self, pcd: o3d.geometry.PointCloud) -> torch.Tensor:
        """Preprocess Open3D point cloud for PointNet2:
        1. pc_normalize (center & scale)
        2. Normal estimation oriented to camera location
        3. Downsample to at most 5000 points
        4. Output shape (1, 6, N) FloatTensor
        """
        pcd_normalized = o3d.geometry.PointCloud()
        pts_normalized, _, _ = pc_normalize(np.asarray(pcd.points))
        pcd_normalized.points = o3d.utility.Vector3dVector(pts_normalized)
        pcd_normalized.estimate_normals()
        pcd_normalized.orient_normals_towards_camera_location(self.normal_orientation)

        if len(np.asarray(pcd_normalized.points)) > 5000:
            pcd_normalized = pcd_normalized.farthest_point_down_sample(5000)

        points = torch.from_numpy(np.asarray(pcd_normalized.points))
        normals = torch.from_numpy(np.asarray(pcd_normalized.normals))
        pts_with_normals = torch.cat((points, normals), dim=1)
        return pts_with_normals.permute(1, 0).unsqueeze(0).float().to(self.device)

    def predict_ranked(
        self, pcd: o3d.geometry.PointCloud, cls: str | None = None
    ) -> list[tuple[str, float]]:
        """Classify a point cloud and return all categories ranked by softmax confidence.

        Returns:
            List of (category_code, confidence_score) sorted descending by score.
            Example: [("1", 0.89), ("0", 0.08), ("2", 0.03)]
        """
        if self.model_type == "none" or self.model is None:
            default_cat = cls if cls is not None else "0"
            return [(default_cat, 1.0)]

        if len(pcd.points) == 0:
            logger.warning("Empty point cloud passed to predict_ranked(); defaulting to '0'.")
            return [("0", 1.0), ("1", 0.0), ("2", 0.0)]

        if self.model_type == "mamba3d":
            input_tensor = self._preprocess_mamba3d(pcd)
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            ranked_indices = np.argsort(-probs)
            return [(CLASS_MAP.get(int(idx), "0"), float(probs[idx])) for idx in ranked_indices]

        elif self.model_type == "pointnet2":
            input_tensor = self._preprocess_pointnet2(pcd)
            with torch.no_grad():
                pred, _ = self.model(input_tensor)
                probs = torch.softmax(pred, dim=1).squeeze(0).cpu().numpy()
            ranked_indices = np.argsort(-probs)
            return [(CLASS_MAP.get(int(idx), "0"), float(probs[idx])) for idx in ranked_indices]

        default_cat = cls if cls is not None else "0"
        return [(default_cat, 1.0)]

    def predict(self, pcd: o3d.geometry.PointCloud, cls: str | None = None) -> str:
        """Classify a point cloud into a canonical shape category.

        Performs true forward network inference.
        Returns '0' (cuboid), '1' (cone/frustum), or '2' (ellipsoid).

        Args:
            pcd: Open3D point cloud.
            cls: Legacy parameter retained for interface compatibility.

        Returns:
            Class code string: '0', '1', or '2'.
        """
        ranked = self.predict_ranked(pcd, cls)
        return ranked[0][0] if ranked else (cls if cls is not None else "0")


# Backward compatibility alias
PcdClassification = ShapeClassifier
