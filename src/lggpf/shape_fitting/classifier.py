"""
Point cloud shape classifier using PointNet2.

Classifies point clouds into basic geometric shape categories
(cuboid, truncated cone, ellipsoid) using a pre-trained PointNet2 SSG model.
"""

import numpy as np
import open3d as o3d
import torch

from ..utils.pointcloud import pc_normalize
from .pointnet2 import get_model


class PcdClassification:
    """PointNet2-based point cloud shape classifier.

    Args:
        path: Path to the trained model checkpoint (.pth file).
        num_class: Number of shape classes. Default 3 (cuboid, cone, ellipsoid).
        normal_orientation: Camera location for orienting surface normals.
            Default [0, 0, 800].
    """

    def __init__(self, path: str, num_class: int = 3, normal_orientation: list = None):
        self.num_class = num_class
        self.normal_orientation = normal_orientation or [0, 0, 800]
        use_normals = True
        self.classifier = get_model(num_class, normal_channel=use_normals)
        self.classifier = self.classifier.cuda()
        checkpoint = torch.load(path, weights_only=False)
        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.eval()

    def predict(self, pcd: o3d.geometry.PointCloud, cls: str) -> str:
        """Classify a point cloud into a geometric shape category.

        Note: The current implementation returns the ``cls`` parameter as-is.
        The PointNet2 prediction is computed and printed for reference, but
        the pipeline uses language-guided brute-force shape selection instead.

        Args:
            pcd: Open3D point cloud (with or without normals).
            cls: Shape class hint from language-guided selection.

        Returns:
            The ``cls`` parameter unchanged.
        """
        # Normalize and compute normals
        pcd_normalized = o3d.geometry.PointCloud()
        pts_normalized, _, _ = pc_normalize(np.asarray(pcd.points))
        pcd_normalized.points = o3d.utility.Vector3dVector(pts_normalized)
        pcd_normalized.estimate_normals()
        pcd_normalized.orient_normals_towards_camera_location(self.normal_orientation)

        # Downsample if needed
        if len(np.asarray(pcd_normalized.points)) > 5000:
            pcd_normalized = pcd_normalized.farthest_point_down_sample(5000)

        # Prepare input tensor: (1, 6, N) — xyz + normals
        points = torch.from_numpy(np.asarray(pcd_normalized.points))
        normals = torch.from_numpy(np.asarray(pcd_normalized.normals))
        pts_with_normals = torch.cat((points, normals), dim=1)
        input_tensor = torch.unsqueeze(pts_with_normals.permute(1, 0), 0).cuda()

        # Inference
        pred, _ = self.classifier(input_tensor.float())
        pred_choice = pred.data.max(1)[1]
        print(f"PointNet2 prediction: {pred}")
        print(f"pred_choice: {pred_choice}, language_hint: {cls}")

        return cls
