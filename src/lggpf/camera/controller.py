"""
Camera controller for Mech-Mind Pro-S 3D camera.

The ``mecheye`` SDK is a vendor-specific package that must be installed
separately.  When the SDK is not available the module can still be imported,
but instantiating :class:`CameraController` will raise an error.
"""

import numpy as np

try:
    from mecheye.shared import show_error
    from mecheye.area_scan_3d_camera import (
        Camera,
        CameraIntrinsics,
        DepthMap,
        Frame2D,
        Frame3D,
        ColorTypeOf2DCamera_Monochrome,
        ColorTypeOf2DCamera_Color,
        UntexturedPointCloud,
    )
    from mecheye.area_scan_3d_camera_utils import find_and_connect

    _HAS_MECHEYE = True
except ImportError:
    _HAS_MECHEYE = False


class CameraController:
    """High-level wrapper around the Mech-Mind camera SDK.

    Provides methods to connect, capture 2D images, and capture depth maps.
    Use :func:`lggpf.utils.depth_to_pointcloud` to convert depth maps to
    point clouds.
    """

    def __init__(self):
        if not _HAS_MECHEYE:
            raise ImportError(
                "The 'mecheye' package is required for CameraController. "
                "Install the Mech-Mind SDK to use this module."
            )
        self.camera = Camera()
        self.intrinsics = None

    def connect(self, ip: str):
        """Connect to the camera by IP address."""
        self.camera.connect(ip)
        self.intrinsics = CameraIntrinsics()
        self.camera.get_camera_intrinsics(self.intrinsics)

    def find_and_connect(self) -> bool:
        """Auto-discover and connect to the first available camera.

        Returns:
            True if connection succeeded, False otherwise.
        """
        if find_and_connect(self.camera):
            self.intrinsics = CameraIntrinsics()
            self.camera.get_camera_intrinsics(self.intrinsics)
            return True
        else:
            print("Connect failed")
            return False

    def disconnect(self):
        """Disconnect from the camera."""
        self.camera.disconnect()

    def capture_2d_image(self) -> np.ndarray:
        """Capture a 2D image (grayscale or color depending on camera type).

        Returns:
            Image as a numpy array.
        """
        frame_2d = Frame2D()
        show_error(self.camera.capture_2d(frame_2d))
        if frame_2d.color_type() == ColorTypeOf2DCamera_Monochrome:
            image2d = frame_2d.get_gray_scale_image()
        elif frame_2d.color_type() == ColorTypeOf2DCamera_Color:
            image2d = frame_2d.get_color_image()
        return image2d.data()

    def capture_depth_map(self) -> np.ndarray:
        """Capture a depth map.

        Returns:
            Depth map as a numpy array (float, values in mm).
        """
        frame3d = Frame3D()
        show_error(self.camera.capture_3d(frame3d))
        depth_map = frame3d.get_depth_map()
        return depth_map.data()

    def convert_depth_map_to_point_cloud(self, depth: "DepthMap") -> np.ndarray:
        """Convert a raw DepthMap object to a point cloud using camera intrinsics.

        Args:
            depth: Mech-Mind DepthMap object.

        Returns:
            Point cloud data as numpy array.
        """
        xyz = UntexturedPointCloud()
        xyz.resize(depth.width(), depth.height())
        for i in range(depth.width() * depth.height()):
            row = int(i / depth.width())
            col = int(i - row * depth.width())
            xyz[i].z = depth[i].z
            xyz[i].x = float(
                (col - self.intrinsics.depth.camera_matrix.cx)
                * depth[i].z
                / self.intrinsics.depth.camera_matrix.fx
            )
            xyz[i].y = float(
                (row - self.intrinsics.depth.camera_matrix.cy)
                * depth[i].z
                / self.intrinsics.depth.camera_matrix.fy
            )
        return xyz.data()
