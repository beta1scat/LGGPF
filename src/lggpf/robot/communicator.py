"""Communication with the RealMan RM-65 robotic arm.

Wraps the vendor-specific ``Robotic_Arm`` SDK for:
  - Modbus RTU control (gripper open/close via registers)
  - Joint-space and Cartesian motion commands
  - Forward / inverse kinematics
  - End-effector pose queries

The ``Robotic_Arm`` package is **not** pip-installable; it must be
obtained from RealMan and placed on ``sys.path`` manually.

Cleaned from the original ``modules/robot/robot_communicator.py``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from spatialmath import SE3, UnitQuaternion

try:
    from Robotic_Arm.rm_robot_interface import (  # type: ignore[import-not-found]
        Algo,
        RoboticArm,
        rm_force_type_e,
        rm_inverse_kinematics_params_t,
        rm_peripheral_read_write_params_t,
        rm_robot_arm_model_e,
        rm_thread_mode_e,
    )

    _HAS_ROBOTIC_ARM = True
except ImportError:
    _HAS_ROBOTIC_ARM = False

logger = logging.getLogger(__name__)

# RM-65 default joint positions and Cartesian poses (degrees / meters)
RM65_DEFAULT_POINTS = [
    [0, 20, 70, 0, 90, 0],
    [0.3, 0, 0.3, 3.14, 0, 0],
    [0.2, 0, 0.3, 3.14, 0, 0],
    [0.3, 0, 0.3, 3.14, 0, 0],
    [0.2, 0.05, 0.3, 3.14, 0, 0],
    [0.2, -0.05, 0.3, 3.14, 0, 0],
]


class RobotArmController:
    """High-level controller for the RealMan RM-65 robotic arm.

    Args:
        ip: IP address of the robot controller.
        port: Communication port.
        level: Connection level (default 3).
        mode: Thread mode — 0: single, 1: dual, 2: triple (default 2).
        gripper_cfg: Optional dict with Modbus gripper configuration.
            Expected keys: ``port``, ``baudrate``, ``timeout``,
            ``force_register``, ``force_value``, ``position_register``,
            ``open_value``, ``close_value``, ``device``.
    """

    def __init__(
        self,
        ip: str,
        port: int = 0,
        level: int = 3,
        mode: int = 2,
        gripper_cfg: dict | None = None,
    ):
        if not _HAS_ROBOTIC_ARM:
            raise ImportError(
                "Robotic_Arm SDK is required. "
                "Obtain it from RealMan and add to sys.path."
            )

        self.thread_mode = rm_thread_mode_e(mode)
        self.robot = RoboticArm(self.thread_mode)
        self.algo = Algo(
            rm_robot_arm_model_e.RM_MODEL_RM_65_E,
            rm_force_type_e.RM_MODEL_RM_B_E,
        )
        self.handle = self.robot.rm_create_robot_arm(ip, port, level)

        # Gripper Modbus configuration
        gcfg = gripper_cfg or {}
        self._gripper_port = gcfg.get("port", 1)
        self._gripper_device = gcfg.get("device", 1)
        self._force_register = gcfg.get("force_register", 0x84)
        self._force_value = gcfg.get("force_value", 500)
        self._position_register = gcfg.get("position_register", 0x80)
        self._open_value = gcfg.get("open_value", 2660)
        self._close_value = gcfg.get("close_value", 3300)

        # Initialize end-effector RS485 as RTU master
        self.set_modbus_mode(
            port=self._gripper_port,
            baudrate=gcfg.get("baudrate", 115200),
            timeout=gcfg.get("timeout", 2),
        )
        self.write_single_register(
            data=self._force_value,
            port=self._gripper_port,
            address=self._force_register,
            device=self._gripper_device,
        )

        if self.handle.id == -1:
            raise ConnectionError("Failed to connect to the robot arm")
        logger.info("Connected to robot arm: %s", self.handle.id)

    def disconnect(self):
        """Disconnect from the robot arm and close Modbus."""
        self.close_modbus_mode(port=self._gripper_port)
        result = self.robot.rm_delete_robot_arm()
        if result == 0:
            logger.info("Disconnected from robot arm")
        else:
            logger.warning("Failed to disconnect from robot arm")

    # =========================================================================
    # Modbus control
    # =========================================================================

    def set_modbus_mode(self, port: int = 0, baudrate: int = 115200, timeout: int = 1):
        """Set the Modbus RTU mode on the specified port.

        Args:
            port: 0 = controller RS485, 1 = end-effector RS485, 2 = controller slave.
            baudrate: One of 9600, 115200, 460800.
            timeout: Timeout in hundred-millisecond units (must be > 0).
        """
        result = self.robot.rm_set_modbus_mode(port, baudrate, timeout)
        if result == 0:
            logger.info("Modbus mode set (port=%d)", port)
        else:
            logger.warning("Failed to set Modbus mode (port=%d)", port)

    def close_modbus_mode(self, port: int = 0):
        """Close the Modbus RTU mode on the specified port."""
        result = self.robot.rm_close_modbus_mode(port)
        if result == 0:
            logger.info("Modbus mode closed (port=%d)", port)
        else:
            logger.warning("Failed to close Modbus mode (port=%d)", port)

    def read_coils(
        self, port: int = 0, address: int = 0, device: int = 2, num: int = 1
    ):
        """Read coils from a Modbus device."""
        params = rm_peripheral_read_write_params_t(port, address, device, num)
        tag = self.robot.rm_read_coils(params)
        if tag[0] == 0:
            logger.info("Read coils data: %s", tag[1])
        else:
            logger.warning("Failed to read coils")

    def write_single_coil(
        self, data: int, port: int = 0, address: int = 0, device: int = 2, num: int = 1
    ):
        """Write a single coil to a Modbus device."""
        params = rm_peripheral_read_write_params_t(port, address, device, num)
        tag = self.robot.rm_write_single_coil(params, data)
        if tag == 0:
            logger.info("Single coil written")
        else:
            logger.warning("Failed to write single coil")

    def write_single_register(
        self, data: int, port: int = 0, address: int = 0, device: int = 2
    ):
        """Write a single register to a Modbus device."""
        params = rm_peripheral_read_write_params_t(port, address, device)
        tag = self.robot.rm_write_single_register(params, data)
        if tag == 0:
            logger.info("Single register written")
        else:
            logger.warning("Failed to write single register")

    def read_holding_registers(self, port: int = 0, address: int = 0, device: int = 2):
        """Read holding registers from a Modbus device."""
        params = rm_peripheral_read_write_params_t(port, address, device)
        tag = self.robot.rm_read_holding_registers(params)
        if tag[0] == 0:
            logger.info("Holding registers data: %s", tag[1])
        else:
            logger.warning("Failed to read holding registers")

    # =========================================================================
    # Motion control
    # =========================================================================

    def get_arm_model(self) -> str | None:
        """Get the robotic arm model string."""
        res, model = self.robot.rm_get_robot_info()
        if res == 0:
            return model["arm_model"]
        logger.warning("Failed to get robot arm model")
        return None

    def movej(
        self, joint: list, v: float = 20, r: float = 0, connect: int = 0, block: int = 1
    ):
        """Move to a joint configuration.

        Args:
            joint: Target joint angles (degrees).
            v: Speed.
            r: Blending radius.
            connect: Trajectory connection flag.
            block: 1 = blocking, 0 = non-blocking.
        """
        result = self.robot.rm_movej(joint, v, r, connect, block)
        if result == 0:
            logger.info("movej succeeded")
        else:
            logger.warning("movej failed (error %s)", result)

    def movel(
        self, pose: list, v: float = 20, r: float = 0, connect: int = 0, block: int = 1
    ):
        """Move linearly to a Cartesian pose [x, y, z, rx, ry, rz].

        Args:
            pose: Target pose (meters, radians).
            v: Speed.
            r: Blending radius.
            connect: Trajectory connection flag.
            block: 1 = blocking, 0 = non-blocking.
        """
        result = self.robot.rm_movel(pose, v, r, connect, block)
        if result == 0:
            logger.info("movel succeeded")
        else:
            logger.warning("movel failed (error %s)", result)

    def movec(
        self,
        pose_via: list,
        pose_to: list,
        v: float = 20,
        r: float = 0,
        loop: int = 0,
        connect: int = 0,
        block: int = 1,
    ):
        """Move along a circular arc.

        Args:
            pose_via: Via-point [x, y, z, rx, ry, rz].
            pose_to: End-point [x, y, z, rx, ry, rz].
            v: Speed.
            r: Blending radius.
            loop: Number of full loops.
            connect: Trajectory connection flag.
            block: 1 = blocking, 0 = non-blocking.
        """
        result = self.robot.rm_movec(pose_via, pose_to, v, r, loop, connect, block)
        if result == 0:
            logger.info("movec succeeded")
        else:
            logger.warning("movec failed (error %s)", result)

    def movej_p(
        self, pose: list, v: float = 20, r: float = 0, connect: int = 0, block: int = 1
    ):
        """Move to a Cartesian pose using joint interpolation.

        Args:
            pose: Target pose [x, y, z, rx, ry, rz].
            v: Speed.
            r: Blending radius.
            connect: Trajectory connection flag.
            block: 1 = blocking, 0 = non-blocking.
        """
        result = self.robot.rm_movej_p(pose, v, r, connect, block)
        if result == 0:
            logger.info("movej_p succeeded")
        else:
            logger.warning("movej_p failed (error %s)", result)

    def movej_canfd(
        self,
        joint: list,
        follow: bool = False,
        expand: float = 0,
        trajectory_mode: int = 1,
        radio: int = 50,
    ):
        """Transparent transmission of joint angles (CAN-FD).

        Args:
            joint: Target joint angles (degrees).
            follow: True for high-follow mode (period <= 10 ms).
            expand: Optional expansion axis value.
            trajectory_mode: 0 = passthrough, 1 = curve fitting, 2 = filter.
            radio: Smoothing coefficient (0-100 for filter, 0-999 for curve fitting).
        """
        result = self.robot.rm_movej_canfd(
            joint, follow, expand, trajectory_mode, radio
        )
        if result == 0:
            logger.debug("movej_canfd succeeded")
        else:
            logger.warning("movej_canfd failed (error %s)", result)

    # =========================================================================
    # Kinematics
    # =========================================================================

    def get_current_end_pose(self) -> SE3:
        """Get the current end-effector pose as SE3 (mm, rotation matrix)."""
        joint_angles = self.robot.rm_get_joint_degree()[1]
        return self.forward_kinematics(joint_angles, flag=0)

    def forward_kinematics(self, joint_angles: list, flag: int = 0) -> SE3:
        """Compute forward kinematics.

        Args:
            joint_angles: Joint angles (degrees).
            flag: 0 = quaternion output, 1 = Euler angle output (internal).

        Returns:
            SE3 pose with translation in mm.
        """
        pose_xyz_quat = self.robot.rm_algo_forward_kinematics(joint_angles, flag)
        pose = SE3(np.array(pose_xyz_quat[:3]) * 1000) * SE3(
            UnitQuaternion(pose_xyz_quat[3], pose_xyz_quat[4:7]).SO3()
        )
        return pose

    def inverse_kinematics(
        self, q_in: list, q_pose: list, flag: int = 1
    ) -> np.ndarray | None:
        """Compute inverse kinematics.

        Args:
            q_in: Initial joint angles (degrees).
            q_pose: Target pose [x, y, z, rx, ry, rz] (meters, radians).
            flag: 1 = Euler angles, 0 = quaternion.

        Returns:
            Joint angles (degrees) as numpy array, or None if IK failed.
        """
        params = rm_inverse_kinematics_params_t(q_in, q_pose, flag)
        joint_angle = self.robot.rm_algo_inverse_kinematics(params)
        if joint_angle[0] == 0:
            return np.array(joint_angle[1])
        elif joint_angle[0] == 1:
            logger.warning("Inverse kinematics failed")
        elif joint_angle[0] == -1:
            logger.warning("Previous joint angles input is empty")
        elif joint_angle[0] == -2:
            logger.warning("Target pose quaternion is invalid")
        return None

    # =========================================================================
    # Gripper
    # =========================================================================

    def close_gripper(self):
        """Close the 2-finger Modbus gripper."""
        self.write_single_register(
            data=self._close_value,
            port=self._gripper_port,
            address=self._position_register,
            device=self._gripper_device,
        )

    def open_gripper(self):
        """Open the 2-finger Modbus gripper."""
        self.write_single_register(
            data=self._open_value,
            port=self._gripper_port,
            address=self._position_register,
            device=self._gripper_device,
        )
