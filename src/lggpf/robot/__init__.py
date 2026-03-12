"""Robot sub-package.

Provides Pinocchio-based robot model with collision detection,
trajectory planning, and RealMan arm communication.
"""

from .planner import Robot, CollisionDetector, CollisionResult, JointSpacePlanner

__all__ = [
    "Robot",
    "CollisionDetector",
    "CollisionResult",
    "JointSpacePlanner",
]

# RobotArmController is imported separately since it requires the
# vendor-specific Robotic_Arm SDK:
#   from lggpf.robot.communicator import RobotArmController
