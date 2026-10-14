"""Analytical kinematics for a two-revolute-joint planar arm.

Angles are radians. Lengths and Cartesian positions use millimeters.
The model assumes both links rotate in the same XY plane.
"""
from dataclasses import dataclass
from math import atan2, cos, sin, acos, hypot, pi
from typing import Tuple


class UnreachableTarget(ValueError):
    """A target cannot be reached within geometry or joint limits."""


@dataclass(frozen=True)
class PlanarArm:
    link1: float = 80.0
    link2: float = 80.0
    shoulder_limits: Tuple[float, float] = (-pi, pi)
    elbow_limits: Tuple[float, float] = (-pi, pi)

    def __post_init__(self) -> None:
        if self.link1 <= 0 or self.link2 <= 0:
            raise ValueError("Link lengths must be positive")
        if any(lo > hi for lo, hi in (self.shoulder_limits, self.elbow_limits)):
            raise ValueError("Invalid joint limits")

    def forward_kinematics(self, theta1: float, theta2: float) -> tuple[float, float]:
        return (
            self.link1 * cos(theta1) + self.link2 * cos(theta1 + theta2),
            self.link1 * sin(theta1) + self.link2 * sin(theta1 + theta2),
        )

    def joint_positions(self, theta1: float, theta2: float):
        elbow = (self.link1 * cos(theta1), self.link1 * sin(theta1))
        return (0.0, 0.0), elbow, self.forward_kinematics(theta1, theta2)

    def inverse_kinematics(self, x: float, y: float, elbow_up: bool = False) -> tuple[float, float]:
        radius = hypot(x, y)
        if radius > self.link1 + self.link2 + 1e-8 or radius < abs(self.link1 - self.link2) - 1e-8:
            raise UnreachableTarget(f"Point ({x}, {y}) is outside the annular workspace")
        c2 = (x*x + y*y - self.link1**2 - self.link2**2) / (2*self.link1*self.link2)
        c2 = max(-1.0, min(1.0, c2))
        t2 = acos(c2) * (-1 if elbow_up else 1)
        t1 = atan2(y, x) - atan2(self.link2*sin(t2), self.link1 + self.link2*cos(t2))
        if not (self.shoulder_limits[0] <= t1 <= self.shoulder_limits[1]
                and self.elbow_limits[0] <= t2 <= self.elbow_limits[1]):
            raise UnreachableTarget("Target violates configured joint limits")
        return t1, t2

    def is_reachable(self, x: float, y: float, elbow_up: bool = False) -> bool:
        try:
            self.inverse_kinematics(x, y, elbow_up=elbow_up)
            return True
        except UnreachableTarget:
            return False
