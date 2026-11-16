
"""Kinematics for the SelfieSketch planar drawing arm.

Lengths and Cartesian coordinates are in centimeters.
Servo angles are in degrees.
"""

from dataclasses import dataclass
import numpy as np


class UnreachableTarget(ValueError):
    """The requested XY position cannot be reached by the arm."""


@dataclass
class DrawingArm:
    link1: float = 10.0
    link2: float = 10.0
    elbow_direction: int = -1

    def forward(self, s1, s2):
        """Convert servo angles to pen coordinates (cm)."""
        t1 = np.radians(s1)
        t2 = np.radians(self.elbow_direction * s2)

        x = self.link1 * np.cos(t1) + self.link2 * np.cos(t1 + t2)
        y = self.link1 * np.sin(t1) + self.link2 * np.sin(t1 + t2)

        return x, y

    def inverse(self, x, y):
        """Convert pen coordinates (cm) to servo angles."""
        r2 = x**2 + y**2

        cos_t2 = (
            r2 - self.link1**2 - self.link2**2
        ) / (2 * self.link1 * self.link2)

        if not -1.0 - 1e-10 <= cos_t2 <= 1.0 + 1e-10:
            raise UnreachableTarget("Target is outside the workspace")

        cos_t2 = np.clip(cos_t2, -1.0, 1.0)

        t2 = self.elbow_direction * np.arccos(cos_t2)

        t1 = np.arctan2(y, x) - np.arctan2(
            self.link2 * np.sin(t2),
            self.link1 + self.link2 * np.cos(t2)
        )

        s1 = np.degrees(t1)
        s2 = np.degrees(t2) / self.elbow_direction

        if not (0 <= s1 <= 180 and 0 <= s2 <= 180):
            raise UnreachableTarget("Target violates servo limits")

        return float(s1), float(s2)