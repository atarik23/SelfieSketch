"""Simple Cartesian test paths in millimeters."""
from math import cos, sin, pi
import numpy as np

def square(x: float, y: float, side: float):
    if side <= 0:
        raise ValueError("side must be positive")
    return [(x, y), (x + side, y), (x + side, y + side),
            (x, y + side), (x, y)]


def circle(cx: float, cy: float, radius: float, segments: int = 96):
    if radius <= 0 or segments < 3:
        raise ValueError("radius must be positive and segments >= 3")
    return [(cx + radius*cos(2*pi*i/segments),
             cy + radius*sin(2*pi*i/segments)) for i in range(segments + 1)]


def interpolate_line(start, end, step=0.1):
    """Generate evenly spaced points along a straight line."""
    if step <= 0:
        raise ValueError("Step must be positive.")

    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)

    distance = np.linalg.norm(end - start)
    n_steps = max(1, int(np.ceil(distance / step)))

    points = np.linspace(start, end, n_steps + 1)

    return points
