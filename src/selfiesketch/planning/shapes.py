"""Simple Cartesian test paths in millimeters."""
from math import cos, sin, pi


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
