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



def fit_paths_to_paper(paths, paper, margin=0.5):
    """Scale pixel paths to fit inside the paper dimensions."""
    if not paths:
        return []

    px, py, pw, ph = paper

    all_points = np.array(
        [point for path in paths for point in path],
        dtype=float
    )

    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)

    width = max_x - min_x
    height = max_y - min_y

    available_width = pw - 2 * margin
    available_height = ph - 2 * margin

    if available_width <= 0 or available_height <= 0:
        raise ValueError("Paper margin is too large.")

    if width == 0 or height == 0:
        raise ValueError("Drawing must have nonzero width and height.")

    scale = min(
        available_width / width,
        available_height / height
    )

    offset_x = px + (pw - width * scale) / 2
    offset_y = py + (ph - height * scale) / 2

    scaled_paths = []

    for path in paths:
        scaled_path = []

        for x, y in path:
            new_x = offset_x + (x - min_x) * scale

            # Flip the Y axis: images grow downward,
            # Cartesian coordinates grow upward.
            new_y = offset_y + (max_y - y) * scale

            scaled_path.append((new_x, new_y))

        scaled_paths.append(scaled_path)

    return scaled_paths

