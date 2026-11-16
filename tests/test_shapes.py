import numpy as np
import pytest

from planning.shapes import circle, square
from planning.shapes import interpolate_line

def test_square_closed():
    points = square(10, 20, 30)
    assert len(points) == 5
    assert points[0] == points[-1]


def test_circle_closed():
    points = circle(100, 100, 10)
    assert len(points) == 97
    assert abs(points[0][0] - points[-1][0]) < 1e-9


def test_interpolate_line():
    points = interpolate_line((3, 10), (7, 10), step=1)

    expected = [
        [3, 10],
        [4, 10],
        [5, 10],
        [6, 10],
        [7, 10],
    ]

    assert np.allclose(points, expected)


def test_interpolate_diagonal():
    points = interpolate_line((0, 0), (3, 4), step=1)

    assert np.allclose(points[0], [0, 0])
    assert np.allclose(points[-1], [3, 4])

    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert np.all(distances <= 1.0 + 1e-9)


def test_interpolate_same_point():
    points = interpolate_line((2, 3), (2, 3))

    assert np.allclose(points[0], [2, 3])
    assert np.allclose(points[-1], [2, 3])


def test_interpolate_invalid_step():
    with pytest.raises(ValueError):
        interpolate_line((0, 0), (1, 1), step=0)
