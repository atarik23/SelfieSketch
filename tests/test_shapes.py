from selfiesketch.planning.shapes import circle, square


def test_square_closed():
    points = square(10, 20, 30)
    assert len(points) == 5
    assert points[0] == points[-1]


def test_circle_closed():
    points = circle(100, 100, 10)
    assert len(points) == 97
    assert abs(points[0][0] - points[-1][0]) < 1e-9
