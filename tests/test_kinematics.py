import numpy as np
import pytest

from robotics.kinematics import DrawingArm, UnreachableTarget


@pytest.fixture
def arm():
    return DrawingArm(link1=10, link2=10)


@pytest.mark.parametrize(
    "s1, s2, expected_x, expected_y",
    [
        (90, 90, 10, 10),
        (90, 0, 0, 20),
        (180, 90, -10, 10),
    ],
)
def test_forward_kinematics(arm, s1, s2, expected_x, expected_y):
    x, y = arm.forward(s1, s2)

    assert np.allclose([x, y], [expected_x, expected_y])


@pytest.mark.parametrize(
    "s1, s2",
    [
        (90, 90),
        (90, 0),
        (180, 90),
        (120, 80),
    ],
)
def test_inverse_kinematics(arm, s1, s2):
    x, y = arm.forward(s1, s2)
    recovered_s1, recovered_s2 = arm.inverse(x, y)

    assert np.allclose(
        [recovered_s1, recovered_s2],
        [s1, s2],
        atol=1e-6,
    )


def test_unreachable_target(arm):
    with pytest.raises(UnreachableTarget):
        arm.inverse(30, 30)


def test_servo_limits(arm):
    with pytest.raises(UnreachableTarget):
        arm.inverse(0, -10)
