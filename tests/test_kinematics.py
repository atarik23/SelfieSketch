from math import isclose, pi
import pytest
from selfiesketch.robotics.kinematics import PlanarArm, UnreachableTarget


@pytest.mark.parametrize("point", [(100, 30), (65, 45), (100, 80), (130, 20)])
def test_ik_fk_roundtrip(point):
    arm = PlanarArm()
    angles = arm.inverse_kinematics(*point)
    actual = arm.forward_kinematics(*angles)
    assert all(isclose(a, b, abs_tol=1e-8) for a, b in zip(actual, point))


def test_outside_workspace():
    arm = PlanarArm()
    with pytest.raises(UnreachableTarget):
        arm.inverse_kinematics(200, 0)


def test_joint_limits():
    arm = PlanarArm(shoulder_limits=(0, pi/4))
    assert not arm.is_reachable(100, 30)


def test_invalid_link():
    with pytest.raises(ValueError):
        PlanarArm(link1=0)
