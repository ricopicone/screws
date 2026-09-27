import dataclasses

import numpy as np
import pytest

from screws import se3
from screws.robot import MissingInertias, Robot

M = np.array([[-1, 0, 0, 0], [0, 1, 0, 6], [0, 0, -1, 2], [0, 0, 0, 1.0]])
AXES = [[0, 0, 1, 4, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, -1, -6, 0, -0.1]]


def test_from_screw_axes_stacks_columns_and_types():
    r = Robot.from_screw_axes(M, AXES)
    assert r.S.shape == (6, 3) and r.n == 3
    assert r.joint_types == ("revolute", "prismatic", "revolute")
    assert r.joint_names == ("joint1", "joint2", "joint3")
    assert np.allclose(r.S[:, 1], AXES[1])


def test_from_screw_axes_rejects_2d_array():
    with pytest.raises(TypeError, match="sequence"):
        Robot.from_screw_axes(M, np.array(AXES).T)


def test_B_is_adjoint_of_M_inverse():
    r = Robot.from_screw_axes(M, AXES)
    assert np.allclose(r.B, se3.adjoint(se3.transform_inv(M)) @ r.S)
    assert np.allclose(r.B[:, 0], [0, 0, -1, 2, 0, 0])


def test_fk_and_ik_roundtrip():
    r = Robot.from_screw_axes(M, AXES)
    th = np.array([np.pi / 2, 3, np.pi])
    T = r.fk(th)
    assert np.allclose(T, [[0, 1, 0, -5], [1, 0, 0, 4], [0, 0, -1, 1.68584073], [0, 0, 0, 1]])
    res = r.ik(T, th + 0.05)
    assert res.converged and np.allclose(r.fk(res.theta), T, atol=1e-3)
    res_s = r.ik(T, th + 0.05, frame="space")
    assert res_s.converged
    with pytest.raises(ValueError, match="frame"):
        r.ik(T, th, frame="world")


def test_jacobians_delegate():
    from screws import kinematics as kin

    r = Robot.from_screw_axes(M, AXES)
    th = [0.1, 0.2, 0.3]
    assert np.allclose(r.jacobian_space(th), kin.jacobian_space(r.S, th))
    assert np.allclose(r.jacobian_body(th), kin.jacobian_body(r.B, th))


def test_frames_fallback_and_last_is_fk():
    r = Robot.from_screw_axes(M, AXES)
    fr = r.frames([0.1, 0.2, 0.3])
    assert len(fr) == 4 and np.allclose(fr[-1], r.fk([0.1, 0.2, 0.3]))
    # revolute joint 1's fallback origin is the point on its axis nearest {s}: omega x v
    fr0 = r.frames([0, 0, 0])
    assert np.allclose(fr0[0][:3, 3], np.cross([0, 0, 1], [4, 0, 0]))
    assert np.allclose(fr0[0][:3, 2], [0, 0, 1])


def test_limits_missing_inertias_and_copies():
    r = Robot.from_screw_axes(M, AXES, joint_limits=[[-1, 1]] * 3)
    assert r.within_limits([0, 0, 0]) and not r.within_limits([2, 0, 0])
    assert Robot.from_screw_axes(M, AXES).within_limits([100, 0, 0])  # no limits: always true
    with pytest.raises(MissingInertias):
        r.mass_matrix([0, 0, 0])
    g = r.with_gravity([0, 0, -1.62])
    assert np.allclose(g.gravity, [0, 0, -1.62]) and np.allclose(r.gravity, [0, 0, -9.81])
    r2 = r.with_limits([[-2, 2]] * 3)
    assert r2.within_limits([1.5, 0, 0]) and not r.within_limits([1.5, 0, 0])
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.name = "x"


def test_robot_is_not_comparable_or_hashable_by_value_and_arrays_are_read_only():
    r = Robot.from_screw_axes(M, AXES)
    r2 = Robot.from_screw_axes(M, AXES)
    assert (r == r2) is False and (r == r) is True  # identity only, never an ndarray truth error
    hash(r)  # identity hash works
    with pytest.raises(ValueError, match="read-only"):
        r.S[0, 0] = 5.0
    with pytest.raises(ValueError, match="read-only"):
        r.M[0, 3] = 5.0


def test_random_theta_uses_pi_for_unbounded_joints():
    from screws import testing as st

    r = Robot.from_screw_axes(M, AXES, joint_limits=[[-0.1, 0.1], [-np.inf, np.inf], [0, 1]])
    for _ in range(20):
        th = st.random_theta(r, np.random.default_rng(1))
        assert r.within_limits(th) and abs(th[1]) <= np.pi
