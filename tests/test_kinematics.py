import numpy as np

from screws import kinematics as kin
from screws import se3

M = np.array([[-1, 0, 0, 0], [0, 1, 0, 6], [0, 0, -1, 2], [0, 0, 0, 1.0]])
S = np.array([[0, 0, 1, 4, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, -1, -6, 0, -0.1]]).T
B = np.array([[0, 0, -1, 2, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0.1]]).T
THETA = np.array([np.pi / 2, 3, np.pi])
T_EXPECT = np.array([[0, 1, 0, -5], [1, 0, 0, 4], [0, 0, -1, 1.68584073], [0, 0, 0, 1]])
T_GOAL = np.array([[0, 1, 0, -5], [1, 0, 0, 4], [0, 0, -1, 1.6858], [0, 0, 0, 1]])


def test_fk_space_and_body_match_mr():
    assert np.allclose(kin.fk_space(M, S, THETA), T_EXPECT)
    assert np.allclose(kin.fk_body(M, B, THETA), T_EXPECT)


def test_jacobians_match_mr():
    Sl = np.array(
        [[0, 0, 1, 0, 0.2, 0.2], [1, 0, 0, 2, 0, 3], [0, 1, 0, 0, 2, 1], [1, 0, 0, 0.2, 0.3, 0.4]]
    ).T
    th = [0.2, 1.1, 0.1, 1.2]
    Js = kin.jacobian_space(Sl, th)
    assert np.allclose(Js[:, 1], [0.98006658, 0.19866933, 0, 1.95218638, 0.43654132, 2.96026613])
    assert np.allclose(Js[:, 0], Sl[:, 0])
    Jb = kin.jacobian_body(Sl, th)
    assert np.allclose(
        Jb[:, 0], [-0.04528405, 0.74359313, -0.66709716, 2.32586047, -1.44321167, -2.06639565]
    )
    assert np.allclose(Jb[:, 3], Sl[:, 3])


def test_ik_body_matches_mr_and_records_history():
    r = kin.ik_body(M, B, T_GOAL, [1.5, 2.5, 3], tol_omega=0.01, tol_v=0.001)
    assert r.converged
    assert np.allclose(r.theta, [1.57073819, 2.999667, 3.14153913], atol=1e-6)
    assert r.history.shape == (r.iterations + 1, 3)
    assert np.allclose(r.history[0], [1.5, 2.5, 3])
    assert np.allclose(r.history[-1], r.theta)
    assert r.error_omega <= 0.01 and r.error_v <= 0.001


def test_ik_space_matches_mr():
    r = kin.ik_space(M, S, T_GOAL, [1.5, 2.5, 3], tol_omega=0.01, tol_v=0.001)
    assert r.converged
    assert np.allclose(r.theta, [1.57073783, 2.99966384, 3.1415342], atol=1e-6)


def test_ik_not_converged_reports_false():
    T = se3.trans([100, 0, 0]) @ M  # unreachable
    r = kin.ik_body(M, B, T, [0, 0, 0], max_iter=3)
    assert not r.converged and r.iterations == 3 and r.history.shape == (4, 3)


def test_joint_frames_first_and_last():
    M_joints = [np.eye(4), se3.trans([0, 0, 1]), se3.trans([0, 1, 1])]
    frames = kin.joint_frames(M_joints, S, THETA)
    assert len(frames) == 3
    assert np.allclose(frames[0], se3.exp6(se3.vec_to_se3(S[:, 0] * THETA[0])) @ M_joints[0])
    # the last joint frame carried to {b} is the forward kinematics
    T_nb = se3.transform_inv(M_joints[2]) @ M
    assert np.allclose(frames[2] @ T_nb, kin.fk_space(M, S, THETA))


def test_manipulability_isotropic_is_one():
    J = np.vstack([np.eye(3), np.zeros((3, 3))])
    m = kin.manipulability(J)
    assert np.isclose(m["angular"][0], 1.0) and np.isclose(m["angular"][2], 1.0)
    J2 = np.vstack([np.zeros((3, 3)), np.diag([2.0, 1.0, 1.0])])
    assert np.isclose(m["angular"][1], 1.0)
    assert np.isclose(kin.manipulability(J2)["linear"][0], 2.0)
    assert np.isclose(kin.manipulability(J2)["linear"][1], 4.0)
    assert np.isclose(kin.manipulability(J2)["linear"][2], 2.0)
