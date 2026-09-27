import numpy as np
import pytest

from screws import dynamics as dyn
from screws import robots

M01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]])
M12 = np.array([[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]])
M23 = np.array([[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]])
M34 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.14225], [0, 0, 0, 1]])
G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
LINK_FRAMES = [M01, M12, M23, M34]
LINK_INERTIAS = [G1, G2, G3]
S = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, -0.089, 0, 0], [0, 1, 0, -0.089, 0, 0.425]]).T
THETA = np.array([0.1, 0.1, 0.1])
DTHETA = np.array([0.1, 0.2, 0.3])
DDTHETA = np.array([2, 1.5, 1])
G = np.array([0, 0, -9.8])
F_TIP = np.ones(6)


def test_ad_matches_mr():
    A = dyn.ad([1, 2, 3, 4, 5, 6])
    assert np.allclose(A[3:, :3], [[0, -6, 5], [6, 0, -4], [-5, 4, 0]])
    assert np.allclose(A[:3, 3:], 0) and np.allclose(A[3:, 3:], A[:3, :3])


def test_inverse_dynamics_matches_mr():
    tau = dyn.inverse_dynamics(THETA, DTHETA, DDTHETA, G, F_TIP, LINK_FRAMES, LINK_INERTIAS, S)
    assert np.allclose(tau, [74.69616155, -33.06766016, -3.23057314])


def test_f_tip_none_is_zero_wrench():
    a = dyn.inverse_dynamics(THETA, DTHETA, DDTHETA, G, None, LINK_FRAMES, LINK_INERTIAS, S)
    b = dyn.inverse_dynamics(THETA, DTHETA, DDTHETA, G, np.zeros(6), LINK_FRAMES, LINK_INERTIAS, S)
    assert np.allclose(a, b)


def test_mass_matrix_symmetric_positive_definite_and_matches_mr():
    Mm = dyn.mass_matrix(THETA, LINK_FRAMES, LINK_INERTIAS, S)
    assert np.allclose(Mm, Mm.T) and np.all(np.linalg.eigvalsh(Mm) > 0)
    assert np.isclose(Mm[0, 0], 22.5433380, atol=1e-5)


def test_pieces_match_mr_and_sum_to_inverse_dynamics():
    c = dyn.velocity_quadratic_forces(THETA, DTHETA, LINK_FRAMES, LINK_INERTIAS, S)
    g = dyn.gravity_forces(THETA, G, LINK_FRAMES, LINK_INERTIAS, S)
    e = dyn.end_effector_forces(THETA, F_TIP, LINK_FRAMES, LINK_INERTIAS, S)
    assert np.allclose(c, [0.26453118, -0.05505157, -0.00689132])
    assert np.allclose(g, [28.40331262, -37.64094817, -5.4415892])
    assert np.allclose(e, [1.40954608, 1.85771497, 1.392409])
    Mm = dyn.mass_matrix(THETA, LINK_FRAMES, LINK_INERTIAS, S)
    tau = dyn.inverse_dynamics(THETA, DTHETA, DDTHETA, G, F_TIP, LINK_FRAMES, LINK_INERTIAS, S)
    assert np.allclose(Mm @ DDTHETA + c + g + e, tau)


def test_forward_dynamics_inverts_inverse_dynamics():
    tau = dyn.inverse_dynamics(THETA, DTHETA, DDTHETA, G, F_TIP, LINK_FRAMES, LINK_INERTIAS, S)
    dd = dyn.forward_dynamics(THETA, DTHETA, tau, G, F_TIP, LINK_FRAMES, LINK_INERTIAS, S)
    assert np.allclose(dd, DDTHETA)
    dd2 = dyn.forward_dynamics(THETA, DTHETA, [0.5, 0.6, 0.7], G, F_TIP, LINK_FRAMES, LINK_INERTIAS, S)
    assert np.allclose(dd2, [-0.97392907, 25.58466784, -32.91499212])


def test_euler_step_and_trajectories():
    th, dth = dyn.euler_step([0.1, 0.1, 0.1], [0.1, 0.2, 0.3], [2, 1.5, 1], 0.1)
    assert np.allclose(th, [0.11, 0.12, 0.13]) and np.allclose(dth, [0.3, 0.35, 0.4])
    N = 5
    theta_mat = np.tile(THETA, (N, 1))
    zeros = np.zeros((N, 3))
    tau_mat = dyn.inverse_dynamics_trajectory(
        theta_mat, zeros, zeros, G, np.zeros((N, 6)), LINK_FRAMES, LINK_INERTIAS, S
    )
    assert tau_mat.shape == (N, 3)
    assert np.allclose(tau_mat[0], dyn.gravity_forces(THETA, G, LINK_FRAMES, LINK_INERTIAS, S))
    th_mat, _dth_mat = dyn.forward_dynamics_trajectory(
        THETA, np.zeros(3), tau_mat, G, np.zeros((N, 6)), LINK_FRAMES, LINK_INERTIAS, S, 0.01, 4
    )
    assert th_mat.shape == (N, 3) and np.allclose(th_mat, THETA, atol=1e-6)  # gravity balanced


def test_robot_methods_use_the_robots_gravity():
    ur5 = robots.ur5()
    g0 = ur5.gravity_forces(np.zeros(6))
    assert np.all(np.isfinite(g0)) and g0.shape == (6,)
    moon = ur5.with_gravity([0, 0, -1.62])
    assert np.allclose(moon.gravity_forces(np.zeros(6)), g0 * 1.62 / 9.81)
    Mm = ur5.mass_matrix(np.zeros(6))
    assert Mm.shape == (6, 6) and np.all(np.linalg.eigvalsh(Mm) > 0)
    tau = ur5.inverse_dynamics(np.zeros(6), np.zeros(6), np.zeros(6))
    assert np.allclose(tau, g0)
    assert np.allclose(ur5.forward_dynamics(np.zeros(6), np.zeros(6), tau), 0, atol=1e-9)
    with pytest.raises(Exception, match="no link inertias"):
        robots.ur5(source="textbook").mass_matrix(np.zeros(6))
