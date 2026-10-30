"""Manipulability and force ellipsoids, against the worked examples of notes 5.5 (MR 5.4)."""

import numpy as np
import pytest

import screws
from screws import robots

PI = np.pi


def two_r_J(t1, t2, L1=1.0, L2=1.0):
    """The notes' J(theta) for the 2R arm, eq. two-r-jacobian."""
    s1, c1, s12, c12 = np.sin(t1), np.cos(t1), np.sin(t1 + t2), np.cos(t1 + t2)
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12], [L1 * c1 + L2 * c12, L2 * c12]])


def test_two_r_ellipses_match_the_notes_first_posture():
    e = screws.manipulability_ellipsoid(two_r_J(0, PI / 4))
    assert np.allclose(e.eigenvalues, [0.1163, 4.2979], atol=1e-4)
    assert np.allclose(e.radii, [0.341, 2.073], atol=1e-3)
    long_axis = e.axes[:, 1]
    angle = np.rad2deg(np.arctan2(long_axis[1], long_axis[0])) % 180
    assert np.isclose(angle, 117, atol=0.5)  # "the long one at 117 deg from x-hat_1"
    mu1, mu2, mu3 = e.measures
    assert np.isclose(mu1, 6.08, atol=0.01) and np.isclose(mu2, 36.9, atol=0.1) and np.isclose(mu3, 0.707, atol=1e-3)
    f = screws.force_ellipsoid(two_r_J(0, PI / 4))
    assert np.allclose(f.radii, [2.932, 0.482], atol=1e-3)
    assert np.allclose(f.axes, e.axes)  # the same principal axes


def test_two_r_second_posture_same_mu3_different_mu1():
    e = screws.manipulability_ellipsoid(two_r_J(0, 3 * PI / 4))
    assert np.allclose(e.radii, [0.659, 1.073], atol=1e-3)
    assert np.allclose(screws.force_ellipsoid(two_r_J(0, 3 * PI / 4)).radii, [1.518, 0.932], atol=1e-3)
    mu1, mu2, mu3 = e.measures
    assert np.isclose(mu1, 1.63, atol=0.01) and np.isclose(mu2, 2.65, atol=0.01) and np.isclose(mu3, 0.707, atol=1e-3)


def test_measures_agree_with_manipulability():
    J = robots.ur5().jacobian_body([0.3, -1.1, 1.4, 0.2, 0.9, -0.5])
    m = screws.manipulability(J)
    assert np.allclose(screws.manipulability_ellipsoid(J[:3]).measures, m["angular"])
    assert np.allclose(screws.manipulability_ellipsoid(J[3:]).measures, m["linear"])


def test_force_is_the_reciprocal_and_the_volumes_multiply_to_one():
    J = robots.ur5().jacobian_body([0.3, -1.1, 1.4, 0.2, 0.9, -0.5])[3:]
    e, f = screws.manipulability_ellipsoid(J), screws.force_ellipsoid(J)
    assert np.allclose(e.radii * f.radii, 1)
    assert np.allclose(e.axes, f.axes)
    assert np.isclose(e.measures[2] * f.measures[2], 1)
    assert np.allclose(e.matrix, J @ J.T) and np.allclose(f.matrix, np.linalg.inv(J @ J.T))
    assert e.kind == "manipulability" and f.kind == "force"


def test_axes_are_unit_eigenvectors_signed_with_largest_entry_positive():
    J = robots.ur5().jacobian_body([0.3, -1.1, 1.4, 0.2, 0.9, -0.5])[3:]
    e = screws.manipulability_ellipsoid(J)
    A = J @ J.T
    for k in range(3):
        u = e.axes[:, k]
        assert np.isclose(np.linalg.norm(u), 1)
        assert np.allclose(A @ u, e.eigenvalues[k] * u)
        assert u[np.argmax(np.abs(u))] > 0


def test_singular_two_r_flattens_and_the_force_ellipse_is_unbounded():
    e = screws.manipulability_ellipsoid(two_r_J(0.4, 0.0))
    f = screws.force_ellipsoid(two_r_J(0.4, 0.0))
    assert e.singular and f.singular
    assert e.radii[0] == 0 and np.isinf(f.radii[0])
    assert e.measures[0] == np.inf and e.measures[2] == 0
    assert f.measures[0] == np.inf and f.measures[2] == np.inf
    with pytest.raises(ValueError, match="unbounded"):
        _ = f.matrix
    # the lost direction is along the arm, (cos 0.4, sin 0.4)
    assert np.isclose(abs(e.axes[:, 0] @ [np.cos(0.4), np.sin(0.4)]), 1)


def test_outline_lies_on_the_ellipse_and_surface_on_the_ellipsoid():
    J = two_r_J(0.2, 1.0)
    e = screws.manipulability_ellipsoid(J, center=[1.0, 2.0])
    pts = e.outline(50)
    Ainv = np.linalg.inv(J @ J.T)
    d = pts - [1.0, 2.0]
    assert pts.shape == (50, 2) and np.allclose(np.einsum("ij,jk,ik->i", d, Ainv, d), 1)
    J3 = robots.ur5().jacobian_body([0.3, -1.1, 1.4, 0.2, 0.9, -0.5])[3:]
    e3 = screws.manipulability_ellipsoid(J3, center=[0.1, 0, 0])
    X, Y, Z = e3.surface(12)
    P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3) - [0.1, 0, 0]
    assert np.allclose(np.einsum("ij,jk,ik->i", P, np.linalg.inv(J3 @ J3.T), P), 1)
    with pytest.raises(ValueError):
        e3.outline()
    with pytest.raises(ValueError):
        e.surface()


def test_rrp_linear_ellipsoid_is_diagonal_in_b():
    """Notes 5.5, the RRP arm at (90 deg, 45 deg, 0.2 m): semi-axes 0.7, 0.495, 1 along x_b, y_b, z_b."""
    Jb = robots.rrp().jacobian_body([PI / 2, PI / 4, 0.2])
    e = screws.manipulability_ellipsoid(Jb[3:])
    assert np.allclose(e.radii, [0.495, 0.7, 1.0], atol=1e-3)
    assert np.allclose(np.abs(e.axes), np.eye(3)[:, [1, 0, 2]], atol=1e-5)  # the URDF's rpy is 1.5708
    assert screws.manipulability_ellipsoid(Jb[:3]).singular  # flat: two axes of turning only


def test_ur5_at_half_pi_matches_the_notes_program():
    ur5 = robots.ur5("textbook")
    theta = np.full(6, PI / 2)
    lin = ur5.manipulability_ellipsoid(theta)
    ang = ur5.manipulability_ellipsoid(theta, kind="angular")
    assert np.allclose(lin.eigenvalues, [0.0578, 0.1109, 0.2547], atol=1e-4)  # from J_b, any axes
    assert np.allclose(ang.eigenvalues, [1, 2, 3], atol=1e-9)
    assert np.allclose(lin.center, ur5.fk(theta)[:3, 3]) and np.allclose(ang.center, lin.center)


def test_robot_ellipsoid_is_the_body_one_turned_into_s():
    ur5 = robots.ur5()
    theta = np.array([0.3, -1.1, 1.4, 0.2, 0.9, -0.5])
    T = ur5.fk(theta)
    R = T[:3, :3]
    Jb = ur5.jacobian_body(theta)
    lin = ur5.manipulability_ellipsoid(theta)
    assert np.allclose(lin.matrix, R @ Jb[3:] @ Jb[3:].T @ R.T)
    ang = ur5.manipulability_ellipsoid(theta, kind="angular")
    Js = ur5.jacobian_space(theta)
    assert np.allclose(ang.matrix, Js[:3] @ Js[:3].T)
    f = ur5.force_ellipsoid(theta)
    assert np.allclose(f.radii * lin.radii, 1) and np.allclose(f.center, T[:3, 3])
    with pytest.raises(ValueError, match="kind"):
        ur5.manipulability_ellipsoid(theta, kind="both")


def test_planar_2r_is_the_notes_arm():
    arm = robots.planar_2r()
    theta = [0.0, PI / 4]
    T = arm.fk(theta)
    assert np.allclose(T[:3, 3], [1 + np.cos(PI / 4), np.sin(PI / 4), 0])
    Jb = arm.jacobian_body(theta)
    tip = T[:3, :3] @ Jb[3:]
    assert np.allclose(tip[:2], two_r_J(*theta)) and np.allclose(tip[2], 0)
    assert np.allclose(arm.frames(theta)[1][:3, 3], [1, 0, 0])  # the elbow
    longer = robots.planar_2r(L1=2.0, L2=0.5)
    assert np.allclose(longer.fk([0, 0])[:3, 3], [2.5, 0, 0])
    e = arm.manipulability_ellipsoid(theta)  # 3D like any robot's, flat in z
    assert np.allclose(sorted(e.radii), [0, 0.341, 2.073], atol=1e-3) and e.measures[0] == np.inf
    flat = arm.manipulability_ellipsoid(theta, planar=True)  # the notes' ellipse, in the plane
    assert np.allclose(flat.radii, [0.341, 2.073], atol=1e-3) and np.isclose(flat.measures[0], 6.08, atol=0.01)
    assert np.allclose(flat.center, T[:2, 3]) and np.allclose(flat.matrix, two_r_J(*theta) @ two_r_J(*theta).T)
    assert np.allclose(arm.force_ellipsoid(theta, planar=True).radii, [2.932, 0.482], atol=1e-3)
    with pytest.raises(ValueError, match="segment"):
        arm.manipulability_ellipsoid(theta, kind="angular", planar=True)
    with pytest.raises(ValueError, match="not planar"):
        robots.ur5().manipulability_ellipsoid(np.zeros(6), planar=True)
