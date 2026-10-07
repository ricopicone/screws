import numpy as np

from screws import motion, se3

DOOR_S = np.array([0, 0, 1, 0, -2, 0.0])
DOOR_T0 = se3.rp_to_transform(np.eye(3), [2.9, 0, 1])


def test_screw_line_door():
    q, s, h = motion.screw_line(DOOR_S)
    assert np.allclose(q, [2, 0, 0]) and np.allclose(s, [0, 0, 1]) and h == 0


def test_screw_line_normalises_a_twist():
    q, s, h = motion.screw_line(0.5 * DOOR_S)
    assert h == 0
    assert np.allclose(q, [2, 0, 0]) and np.allclose(s, [0, 0, 1])


def test_screw_line_translation():
    q, s, h = motion.screw_line([0, 0, 0, 0, 2, 0])
    assert q is None and np.allclose(s, [0, 1, 0]) and h == np.inf


def test_screw_motion_matches_the_notes_closed_form():
    th = np.linspace(0, 2, 7)
    Ts = motion.screw_motion(DOOR_S, DOOR_T0, th)
    for t, T in zip(th, Ts):
        c, s = np.cos(t), np.sin(t)
        expected = np.array([[c, -s, 0, 2 + 0.9 * c], [s, c, 0, 0.9 * s], [0, 0, 1, 1], [0, 0, 0, 1]])
        assert np.allclose(T, expected)


def test_helix_advances_two_pi_h_per_turn():
    S = se3.screw_axis(q=[0.1, 0, 0], s_hat=[0, 0, -1], h=0.02)
    path = motion.helix(S, [0.2, 0, 0.3], [0, 2 * np.pi])
    assert np.allclose(path[1] - path[0], [0, 0, -2 * np.pi * 0.02])


def test_point_velocity_is_the_derivative_of_the_motion():
    S = se3.screw_axis(q=[0.3, -0.2, 0], s_hat=[0, 1, 0], h=0.1)
    p0 = np.array([[0.5, 0.1, 0.4], [-0.2, 0.3, 1.0]])
    t, dt = 0.7, 1e-6
    a = np.array([motion.helix(S, p, [t, t + dt]) for p in p0])
    fd = (a[:, 1] - a[:, 0]) / dt
    assert np.allclose(motion.point_velocity(S, a[:, 0]), fd, atol=1e-5)


def test_space_and_body_twists_related_by_the_adjoint_at_every_theta():
    S_b = se3.adjoint(se3.transform_inv(DOOR_T0)) @ DOOR_S
    assert np.allclose(S_b, [0, 0, 1, 0, 0.9, 0])
    for T in motion.screw_motion(DOOR_S, DOOR_T0, np.linspace(0, 1.7, 5)):
        assert np.allclose(se3.adjoint(T) @ S_b, DOOR_S)
