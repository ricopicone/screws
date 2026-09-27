import numpy as np

from screws import so3


def test_vec_so3_roundtrip():
    w = np.array([1.0, 2.0, 3.0])
    assert np.allclose(so3.so3_to_vec(so3.vec_to_so3(w)), w)


def test_exp3_matches_mr_example():
    W = so3.vec_to_so3([1, 2, 3])
    R = so3.exp3(W)
    assert np.allclose(
        R,
        [
            [-0.69492056, 0.71352099, 0.08929286],
            [-0.19200697, -0.30378504, 0.93319235],
            [0.69297817, 0.6313497, 0.34810748],
        ],
    )


def test_log3_matches_mr_example():
    R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    L = so3.log3(R)
    assert np.allclose(L, [[0, -1.20919958, 1.20919958], [1.20919958, 0, -1.20919958],
                           [-1.20919958, 1.20919958, 0]])


def test_log3_identity_is_zero():
    assert np.array_equal(so3.log3(np.eye(3)), np.zeros((3, 3)))


def test_log3_half_turn():
    R = so3.rot([0, 0, 1], np.pi)
    w = so3.so3_to_vec(so3.log3(R))
    assert np.isclose(abs(w[2]), np.pi) and np.allclose(w[:2], 0)


def test_axis_angle3_matches_mr():
    w_hat, theta = so3.axis_angle3([1, 2, 3])
    assert np.allclose(w_hat, [0.26726124, 0.53452248, 0.80178373])
    assert np.isclose(theta, 3.7416573867739413)


def test_rot_and_exp3_agree():
    assert np.allclose(so3.rot([0, 1, 0], 0.3), so3.exp3(so3.vec_to_so3([0, 0.3, 0])))


def test_rot_inv_is_transpose():
    R = so3.rot([1, 0, 0], 0.7)
    assert np.allclose(so3.rot_inv(R) @ R, np.eye(3))


def test_rotation_from_rpy_order():
    # URDF: fixed-axis roll, pitch, yaw = Rot(z, yaw) Rot(y, pitch) Rot(x, roll)
    R = so3.rotation_from_rpy(0.1, 0.2, 0.3)
    expect = so3.rot([0, 0, 1], 0.3) @ so3.rot([0, 1, 0], 0.2) @ so3.rot([1, 0, 0], 0.1)
    assert np.allclose(R, expect)


def test_is_so3_and_project():
    bad = np.array([[0.675, 0.150, 0.720], [0.370, 0.771, -0.511], [-0.630, 0.619, 0.472]])
    assert not so3.is_so3(bad)
    good = so3.project_so3(bad)
    assert so3.is_so3(good)
    assert np.allclose(good[0], [0.67901136, 0.14894516, 0.71885945])
    assert np.isclose(so3.distance_so3([[1, 0, 0], [0, 0.1, -0.95], [0, 1, 0.1]]), 0.08835, atol=1e-5)


def test_returns_float64_and_near_zero():
    assert so3.exp3(np.zeros((3, 3))).dtype == np.float64
    assert so3.near_zero(-1e-7) and not so3.near_zero(1e-3)
    assert np.allclose(so3.normalize([3, 0, 4]), [0.6, 0, 0.8])


def test_log3_half_turn_about_general_axes_roundtrips():
    # Traces round to -1 ± 1e-15 here; the log must still return a pi rotation.
    for axis in ([1, 1, 1], [1, 1, 0], [0.3, -0.5, 0.8]):
        R = so3.rot(so3.normalize(axis), np.pi)
        L = so3.log3(R)
        assert np.all(np.isfinite(L))
        assert np.isclose(np.linalg.norm(so3.so3_to_vec(L)), np.pi, atol=1e-6)
        assert np.allclose(so3.exp3(L), R, atol=1e-7)
