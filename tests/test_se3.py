import numpy as np

from screws import se3, so3

T_MR = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 3], [0, 0, 0, 1.0]])


def test_rp_transform_roundtrip():
    R = so3.rot([0, 0, 1], 0.4)
    T = se3.rp_to_transform(R, [1, 2, 5])
    R2, p2 = se3.transform_to_rp(T)
    assert np.allclose(R2, R) and np.allclose(p2, [1, 2, 5]) and T.shape == (4, 4)


def test_transform_inv_matches_mr():
    assert np.allclose(
        se3.transform_inv(T_MR), [[1, 0, 0, 0], [0, 0, 1, -3], [0, -1, 0, 0], [0, 0, 0, 1]]
    )


def test_vec_se3_roundtrip():
    V = np.arange(1.0, 7.0)
    m = se3.vec_to_se3(V)
    assert np.allclose(m, [[0, -3, 2, 4], [3, 0, -1, 5], [-2, 1, 0, 6], [0, 0, 0, 0]])
    assert np.allclose(se3.se3_to_vec(m), V)


def test_adjoint_matches_mr():
    A = se3.adjoint(T_MR)
    assert np.allclose(A[3:, :3], [[0, 0, 3], [3, 0, 0], [0, 0, 0]])
    assert np.allclose(A[:3, 3:], 0) and np.allclose(A[3:, 3:], T_MR[:3, :3])


def test_screw_axis_matches_mr():
    assert np.allclose(se3.screw_axis([3, 0, 0], [0, 0, 1], 2), [0, 0, 1, 0, -3, 2])


def test_revolute_axis_is_minus_omega_cross_q():
    S = se3.revolute_axis(q=[0.425, 0, 0.089], s_hat=[0, 1, 0])
    assert np.allclose(S, [0, 1, 0, -0.089, 0, 0.425])


def test_prismatic_axis():
    assert np.allclose(se3.prismatic_axis([1, 0, 0]), [0, 0, 0, 1, 0, 0])


def test_axis_angle6_matches_mr():
    S, theta = se3.axis_angle6([1, 0, 0, 1, 2, 3])
    assert np.allclose(S, [1, 0, 0, 1, 2, 3]) and theta == 1.0
    S, theta = se3.axis_angle6([0, 0, 0, 0, 2, 0])
    assert np.allclose(S, [0, 0, 0, 0, 1, 0]) and theta == 2.0


def test_exp6_matches_mr_example():
    se3mat = np.array(
        [[0, 0, 0, 0], [0, 0, -1.57079632, 2.35619449], [0, 1.57079632, 0, 2.35619449], [0, 0, 0, 0]]
    )
    assert np.allclose(se3.exp6(se3mat), T_MR, atol=1e-6)


def test_exp6_pure_translation():
    S = se3.prismatic_axis([0, 0, 1])
    T = se3.exp6(se3.vec_to_se3(S * 0.7))
    assert np.allclose(T, se3.trans([0, 0, 0.7]))


def test_log6_matches_mr_and_roundtrips():
    L = se3.log6(T_MR)
    assert np.allclose(L[1], [0, 0, -1.57079633, 2.35619449])
    assert np.allclose(se3.exp6(L), T_MR)
    assert np.array_equal(se3.log6(np.eye(4)), np.zeros((4, 4)))


def test_log6_pure_translation():
    L = se3.log6(se3.trans([1, 2, 3]))
    assert np.allclose(L[:3, 3], [1, 2, 3]) and np.allclose(L[:3, :3], 0)


def test_project_distance_is_se3():
    bad = np.array(
        [[0.675, 0.150, 0.720, 1.2], [0.370, 0.771, -0.511, 5.4],
         [-0.630, 0.619, 0.472, 3.6], [0.003, 0.002, 0.010, 0.9]]
    )
    good = se3.project_se3(bad)
    assert se3.is_se3(good) and np.allclose(good[:3, 3], [1.2, 5.4, 3.6]) and good[3, 3] == 1
    assert se3.is_se3(T_MR) and not se3.is_se3(np.eye(4) * 1.1)
    d = se3.distance_se3([[1, 0, 0, 1.2], [0, 0.1, -0.95, 1.5], [0, 1, 0.1, -0.9], [0, 0, 0.1, 0.98]])
    assert np.isclose(d, 0.134931, atol=1e-5)


def test_log6_half_turn_is_finite_and_roundtrips():
    for axis in ([1, 1, 1], [1, 1, 0]):
        T = se3.rp_to_transform(so3.rot(so3.normalize(axis), np.pi), [0.2, -0.1, 0.4])
        L = se3.log6(T)
        assert np.all(np.isfinite(L))
        assert np.allclose(se3.exp6(L), T, atol=1e-7)
