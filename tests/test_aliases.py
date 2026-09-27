import modern_robotics as mr
import numpy as np

import screws as sc
from screws.aliases import ALIASES

B = np.array([[0, 0, -1, 2, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0.1]]).T
S = np.array([[0, 0, 1, 4, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, -1, -6, 0, -0.1]]).T
M = np.array([[-1, 0, 0, 0], [0, 1, 0, 6], [0, 0, -1, 2], [0, 0, 0, 1.0]])
T = np.array([[0, 1, 0, -5], [1, 0, 0, 4], [0, 0, -1, 1.6858], [0, 0, 0, 1]])


def test_identity_aliases():
    assert sc.FKinSpace is sc.fk_space
    assert sc.MatrixExp6 is sc.exp6
    assert sc.Adjoint is sc.adjoint
    assert sc.VecToso3 is sc.vec_to_so3
    assert sc.JacobianBody is sc.jacobian_body


def test_ik_aliases_return_mr_tuple():
    th, ok = sc.IKinBody(B, M, T, [1.5, 2.5, 3], 0.01, 0.001)
    assert ok and np.allclose(th, [1.57073819, 2.999667, 3.14153913], atol=1e-6)
    th, ok = sc.IKinSpace(S, M, T, [1.5, 2.5, 3], 0.01, 0.001)
    assert ok and np.allclose(th, [1.57073783, 2.99966384, 3.1415342], atol=1e-6)


def test_every_mr_kinematics_name_is_aliased():
    wanted = {
        "NearZero", "Normalize", "RotInv", "VecToso3", "so3ToVec", "AxisAng3", "MatrixExp3",
        "MatrixLog3", "RpToTrans", "TransToRp", "TransInv", "VecTose3", "se3ToVec", "Adjoint",
        "ScrewToAxis", "AxisAng6", "MatrixExp6", "MatrixLog6", "ProjectToSO3", "ProjectToSE3",
        "DistanceToSO3", "DistanceToSE3", "TestIfSO3", "TestIfSE3", "FKinBody", "FKinSpace",
        "JacobianBody", "JacobianSpace", "IKinBody", "IKinSpace",
    }
    assert wanted <= set(ALIASES)
    for name in wanted:
        assert hasattr(sc, name), name
        assert hasattr(sc, ALIASES[name]), ALIASES[name]


def test_flat_namespace_exports_everything():
    for name in ("fk_space", "exp6", "rot", "Robot", "IKResult", "MissingInertias", "robots",
                 "testing", "urdf", "so3", "se3", "kinematics"):
        assert hasattr(sc, name), name
    assert "fk_space" in sc.__all__ and "FKinSpace" in sc.__all__
    assert sc.se3.exp6 is sc.exp6


def test_random_agreement_with_modern_robotics():
    rng = np.random.default_rng(7)
    for _ in range(20):
        T = sc.testing.random_transform(rng)
        V = rng.normal(size=6)
        assert np.allclose(sc.log6(T), mr.MatrixLog6(T))
        assert np.allclose(sc.exp6(sc.vec_to_se3(V)), mr.MatrixExp6(mr.VecTose3(V)))
        assert np.allclose(sc.adjoint(T), mr.Adjoint(T))
        assert np.allclose(sc.log3(T[:3, :3]), mr.MatrixLog3(T[:3, :3]))
        th = rng.uniform(-np.pi, np.pi, size=3)
        assert np.allclose(sc.fk_space(M, S, th), mr.FKinSpace(M, S, th))
        assert np.allclose(sc.jacobian_space(S, th), mr.JacobianSpace(S, th))
        assert np.allclose(sc.jacobian_body(B, th), mr.JacobianBody(B, th))
