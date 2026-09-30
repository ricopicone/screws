"""Rigid-body motions: SE(3), se(3), twists, screw axes, the adjoint. MR chapter 3.3-3.4.

Portions derived from the Modern Robotics code library,
Copyright (c) 2018 Huan Weng, Bill Hunt, Jarvis Schultz, Mikhail Todes (MIT).
"""

from __future__ import annotations

import numpy as np

from .so3 import axis_angle3, exp3, log3, near_zero, project_so3, so3_to_vec, vec_to_so3

__all__ = [
    "adjoint",
    "axis_angle6",
    "distance_se3",
    "exp6",
    "is_se3",
    "log6",
    "prismatic_axis",
    "project_se3",
    "revolute_axis",
    "rp_to_transform",
    "screw_axis",
    "se3_to_vec",
    "trans",
    "transform_inv",
    "transform_to_rp",
    "vec_to_se3",
]


def rp_to_transform(R, p) -> np.ndarray:
    """The homogeneous transformation T = (R, p).

    MR 3.3.1, eq. 3.62; notes 3.4 (Rigid-body motions).
    """
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(p, dtype=float)
    return T


def transform_to_rp(T) -> tuple[np.ndarray, np.ndarray]:
    """(R, p) from the homogeneous transformation T.

    MR 3.3.1; notes 3.4.
    """
    T = np.asarray(T, dtype=float)
    return T[:3, :3].copy(), T[:3, 3].copy()


def transform_inv(T) -> np.ndarray:
    """T^{-1} = (R^T, -R^T p), without a general matrix inverse.

    MR 3.3.1, eq. 3.64; notes 3.4.
    """
    R, p = transform_to_rp(T)
    return rp_to_transform(R.T, -R.T @ p)


def vec_to_se3(V) -> np.ndarray:
    """[V], the 4x4 se(3) matrix of the twist V = (omega, v).

    MR 3.3.2.1, eq. 3.72; notes 3.5.
    """
    V = np.asarray(V, dtype=float)
    m = np.zeros((4, 4))
    m[:3, :3] = vec_to_so3(V[:3])
    m[:3, 3] = V[3:]
    return m


def se3_to_vec(se3mat) -> np.ndarray:
    """The twist V = (omega, v) from its se(3) matrix [V].

    MR 3.3.2.1; notes 3.5.
    """
    m = np.asarray(se3mat, dtype=float)
    return np.concatenate([so3_to_vec(m[:3, :3]), m[:3, 3]])


def adjoint(T) -> np.ndarray:
    """[Ad_T], the 6x6 matrix that changes the frame of a twist: V' = [Ad_T] V.

    MR 3.3.2.1, definition 3.20; notes 3.5.
    """
    R, p = transform_to_rp(T)
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[3:, :3] = vec_to_so3(p) @ R
    A[3:, 3:] = R
    return A


def screw_axis(q, s_hat, h) -> np.ndarray:
    """The screw axis S = (s_hat, -s_hat x q + h s_hat) through q, direction s_hat, pitch h.

    Finite pitch only; use prismatic_axis for h = infinity. MR 3.3.2.2, eq. 3.77;
    notes 3.5.
    """
    q, s_hat = np.asarray(q, dtype=float), np.asarray(s_hat, dtype=float)
    return np.concatenate([s_hat, np.cross(q, s_hat) + float(h) * s_hat])


def revolute_axis(q, s_hat) -> np.ndarray:
    """Zero-pitch screw axis (omega, v) = (s_hat, -s_hat x q) through the point q.

    s_hat is the unit direction of the axis, the textbook's s-hat for a general screw
    (MR 3.3.2.2); for a revolute joint it is the omega-hat of the notes' Definition 3.12,
    so revolute_axis(q, omega_hat) reads naturally too. Addition, not in the MR library;
    the joint-frame derivation of notes 4.1.
    """
    q, s_hat = np.asarray(q, dtype=float), np.asarray(s_hat, dtype=float)
    return np.concatenate([s_hat, -np.cross(s_hat, q)])


def prismatic_axis(s_hat) -> np.ndarray:
    """Infinite-pitch screw axis (0, s_hat).

    Addition, not in the MR library. MR 3.3.2.2.
    """
    return np.concatenate([np.zeros(3), np.asarray(s_hat, dtype=float)])


def axis_angle6(expc6) -> tuple[np.ndarray, float]:
    """(S, theta) from the exponential coordinates S theta of a rigid-body motion.

    theta is ||omega||, or ||v|| when omega = 0. MR 3.3.3; notes 3.7.
    """
    e = np.asarray(expc6, dtype=float)
    theta = float(np.linalg.norm(e[:3]))
    if near_zero(theta):
        theta = float(np.linalg.norm(e[3:]))
    return e / theta, theta


def exp6(se3mat) -> np.ndarray:
    """e^{[S] theta}: the homogeneous transformation from the se(3) matrix [S] theta.

    MR 3.3.3, proposition 3.25; notes 3.7 (Exponential coordinates of rigid-body motions).
    """
    m = np.asarray(se3mat, dtype=float)
    omega_theta = so3_to_vec(m[:3, :3])
    if near_zero(np.linalg.norm(omega_theta)):
        return rp_to_transform(np.eye(3), m[:3, 3])
    theta = axis_angle3(omega_theta)[1]
    W = m[:3, :3] / theta
    G = np.eye(3) * theta + (1.0 - np.cos(theta)) * W + (theta - np.sin(theta)) * (W @ W)
    return rp_to_transform(exp3(m[:3, :3]), G @ m[:3, 3] / theta)


def log6(T) -> np.ndarray:
    """The matrix logarithm [S] theta of a homogeneous transformation T.

    MR 3.3.3, algorithm after eq. 3.92; notes 3.7.
    """
    R, p = transform_to_rp(T)
    W = log3(R)
    out = np.zeros((4, 4))
    if np.array_equal(W, np.zeros((3, 3))):
        out[:3, 3] = p
        return out
    theta = float(np.linalg.norm(so3_to_vec(W)))
    Ginv = (
        np.eye(3)
        - W / 2.0
        + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2.0) * (W @ W) / theta
    )
    out[:3, :3] = W
    out[:3, 3] = Ginv @ p
    return out


def project_se3(mat) -> np.ndarray:
    """The transformation nearest to mat: its rotation block projected, its position kept.

    MR library; MR Appendix E.
    """
    m = np.asarray(mat, dtype=float)
    return rp_to_transform(project_so3(m[:3, :3]), m[:3, 3])


def distance_se3(mat) -> float:
    """How far mat is from SE(3): the MR library's Frobenius measure, 1e9 if det R <= 0.

    MR library; MR Appendix E.
    """
    m = np.asarray(mat, dtype=float)
    R = m[:3, :3]
    if np.linalg.det(R) > 0:
        probe = np.zeros((4, 4))
        probe[:3, :3] = R.T @ R
        probe[3, :] = m[3, :]
        return float(np.linalg.norm(probe - np.eye(4)))
    return 1e9


def is_se3(mat) -> bool:
    """True if mat is within 1e-3 of SE(3) by distance_se3.

    MR library.
    """
    return abs(distance_se3(mat)) < 1e-3


def trans(p) -> np.ndarray:
    """Trans(p): the pure translation by p.

    Addition, not in the MR library. MR 3.3.1; notes 3.4.
    """
    return rp_to_transform(np.eye(3), p)
