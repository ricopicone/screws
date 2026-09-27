"""Rotations: SO(3), so(3), exponential coordinates. MR chapter 3.2.

Portions derived from the Modern Robotics code library,
Copyright (c) 2018 Huan Weng, Bill Hunt, Jarvis Schultz, Mikhail Todes (MIT).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "axis_angle3",
    "distance_so3",
    "exp3",
    "is_so3",
    "log3",
    "near_zero",
    "normalize",
    "project_so3",
    "rot",
    "rot_inv",
    "rotation_from_rpy",
    "so3_to_vec",
    "vec_to_so3",
]


def near_zero(z) -> bool:
    """True if the scalar z is small enough to be treated as zero (|z| < 1e-6).

    MR library helper; notes 3.6 uses it for the theta = 0 case.
    """
    return abs(float(z)) < 1e-6


def normalize(v) -> np.ndarray:
    """The unit vector in the direction of v.

    MR library helper.
    """
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def rot_inv(R) -> np.ndarray:
    """The inverse of a rotation matrix, its transpose.

    MR 3.2.1; notes 3.3 (Rotation matrices).
    """
    return np.asarray(R, dtype=float).T


def vec_to_so3(omega) -> np.ndarray:
    """[omega], the 3x3 skew-symmetric matrix with [omega] y = omega x y.

    MR 3.2.3.1, eq. 3.30; notes 3.5 (Angular velocities).
    """
    w = np.asarray(omega, dtype=float)
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def so3_to_vec(so3mat) -> np.ndarray:
    """The 3-vector omega from its skew-symmetric matrix [omega].

    MR 3.2.3.1; notes 3.5.
    """
    m = np.asarray(so3mat, dtype=float)
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def axis_angle3(expc3) -> tuple[np.ndarray, float]:
    """(omega_hat, theta) from the exponential coordinates omega_hat theta.

    MR 3.2.3.3; notes 3.6 (Exponential coordinates of rotations).
    """
    e = np.asarray(expc3, dtype=float)
    return normalize(e), float(np.linalg.norm(e))


def exp3(so3mat) -> np.ndarray:
    """e^{[omega] theta}: the rotation matrix from the so(3) matrix [omega] theta.

    Rodrigues' formula. MR 3.2.3.3, eq. 3.51; notes 3.6.
    """
    m = np.asarray(so3mat, dtype=float)
    omega_theta = so3_to_vec(m)
    if near_zero(np.linalg.norm(omega_theta)):
        return np.eye(3)
    theta = axis_angle3(omega_theta)[1]
    W = m / theta
    return np.eye(3) + np.sin(theta) * W + (1.0 - np.cos(theta)) * (W @ W)


def log3(R) -> np.ndarray:
    """The matrix logarithm [omega] theta of a rotation matrix R.

    MR 3.2.3.3, algorithm after eq. 3.53; notes 3.6.
    """
    R = np.asarray(R, dtype=float)
    acos_input = (np.trace(R) - 1.0) / 2.0
    if acos_input >= 1.0:
        return np.zeros((3, 3))
    if acos_input <= -1.0:
        if not near_zero(1.0 + R[2, 2]):
            omega = (1.0 / np.sqrt(2.0 * (1.0 + R[2, 2]))) * np.array(
                [R[0, 2], R[1, 2], 1.0 + R[2, 2]]
            )
        elif not near_zero(1.0 + R[1, 1]):
            omega = (1.0 / np.sqrt(2.0 * (1.0 + R[1, 1]))) * np.array(
                [R[0, 1], 1.0 + R[1, 1], R[2, 1]]
            )
        else:
            omega = (1.0 / np.sqrt(2.0 * (1.0 + R[0, 0]))) * np.array(
                [1.0 + R[0, 0], R[1, 0], R[2, 0]]
            )
        return vec_to_so3(np.pi * omega)
    theta = np.arccos(acos_input)
    return theta / 2.0 / np.sin(theta) * (R - R.T)


def project_so3(mat) -> np.ndarray:
    """The rotation matrix nearest to mat (singular-value decomposition).

    Only meaningful for matrices already close to SO(3). MR library; MR Appendix E.
    """
    U, _, Vh = np.linalg.svd(np.asarray(mat, dtype=float))
    R = U @ Vh
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    return R


def distance_so3(mat) -> float:
    """||mat^T mat - I|| if det mat > 0, else 1e9: how far mat is from SO(3).

    MR library; MR Appendix E.
    """
    m = np.asarray(mat, dtype=float)
    if np.linalg.det(m) > 0:
        return float(np.linalg.norm(m.T @ m - np.eye(3)))
    return 1e9


def is_so3(mat) -> bool:
    """True if mat is within 1e-3 of SO(3) by distance_so3.

    MR library.
    """
    return abs(distance_so3(mat)) < 1e-3


def rot(omega_hat, theta) -> np.ndarray:
    """Rot(omega_hat, theta): the rotation about the unit axis omega_hat by theta.

    Addition, not in the MR library. MR 3.2.3.3; notes 3.6.
    """
    return exp3(vec_to_so3(np.asarray(omega_hat, dtype=float)) * float(theta))


def rotation_from_rpy(roll, pitch, yaw) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: Rot(z, yaw) Rot(y, pitch) Rot(x, roll).

    Addition, not in the MR library. MR Appendix B.1; notes 4.3 (URDF).
    """
    return rot([0, 0, 1], yaw) @ rot([0, 1, 0], pitch) @ rot([1, 0, 0], roll)
