"""Forward kinematics, Jacobians, numerical inverse kinematics. MR chapters 4, 5, 6.2.

Portions derived from the Modern Robotics code library,
Copyright (c) 2018 Huan Weng, Bill Hunt, Jarvis Schultz, Mikhail Todes (MIT).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .se3 import adjoint, exp6, log6, se3_to_vec, transform_inv, vec_to_se3

__all__ = [
    "IKResult",
    "fk_body",
    "fk_space",
    "ik_body",
    "ik_space",
    "jacobian_body",
    "jacobian_space",
    "joint_frames",
    "manipulability",
]


def _screws(S) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != 6:
        raise ValueError(f"screw axes must be a 6xn array, one axis per column; got shape {S.shape}")
    return S


def fk_space(M, S, theta) -> np.ndarray:
    """Space form of the product of exponentials: e^{[S1]t1} ... e^{[Sn]tn} M.

    S is 6xn, one screw axis per column, in {s} at the home position.
    MR 4.1.1, eq. 4.14; notes 4.1 (Product of exponentials formula).
    """
    S, theta = _screws(S), np.asarray(theta, dtype=float)
    T = np.asarray(M, dtype=float).copy()
    for i in range(len(theta) - 1, -1, -1):
        T = exp6(vec_to_se3(S[:, i] * theta[i])) @ T
    return T


def fk_body(M, B, theta) -> np.ndarray:
    """Body form of the product of exponentials: M e^{[B1]t1} ... e^{[Bn]tn}.

    B is 6xn, one screw axis per column, in {b} at the home position.
    MR 4.1.3, eq. 4.16; notes 4.2 (Screw axes in the end-effector frame).
    """
    B, theta = _screws(B), np.asarray(theta, dtype=float)
    T = np.asarray(M, dtype=float).copy()
    for i in range(len(theta)):
        T = T @ exp6(vec_to_se3(B[:, i] * theta[i]))
    return T


def joint_frames(M_joints, S, theta) -> list[np.ndarray]:
    """Every joint frame T_{0i}(theta) = e^{[S1]t1} ... e^{[Si]ti} M_joints[i-1], i = 1..n.

    M_joints holds each joint frame at the zero position. Addition, not in the MR
    library; the space form truncated after each joint. Notes 4.1 and problems.
    """
    S, theta = _screws(S), np.asarray(theta, dtype=float)
    T = np.eye(4)
    out = []
    for i, Mi in enumerate(M_joints):
        T = T @ exp6(vec_to_se3(S[:, i] * theta[i]))
        out.append(T @ np.asarray(Mi, dtype=float))
    return out


def jacobian_space(S, theta) -> np.ndarray:
    """The space Jacobian J_s(theta), 6xn, column i = [Ad_{e^{[S1]t1}...e^{[S_{i-1}]t_{i-1}}}] S_i.

    MR 5.1.1, eq. 5.11; notes 5.1.
    """
    S, theta = _screws(S), np.asarray(theta, dtype=float)
    Js = S.copy()
    T = np.eye(4)
    for i in range(1, len(theta)):
        T = T @ exp6(vec_to_se3(S[:, i - 1] * theta[i - 1]))
        Js[:, i] = adjoint(T) @ S[:, i]
    return Js


def jacobian_body(B, theta) -> np.ndarray:
    """The body Jacobian J_b(theta), 6xn, column i = [Ad_{e^{-[Bn]tn}...e^{-[B_{i+1}]t_{i+1}}}] B_i.

    MR 5.1.2, eq. 5.18; notes 5.1.
    """
    B, theta = _screws(B), np.asarray(theta, dtype=float)
    Jb = B.copy()
    T = np.eye(4)
    for i in range(len(theta) - 2, -1, -1):
        T = T @ exp6(vec_to_se3(B[:, i + 1] * -theta[i + 1]))
        Jb[:, i] = adjoint(T) @ B[:, i]
    return Jb


def _ellipsoid_measures(A: np.ndarray) -> tuple[float, float, float]:
    lam = np.sort(np.linalg.eigvalsh(A))
    lam_min, lam_max = float(lam[0]), float(lam[-1])
    if lam_min <= 0.0:
        return float("inf"), float("inf"), 0.0
    return float(np.sqrt(lam_max / lam_min)), lam_max / lam_min, float(np.sqrt(np.prod(lam)))


def manipulability(J) -> dict[str, tuple[float, float, float]]:
    """MR's three manipulability measures (mu1, mu2, mu3) of the angular and linear blocks.

    For A = J_w J_w^T and A = J_v J_v^T: mu1 = sqrt(lambda_max / lambda_min), the ratio of
    the longest to the shortest ellipsoid semi-axis; mu2 = lambda_max / lambda_min, its
    square; mu3 = sqrt(det A), proportional to the ellipsoid's volume. Infinite at a
    singularity. Addition, not in the MR library. MR 5.4; notes 5.4.
    """
    J = np.asarray(J, dtype=float)
    return {
        "angular": _ellipsoid_measures(J[:3] @ J[:3].T),
        "linear": _ellipsoid_measures(J[3:] @ J[3:].T),
    }


@dataclass(frozen=True)
class IKResult:
    """The outcome of a Newton-Raphson inverse-kinematics run.

    theta: the final iterate. converged: both error norms within tolerance.
    iterations: Newton steps taken. history: every iterate, (iterations + 1) x n,
    history[0] the initial guess and history[-1] == theta. error_omega, error_v:
    the final rotational and translational error norms.
    """

    theta: np.ndarray
    converged: bool
    iterations: int
    history: np.ndarray
    error_omega: float
    error_v: float


def _newton(theta0, twist, jacobian, tol_omega, tol_v, max_iter) -> IKResult:
    theta = np.asarray(theta0, dtype=float).copy()
    history = [theta.copy()]
    V = twist(theta)
    i = 0

    def err(V):
        return np.linalg.norm(V[:3]) > tol_omega or np.linalg.norm(V[3:]) > tol_v

    while err(V) and i < max_iter:
        theta = theta + np.linalg.pinv(jacobian(theta)) @ V
        i += 1
        history.append(theta.copy())
        V = twist(theta)
    return IKResult(
        theta=theta,
        converged=not err(V),
        iterations=i,
        history=np.array(history),
        error_omega=float(np.linalg.norm(V[:3])),
        error_v=float(np.linalg.norm(V[3:])),
    )


def ik_body(M, B, T, theta0, *, tol_omega=1e-3, tol_v=1e-4, max_iter=20) -> IKResult:
    """Numerical inverse kinematics in the body frame: Newton-Raphson on the body twist.

    Iterates theta <- theta + J_b^+(theta) V_b with [V_b] = log(T_sb^{-1}(theta) T).
    Stops when ||omega_b|| <= tol_omega and ||v_b|| <= tol_v, or after max_iter steps.
    MR 6.2.2, algorithm on p. 230; notes 6.2.
    """
    M, B, T = (np.asarray(a, dtype=float) for a in (M, _screws(B), T))

    def twist(theta):
        return se3_to_vec(log6(transform_inv(fk_body(M, B, theta)) @ T))

    return _newton(theta0, twist, lambda th: jacobian_body(B, th), tol_omega, tol_v, max_iter)


def ik_space(M, S, T, theta0, *, tol_omega=1e-3, tol_v=1e-4, max_iter=20) -> IKResult:
    """Numerical inverse kinematics in the space frame: Newton-Raphson on the space twist.

    Iterates theta <- theta + J_s^+(theta) V_s with V_s = [Ad_{T_sb}] V_b.
    MR 6.2.2; notes 6.2.
    """
    M, S, T = (np.asarray(a, dtype=float) for a in (M, _screws(S), T))

    def twist(theta):
        Tsb = fk_space(M, S, theta)
        return adjoint(Tsb) @ se3_to_vec(log6(transform_inv(Tsb) @ T))

    return _newton(theta0, twist, lambda th: jacobian_space(S, th), tol_omega, tol_v, max_iter)
