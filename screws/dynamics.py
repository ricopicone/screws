"""Dynamics of open chains: Newton-Euler inverse dynamics and what is built from it. MR chapter 8.

Portions derived from the Modern Robotics code library,
Copyright (c) 2018 Huan Weng, Bill Hunt, Jarvis Schultz, Mikhail Todes (MIT).
"""

from __future__ import annotations

import numpy as np

from .se3 import adjoint, exp6, transform_inv, vec_to_se3
from .so3 import vec_to_so3

__all__ = [
    "ad",
    "end_effector_forces",
    "euler_step",
    "forward_dynamics",
    "forward_dynamics_trajectory",
    "gravity_forces",
    "inverse_dynamics",
    "inverse_dynamics_trajectory",
    "mass_matrix",
    "velocity_quadratic_forces",
]


def ad(V) -> np.ndarray:
    """[ad_V], the 6x6 matrix of the Lie bracket: [V1, V2] = [ad_V1] V2.

    MR 8.2.2, eq. 8.39; notes 8.2.
    """
    V = np.asarray(V, dtype=float)
    W = vec_to_so3(V[:3])
    A = np.zeros((6, 6))
    A[:3, :3] = W
    A[3:, :3] = vec_to_so3(V[3:])
    A[3:, 3:] = W
    return A


def _model(link_frames, link_inertias, S, n):
    Ms = [np.asarray(m, dtype=float) for m in link_frames]
    Gs = [np.asarray(g, dtype=float) for g in link_inertias]
    S = np.asarray(S, dtype=float)
    if len(Ms) != n + 1 or len(Gs) != n or S.shape != (6, n):
        raise ValueError(
            f"a {n}-joint chain needs {n + 1} link frames, {n} inertias and a 6x{n} S; got "
            f"{len(Ms)}, {len(Gs)} and {S.shape}"
        )
    return Ms, Gs, S


def inverse_dynamics(theta, dtheta, ddtheta, g, F_tip, link_frames, link_inertias, S) -> np.ndarray:
    """Joint forces/torques tau = M(theta) ddtheta + c(theta, dtheta) + g(theta) + J^T F_tip.

    Newton-Euler forward and backward iterations. link_frames are MR's M_{i-1,i} (n + 1 of
    them, the last carrying frame {n} to {b}); link_inertias are the spatial inertias G_i in
    frame {i}; S the space-frame screw axes at home; F_tip the wrench on the end-effector in
    {n+1} (None means zero). MR 8.3.2, algorithm on p. 296; notes 8.3.
    """
    theta = np.asarray(theta, dtype=float)
    dtheta = np.asarray(dtheta, dtype=float)
    ddtheta = np.asarray(ddtheta, dtype=float)
    n = theta.shape[0]
    Ms, Gs, S = _model(link_frames, link_inertias, S, n)
    F = np.zeros(6) if F_tip is None else np.asarray(F_tip, dtype=float).copy()
    Mi = np.eye(4)
    A = np.zeros((6, n))
    AdT = [None] * (n + 1)
    V = np.zeros((6, n + 1))
    Vd = np.zeros((6, n + 1))
    Vd[3:, 0] = -np.asarray(g, dtype=float)
    AdT[n] = adjoint(transform_inv(Ms[n]))
    tau = np.zeros(n)
    for i in range(n):
        Mi = Mi @ Ms[i]
        A[:, i] = adjoint(transform_inv(Mi)) @ S[:, i]
        AdT[i] = adjoint(exp6(vec_to_se3(A[:, i] * -theta[i])) @ transform_inv(Ms[i]))
        V[:, i + 1] = AdT[i] @ V[:, i] + A[:, i] * dtheta[i]
        Vd[:, i + 1] = AdT[i] @ Vd[:, i] + A[:, i] * ddtheta[i] + ad(V[:, i + 1]) @ A[:, i] * dtheta[i]
    for i in range(n - 1, -1, -1):
        F = AdT[i + 1].T @ F + Gs[i] @ Vd[:, i + 1] - ad(V[:, i + 1]).T @ (Gs[i] @ V[:, i + 1])
        tau[i] = F @ A[:, i]
    return tau


def mass_matrix(theta, link_frames, link_inertias, S) -> np.ndarray:
    """M(theta), one inverse-dynamics call per unit joint acceleration. MR 8.3.3; notes 8.3."""
    theta = np.asarray(theta, dtype=float)
    n = theta.shape[0]
    M = np.zeros((n, n))
    for i in range(n):
        dd = np.zeros(n)
        dd[i] = 1.0
        M[:, i] = inverse_dynamics(theta, np.zeros(n), dd, np.zeros(3), None, link_frames, link_inertias, S)
    return M


def velocity_quadratic_forces(theta, dtheta, link_frames, link_inertias, S) -> np.ndarray:
    """c(theta, dtheta), the Coriolis and centripetal joint forces. MR 8.3.3; notes 8.3."""
    n = len(theta)
    return inverse_dynamics(theta, dtheta, np.zeros(n), np.zeros(3), None, link_frames, link_inertias, S)


def gravity_forces(theta, g, link_frames, link_inertias, S) -> np.ndarray:
    """g(theta), the joint forces that hold the chain against gravity g. MR 8.3.3; notes 8.3."""
    n = len(theta)
    return inverse_dynamics(theta, np.zeros(n), np.zeros(n), g, None, link_frames, link_inertias, S)


def end_effector_forces(theta, F_tip, link_frames, link_inertias, S) -> np.ndarray:
    """J^T(theta) F_tip, the joint forces that create the end-effector wrench alone. MR 8.3.3."""
    n = len(theta)
    return inverse_dynamics(theta, np.zeros(n), np.zeros(n), np.zeros(3), F_tip, link_frames, link_inertias, S)


def forward_dynamics(theta, dtheta, tau, g, F_tip, link_frames, link_inertias, S) -> np.ndarray:
    """ddtheta = M(theta)^{-1} (tau - c - g - J^T F_tip). MR 8.5; notes 8.5."""
    tau = np.asarray(tau, dtype=float)
    rhs = (
        tau
        - velocity_quadratic_forces(theta, dtheta, link_frames, link_inertias, S)
        - gravity_forces(theta, g, link_frames, link_inertias, S)
        - end_effector_forces(theta, F_tip, link_frames, link_inertias, S)
    )
    return np.linalg.solve(mass_matrix(theta, link_frames, link_inertias, S), rhs)


def euler_step(theta, dtheta, ddtheta, dt) -> tuple[np.ndarray, np.ndarray]:
    """One first-order Euler step: (theta + dt dtheta, dtheta + dt ddtheta). MR 8.5."""
    theta, dtheta, ddtheta = (np.asarray(a, dtype=float) for a in (theta, dtheta, ddtheta))
    return theta + dt * dtheta, dtheta + dt * ddtheta


def inverse_dynamics_trajectory(
    theta_mat, dtheta_mat, ddtheta_mat, g, F_tip_mat, link_frames, link_inertias, S
) -> np.ndarray:
    """tau along a trajectory: N x n rows of joint values in, N x n rows of joint forces out.

    F_tip_mat is N x 6 (or None for no tip wrench). MR 8.5; notes 8.5.
    """
    theta_mat, dtheta_mat, ddtheta_mat = (np.asarray(a, dtype=float) for a in (theta_mat, dtheta_mat, ddtheta_mat))
    N = theta_mat.shape[0]
    F_mat = np.zeros((N, 6)) if F_tip_mat is None else np.asarray(F_tip_mat, dtype=float)
    tau_mat = np.zeros_like(theta_mat)
    for k in range(N):
        tau_mat[k] = inverse_dynamics(
            theta_mat[k], dtheta_mat[k], ddtheta_mat[k], g, F_mat[k], link_frames, link_inertias, S
        )
    return tau_mat


def forward_dynamics_trajectory(
    theta, dtheta, tau_mat, g, F_tip_mat, link_frames, link_inertias, S, dt, int_res
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate an open-loop force history: N x n tau_mat in, (N x n theta, N x n dtheta) out.

    dt is the time between rows; int_res Euler substeps are taken between rows. MR 8.5.
    """
    theta = np.asarray(theta, dtype=float).copy()
    dtheta = np.asarray(dtheta, dtype=float).copy()
    tau_mat = np.asarray(tau_mat, dtype=float)
    N = tau_mat.shape[0]
    F_mat = np.zeros((N, 6)) if F_tip_mat is None else np.asarray(F_tip_mat, dtype=float)
    theta_mat = np.zeros_like(tau_mat)
    dtheta_mat = np.zeros_like(tau_mat)
    theta_mat[0], dtheta_mat[0] = theta, dtheta
    for k in range(N - 1):
        for _ in range(int(int_res)):
            dd = forward_dynamics(theta, dtheta, tau_mat[k], g, F_mat[k], link_frames, link_inertias, S)
            theta, dtheta = euler_step(theta, dtheta, dd, dt / int_res)
        theta_mat[k + 1], dtheta_mat[k + 1] = theta, dtheta
    return theta_mat, dtheta_mat
