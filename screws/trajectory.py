"""Trajectory generation: time scalings and point-to-point paths. MR chapter 9.

Portions derived from the Modern Robotics code library,
Copyright (c) 2018 Huan Weng, Bill Hunt, Jarvis Schultz, Mikhail Todes (MIT).
"""

from __future__ import annotations

import numpy as np

from .se3 import exp6, log6, rp_to_transform, transform_inv, transform_to_rp
from .so3 import exp3, log3

__all__ = [
    "cartesian_trajectory",
    "cubic_time_scaling",
    "joint_trajectory",
    "quintic_time_scaling",
    "screw_trajectory",
]

_SCALINGS = ("cubic", "quintic")


def cubic_time_scaling(T_final, t) -> float:
    """s(t) = 3 (t/T)^2 - 2 (t/T)^3: rest-to-rest with zero end velocities. MR 9.2.1, eq. 9.9; notes 9.2."""
    u = float(t) / float(T_final)
    return 3.0 * u**2 - 2.0 * u**3


def quintic_time_scaling(T_final, t) -> float:
    """s(t) = 10 (t/T)^3 - 15 (t/T)^4 + 6 (t/T)^5: zero end velocities and accelerations. MR 9.2.1, eq. 9.12."""
    u = float(t) / float(T_final)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _s_values(T_final, N, scaling):
    if scaling not in _SCALINGS:
        raise ValueError(f"scaling must be one of {_SCALINGS}; got {scaling!r}")
    N = int(N)
    if N < 1:
        raise ValueError("N must be at least 1")
    if N == 1:
        return [0.0]
    f = cubic_time_scaling if scaling == "cubic" else quintic_time_scaling
    gap = float(T_final) / (N - 1)
    return [f(T_final, gap * k) for k in range(N)]


def joint_trajectory(theta_start, theta_end, T_final, N, scaling: str = "quintic") -> np.ndarray:
    """A straight line in joint space, N rows from theta_start to theta_end over T_final seconds.

    Row k is at time k T_final / (N - 1). MR 9.2, eq. 9.7; notes 9.2.
    """
    a, b = np.asarray(theta_start, dtype=float), np.asarray(theta_end, dtype=float)
    return np.array([s * b + (1.0 - s) * a for s in _s_values(T_final, N, scaling)])


def screw_trajectory(X_start, X_end, T_final, N, scaling: str = "quintic") -> list[np.ndarray]:
    """A screw motion from X_start to X_end: X(s) = X_start exp(log(X_start^{-1} X_end) s).

    N transformations; the body follows one constant screw axis. MR 9.2.3, eq. 9.14; notes 9.2.
    """
    X_start, X_end = np.asarray(X_start, dtype=float), np.asarray(X_end, dtype=float)
    L = log6(transform_inv(X_start) @ X_end)
    return [X_start @ exp6(L * s) for s in _s_values(T_final, N, scaling)]


def cartesian_trajectory(X_start, X_end, T_final, N, scaling: str = "quintic") -> list[np.ndarray]:
    """A straight-line motion of the origin with a rotation interpolated separately.

    p(s) = p_start + s (p_end - p_start), R(s) = R_start exp(log(R_start^T R_end) s).
    MR 9.2.3, eq. 9.15; notes 9.2.
    """
    R0, p0 = transform_to_rp(X_start)
    R1, p1 = transform_to_rp(X_end)
    L = log3(R0.T @ R1)
    return [rp_to_transform(R0 @ exp3(L * s), p0 + s * (p1 - p0)) for s in _s_values(T_final, N, scaling)]
