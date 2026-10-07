"""A screw axis as a motion: its line and pitch, the motion it generates, and the
velocities it gives the points of a body.

These are the numbers screws.viz animates, kept here so they can be checked without
drawing anything. numpy only. MR 3.3.2; notes 3.5, 3.10.
"""

from __future__ import annotations

import numpy as np

from .se3 import exp6, vec_to_se3
from .so3 import near_zero

__all__ = ["helix", "point_velocity", "screw_line", "screw_motion"]


def screw_line(S) -> tuple[np.ndarray | None, np.ndarray, float]:
    """The line and pitch of a screw axis or twist S = (omega, v): (q, s_hat, h).

    q = omega x v / |omega|^2 is the point on the axis nearest the origin, s_hat = omega /
    |omega| its direction and h = omega . v / |omega|^2 the pitch, so a twist V = S thetadot
    gives the same line as S. For a pure translation (omega = 0) there is no line: q is
    None, s_hat = v / |v| and h = inf. MR 3.3.2.2, eq. 3.77 read backwards; notes 3.10.
    """
    S = np.asarray(S, dtype=float)
    omega, v = S[:3], S[3:]
    n2 = float(omega @ omega)
    if near_zero(np.sqrt(n2)):
        return None, v / np.linalg.norm(v), np.inf
    return np.cross(omega, v) / n2, omega / np.sqrt(n2), float(omega @ v) / n2


def screw_motion(S, T0, thetas) -> np.ndarray:
    """e^{[S] theta} T0 for each theta, shape (N, 4, 4): the body that starts at T0 turned
    about S (written in the same frame as T0, the space form) by theta. MR 3.3.3.2, eq. 3.85.
    """
    S, T0 = np.asarray(S, dtype=float), np.asarray(T0, dtype=float)
    return np.array([exp6(vec_to_se3(S * float(t))) @ T0 for t in np.atleast_1d(thetas)])


def point_velocity(V, p) -> np.ndarray:
    """The velocity omega x p + v of the body point at p under the twist V = (omega, v),
    p and V in the same frame; p may be (3,) or (N, 3). MR 3.3.2.1, eq. 3.73; notes 3.10.
    """
    V, p = np.asarray(V, dtype=float), np.asarray(p, dtype=float)
    return np.cross(V[:3], p) + V[3:]


def helix(S, p, thetas) -> np.ndarray:
    """The path, shape (N, 3), of the point p under e^{[S] theta}: a helix about the axis
    that advances 2 pi h per turn, a circle when h = 0, a straight line when omega = 0."""
    ph = np.append(np.asarray(p, dtype=float), 1.0)
    return np.array([(exp6(vec_to_se3(np.asarray(S, float) * float(t))) @ ph)[:3] for t in np.atleast_1d(thetas)])
