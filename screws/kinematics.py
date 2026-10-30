"""Forward kinematics, Jacobians, numerical inverse kinematics. MR chapters 4, 5, 6.2.

Portions derived from the Modern Robotics code library,
Copyright (c) 2018 Huan Weng, Bill Hunt, Jarvis Schultz, Mikhail Todes (MIT).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .se3 import adjoint, exp6, log6, se3_to_vec, transform_inv, vec_to_se3

__all__ = [
    "Ellipsoid",
    "IKResult",
    "fk_body",
    "fk_space",
    "force_ellipsoid",
    "ik_body",
    "ik_space",
    "jacobian_body",
    "jacobian_space",
    "joint_frames",
    "manipulability",
    "manipulability_ellipsoid",
]


def _screws(S) -> np.ndarray:
    if isinstance(S, (list, tuple)) and S and all(np.shape(a) == (6,) for a in S):
        raise TypeError(
            "screw axes must be a 6xn array, one axis per column, not a sequence of 6-vectors: "
            "stack them with np.column_stack(axes), or build a Robot.from_screw_axes(M, axes)"
        )
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != 6:
        raise ValueError(f"screw axes must be a 6xn array, one axis per column; got shape {S.shape}")
    return S


def _theta(theta, S) -> np.ndarray:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.shape[0] != S.shape[1]:
        raise ValueError(f"theta has {theta.shape[0]} entries for {S.shape[1]} joints")
    return theta


def fk_space(M, S, theta) -> np.ndarray:
    """Space form of the product of exponentials: e^{[S1]t1} ... e^{[Sn]tn} M.

    S is 6xn, one screw axis per column, in {s} at the home position.
    MR 4.1.1, eq. 4.14; notes 4.1 (Product of exponentials formula).
    """
    S = _screws(S)
    theta = _theta(theta, S)
    T = np.asarray(M, dtype=float).copy()
    for i in range(len(theta) - 1, -1, -1):
        T = exp6(vec_to_se3(S[:, i] * theta[i])) @ T
    return T


def fk_body(M, B, theta) -> np.ndarray:
    """Body form of the product of exponentials: M e^{[B1]t1} ... e^{[Bn]tn}.

    B is 6xn, one screw axis per column, in {b} at the home position.
    MR 4.1.3, eq. 4.16; notes 4.2 (Screw axes in the end-effector frame).
    """
    B = _screws(B)
    theta = _theta(theta, B)
    T = np.asarray(M, dtype=float).copy()
    for i in range(len(theta)):
        T = T @ exp6(vec_to_se3(B[:, i] * theta[i]))
    return T


def joint_frames(M_joints, S, theta) -> list[np.ndarray]:
    """Every joint frame T_{0i}(theta) = e^{[S1]t1} ... e^{[Si]ti} M_joints[i-1], i = 1..n.

    M_joints holds each joint frame at the zero position. Addition, not in the MR
    library; the space form truncated after each joint. Notes 4.1 and problems.
    """
    S = _screws(S)
    theta = _theta(theta, S)
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
    S = _screws(S)
    theta = _theta(theta, S)
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
    B = _screws(B)
    theta = _theta(theta, B)
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


@dataclass(frozen=True, eq=False)
class Ellipsoid:
    """A manipulability or force ellipsoid of A = J J^T, the image of the unit sphere of
    joint rates (or joint torques) at the end-effector.

    center: where it sits (the origin unless given). axes: m x m, column i the unit
    direction of semi-axis i, an eigenvector of A, signed so its largest entry is positive.
    eigenvalues: lambda_i of A, ascending as numpy.linalg.eigh returns them. radii: the
    semi-axes, sqrt(lambda_i) for manipulability and 1/sqrt(lambda_i) for force; a zero
    eigenvalue gives radius 0 (the velocity lost at a singularity) or inf (the force the
    joints need not resist). kind: "manipulability" or "force". MR 5.4; notes 5.5.
    """

    center: np.ndarray
    axes: np.ndarray
    eigenvalues: np.ndarray
    radii: np.ndarray
    kind: str

    @property
    def singular(self) -> bool:
        """True when some eigenvalue of A is zero: J has lost rank."""
        return bool(np.any(self.eigenvalues == 0))

    @property
    def matrix(self) -> np.ndarray:
        """A = J J^T for a manipulability ellipsoid, B = A^{-1} for a force ellipsoid."""
        if self.kind == "force" and self.singular:
            raise ValueError("B = (J J^T)^{-1} does not exist: the force ellipsoid is unbounded")
        lam = self.eigenvalues if self.kind == "manipulability" else 1.0 / self.eigenvalues
        return self.axes @ np.diag(lam) @ self.axes.T

    @property
    def measures(self) -> tuple[float, float, float]:
        """(mu1, mu2, mu3) of this ellipsoid's matrix, read from its semi-axes: the longest
        over the shortest, its square, and their product (sqrt det). For a manipulability
        ellipsoid these are manipulability()'s. Infinite at a singularity, where mu3 is 0
        for manipulability and infinite for force. MR 5.4; notes 5.5."""
        if self.singular:
            return float("inf"), float("inf"), 0.0 if self.kind == "manipulability" else float("inf")
        r = self.radii
        mu1 = float(r.max() / r.min())
        return mu1, mu1**2, float(np.prod(r))

    def outline(self, n: int = 100) -> np.ndarray:
        """n points around a 2D ellipse, shape (n, 2), the first repeated at the end."""
        if len(self.radii) != 2:
            raise ValueError(f"outline() is for a 2D ellipse; this one has {len(self.radii)} axes (use surface())")
        t = np.linspace(0, 2 * np.pi, n)
        return self.center + (self.axes @ (self.radii[:, None] * np.vstack([np.cos(t), np.sin(t)]))).T

    def surface(self, n: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """X, Y, Z grids (n x n) over a 3D ellipsoid, for plot_surface. Bounded only."""
        if len(self.radii) != 3:
            raise ValueError(f"surface() is for a 3D ellipsoid; this one has {len(self.radii)} axes (use outline())")
        if np.any(np.isinf(self.radii)):
            raise ValueError("the ellipsoid is unbounded along an axis; it has no surface to draw")
        u, w = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n))
        unit = np.stack([np.cos(u) * np.sin(w), np.sin(u) * np.sin(w), np.cos(w)], axis=-1)
        P = (unit * self.radii) @ self.axes.T + self.center
        return P[..., 0], P[..., 1], P[..., 2]


def _ellipsoid(J, center, kind) -> Ellipsoid:
    J = np.atleast_2d(np.asarray(J, dtype=float))
    lam, vecs = np.linalg.eigh(J @ J.T)
    lam = np.clip(lam, 0.0, None)
    lam[lam <= 1e-12 * max(float(lam[-1]), 1e-300)] = 0.0  # rounding residue at a singularity
    for k in range(vecs.shape[1]):
        vecs[:, k] *= np.sign(vecs[np.argmax(np.abs(vecs[:, k])), k])
    with np.errstate(divide="ignore"):
        radii = np.sqrt(lam) if kind == "manipulability" else 1.0 / np.sqrt(lam)
    center = np.zeros(J.shape[0]) if center is None else np.asarray(center, dtype=float)
    return Ellipsoid(center, vecs, lam, radii, kind)


def manipulability_ellipsoid(J, center=None) -> Ellipsoid:
    """The manipulability ellipsoid of a Jacobian block J (m x n, m <= n): the end-effector
    velocities qdot = J thetadot with ||thetadot|| = 1, qdot^T A^{-1} qdot = 1, A = J J^T.
    Pass J_omega or J_v, not a whole 6 x n Jacobian, whose rows mix units; for the linear
    one MR advises J_b's rows. Semi-axes sqrt(lambda_i) along A's eigenvectors.
    Addition, not in the MR library. MR 5.4; notes 5.5.
    """
    return _ellipsoid(J, center, "manipulability")


def force_ellipsoid(J, center=None) -> Ellipsoid:
    """The force ellipsoid of a Jacobian block J: the end-effector forces f with
    tau = J^T f and ||tau|| = 1, f^T A f = 1. The same axes as the manipulability
    ellipsoid, semi-axes 1/sqrt(lambda_i). Addition, not in the MR library. MR 5.4; notes 5.5.
    """
    return _ellipsoid(J, center, "force")


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


def _newton(theta0, S, twist, jacobian, tol_omega, tol_v, max_iter) -> IKResult:
    theta = _theta(theta0, S).copy()
    history = [theta.copy()]
    V = twist(theta)
    i = 0

    def within(V):
        # Written so that a NaN twist is never "within tolerance".
        return bool(np.linalg.norm(V[:3]) <= tol_omega and np.linalg.norm(V[3:]) <= tol_v)

    while not within(V) and i < max_iter and np.all(np.isfinite(V)):
        theta = theta + np.linalg.pinv(jacobian(theta)) @ V
        i += 1
        history.append(theta.copy())
        V = twist(theta)
    return IKResult(
        theta=theta,
        converged=within(V) and bool(np.all(np.isfinite(theta))),
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

    return _newton(theta0, B, twist, lambda th: jacobian_body(B, th), tol_omega, tol_v, max_iter)


def ik_space(M, S, T, theta0, *, tol_omega=1e-3, tol_v=1e-4, max_iter=20) -> IKResult:
    """Numerical inverse kinematics in the space frame: Newton-Raphson on the space twist.

    Iterates theta <- theta + J_s^+(theta) V_s with V_s = [Ad_{T_sb}] V_b.
    MR 6.2.2; notes 6.2.
    """
    M, S, T = (np.asarray(a, dtype=float) for a in (M, _screws(S), T))

    def twist(theta):
        Tsb = fk_space(M, S, theta)
        return adjoint(Tsb) @ se3_to_vec(log6(transform_inv(Tsb) @ T))

    return _newton(theta0, S, twist, lambda th: jacobian_space(S, th), tol_omega, tol_v, max_iter)
