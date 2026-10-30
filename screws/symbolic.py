"""Symbolic twins of the chapter functions, in SymPy: the product of exponentials, the
Jacobians, the adjoint and Newton-Euler dynamics as formulas in the joint variables.

``import screws.symbolic as sym`` (needs SymPy: ``uv add "screws[symbolic]"``). Each
function has the name and the arguments of its numerical twin (``sym.fk_space(M, S,
theta)`` beside ``screws.fk_space(M, S, theta)``, MR's names too: ``sym.FKinSpace``) and
returns SymPy matrices; a vector is a column. Two differences:

- theta may be left out (or None): the joint variables are then the real symbols
  theta1 ... thetan, and dtheta, ddtheta in the dynamics thetadot1 ..., thetaddot1 ...
  (see joint_symbols).
- Numbers are read exactly. A Python float becomes the decimal it prints as (0.425 is
  17/40), unless it is, to the last bit, a simple fraction or a simple fraction of pi,
  sqrt(2) or sqrt(3) (1/3, np.pi / 2, np.cos(np.pi / 4) is sqrt(2)/2); round-off below
  1e-12 is zero. So a Robot's float arrays give clean formulas. SymPy objects pass
  through untouched (a sympy.Float stays a float).

Results are simplified (sympy.simplify, entry by entry) unless simplify=False, or by the
function given instead (simplify=sympy.trigsimp is sometimes quicker). The kinematics of a
six-joint arm take seconds, its gravity forces half a minute, its mass matrix far longer;
and a robot read from floats that carry round-off simplifies slowly or not at all, which
is what Robot.symbolic(decimals=...) is for.

Not here, being numerical by nature: the logarithms, inverse kinematics, manipulability,
projections and distances, time stepping and trajectories.
MR chapters 3, 4, 5 and 8; notes 3.5-3.7, 4.1-4.2, 5.1, 8.3.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np

try:
    import sympy as sp
except ImportError as e:  # pragma: no cover - exercised by a test that hides sympy
    raise ImportError(
        'screws.symbolic needs SymPy: uv add "screws[symbolic]" (or pip install sympy)'
    ) from e

from .aliases import ALIASES as _MR_ALIASES

__all__ = [
    "SymbolicRobot",
    "ad",
    "adjoint",
    "end_effector_forces",
    "exact",
    "exp3",
    "exp6",
    "fk_body",
    "fk_space",
    "forward_dynamics",
    "gravity_forces",
    "inverse_dynamics",
    "jacobian_body",
    "jacobian_space",
    "joint_frames",
    "joint_symbols",
    "mass_matrix",
    "prismatic_axis",
    "revolute_axis",
    "rot",
    "rot_inv",
    "rotation_from_rpy",
    "rp_to_transform",
    "screw_axis",
    "se3_to_vec",
    "so3_to_vec",
    "trans",
    "transform_inv",
    "transform_to_rp",
    "vec_to_se3",
    "vec_to_so3",
    "velocity_quadratic_forces",
]

_CHOP = 1e-12
_NICE = (sp.Integer(1), sp.pi, sp.sqrt(2), sp.sqrt(3))


def _float(x: float, decimals: int | None):
    if not math.isfinite(x):
        raise ValueError(f"{x} has no exact value")
    if decimals is not None:
        x = round(x, decimals)
    if abs(x) < _CHOP:
        return sp.Integer(0)
    for base in _NICE:
        c = sp.Rational(x / float(base)).limit_denominator(1000)
        if abs(float(c * base) - x) <= 4 * np.finfo(float).eps * abs(x):
            return c * base
    return sp.Rational(repr(x))


def _scalar(x, decimals: int | None = None):
    if isinstance(x, sp.Basic):
        return x
    if isinstance(x, (bool, np.bool_, int, np.integer)):
        return sp.Integer(int(x))
    if isinstance(x, (float, np.floating)):
        return _float(float(x), decimals)
    return sp.sympify(x)


def exact(a, *, decimals: int | None = None):
    """a as an exact SymPy object: a scalar, or a Matrix for anything array-like.

    A float becomes the decimal it prints as (0.1 is 1/10), unless it is, to the last bit,
    a fraction with denominator up to 1000, or such a fraction of pi, sqrt(2) or sqrt(3)
    (0.8660254037844386 is sqrt(3)/2); below 1e-12 it is zero. decimals=k rounds every
    float to k places first: a URDF that writes pi/2 as 1.5708 leaves cos 1.5708 =
    -3.7e-6 where 0 was meant, and decimals=4 drops it. SymPy objects are left as they
    are. A 1-D sequence becomes a column. Addition, not in the MR library.
    """
    if isinstance(a, sp.MatrixBase):
        return sp.Matrix(a).applyfunc(lambda x: _scalar(x, decimals))
    if isinstance(a, (list, tuple, np.ndarray)):
        arr = np.asarray(a, dtype=object)
        if arr.ndim == 1:
            return sp.Matrix([_scalar(x, decimals) for x in arr])
        if arr.ndim == 2:
            return sp.Matrix([[_scalar(x, decimals) for x in row] for row in arr])
        raise ValueError(f"expected a scalar, vector or matrix; got shape {arr.shape}")
    return _scalar(a, decimals)


def _matrix(a, shape) -> sp.Matrix:
    m = exact(a)
    if not isinstance(m, sp.MatrixBase) or m.shape != shape:
        got = m.shape if isinstance(m, sp.MatrixBase) else "a scalar"
        raise ValueError(f"expected a {shape[0]}x{shape[1]} matrix; got {got}")
    return m


def _vector(a, n) -> sp.Matrix:
    m = exact(a)
    if not isinstance(m, sp.MatrixBase) or n not in m.shape or 1 not in m.shape:
        got = m.shape if isinstance(m, sp.MatrixBase) else "a scalar"
        raise ValueError(f"expected a {n}-vector; got {got}")
    return m.reshape(n, 1)


Simplify = bool | Callable


def _done(m, simplify: Simplify):
    if callable(simplify):
        return m.applyfunc(simplify)
    return m.applyfunc(sp.simplify) if simplify else m


def joint_symbols(n: int, name: str = "theta") -> tuple[sp.Symbol, ...]:
    """The real symbols name1 ... namen, a tuple: theta1 ... thetan by default.

    SymPy prints a trailing "dot" or "ddot" as an accent, so joint_symbols(2, "thetadot")
    displays as theta-dot-1, theta-dot-2. Addition, not in the MR library.
    """
    return tuple(sp.Symbol(f"{name}{i}", real=True) for i in range(1, n + 1))


# ----- rotations: MR 3.2 ------------------------------------------------------------------


def rot_inv(R) -> sp.Matrix:
    """The inverse of a rotation matrix, its transpose. MR 3.2.1; notes 3.3."""
    return _matrix(R, (3, 3)).T


def vec_to_so3(omega) -> sp.Matrix:
    """[omega], the 3x3 skew-symmetric matrix with [omega] y = omega x y. MR eq. 3.30; notes 3.5."""
    w = _vector(omega, 3)
    return sp.Matrix([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def so3_to_vec(so3mat) -> sp.Matrix:
    """The 3-vector omega from its skew-symmetric matrix [omega]. MR 3.2.3.1; notes 3.5."""
    m = _matrix(so3mat, (3, 3))
    return sp.Matrix([m[2, 1], m[0, 2], m[1, 0]])


def _has_root(e) -> bool:
    return any(p.exp.is_Rational and not p.exp.is_integer for p in e.atoms(sp.Pow))


def _angle(w: sp.Matrix):
    """theta from the exponential coordinates w = omega_hat theta: a root of theta^2 = w . w.

    The exponentials use only even functions of theta (sin theta / theta, cos theta, ...),
    so either root serves, and the absolute values SymPy puts round sqrt(theta^2) can go.
    An axis read from floats is a unit vector only to round-off; a factor within 1e-12 of
    1 in w . w is that round-off, and is dropped as smaller numbers are by exact.
    """
    n2 = w.dot(w)
    c, rest = n2.as_coeff_Mul()
    if c != 1 and c.is_Rational and abs(c - 1) < _CHOP:
        n2 = rest
    r = _root(n2)
    if r is None:  # e.g. theta^2 (cos^2 a + sin^2 a), whose root is plain only once simplified
        n2 = sp.simplify(n2)
        r = _root(n2)
    return sp.sqrt(n2) if r is None else r


def _root(n2):
    if n2 == 0:
        return sp.Integer(0)
    r = sp.powdenest(sp.sqrt(n2), force=True).replace(sp.Abs, lambda a: a)
    return r if not _has_root(r) and sp.expand(r**2 - n2) == 0 else None


def exp3(so3mat, *, simplify: Simplify = True) -> sp.Matrix:
    """e^{[omega] theta}: the rotation matrix from the so(3) matrix [omega] theta.

    Rodrigues' formula, I + sin(theta) [omega] + (1 - cos theta) [omega]^2, written in
    [omega] theta itself so the angle need not be passed separately. MR eq. 3.51; notes 3.6.
    """
    W = _matrix(so3mat, (3, 3))
    r = _angle(so3_to_vec(W))
    if r == 0:
        return sp.eye(3)
    R = sp.eye(3) + sp.sin(r) / r * W + (1 - sp.cos(r)) / r**2 * (W * W)
    return _done(R, simplify)


def rot(omega_hat, theta, *, simplify: Simplify = True) -> sp.Matrix:
    """Rot(omega_hat, theta): the rotation about the unit axis omega_hat by theta.

    Addition, not in the MR library. MR 3.2.3.3; notes 3.6.
    """
    return exp3(vec_to_so3(omega_hat) * _scalar(theta), simplify=simplify)


def rotation_from_rpy(roll, pitch, yaw, *, simplify: Simplify = True) -> sp.Matrix:
    """URDF fixed-axis roll-pitch-yaw: Rot(z, yaw) Rot(y, pitch) Rot(x, roll).

    Addition, not in the MR library. MR Appendix B.1; notes 4.3.
    """
    R = rot([0, 0, 1], yaw, simplify=False) * rot([0, 1, 0], pitch, simplify=False)
    return _done(R * rot([1, 0, 0], roll, simplify=False), simplify)


# ----- rigid-body motions: MR 3.3 ---------------------------------------------------------


def rp_to_transform(R, p) -> sp.Matrix:
    """The homogeneous transformation T = (R, p). MR eq. 3.62; notes 3.4."""
    T = sp.eye(4)
    T[:3, :3] = _matrix(R, (3, 3))
    T[:3, 3] = _vector(p, 3)
    return T


def transform_to_rp(T) -> tuple[sp.Matrix, sp.Matrix]:
    """(R, p) from the homogeneous transformation T. MR 3.3.1; notes 3.4."""
    T = _matrix(T, (4, 4))
    return T[:3, :3], T[:3, 3]


def transform_inv(T) -> sp.Matrix:
    """T^{-1} = (R^T, -R^T p). MR eq. 3.64; notes 3.4."""
    R, p = transform_to_rp(T)
    return rp_to_transform(R.T, -R.T * p)


def trans(p) -> sp.Matrix:
    """Trans(p): the pure translation by p. Addition, not in the MR library. MR 3.3.1."""
    return rp_to_transform(sp.eye(3), p)


def vec_to_se3(V) -> sp.Matrix:
    """[V], the 4x4 se(3) matrix of the twist V = (omega, v). MR eq. 3.72; notes 3.5."""
    V = _vector(V, 6)
    m = sp.zeros(4, 4)
    m[:3, :3] = vec_to_so3(V[:3, 0])
    m[:3, 3] = V[3:, 0]
    return m


def se3_to_vec(se3mat) -> sp.Matrix:
    """The twist V = (omega, v) from its se(3) matrix [V]. MR 3.3.2.1; notes 3.5."""
    m = _matrix(se3mat, (4, 4))
    return sp.Matrix.vstack(so3_to_vec(m[:3, :3]), m[:3, 3])


def adjoint(T) -> sp.Matrix:
    """[Ad_T], the 6x6 matrix that changes the frame of a twist. MR definition 3.20; notes 3.5."""
    R, p = transform_to_rp(T)
    A = sp.zeros(6, 6)
    A[:3, :3] = R
    A[3:, :3] = vec_to_so3(p) * R
    A[3:, 3:] = R
    return A


def screw_axis(q, s_hat, h) -> sp.Matrix:
    """The screw axis S = (s_hat, -s_hat x q + h s_hat) through q, direction s_hat, pitch h.

    Finite pitch only; use prismatic_axis for h = infinity. MR eq. 3.77; notes 3.5.
    """
    q, s = _vector(q, 3), _vector(s_hat, 3)
    return sp.Matrix.vstack(s, q.cross(s) + _scalar(h) * s)


def revolute_axis(q, s_hat) -> sp.Matrix:
    """Zero-pitch screw axis (s_hat, -s_hat x q) through the point q.

    Addition, not in the MR library; the joint-frame derivation of notes 4.1.
    """
    q, s = _vector(q, 3), _vector(s_hat, 3)
    return sp.Matrix.vstack(s, -s.cross(q))


def prismatic_axis(s_hat) -> sp.Matrix:
    """Infinite-pitch screw axis (0, s_hat). Addition, not in the MR library. MR 3.3.2.2."""
    return sp.Matrix.vstack(sp.zeros(3, 1), _vector(s_hat, 3))


def exp6(se3mat, *, simplify: Simplify = True) -> sp.Matrix:
    """e^{[S] theta}: the homogeneous transformation from the se(3) matrix [S] theta.

    MR proposition 3.25, (e^{[omega] theta}, G(theta) v), with G(theta) v written in
    [omega] theta and v theta; a pure translation when omega = 0. MR 3.3.3; notes 3.7.
    """
    m = _matrix(se3mat, (4, 4))
    W, w = m[:3, :3], m[:3, 3]
    r = _angle(so3_to_vec(W))
    if r == 0:
        return rp_to_transform(sp.eye(3), w)
    W2 = W * W
    R = sp.eye(3) + sp.sin(r) / r * W + (1 - sp.cos(r)) / r**2 * W2
    p = w + (1 - sp.cos(r)) / r**2 * (W * w) + (r - sp.sin(r)) / r**3 * (W2 * w)
    return _done(rp_to_transform(R, p), simplify)


# ----- forward kinematics and Jacobians: MR 4.1, 5.1 --------------------------------------


def _screws(S) -> sp.Matrix:
    if isinstance(S, (list, tuple)) and S and all(_is_6_vector(a) for a in S):
        raise TypeError(
            "screw axes must be a 6xn matrix, one axis per column, not a sequence of 6-vectors: "
            "stack them with sympy.Matrix.hstack(*axes), or build a "
            "SymbolicRobot.from_screw_axes(M, axes)"
        )
    S = exact(S)
    if not isinstance(S, sp.MatrixBase) or S.shape[0] != 6:
        got = S.shape if isinstance(S, sp.MatrixBase) else "a scalar"
        raise ValueError(f"screw axes must be a 6xn matrix, one axis per column; got {got}")
    return S


def _is_6_vector(a) -> bool:
    if isinstance(a, sp.MatrixBase):
        return a.shape in ((6, 1), (1, 6))
    return np.shape(a) == (6,)


def _joints(theta, n: int, name: str = "theta") -> list:
    if theta is None:
        return list(joint_symbols(n, name))
    if isinstance(theta, sp.MatrixBase):
        theta = list(theta)
    theta = [_scalar(t) for t in np.asarray(theta, dtype=object).reshape(-1)]
    if len(theta) != n:
        raise ValueError(f"{name} has {len(theta)} entries for {n} joints")
    return theta


def _exp(axis: sp.Matrix, theta) -> sp.Matrix:
    return exp6(vec_to_se3(axis * theta), simplify=False)


def fk_space(M, S, theta=None, *, simplify: Simplify = True) -> sp.Matrix:
    """Space form of the product of exponentials: e^{[S1]t1} ... e^{[Sn]tn} M.

    S is 6xn, one screw axis per column, in {s} at the home position; theta defaults to
    theta1 ... thetan. MR eq. 4.14; notes 4.1.
    """
    S = _screws(S)
    theta = _joints(theta, S.shape[1])
    T = _matrix(M, (4, 4))
    for i in range(len(theta) - 1, -1, -1):
        T = _exp(S[:, i], theta[i]) * T
    return _done(T, simplify)


def fk_body(M, B, theta=None, *, simplify: Simplify = True) -> sp.Matrix:
    """Body form of the product of exponentials: M e^{[B1]t1} ... e^{[Bn]tn}.

    B is 6xn, one screw axis per column, in {b} at the home position. MR eq. 4.16; notes 4.2.
    """
    B = _screws(B)
    theta = _joints(theta, B.shape[1])
    T = _matrix(M, (4, 4))
    for i in range(len(theta)):
        T = T * _exp(B[:, i], theta[i])
    return _done(T, simplify)


def joint_frames(M_joints, S, theta=None, *, simplify: Simplify = True) -> list[sp.Matrix]:
    """Every joint frame T_{0i}(theta) = e^{[S1]t1} ... e^{[Si]ti} M_joints[i-1], i = 1..n.

    Addition, not in the MR library; the space form truncated after each joint. Notes 4.1.
    """
    S = _screws(S)
    theta = _joints(theta, S.shape[1])
    T = sp.eye(4)
    out = []
    for i, Mi in enumerate(M_joints):
        T = T * _exp(S[:, i], theta[i])
        out.append(_done(T * _matrix(Mi, (4, 4)), simplify))
    return out


def jacobian_space(S, theta=None, *, simplify: Simplify = True) -> sp.Matrix:
    """The space Jacobian J_s(theta), 6xn, column i = [Ad_{e^{[S1]t1}...e^{[S_{i-1}]t_{i-1}}}] S_i.

    MR eq. 5.11; notes 5.1.
    """
    S = _screws(S)
    theta = _joints(theta, S.shape[1])
    Js = S.copy()
    T = sp.eye(4)
    for i in range(1, len(theta)):
        T = T * _exp(S[:, i - 1], theta[i - 1])
        Js[:, i] = adjoint(T) * S[:, i]
    return _done(Js, simplify)


def jacobian_body(B, theta=None, *, simplify: Simplify = True) -> sp.Matrix:
    """The body Jacobian J_b(theta), 6xn, column i = [Ad_{e^{-[Bn]tn}...e^{-[B_{i+1}]t_{i+1}}}] B_i.

    MR eq. 5.18; notes 5.1.
    """
    B = _screws(B)
    theta = _joints(theta, B.shape[1])
    Jb = B.copy()
    T = sp.eye(4)
    for i in range(len(theta) - 2, -1, -1):
        T = T * _exp(B[:, i + 1], -theta[i + 1])
        Jb[:, i] = adjoint(T) * B[:, i]
    return _done(Jb, simplify)


# ----- dynamics: MR 8.2-8.5 ---------------------------------------------------------------


def ad(V) -> sp.Matrix:
    """[ad_V], the 6x6 matrix of the Lie bracket: [V1, V2] = [ad_V1] V2. MR eq. 8.39; notes 8.2."""
    V = _vector(V, 6)
    W = vec_to_so3(V[:3, 0])
    A = sp.zeros(6, 6)
    A[:3, :3] = W
    A[3:, :3] = vec_to_so3(V[3:, 0])
    A[3:, 3:] = W
    return A


def _model(link_frames, link_inertias, S, n):
    Ms = [_matrix(m, (4, 4)) for m in link_frames]
    Gs = [_matrix(g, (6, 6)) for g in link_inertias]
    if len(Ms) != n + 1 or len(Gs) != n or S.shape != (6, n):
        raise ValueError(
            f"a {n}-joint chain needs {n + 1} link frames, {n} inertias and a 6x{n} S; got "
            f"{len(Ms)}, {len(Gs)} and {S.shape}"
        )
    return Ms, Gs


def inverse_dynamics(
    theta, dtheta, ddtheta, g, F_tip, link_frames, link_inertias, S, *, simplify: Simplify = True
) -> sp.Matrix:
    """Joint forces/torques tau = M(theta) ddtheta + c(theta, dtheta) + g(theta) + J^T F_tip.

    Newton-Euler forward and backward iterations, as screws.inverse_dynamics. theta, dtheta
    and ddtheta may be None: theta_i, thetadot_i, thetaddot_i. F_tip None means zero.
    MR 8.3.2, algorithm on p. 296; notes 8.3.
    """
    S = _screws(S)
    n = S.shape[1]
    theta = _joints(theta, n)
    dtheta = _joints(dtheta, n, "thetadot")
    ddtheta = _joints(ddtheta, n, "thetaddot")
    Ms, Gs = _model(link_frames, link_inertias, S, n)
    F = sp.zeros(6, 1) if F_tip is None else _vector(F_tip, 6)
    Mi = sp.eye(4)
    A, AdT = [], [None] * (n + 1)
    V = [sp.zeros(6, 1)]
    Vd = [sp.Matrix.vstack(sp.zeros(3, 1), -_vector(g, 3))]
    AdT[n] = adjoint(transform_inv(Ms[n]))
    for i in range(n):
        Mi = Mi * Ms[i]
        A.append(adjoint(transform_inv(Mi)) * S[:, i])
        AdT[i] = adjoint(_exp(A[i], -theta[i]) * transform_inv(Ms[i]))
        V.append(AdT[i] * V[i] + A[i] * dtheta[i])
        Vd.append(AdT[i] * Vd[i] + A[i] * ddtheta[i] + ad(V[i + 1]) * A[i] * dtheta[i])
    tau = sp.zeros(n, 1)
    for i in range(n - 1, -1, -1):
        F = AdT[i + 1].T * F + Gs[i] * Vd[i + 1] - ad(V[i + 1]).T * (Gs[i] * V[i + 1])
        tau[i] = F.dot(A[i])
    return _done(tau, simplify)


def mass_matrix(theta, link_frames, link_inertias, S, *, simplify: Simplify = True) -> sp.Matrix:
    """M(theta), one inverse-dynamics call per unit joint acceleration. MR 8.3.3; notes 8.3."""
    S = _screws(S)
    n = S.shape[1]
    theta = _joints(theta, n)
    cols = [
        inverse_dynamics(
            theta, [0] * n, [int(j == i) for j in range(n)], [0, 0, 0], None,
            link_frames, link_inertias, S, simplify=False,
        )
        for i in range(n)
    ]
    return _done(sp.Matrix.hstack(*cols), simplify)


def velocity_quadratic_forces(
    theta, dtheta, link_frames, link_inertias, S, *, simplify: Simplify = True
) -> sp.Matrix:
    """c(theta, dtheta), the Coriolis and centripetal joint forces. MR 8.3.3; notes 8.3."""
    n = _screws(S).shape[1]
    return inverse_dynamics(
        theta, dtheta, [0] * n, [0, 0, 0], None, link_frames, link_inertias, S, simplify=simplify
    )


def gravity_forces(theta, g, link_frames, link_inertias, S, *, simplify: Simplify = True) -> sp.Matrix:
    """g(theta), the joint forces that hold the chain against gravity g. MR 8.3.3; notes 8.3."""
    n = _screws(S).shape[1]
    return inverse_dynamics(
        theta, [0] * n, [0] * n, g, None, link_frames, link_inertias, S, simplify=simplify
    )


def end_effector_forces(
    theta, F_tip, link_frames, link_inertias, S, *, simplify: Simplify = True
) -> sp.Matrix:
    """J^T(theta) F_tip, the joint forces that create the end-effector wrench alone. MR 8.3.3.

    F_tip may be None: the symbols m_x, m_y, m_z, f_x, f_y, f_z.
    """
    n = _screws(S).shape[1]
    if F_tip is None:
        F_tip = sp.symbols("m_x m_y m_z f_x f_y f_z", real=True)
    return inverse_dynamics(
        theta, [0] * n, [0] * n, [0, 0, 0], F_tip, link_frames, link_inertias, S, simplify=simplify
    )


def forward_dynamics(
    theta, dtheta, tau, g, F_tip, link_frames, link_inertias, S, *, simplify: Simplify = True
) -> sp.Matrix:
    """ddtheta = M(theta)^{-1} (tau - c - g - J^T F_tip). tau may be None: tau1 ... taun.

    The symbolic inverse grows quickly; past two or three joints, expect to wait. MR 8.5.
    """
    S = _screws(S)
    n = S.shape[1]
    theta = _joints(theta, n)
    dtheta = _joints(dtheta, n, "thetadot")
    tau = sp.Matrix(_joints(tau, n, "tau"))
    rest = inverse_dynamics(
        theta, dtheta, [0] * n, g, F_tip, link_frames, link_inertias, S, simplify=False
    )
    M = mass_matrix(theta, link_frames, link_inertias, S, simplify=simplify)
    # M is positive definite, so elimination never meets a true zero pivot and a structural
    # zero test will do; SymPy's default test tries to prove each pivot nonzero, and on an
    # unsimplified M that takes minutes.
    return _done(M.LUsolve(tau - rest, iszerofunc=lambda e: e == 0), simplify)


# ----- a robot ----------------------------------------------------------------------------


def _frozen(a, shape) -> sp.ImmutableMatrix:
    return sp.ImmutableMatrix(_matrix(a, shape))


@dataclass(frozen=True, eq=False)
class SymbolicRobot:
    """A serial chain whose description may hold symbols: the symbolic twin of screws.Robot.

    M: 4x4 home configuration of {b} in {s}. S: 6xn space-frame screw axes, one per
    column. link_frames, link_inertias: MR's M_{i-1,i} and G_i, or None. gravity: the
    gravity vector in {s}, (0, 0, -g) with g a positive symbol unless given. Every method
    takes the joint values as optional (theta1 ... thetan when left out) and simplify=.

    Build one from a numerical robot with Robot.symbolic() (its floats read exactly), or by
    hand with from_screw_axes. MR 4.1, 8.3; notes chapter 4.
    """

    name: str
    M: sp.ImmutableMatrix
    S: sp.ImmutableMatrix
    joint_types: tuple[str, ...]
    joint_names: tuple[str, ...]
    link_frames: tuple[sp.ImmutableMatrix, ...] | None = None
    link_inertias: tuple[sp.ImmutableMatrix, ...] | None = None
    gravity: sp.ImmutableMatrix = field(
        default_factory=lambda: sp.ImmutableMatrix([0, 0, -sp.Symbol("g", positive=True)])
    )

    def __post_init__(self):
        S = sp.ImmutableMatrix(_screws(self.S))
        n = S.shape[1]
        if len(self.joint_types) != n or len(self.joint_names) != n:
            raise ValueError("joint_types and joint_names must have one entry per joint")
        object.__setattr__(self, "M", _frozen(self.M, (4, 4)))
        object.__setattr__(self, "S", S)
        object.__setattr__(self, "joint_types", tuple(self.joint_types))
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        object.__setattr__(self, "gravity", sp.ImmutableMatrix(_vector(self.gravity, 3)))
        if self.link_frames is not None:
            object.__setattr__(self, "link_frames", tuple(_frozen(m, (4, 4)) for m in self.link_frames))
        if self.link_inertias is not None:
            object.__setattr__(
                self, "link_inertias", tuple(_frozen(G, (6, 6)) for G in self.link_inertias)
            )

    @classmethod
    def from_robot(cls, robot, *, decimals: int | None = None) -> SymbolicRobot:
        """The symbolic twin of a screws.Robot, its floats read exactly (see exact for
        decimals); Robot.symbolic calls this.

        Warns when, without decimals, the kinematics hold a number below 1e-5 that is not
        zero: almost always the round-off of a URDF's pi/2, which the formulas then carry
        and which keeps simplify from cancelling anything.
        """

        def ex(a):
            return exact(a, decimals=decimals)

        if decimals is None:
            geometry = [robot.M, robot.S, *(robot.link_frames or ())]
            tiny = [x for a in geometry for x in np.ravel(a) if _CHOP <= abs(x) < 1e-5]
            if tiny:
                warnings.warn(
                    f"{robot.name}: {tiny[0]:.1e} in its kinematics is probably round-off (a "
                    "URDF's pi/2, say); its formulas will carry it and simplify will crawl. "
                    "Pass decimals= (decimals=6 rounds to the micrometre).",
                    stacklevel=3,
                )

        return cls(
            name=robot.name,
            M=ex(robot.M),
            S=ex(robot.S),
            joint_types=robot.joint_types,
            joint_names=robot.joint_names,
            link_frames=None if robot.link_frames is None else tuple(map(ex, robot.link_frames)),
            link_inertias=None if robot.link_inertias is None else tuple(map(ex, robot.link_inertias)),
            gravity=ex(robot.gravity),
        )

    @classmethod
    def from_screw_axes(
        cls, M, axes, *, name="robot", joint_types=None, joint_names=None, gravity=None
    ) -> SymbolicRobot:
        """Build a robot from M and a sequence of space-frame screw axes, one 6-vector per joint.

        As Robot.from_screw_axes: joint types default to reading each axis (omega = 0 means
        prismatic). Notes 4.1.
        """
        if isinstance(axes, (sp.MatrixBase, np.ndarray)) and len(axes.shape) == 2 and min(axes.shape) > 1:
            raise TypeError(
                "axes must be a sequence of 6-vectors, one per joint; "
                "a 6xn matrix is ambiguous when n = 6 (pass list of its columns)"
            )
        vectors = [_vector(a, 6) for a in axes]
        n = len(vectors)
        if joint_types is None:
            joint_types = tuple(
                "prismatic" if all(sp.simplify(x) == 0 for x in v[:3, 0]) else "revolute"
                for v in vectors
            )
        if joint_names is None:
            joint_names = tuple(f"joint{i + 1}" for i in range(n))
        extra = {} if gravity is None else {"gravity": gravity}
        return cls(
            name=name,
            M=M,
            S=sp.Matrix.hstack(*vectors),
            joint_types=tuple(joint_types),
            joint_names=tuple(joint_names),
            **extra,
        )

    @property
    def n(self) -> int:
        """The number of joints."""
        return self.S.shape[1]

    @property
    def theta(self) -> tuple[sp.Symbol, ...]:
        """theta1 ... thetan, the joint variables the methods use when none are passed."""
        return joint_symbols(self.n)

    @cached_property
    def B(self) -> sp.Matrix:
        """The body-frame screw axes at home, B = [Ad_{M^{-1}}] S. MR 4.1.3; notes 4.2."""
        return _done(adjoint(transform_inv(self.M)) * self.S, True)

    def fk(self, theta=None, *, simplify: Simplify = True) -> sp.Matrix:
        """T_sb(theta), by the space form. Notes 4.1."""
        return fk_space(self.M, self.S, theta, simplify=simplify)

    def jacobian_space(self, theta=None, *, simplify: Simplify = True) -> sp.Matrix:
        """J_s(theta). MR 5.1.1; notes 5.1."""
        return jacobian_space(self.S, theta, simplify=simplify)

    def jacobian_body(self, theta=None, *, simplify: Simplify = True) -> sp.Matrix:
        """J_b(theta). MR 5.1.2; notes 5.1."""
        return jacobian_body(self.B, theta, simplify=simplify)

    def _dyn(self):
        if self.link_inertias is None or self.link_frames is None:
            from .robot import MissingInertias

            raise MissingInertias(
                f"{self.name} has no link inertias; build it from a robot that has them, "
                "or call with_inertias(link_frames, link_inertias)"
            )
        return self.link_frames, self.link_inertias, self.S

    def inverse_dynamics(
        self, theta=None, dtheta=None, ddtheta=None, F_tip=None, *, simplify: Simplify = True
    ) -> sp.Matrix:
        """tau = M(theta) ddtheta + c(theta, dtheta) + g(theta) + J^T F_tip. MR 8.3."""
        return inverse_dynamics(
            theta, dtheta, ddtheta, self.gravity, F_tip, *self._dyn(), simplify=simplify
        )

    def mass_matrix(self, theta=None, *, simplify: Simplify = True) -> sp.Matrix:
        """M(theta). MR 8.3."""
        return mass_matrix(theta, *self._dyn(), simplify=simplify)

    def velocity_quadratic_forces(self, theta=None, dtheta=None, *, simplify: Simplify = True) -> sp.Matrix:
        """c(theta, dtheta). MR 8.3."""
        return velocity_quadratic_forces(theta, dtheta, *self._dyn(), simplify=simplify)

    def gravity_forces(self, theta=None, *, simplify: Simplify = True) -> sp.Matrix:
        """g(theta), with this robot's gravity. MR 8.3."""
        return gravity_forces(theta, self.gravity, *self._dyn(), simplify=simplify)

    def end_effector_forces(self, theta=None, F_tip=None, *, simplify: Simplify = True) -> sp.Matrix:
        """J^T(theta) F_tip; F_tip defaults to the symbols m_x ... f_z. MR 8.3."""
        return end_effector_forces(theta, F_tip, *self._dyn(), simplify=simplify)

    def forward_dynamics(
        self, theta=None, dtheta=None, tau=None, F_tip=None, *, simplify: Simplify = True
    ) -> sp.Matrix:
        """ddtheta from tau (tau1 ... taun when left out), with this robot's gravity. MR 8.5."""
        return forward_dynamics(
            theta, dtheta, tau, self.gravity, F_tip, *self._dyn(), simplify=simplify
        )

    def with_inertias(self, link_frames, link_inertias) -> SymbolicRobot:
        """A copy carrying MR's M_{i-1,i} list and G_i list."""
        return replace(self, link_frames=tuple(link_frames), link_inertias=tuple(link_inertias))

    def with_gravity(self, g) -> SymbolicRobot:
        """A copy with a different gravity vector."""
        return replace(self, gravity=g)


# MR's CamelCase names for the functions above, as in screws.aliases.
_OURS = frozenset(__all__)
for _mr, _ours in _MR_ALIASES.items():
    if _ours in _OURS and _mr not in _OURS:
        globals()[_mr] = globals()[_ours]
        __all__.append(_mr)
del _OURS, _mr, _ours
