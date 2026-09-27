"""The Robot class: one object holding a serial chain's kinematic and inertial description.

Every method delegates to the chapter functions (so3, se3, kinematics), which remain
the reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np

from . import dynamics as dyn
from . import kinematics as kin
from .se3 import adjoint, rp_to_transform, transform_inv
from .so3 import near_zero, normalize

__all__ = ["MissingInertias", "Robot"]


class MissingInertias(Exception):
    """Raised when a dynamics method is called on a robot without link inertias."""


def _gravity_default() -> np.ndarray:
    return np.array([0.0, 0.0, -9.81])


def _frozen(a) -> np.ndarray:
    """A float64 copy with the writeable flag off, so a frozen Robot cannot be edited in place."""
    a = np.array(a, dtype=float)
    a.flags.writeable = False
    return a


@dataclass(frozen=True, eq=False)
class Robot:
    """A serial-chain robot in the product-of-exponentials description.

    M: 4x4 home configuration of {b} in {s}. S: 6xn space-frame screw axes at home,
    one per column. joint_types: "revolute" or "prismatic" per joint. joint_limits:
    nx2 lower/upper bounds, or None. joint_frames_home: each joint frame T_{0i}(0), or
    None. link_frames: MR's M_{i-1,i}, n+1 of them, or None. link_inertias: MR's spatial
    inertias G_i, 6x6 each, or None. gravity: the gravity vector in {s}.

    MR 4.1 (M, S), 4.1.3 (B), 8.3 (link_frames, link_inertias); notes chapter 4.
    """

    name: str
    M: np.ndarray
    S: np.ndarray
    joint_types: tuple[str, ...]
    joint_names: tuple[str, ...]
    joint_limits: np.ndarray | None = None
    joint_frames_home: tuple[np.ndarray, ...] | None = None
    link_frames: tuple[np.ndarray, ...] | None = None
    link_inertias: tuple[np.ndarray, ...] | None = None
    gravity: np.ndarray = field(default_factory=_gravity_default)

    def __post_init__(self):
        M = np.asarray(self.M, dtype=float)
        S = np.asarray(self.S, dtype=float)
        if M.shape != (4, 4):
            raise ValueError(f"M must be 4x4; got shape {M.shape}")
        if S.ndim != 2 or S.shape[0] != 6:
            raise ValueError(f"S must be 6xn, one screw axis per column; got shape {S.shape}")
        n = S.shape[1]
        if len(self.joint_types) != n or len(self.joint_names) != n:
            raise ValueError("joint_types and joint_names must have one entry per joint")
        object.__setattr__(self, "M", M)
        object.__setattr__(self, "S", S)
        object.__setattr__(self, "joint_types", tuple(self.joint_types))
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        object.__setattr__(self, "gravity", np.asarray(self.gravity, dtype=float))
        if self.joint_limits is not None:
            lim = np.asarray(self.joint_limits, dtype=float)
            if lim.shape != (n, 2):
                raise ValueError(f"joint_limits must be nx2; got shape {lim.shape}")
            object.__setattr__(self, "joint_limits", lim)
        for name in ("joint_frames_home", "link_frames", "link_inertias"):
            v = getattr(self, name)
            if v is not None:
                object.__setattr__(self, name, tuple(_frozen(a) for a in v))
        for name in ("M", "S", "gravity", "joint_limits"):
            v = getattr(self, name)
            if v is not None:
                object.__setattr__(self, name, _frozen(v))

    # ----- construction -------------------------------------------------------------

    @classmethod
    def from_screw_axes(
        cls,
        M,
        axes,
        *,
        name: str = "robot",
        joint_types=None,
        joint_names=None,
        joint_limits=None,
    ) -> Robot:
        """Build a robot from M and a sequence of space-frame screw axes, one 6-vector per joint.

        The axes must be a sequence (list or tuple), not a 2-D array: a 6x6 array is
        ambiguous when n = 6. Joint types default to reading each axis (omega = 0 means
        prismatic). Notes 4.1.
        """
        if isinstance(axes, np.ndarray) and axes.ndim == 2:
            raise TypeError(
                "axes must be a sequence of 6-vectors, one per joint; "
                "a 6xn array is ambiguous when n = 6 (pass list(S.T) for columns)"
            )
        vectors = [np.asarray(a, dtype=float).reshape(-1) for a in axes]
        for v in vectors:
            if v.shape != (6,):
                raise ValueError(f"each screw axis must be a 6-vector; got shape {v.shape}")
        S = np.column_stack(vectors)
        n = S.shape[1]
        if joint_types is None:
            joint_types = tuple(
                "prismatic" if near_zero(np.linalg.norm(v[:3])) else "revolute" for v in vectors
            )
        if joint_names is None:
            joint_names = tuple(f"joint{i + 1}" for i in range(n))
        return cls(
            name=name,
            M=M,
            S=S,
            joint_types=tuple(joint_types),
            joint_names=tuple(joint_names),
            joint_limits=joint_limits,
        )

    @classmethod
    def from_urdf(cls, path, *, base_link=None, ee_link=None) -> Robot:
        """Build a robot from a URDF file or string; see screws.urdf.load."""
        from .urdf import load

        return load(path, base_link=base_link, ee_link=ee_link)

    # ----- derived quantities ---------------------------------------------------------

    @property
    def n(self) -> int:
        """The number of joints."""
        return self.S.shape[1]

    @cached_property
    def B(self) -> np.ndarray:
        """The body-frame screw axes at home, B = [Ad_{M^{-1}}] S. MR 4.1.3; notes 4.2."""
        return adjoint(transform_inv(self.M)) @ self.S

    # ----- kinematics -----------------------------------------------------------------

    def fk(self, theta) -> np.ndarray:
        """T_sb(theta), the end-effector configuration, by the space form. Notes 4.1."""
        return kin.fk_space(self.M, self.S, theta)

    def frames(self, theta) -> list[np.ndarray]:
        """Every joint frame T_{0i}(theta), i = 1..n, followed by T_sb(theta).

        Uses joint_frames_home when the robot has them. Otherwise each joint frame is a
        stand-in for a stick figure: origin at the point of the axis nearest {s},
        q = omega x v, z along omega; a prismatic joint reuses the previous origin.
        """
        homes = self.joint_frames_home
        if homes is None:
            homes = self._fallback_joint_frames()
        out = kin.joint_frames(homes, self.S, theta)
        out.append(self.fk(theta))
        return out

    def _fallback_joint_frames(self) -> list[np.ndarray]:
        frames, prev_q = [], np.zeros(3)
        for i in range(self.n):
            omega, v = self.S[:3, i], self.S[3:, i]
            if self.joint_types[i] == "prismatic":
                q, z = prev_q, normalize(v)
            else:
                q, z = np.cross(omega, v), normalize(omega)
            helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x = normalize(np.cross(helper, z))
            y = np.cross(z, x)
            frames.append(rp_to_transform(np.column_stack([x, y, z]), q))
            prev_q = q
        return frames

    def jacobian_space(self, theta) -> np.ndarray:
        """J_s(theta). MR 5.1.1; notes 5.1."""
        return kin.jacobian_space(self.S, theta)

    def jacobian_body(self, theta) -> np.ndarray:
        """J_b(theta). MR 5.1.2; notes 5.1."""
        return kin.jacobian_body(self.B, theta)

    def ik(self, T, theta0, *, frame: str = "body", **tol) -> kin.IKResult:
        """Numerical inverse kinematics for the goal T from the guess theta0.

        frame is "body" (default) or "space"; tol may carry tol_omega, tol_v, max_iter.
        MR 6.2; notes 6.2.
        """
        if frame == "body":
            return kin.ik_body(self.M, self.B, T, theta0, **tol)
        if frame == "space":
            return kin.ik_space(self.M, self.S, T, theta0, **tol)
        raise ValueError(f'frame must be "body" or "space"; got {frame!r}')

    def within_limits(self, theta) -> bool:
        """True if every joint value lies within joint_limits (always true without limits)."""
        if self.joint_limits is None:
            return True
        theta = np.asarray(theta, dtype=float)
        return bool(np.all(theta >= self.joint_limits[:, 0]) and np.all(theta <= self.joint_limits[:, 1]))

    # ----- dynamics ---------------------------------------------------------------------

    def _need_inertias(self):
        if self.link_inertias is None or self.link_frames is None:
            raise MissingInertias(
                f"{self.name} has no link inertias; load a URDF with <inertial> elements, read "
                "them off a scene with Arm.robot(inertias=True), or call with_inertias(...)"
            )

    def inverse_dynamics(self, theta, dtheta, ddtheta, F_tip=None) -> np.ndarray:
        """tau = M(theta) ddtheta + c(theta, dtheta) + g(theta) + J^T F_tip, with this robot's gravity. MR 8.3."""
        self._need_inertias()
        return dyn.inverse_dynamics(
            theta, dtheta, ddtheta, self.gravity, F_tip, self.link_frames, self.link_inertias, self.S
        )

    def mass_matrix(self, theta) -> np.ndarray:
        """M(theta). MR 8.3."""
        self._need_inertias()
        return dyn.mass_matrix(theta, self.link_frames, self.link_inertias, self.S)

    def velocity_quadratic_forces(self, theta, dtheta) -> np.ndarray:
        """c(theta, dtheta). MR 8.3."""
        self._need_inertias()
        return dyn.velocity_quadratic_forces(theta, dtheta, self.link_frames, self.link_inertias, self.S)

    def gravity_forces(self, theta) -> np.ndarray:
        """g(theta), with this robot's gravity. MR 8.3."""
        self._need_inertias()
        return dyn.gravity_forces(theta, self.gravity, self.link_frames, self.link_inertias, self.S)

    def end_effector_forces(self, theta, F_tip) -> np.ndarray:
        """J^T(theta) F_tip. MR 8.3."""
        self._need_inertias()
        return dyn.end_effector_forces(theta, F_tip, self.link_frames, self.link_inertias, self.S)

    def forward_dynamics(self, theta, dtheta, tau, F_tip=None) -> np.ndarray:
        """ddtheta from tau, with this robot's gravity. MR 8.5."""
        self._need_inertias()
        return dyn.forward_dynamics(
            theta, dtheta, tau, self.gravity, F_tip, self.link_frames, self.link_inertias, self.S
        )

    # ----- copies ---------------------------------------------------------------------

    def with_gravity(self, g) -> Robot:
        """A copy with a different gravity vector."""
        return replace(self, gravity=np.asarray(g, dtype=float))

    def with_inertias(self, link_frames, link_inertias) -> Robot:
        """A copy carrying MR's M_{i-1,i} list and G_i list."""
        return replace(self, link_frames=tuple(link_frames), link_inertias=tuple(link_inertias))

    def with_limits(self, joint_limits) -> Robot:
        """A copy with nx2 joint limits."""
        return replace(self, joint_limits=joint_limits)
