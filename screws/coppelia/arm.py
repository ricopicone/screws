"""Arm: the joints under one scene tree, read and commanded through ``sim``."""

from __future__ import annotations

import numpy as np

from ..robot import Robot
from ..se3 import prismatic_axis, revolute_axis, transform_inv

__all__ = ["Arm"]

_MODES = ("position", "velocity", "torque")


class Arm:
    """A serial chain in the scene: its joints ordered base to tip, and its tip object.

    Joints are every joint under ``path``, ordered by depth, or the ones named in ``joints``
    (aliases or paths, in order) when a gripper or other tool adds joints of its own. The tip is the first dummy
    under ``path`` whose alias is "tip" or "ee" (or contains "tip"), else the last joint.
    """

    def __init__(self, scene, path: str, joints=None):
        self.scene = scene
        self.sim = scene.sim
        self.path = path
        self.base = scene._handle(path)
        self.alias = self.sim.getObjectAlias(self.base, -1)
        found = list(self.sim.getObjectsInTree(self.base, self.sim.sceneobject_joint, 0))
        found.sort(key=self._depth)
        if joints is None:
            joints = found
        else:
            by_alias = {self.sim.getObjectAlias(h, -1): h for h in found}
            picked = []
            for j in joints:
                if isinstance(j, str) and j in by_alias:
                    picked.append(by_alias[j])
                elif isinstance(j, str):
                    try:
                        picked.append(scene._handle(j))
                    except LookupError as exc:
                        raise LookupError(
                            f"no joint {j!r} under {path}; found {sorted(by_alias)}"
                        ) from exc
                else:
                    picked.append(int(j))
            joints = picked
        self.handles: tuple[int, ...] = tuple(joints)
        self.joint_names: tuple[str, ...] = tuple(self.sim.getObjectAlias(h, -1) for h in joints)
        self.tip, self.tip_alias = self._find_tip()
        self._mode: str | None = None

    def _depth(self, h: int) -> int:
        d = 0
        while h != self.sim.handle_world and h != self.base:
            h = self.sim.getObjectParent(h)
            d += 1
        return d

    def _find_tip(self) -> tuple[int, str]:
        dummies = list(self.sim.getObjectsInTree(self.base, self.sim.sceneobject_dummy, 0))
        named = {self.sim.getObjectAlias(h, -1): h for h in dummies}
        for want in ("tip", "ee", "connection"):
            if want in named:
                return named[want], want
        for alias, h in named.items():
            if any(key in alias.lower() for key in ("tip", "connection")):
                return h, alias
        if not self.handles:
            raise LookupError(f"{self.path} has no joints and no tip dummy")
        # Fall back to the last joint's first non-joint child (the last link), else the joint.
        last = self.handles[-1]
        for kind in (self.sim.sceneobject_shape, self.sim.sceneobject_dummy):
            for h in self.sim.getObjectsInTree(last, kind, 1 + 2):  # exclude base, first children only
                return h, self.sim.getObjectAlias(h, -1)
        return last, self.joint_names[-1]

    @property
    def n(self) -> int:
        return len(self.handles)

    # ----- reading ----------------------------------------------------------------------

    def theta(self) -> np.ndarray:
        """Joint positions (rad or m)."""
        return np.array([self.sim.getJointPosition(h) for h in self.handles], dtype=float)

    def dtheta(self) -> np.ndarray:
        """Joint velocities."""
        return np.array([self.sim.getJointVelocity(h) for h in self.handles], dtype=float)

    def tau(self) -> np.ndarray:
        """Measured joint forces or torques; NaN for a joint that reports none (not dynamic)."""
        out = []
        for h in self.handles:
            try:
                out.append(float(self.sim.getJointForce(h)))
            except Exception:  # noqa: BLE001 - the remote API raises a plain Exception
                out.append(float("nan"))
        return np.array(out, dtype=float)

    def tip_frame(self) -> np.ndarray:
        """The tip's 4x4 configuration in the world frame."""
        return self.scene.frame(self.tip)

    def joint_frames(self) -> list[np.ndarray]:
        """Each joint's 4x4 frame in the world frame; the joint axis is the local z axis."""
        return [self.scene.frame(h) for h in self.handles]

    def joint_types(self) -> tuple[str, ...]:
        return tuple(
            "prismatic" if self.sim.getJointType(h) == self.sim.joint_prismatic else "revolute"
            for h in self.handles
        )

    def joint_limits(self) -> np.ndarray | None:
        """nx2 limits from the joints' intervals, or None if any joint is cyclic."""
        out = []
        for h in self.handles:
            cyclic, interval = self.sim.getJointInterval(h)
            if cyclic:
                return None
            out.append([interval[0], interval[0] + interval[1]])
        return np.array(out, dtype=float)

    # ----- commanding -------------------------------------------------------------------

    def mode(self, kind: str) -> None:
        """Set every joint's dynamic control mode: "position", "velocity" or "torque"."""
        if kind not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}; got {kind!r}")
        value = {
            "position": self.sim.jointdynctrl_position,
            "velocity": self.sim.jointdynctrl_velocity,
            "torque": self.sim.jointdynctrl_force,
        }[kind]
        for h in self.handles:
            self.sim.setObjectInt32Param(h, self.sim.jointintparam_dynctrlmode, value)
        self._mode = kind

    def _require(self, kind: str) -> None:
        if self._mode != kind:
            raise RuntimeError(
                f"the arm is in {self._mode!r} mode; call arm.mode({kind!r}) before a {kind} command"
            )

    def command(self, u) -> None:
        """Dispatch on the current mode: positions, velocities or torques."""
        if self._mode is None:
            raise RuntimeError('set a mode first: arm.mode("position" | "velocity" | "torque")')
        dispatch = {
            "position": self.command_positions,
            "velocity": self.command_velocities,
            "torque": self.command_torques,
        }
        dispatch[self._mode](u)

    def _vector(self, u) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.shape[0] != self.n:
            raise ValueError(f"got {u.shape[0]} values for {self.n} joints")
        return u

    def command_positions(self, theta) -> None:
        self._require("position")
        for h, v in zip(self.handles, self._vector(theta)):
            self.sim.setJointTargetPosition(h, float(v))

    def command_velocities(self, dtheta) -> None:
        self._require("velocity")
        for h, v in zip(self.handles, self._vector(dtheta)):
            self.sim.setJointTargetVelocity(h, float(v))

    def command_torques(self, tau) -> None:
        """Force mode: the signed joint force or torque is applied directly
        (sim.setJointTargetForce with signedValue, CoppeliaSim 4.3+)."""
        self._require("torque")
        for h, v in zip(self.handles, self._vector(tau)):
            self.sim.setJointTargetForce(h, float(v), True)

    def teleport(self, theta) -> None:
        """Set joint positions directly, without physics: for animating IK iterates."""
        for h, v in zip(self.handles, self._vector(theta)):
            self.sim.setJointPosition(h, float(v))

    # ----- the robot off the scene ------------------------------------------------------

    def robot(self, *, inertias: bool = False, relative_to: str = "world") -> Robot:
        """A screws.Robot read from the scene at the zero position.

        M is the tip frame; joint i's screw axis has omega = its frame's z axis and
        q = its origin (notes 4.1: v = -omega x q). relative_to="world" (default) makes {s}
        CoppeliaSim's world frame; relative_to="base" makes {s} the frame of the object at
        ``path``, so the result is independent of where the model stands in the scene.
        The arm is teleported to zero for the reading and put back afterwards, so call this
        before starting the simulation or accept a jump. inertias=True arrives in 0.2.
        """
        if inertias:
            raise NotImplementedError("scene inertias arrive in screws 0.2")
        if relative_to not in ("world", "base"):
            raise ValueError(f'relative_to must be "world" or "base"; got {relative_to!r}')
        here = self.theta()
        self.teleport(np.zeros(self.n))
        try:
            frames = self.joint_frames()
            M = self.tip_frame()
            if relative_to == "base":
                T_ws_inv = transform_inv(self.scene.frame(self.base))
                frames = [T_ws_inv @ F for F in frames]
                M = T_ws_inv @ M
        finally:
            self.teleport(here)
        axes = []
        for F, kind in zip(frames, self.joint_types()):
            z, q = F[:3, 2], F[:3, 3]
            axes.append(prismatic_axis(z) if kind == "prismatic" else revolute_axis(q, z))
        return Robot(
            name=self.alias,
            M=M,
            S=np.column_stack(axes),
            joint_types=self.joint_types(),
            joint_names=self.joint_names,
            joint_limits=self.joint_limits(),
            joint_frames_home=tuple(frames),
        )

    def __repr__(self) -> str:
        return f"Arm({self.path!r}, joints={list(self.joint_names)}, tip={self.tip_alias!r})"

