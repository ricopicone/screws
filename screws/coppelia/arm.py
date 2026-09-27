"""Arm: the joints under one scene tree, read and commanded through ``sim``."""

from __future__ import annotations

import numpy as np

from ..robot import Robot
from ..se3 import prismatic_axis, revolute_axis

__all__ = ["Arm"]

_MODES = ("position", "velocity", "torque")


class Arm:
    """A serial chain in the scene: its joints ordered base to tip, and its tip object.

    Joints are every joint under ``path``, ordered by depth. The tip is the first dummy
    under ``path`` whose alias is "tip" or "ee" (or contains "tip"), else the last joint.
    """

    def __init__(self, scene, path: str):
        self.scene = scene
        self.sim = scene.sim
        self.path = path
        self.base = scene._handle(path)
        self.alias = self.sim.getObjectAlias(self.base, 0)
        joints = list(self.sim.getObjectsInTree(self.base, self.sim.sceneobject_joint, 0))
        joints.sort(key=self._depth)
        self.handles: tuple[int, ...] = tuple(joints)
        self.joint_names: tuple[str, ...] = tuple(self.sim.getObjectAlias(h, 0) for h in joints)
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
        named = {self.sim.getObjectAlias(h, 0): h for h in dummies}
        for want in ("tip", "ee"):
            if want in named:
                return named[want], want
        for alias, h in named.items():
            if "tip" in alias.lower():
                return h, alias
        if not self.handles:
            raise LookupError(f"{self.path} has no joints and no tip dummy")
        return self.handles[-1], self.joint_names[-1]

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
        """Measured joint forces or torques."""
        return np.array([self.sim.getJointForce(h) for h in self.handles], dtype=float)

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

    def command_positions(self, theta) -> None:
        self._require("position")
        for h, v in zip(self.handles, np.asarray(theta, dtype=float)):
            self.sim.setJointTargetPosition(h, float(v))

    def command_velocities(self, dtheta) -> None:
        self._require("velocity")
        for h, v in zip(self.handles, np.asarray(dtheta, dtype=float)):
            self.sim.setJointTargetVelocity(h, float(v))

    def command_torques(self, tau) -> None:
        """Force mode: target force |tau| with a large target velocity in the sign of tau."""
        self._require("torque")
        for h, v in zip(self.handles, np.asarray(tau, dtype=float)):
            self.sim.setJointTargetForce(h, float(abs(v)))
            self.sim.setJointTargetVelocity(h, float(np.sign(v) if v != 0 else 1.0) * 1e3)

    def teleport(self, theta) -> None:
        """Set joint positions directly, without physics: for animating IK iterates."""
        for h, v in zip(self.handles, np.asarray(theta, dtype=float)):
            self.sim.setJointPosition(h, float(v))

    # ----- the robot off the scene ------------------------------------------------------

    def robot(self, *, inertias: bool = False) -> Robot:
        """A screws.Robot read from the scene at the zero position.

        M is the tip frame; joint i's screw axis has omega = its frame's z axis and
        q = its origin (notes 4.1: v = -omega x q). The arm is teleported to zero for the
        reading and put back afterwards. inertias=True arrives in screws 0.2.
        """
        if inertias:
            raise NotImplementedError("scene inertias arrive in screws 0.2")
        here = self.theta()
        self.teleport(np.zeros(self.n))
        try:
            frames = self.joint_frames()
            M = self.tip_frame()
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

