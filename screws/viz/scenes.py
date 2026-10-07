"""Bodies and the three screw motions of the chapter 3 review: a door (zero pitch), a drawer
(infinite pitch) and a screwdriver (finite pitch). numpy only; the animators draw them.

A Scene holds a screw axis S written in {s}, the body frame's home T0 = T_sb(0), the body
as meshes in {b} coordinates, fixed meshes in {s}, and the points whose velocities are
drawn. The body at theta is e^{[S] theta} T0 applied to its {b} coordinates. Notes 3.10.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..motion import screw_line
from ..se3 import adjoint, axis_angle6, log6, rp_to_transform, screw_axis, se3_to_vec, transform_inv

__all__ = ["Mesh", "Scene", "box", "door", "drawer", "prism", "screw_between", "screwdriver"]

WOOD = "#c8955c"
STEEL = "#9aa3ab"
WALL = "#d9d9d9"
HANDLE = "#3b3b3b"
GRIP = "#d62728"


@dataclass
class Mesh:
    """Polygons sharing a vertex array: vertices (V, 3), faces as lists of vertex indices,
    one colour per face (or one for all)."""

    vertices: np.ndarray
    faces: list
    colours: list | str = WOOD
    alpha: float = 1.0

    def face_colours(self) -> list:
        if isinstance(self.colours, str):
            return [self.colours] * len(self.faces)
        return list(self.colours)

    def transformed(self, T) -> Mesh:
        T = np.asarray(T, float)
        return Mesh(self.vertices @ T[:3, :3].T + T[:3, 3], self.faces, self.colours, self.alpha)


def box(lo, hi, colours=WOOD, alpha=1.0) -> Mesh:
    """An axis-aligned box from corner lo to corner hi."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], float)
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    return Mesh(v, faces, colours, alpha)


def prism(base, axis, radius, length, sides=6, colours=STEEL, alpha=1.0) -> Mesh:
    """A right prism with a regular polygon section: from base along the unit axis."""
    base, axis = np.asarray(base, float), np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    e1 = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    ang = 2 * np.pi * np.arange(sides) / sides
    ring = radius * (np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2))
    v = np.vstack([base + ring, base + length * axis + ring])
    faces = [[k, (k + 1) % sides, sides + (k + 1) % sides, sides + k] for k in range(sides)]
    faces += [list(range(sides)), list(range(sides, 2 * sides))]
    return Mesh(v, faces, colours, alpha)


@dataclass
class Scene:
    """One screw motion, ready to animate. S is written in {s}; body meshes and points are
    in {b} coordinates; theta is (start, end) in radians (metres when S is a translation)."""

    name: str
    S: np.ndarray
    T0: np.ndarray
    body: list
    theta: tuple
    points: np.ndarray
    fixed: list = field(default_factory=list)
    trace: int = 0
    arrow_scale: float = 0.5
    view: tuple = (24, 32)  # the notes' house view: x toward the lower left, y right, z up
    note: str = ""
    goal: np.ndarray | None = None  # where the motion ends, drawn faded with the start

    @property
    def S_b(self) -> np.ndarray:
        """The same screw axis in {b} at home: [Ad_{T_sb(0)^{-1}}] S_s."""
        return adjoint(transform_inv(self.T0)) @ self.S

    @property
    def is_translation(self) -> bool:
        return bool(np.allclose(self.S[:3], 0))


def _to_body(T0, meshes):
    Tinv = transform_inv(T0)
    return [m.transformed(Tinv) for m in meshes]


def door() -> Scene:
    """The door of notes 3.10: a vertical hinge through q = (2, 0, 0), {b} on the handle at
    (2.9, 0, 1), S_s = (0, 0, 1, 0, -2, 0), S_b = (0, 0, 1, 0, 0.9, 0), opened 0 to 100 deg."""
    T0 = rp_to_transform(np.eye(3), [2.9, 0, 1])
    slab = box([2.0, -0.025, 0.0], [3.1, 0.025, 2.0], WOOD)
    knob = box([2.86, 0.025, 0.97], [2.94, 0.09, 1.03], HANDLE)
    wall = [
        box([1.3, -0.1, 0.0], [1.98, -0.03, 2.3], WALL, 0.35),
        box([3.12, -0.1, 0.0], [3.6, -0.03, 2.3], WALL, 0.35),
    ]
    pts_s = np.array([[2.9, 0, 1], [3.1, 0, 2.0], [2.45, 0, 0.3]])
    return Scene(
        "door",
        np.array([0, 0, 1, 0, -2, 0.0]),
        T0,
        _to_body(T0, [slab, knob]),
        (0.0, np.deg2rad(100)),
        pts_s - T0[:3, 3],
        fixed=wall,
        arrow_scale=0.5,
        note="hinge through q = (2, 0, 0), pitch 0",
    )


def drawer() -> Scene:
    """A drawer sliding 0.45 m out of a cabinet along +y: omega = 0, infinite pitch, every
    point with the same velocity."""
    T0 = rp_to_transform(np.eye(3), [0.5, 0.07, 1.125])
    tray = box([0.08, -1.0, 0.95], [0.92, 0.03, 1.3], WOOD)
    pull = box([0.4, 0.03, 1.1], [0.6, 0.07, 1.15], HANDLE)
    cabinet = box([0.0, -1.1, 0.0], [1.0, 0.0, 1.4], WALL, 0.25)
    pts_s = np.array([[0.5, 0.07, 1.125], [0.08, 0.03, 0.95], [0.92, -1.0, 1.3]])
    return Scene(
        "drawer",
        np.array([0, 0, 0, 0, 1, 0.0]),
        T0,
        _to_body(T0, [tray, pull]),
        (0.0, 0.45),
        pts_s - T0[:3, 3],
        fixed=[cabinet],
        arrow_scale=0.5,
        note="a translation, pitch infinite",
    )


def screwdriver(pitch: float = 0.01) -> Scene:
    """A screwdriver driving down: s_hat = -z through the origin, two turns. The pitch,
    0.01 m/rad by default (6.3 cm per turn), is exaggerated so the advance shows."""
    T0 = np.eye(4)
    T0[:3, 3] = [0, 0, 0.13]  # {b} at the tip
    shaft = prism([0, 0, 0.13], [0, 0, 1], 0.006, 0.15, 8, STEEL)
    grip = prism([0, 0, 0.28], [0, 0, 1], 0.024, 0.12, 6, [GRIP] + [HANDLE] * 5 + [HANDLE, HANDLE])
    board = box([-0.12, -0.12, -0.03], [0.12, 0.12, 0.0], WALL, 0.5)
    pts_s = np.array([[0, 0.024, 0.395], [0, 0, 0.13], [-0.0207, 0.012, 0.3]])  # on the grip, the tip
    return Scene(
        "screwdriver",
        screw_axis([0, 0, 0], [0, 0, -1], pitch),
        T0,
        _to_body(T0, [shaft, grip]),
        (0.0, 4 * np.pi),
        pts_s - T0[:3, 3],
        fixed=[board],
        arrow_scale=2.5,
        view=(30, 32),
        note=f"pitch {pitch:g} m/rad, exaggerated",
    )


def screw_between(T_start, T_end, *, body=None, name: str = "screw motion") -> Scene:
    """The one screw motion that carries the frame T_start to T_end (both in {s}): from
    [S] theta = log(T_end T_start^{-1}), with S a unit screw axis (|omega| = 1, or |v| = 1
    for a translation) and theta the angle turned (the distance slid). Every rigid-body
    displacement is a screw motion (Chasles); this shows which one. The frame's origin
    traces the helix, and the start and end frames are drawn faded. MR 3.3.3.2; notes 3.7.
    """
    T_start, T_end = np.asarray(T_start, float), np.asarray(T_end, float)
    S, theta = axis_angle6(se3_to_vec(log6(T_end @ transform_inv(T_start))))
    reach = float(np.linalg.norm(T_end[:3, 3] - T_start[:3, 3])) + 0.3
    a = 0.12 * reach  # a phone-shaped slab, sized to the motion
    if body is None:
        body = [box([-0.5 * a, -a, -0.08 * a], [0.5 * a, a, 0.08 * a], "#3B83CD")]
    points = np.array([[0.0, 0, 0], [0.5 * a, a, 0.08 * a], [-0.5 * a, -a, 0.08 * a]])
    _, _, h = screw_line(S)
    turned = f"{theta:.3g} m slid" if np.allclose(S[:3], 0) else f"{np.rad2deg(theta):.0f}° turned"
    return Scene(
        name,
        S,
        T_start,
        list(body),
        (0.0, theta),
        points,
        arrow_scale=0.15 * reach / max(theta, 1e-9),
        note=f"{turned}, pitch h = {h:.3g}",
        goal=T_end,
    )
