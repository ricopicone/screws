"""URDF -> Robot: joint origins and axes to M and the screw axes; inertial elements to
MR's link frames and spatial inertias. Notes 4.3 (The Universal Robot Description Format);
MR 4.2 and 8.3.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib import resources

import numpy as np

from .robot import Robot
from .se3 import adjoint, prismatic_axis, revolute_axis, rp_to_transform, transform_inv
from .so3 import rotation_from_rpy

__all__ = ["load", "packaged"]

_REVOLUTE = {"revolute", "continuous"}
_MOVING = _REVOLUTE | {"prismatic"}


@dataclass
class _Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray | None
    limits: tuple[float, float] | None


@dataclass
class _Inertial:
    mass: float
    origin: np.ndarray
    inertia: np.ndarray


def packaged(filename: str) -> str:
    """The filesystem path of a URDF shipped inside screws.robots."""
    return str(resources.files("screws.robots").joinpath(filename))


def _floats(text: str | None, default: str) -> np.ndarray:
    return np.array([float(x) for x in (text or default).split()])


def _origin(el: ET.Element | None) -> np.ndarray:
    if el is None:
        return np.eye(4)
    xyz = _floats(el.get("xyz"), "0 0 0")
    rpy = _floats(el.get("rpy"), "0 0 0")
    return rp_to_transform(rotation_from_rpy(*rpy), xyz)


def _parse_joint(el: ET.Element) -> _Joint:
    jtype = el.get("type", "")
    if jtype not in _MOVING | {"fixed"}:
        raise ValueError(
            f"joint {el.get('name')!r} has type {jtype!r}; screws reads revolute, continuous, "
            "prismatic and fixed joints only"
        )
    axis_el = el.find("axis")
    axis = _floats(axis_el.get("xyz"), "1 0 0") if axis_el is not None else None
    if jtype in _MOVING and axis is None:
        axis = np.array([1.0, 0.0, 0.0])  # the URDF default
    if axis is not None:
        norm = np.linalg.norm(axis)
        if norm == 0:
            raise ValueError(f"joint {el.get('name')!r} has a zero <axis>")
        axis = axis / norm
    lim_el = el.find("limit")
    limits = None
    if lim_el is not None and lim_el.get("lower") is not None and lim_el.get("upper") is not None:
        limits = (float(lim_el.get("lower")), float(lim_el.get("upper")))
    return _Joint(
        name=el.get("name", ""),
        type=jtype,
        parent=el.find("parent").get("link"),
        child=el.find("child").get("link"),
        origin=_origin(el.find("origin")),
        axis=axis,
        limits=limits,
    )


def _parse_inertial(link_el: ET.Element) -> _Inertial | None:
    el = link_el.find("inertial")
    if el is None:
        return None
    mass = float(el.find("mass").get("value"))
    i = el.find("inertia")
    g = {k: float(i.get(k, "0")) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
    inertia = np.array(
        [
            [g["ixx"], g["ixy"], g["ixz"]],
            [g["ixy"], g["iyy"], g["iyz"]],
            [g["ixz"], g["iyz"], g["izz"]],
        ]
    )
    return _Inertial(mass=mass, origin=_origin(el.find("origin")), inertia=inertia)


def _pick(kind: str, candidates: list[str], chosen: str | None) -> str:
    if chosen is not None:
        return chosen
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError("the URDF has no joints, so there is no chain to read")
    raise ValueError(
        f"cannot choose the {kind}: candidates are {', '.join(sorted(candidates))}; "
        f"pass {kind.split()[0]}_link=..."
    )


def load(path_or_xml, *, base_link: str | None = None, ee_link: str | None = None) -> Robot:
    """Build a Robot from a URDF file path or a URDF string.

    The chain runs from base_link (default: the one link that is nobody's child) to ee_link
    (default: the one link that is nobody's parent). Each moving joint's <origin> is
    T_{i-1,i}(0); its <axis> in its own frame is A_i = (axis, 0) or (0, axis); the screw
    axis in {s} is S_i = [Ad_{T_0i(0)}] A_i, and M is the product carried through the
    fixed joints to ee_link. Fixed joints add no column. <axis> is normalised. <limit>
    bounds are recorded; a joint without them (e.g. continuous) gets (-inf, inf) when any
    other joint has limits.

    Inertia follows MR 8.3: link frame {i} sits at link i's centre of mass with the
    <inertial><origin> axes, so G_i = diag(I_b, m I); link_frames[i-1] is M_{i-1,i} and
    link_frames[n] carries the last link frame to {b}. A moving link without <inertial>
    leaves link_frames and link_inertias as None (a kinematic-only robot).
    Notes 4.3; MR 4.2, 8.3.
    """
    text = str(path_or_xml)
    if "<robot" in text:
        root = ET.fromstring(text)
    else:
        if not os.path.exists(text):
            raise FileNotFoundError(f"no URDF at {text}")
        root = ET.parse(text).getroot()
    if root.tag != "robot":
        raise ValueError(f"expected a <robot> root element, got <{root.tag}>")

    joints = [_parse_joint(j) for j in root.findall("joint")]
    inertials = {link.get("name"): _parse_inertial(link) for link in root.findall("link")}
    by_parent: dict[str, list[_Joint]] = {}
    parents, children = set(), set()
    for j in joints:
        by_parent.setdefault(j.parent, []).append(j)
        parents.add(j.parent)
        children.add(j.child)
    roots = sorted(parents - children)
    leaves = sorted(children - parents)
    base = _pick("base link", roots, base_link)
    ee = _pick("ee link", leaves, ee_link)

    # Walk from base to ee along the unique parent->child path.
    path: list[_Joint] = []
    link = base
    seen = set()
    while link != ee:
        if link in seen:
            raise ValueError("the URDF's joint graph has a cycle")
        seen.add(link)
        nxt = [j for j in by_parent.get(link, []) if _reaches(j.child, ee, by_parent)]
        if not nxt:
            raise ValueError(f"no chain of joints from {base!r} to {ee!r}")
        path.append(nxt[0])
        link = nxt[0].child

    T = np.eye(4)
    S_cols, types, names, limits, joint_frames = [], [], [], [], []
    moving_links: list[tuple[str, np.ndarray]] = []  # (child link, T_0i(0))
    for j in path:
        T = T @ j.origin
        if j.type == "fixed":
            continue
        joint_frames.append(T.copy())
        if j.type in _REVOLUTE:
            S_cols.append(adjoint(T) @ revolute_axis(np.zeros(3), j.axis))
            types.append("revolute")
        else:
            S_cols.append(adjoint(T) @ prismatic_axis(j.axis))
            types.append("prismatic")
        names.append(j.name)
        limits.append(j.limits)
        moving_links.append((j.child, T.copy()))
    M = T
    n = len(S_cols)
    S = np.column_stack(S_cols) if n else np.zeros((6, 0))
    joint_limits = None
    if n and any(lim is not None for lim in limits):
        joint_limits = np.array(
            [lim if lim is not None else (-np.inf, np.inf) for lim in limits], dtype=float
        )

    link_frames = link_inertias = None
    if n and all(inertials.get(name) is not None for name, _ in moving_links):
        com_frames = [T0i @ inertials[name].origin for name, T0i in moving_links]
        link_frames, link_inertias = [], []
        prev = np.eye(4)
        for F, (name, _) in zip(com_frames, moving_links):
            link_frames.append(transform_inv(prev) @ F)
            inert = inertials[name]
            G = np.zeros((6, 6))
            G[:3, :3] = inert.inertia
            G[3:, 3:] = inert.mass * np.eye(3)
            link_inertias.append(G)
            prev = F
        link_frames.append(transform_inv(prev) @ M)

    return Robot(
        name=root.get("name", "robot"),
        M=M,
        S=S,
        joint_types=tuple(types),
        joint_names=tuple(names),
        joint_limits=joint_limits,
        joint_frames_home=tuple(joint_frames),
        link_frames=None if link_frames is None else tuple(link_frames),
        link_inertias=None if link_inertias is None else tuple(link_inertias),
    )


def _reaches(link: str, target: str, by_parent: dict[str, list[_Joint]]) -> bool:
    stack = [link]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur == target:
            return True
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(j.child for j in by_parent.get(cur, []))
    return False
