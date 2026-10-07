"""Pictures of screw axes: the product of exponentials as a drawing.

draw_robot draws a robot's skeleton, its reference frame and one screw axis per joint, in
the space frame {s} (the S_i) or the body frame {b} (the B_i), and can show how each v_i
is built: the point q_i on the axis, the cross product -omega_i x q_i, the pitch term
h_i omega_i, and their sum v_i. draw_screw does the same for one axis on its own. explore
adds joint sliders in a notebook. Everything needs matplotlib (``screws[plot]``).
MR 3.3.2, 4.1; notes 3.5, 4.1, 4.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..robot import Robot
from ..se3 import transform_inv
from ..so3 import near_zero

__all__ = ["JointDrawing", "RobotDrawing", "ScrewDrawing", "draw_robot", "draw_screw", "explore"]

COLOURS = {
    "skeleton": "#F78B11",
    "joint": "#3B83CD",
    "axis": "#3B83CD",
    "omega": "#1f4e9c",
    "q": "#7f7f7f",
    "cross": "#e07b00",
    "pitch": "#8e44ad",
    "v": "#c0392b",
    "x": "#c0392b",
    "y": "#27ae60",
    "z": "#2980b9",
}
SHOW_DEFAULT = ("skeleton", "frames", "axes", "omega")


@dataclass
class JointDrawing:
    """What was drawn for one joint, and the numbers behind it (all in the drawing's frame)."""

    index: int
    kind: str
    omega: np.ndarray
    v: np.ndarray
    q: np.ndarray
    pitch: float
    cross: np.ndarray  # -omega x q
    pitch_term: np.ndarray  # h omega
    axis_line: object = None
    omega_arrow: object = None
    q_line: object = None
    cross_arrow: object = None
    pitch_arrow: object = None
    v_arrow: object = None


@dataclass
class RobotDrawing:
    fig: object
    ax: object
    frame: str
    skeleton: np.ndarray  # (n + 2, 3): base, joint origins, tool, in the drawing's frame
    joints: list[JointDrawing] = field(default_factory=list)


@dataclass
class ScrewDrawing:
    fig: object
    ax: object
    omega: np.ndarray
    v: np.ndarray
    q: np.ndarray
    pitch: float
    cross: np.ndarray
    pitch_term: np.ndarray
    axis_line: object = None
    omega_arrow: object = None
    q_line: object = None
    cross_arrow: object = None
    pitch_arrow: object = None
    v_arrow: object = None


def _axes(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.figure
    return fig, ax


def _arrow(ax, origin, vec, colour, label=None, lw=2.0):
    origin, vec = np.asarray(origin, float), np.asarray(vec, float)
    if near_zero(np.linalg.norm(vec)):
        return None
    return ax.quiver(*origin, *vec, color=colour, linewidth=lw, arrow_length_ratio=0.15, label=label)


def _triad(ax, T, length, label=None):
    T = np.asarray(T, float)
    o = T[:3, 3]
    arts = [_arrow(ax, o, length * T[:3, k], COLOURS[c], lw=1.5) for k, c in enumerate("xyz")]
    if label:
        ax.text(*(o + 0.04 * length), label, fontsize=9)
    return arts


def _decompose(omega, v):
    """(q, pitch, cross, pitch_term) for a screw axis (omega, v): q the point on the axis
    nearest the origin, pitch h = omega . v, cross = -omega x q, pitch_term = h omega."""
    omega, v = np.asarray(omega, float), np.asarray(v, float)
    if near_zero(np.linalg.norm(omega)):
        return np.zeros(3), float("inf"), np.zeros(3), v.copy()
    q = np.cross(omega, v)
    h = float(omega @ v)
    return q, h, -np.cross(omega, q), h * omega


def _palette():
    import matplotlib

    return [matplotlib.colormaps["tab10"](k) for k in range(10)]


def _set_equal(ax, points, pad=0.15):
    pts = np.asarray(points, float)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c, r = (lo + hi) / 2, max(float((hi - lo).max()) / 2, 0.1) + pad
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def draw_robot(
    robot: Robot,
    theta=None,
    *,
    frame: str = "space",
    joints=None,
    show=SHOW_DEFAULT,
    ax=None,
    axis_length: float | None = None,
    figsize=(8, 7),
    labels: bool = True,
) -> RobotDrawing:
    """Draw a robot's skeleton at theta (default: home) with its screw axes at home.

    frame="space" draws in {s} with the S_i; frame="body" draws in {b} with the B_i, so
    the tool frame sits at the origin and the base is wherever M^{-1} puts it. The skeleton
    joins the joints' home positions when the robot knows them (a URDF, a scene, or
    joint_frames_home); otherwise only the base and the tool are marked. joints
    selects which joints get axes (indices from 0; default all). show is any of
    "skeleton", "frames", "axes" (the axis lines), "omega" (the omega arrows at q_i),
    "construction" (q_i, -omega_i x q_i, h_i omega_i and v_i at the origin), "labels".
    Returns a RobotDrawing whose joints carry the vectors drawn, for checking by hand.
    MR 4.1.1 and 4.1.3; notes 4.1, 4.2.
    """
    if frame not in ("space", "body"):
        raise ValueError(f'frame must be "space" or "body"; got {frame!r}')
    theta = np.zeros(robot.n) if theta is None else np.asarray(theta, float)
    fig, ax = _axes(ax, figsize)
    T_ref = np.eye(4) if frame == "space" else transform_inv(robot.M)
    axes_mat = robot.S if frame == "space" else robot.B
    known = robot.joint_frames_home is not None
    homes = robot.joint_frames_home or tuple(robot._fallback_joint_frames())
    homes_ref = [T_ref @ F for F in homes]
    posed = [T_ref @ F for F in robot.frames(theta)]  # joint frames then the tool, at theta
    if known:
        skeleton = np.array([T_ref[:3, 3]] + [F[:3, 3] for F in posed])
    else:  # joint positions unknown: only the base and the tool, no made-up links
        skeleton = np.array([T_ref[:3, 3], posed[-1][:3, 3]])
    joints_sel = list(range(robot.n)) if joints is None else list(joints)
    reach = max(float(np.linalg.norm(p)) for p in skeleton) or 1.0
    L = axis_length if axis_length is not None else 0.35 * reach

    if "skeleton" in show:
        if known:
            ax.plot(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2], "-", color=COLOURS["skeleton"], lw=3)
            ax.plot(skeleton[1:-1, 0], skeleton[1:-1, 1], skeleton[1:-1, 2], "o", color=COLOURS["joint"], ms=7)
        ax.plot(skeleton[-1:, 0], skeleton[-1:, 1], skeleton[-1:, 2], "s", color="black", ms=6)
    if "frames" in show:
        _triad(ax, np.eye(4), 0.25 * reach, "{s}" if frame == "space" else "{b}")
        other = T_ref @ robot.M if frame == "space" else T_ref
        _triad(ax, other, 0.18 * reach, "{b} at home" if frame == "space" else "{s}")

    drawing = RobotDrawing(fig, ax, frame, skeleton)
    extents = [skeleton]
    palette = _palette()
    for i in joints_sel:
        omega, v = axes_mat[:3, i].copy(), axes_mat[3:, i].copy()
        kind = robot.joint_types[i]
        q_near, h, cross, pitch_term = _decompose(omega, v)
        q = homes_ref[i][:3, 3].copy() if kind == "revolute" else q_near
        if kind == "revolute":  # the joint's own origin is on the axis; rebuild the pieces from it
            cross = -np.cross(omega, q)
            pitch_term = h * omega
        jd = JointDrawing(i, kind, omega, v, q, h, cross, pitch_term)
        colour = palette[i % len(palette)]
        direction = omega if kind == "revolute" else v / np.linalg.norm(v)
        anchor = q if kind == "revolute" else homes_ref[i][:3, 3]
        if "axes" in show:
            a, b = anchor - L * direction, anchor + L * direction
            (jd.axis_line,) = ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], "--", color=colour, lw=1.4)
            extents.append(np.array([a, b]))
            if labels:  # at the far end of the line, clear of the joint and its neighbours
                name = ("S" if frame == "space" else "B") + str(i + 1)
                ax.text(*b, name, color=colour, fontsize=10, fontweight="bold")
        if "omega" in show and kind == "revolute":
            jd.omega_arrow = _arrow(ax, q, 0.6 * L * omega, colour, lw=2.5)
        if "construction" in show:
            o = np.zeros(3)
            if kind == "revolute":
                (jd.q_line,) = ax.plot([0, q[0]], [0, q[1]], [0, q[2]], ":", color=COLOURS["q"], lw=1.5)
                ax.text(*(q / 2), f"q{i + 1}", color=COLOURS["q"], fontsize=9)
                if near_zero(h):  # v is the cross product itself: one arrow, labelled as both
                    jd.v_arrow = _arrow(ax, o, v, COLOURS["v"], lw=2.5)
                    if labels:
                        ax.text(*(v + 0.02 * reach), f"v{i + 1} = -ω{i + 1}×q{i + 1}", color=COLOURS["v"], fontsize=9)
                else:
                    jd.cross_arrow = _arrow(ax, o, cross, COLOURS["cross"], lw=2.0)
                    jd.pitch_arrow = _arrow(ax, o + cross, pitch_term, COLOURS["pitch"], lw=2.0)
                    jd.v_arrow = _arrow(ax, o, v, COLOURS["v"], lw=2.5)
                    if labels:
                        ax.text(*(cross * 0.5 + 0.02 * reach), f"-ω{i + 1}×q{i + 1}", color=COLOURS["cross"], fontsize=9)
                        ax.text(*(cross + 0.5 * pitch_term), f"h{i + 1}ω{i + 1}", color=COLOURS["pitch"], fontsize=9)
                        ax.text(*(v + 0.02 * reach), f"v{i + 1}", color=COLOURS["v"], fontsize=9)
                extents.append(np.array([cross, v]))
            else:
                jd.v_arrow = _arrow(ax, anchor, 0.6 * L * v, COLOURS["v"], lw=2.5)
                if labels:
                    ax.text(*(anchor + 0.6 * L * v), f"v{i + 1}", color=COLOURS["v"], fontsize=9)
        drawing.joints.append(jd)
    _set_equal(ax, np.vstack(extents))
    ax.view_init(elev=22, azim=-58)
    ax.set_title(f"screw axes in {{{'s' if frame == 'space' else 'b'}}} at home" + ("" if not np.any(theta) else ", arm at θ"))
    return drawing


def draw_screw(S, *, ax=None, figsize=(6, 6), labels: bool = True) -> ScrewDrawing:
    """One screw axis S = (omega, v) on its own: the axis, omega at q, and how v is built
    from q (the point on the axis nearest the origin), -omega x q and the pitch term h omega.
    MR 3.3.2.2, eq. 3.77; notes 3.5.
    """
    S = np.asarray(S, float)
    omega, v = S[:3].copy(), S[3:].copy()
    q, h, cross, pitch_term = _decompose(omega, v)
    fig, ax = _axes(ax, figsize)
    d = ScrewDrawing(fig, ax, omega, v, q, h, cross, pitch_term)
    o = np.zeros(3)
    _triad(ax, np.eye(4), 0.25 * max(0.3, float(np.linalg.norm(q)), float(np.linalg.norm(v))), "{s}")
    if near_zero(np.linalg.norm(omega)):
        d.axis_line = ax.plot([0, v[0]], [0, v[1]], [0, v[2]], "--", color=COLOURS["axis"])[0]
        d.v_arrow = _arrow(ax, o, v, COLOURS["v"], lw=2.5)
        pts = np.array([o, v])
    else:
        size = max(0.3, float(np.linalg.norm(q)), float(np.linalg.norm(v)))
        L = 0.8 * size
        a, b = q - L * omega, q + L * omega
        (d.axis_line,) = ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], "--", color=COLOURS["axis"], lw=1.2)
        d.omega_arrow = _arrow(ax, q, 0.5 * size * omega, COLOURS["omega"], lw=2.5)
        (d.q_line,) = ax.plot([0, q[0]], [0, q[1]], [0, q[2]], ":", color=COLOURS["q"], lw=1.5)
        if near_zero(h):
            d.v_arrow = _arrow(ax, o, v, COLOURS["v"], lw=2.5)
        else:
            d.cross_arrow = _arrow(ax, o, cross, COLOURS["cross"], lw=2.0)
            d.pitch_arrow = _arrow(ax, o + cross, pitch_term, COLOURS["pitch"], lw=2.0)
            d.v_arrow = _arrow(ax, o, v, COLOURS["v"], lw=2.5)
        pts = np.array([a, b, o, q, cross, v, q + 0.5 * size * omega])
        if labels:
            ax.text(*(q + 0.5 * size * omega), "ω", color=COLOURS["omega"])
            ax.text(*(q / 2), "q", color=COLOURS["q"])
            if near_zero(h):
                ax.text(*v, "v = -ω×q", color=COLOURS["v"])
            else:
                ax.text(*(cross * 0.5), "-ω×q", color=COLOURS["cross"])
                ax.text(*(cross + 0.5 * pitch_term), "hω", color=COLOURS["pitch"])
                ax.text(*v, "v", color=COLOURS["v"])
    _set_equal(ax, pts, pad=0.05)
    ax.set_title(f"screw axis (omega, v), pitch h = {h:.3g}" if np.isfinite(h) else "screw axis, infinite pitch")
    return d


def explore(robot: Robot, *, frame: str = "space", show=SHOW_DEFAULT, **kwargs):
    """Joint sliders in a notebook: redraws draw_robot at the slider angles. Needs ipywidgets."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display

    sliders = [
        widgets.FloatSlider(value=0.0, min=-np.pi, max=np.pi, step=0.01, description=f"θ{i + 1}", readout_format=".2f")
        for i in range(robot.n)
    ]
    out = widgets.Output()

    def redraw(*_):
        with out:
            out.clear_output(wait=True)
            d = draw_robot(robot, [s.value for s in sliders], frame=frame, show=show, **kwargs)
            plt.show(d.fig)
            plt.close(d.fig)

    for s in sliders:
        s.observe(redraw, names="value")
    display(widgets.VBox(sliders), out)
    redraw()
    return sliders
