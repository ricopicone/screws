"""Pictures of screw axes: the product of exponentials as a drawing.

draw_robot draws a robot's skeleton, its reference frame and one screw axis per joint, in
the space frame {s} (the S_i) or the body frame {b} (the B_i), and can show how each v_i
is built: the point q_i on the axis, the cross product -omega_i x q_i, the pitch term
h_i omega_i, and their sum v_i. With axes="current" the axes are where the joints have
carried them at theta, the columns of J_s(theta) (or J_b(theta)), and the same drawing
builds the Jacobian column by column. draw_screw does the same for one axis on its own.
explore adds joint sliders in a notebook. Everything needs matplotlib (``screws[plot]``).

One colour per meaning, never one per joint, here and in every picture the package draws.
Red, green and blue are a frame's x, y and z and nothing else. Links are orange and joints
blue, as in the notes; a screw axis, and its omega-hat, is the joints' blue. A velocity is
purple (the pieces of v a lighter purple); a point's position and a path are grey. Labels
tell the joints apart. MR 3.3.2, 4.1, 5.1.1; notes 3.5, 4.1, 4.2, 5.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..robot import Robot
from ..se3 import transform_inv
from ..so3 import near_zero

__all__ = ["JointDrawing", "RobotDrawing", "ScrewDrawing", "draw_robot", "draw_screw", "explore"]

COLOURS = {
    "skeleton": "#F78B11",  # links
    "joint": "#3B83CD",  # joints
    "axis": "#3B83CD",  # a screw axis and its omega-hat: the joints' blue
    "omega": "#3B83CD",
    "velocity": "#8E44AD",  # v, a point's omega x p + v, the velocity a joint gives the tool
    "v": "#8E44AD",
    "cross": "#B585CA",  # the pieces of v, -omega x q and h omega
    "pitch": "#B585CA",
    "q": "#7f7f7f",  # a point's position
    "path": "#8C8C8C",
    "force": "#555555",
    "x": "#c0392b",  # a frame's axes, and nothing else
    "y": "#27ae60",
    "z": "#2980b9",
}
SHOW_DEFAULT = ("skeleton", "frames", "axes", "omega")
AXES = ("home", "current")


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
    """What draw_robot drew: the skeleton, and one JointDrawing per joint drawn (all in the
    drawing's frame)."""

    fig: object
    ax: object
    frame: str
    skeleton: np.ndarray  # (n + 2, 3): base, joint origins, tool, in the drawing's frame
    joints: list[JointDrawing] = field(default_factory=list)
    axes: str = "home"  # "home": the S_i (B_i); "current": the columns of J_s (J_b) at theta


@dataclass
class ScrewDrawing:
    """What draw_screw drew: the axis (omega, v), a point q on it, the pitch, and the two
    terms of v = -omega x q + h omega."""

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


def _arrow(ax, origin, vec, colour, label=None, lw=2.0, **kw):
    origin, vec = np.asarray(origin, float), np.asarray(vec, float)
    if near_zero(np.linalg.norm(vec)):
        return None
    return ax.quiver(*origin, *vec, color=colour, linewidth=lw, arrow_length_ratio=0.15, label=label, **kw)


def _screw_axes(robot: Robot, theta, frame: str, at: str):
    """The joints' screw axes as a drawing shows them: (T_ref, {s} to the drawing's frame;
    the 6 x n axes in it; each joint's origin in it, the point its axis is drawn through).
    at="home": the S_i (B_i) through the joints at home, in {s} ({b} at home).
    at="current": the columns of J_s(theta) (J_b(theta)), J_si = S_i carried by the joints
    inboard of it, through the joints at theta, in {s} ({b} at theta). MR 4.1, 5.1."""
    if frame not in ("space", "body"):
        raise ValueError(f'frame must be "space" or "body"; got {frame!r}')
    if at == "home":
        T_ref = np.eye(4) if frame == "space" else transform_inv(robot.M)
        A = robot.S if frame == "space" else robot.B
        F = robot.joint_frames_home or tuple(robot._fallback_joint_frames())
    elif at == "current":
        F = robot.frames(theta)
        T_ref = np.eye(4) if frame == "space" else transform_inv(F[-1])
        A = robot.jacobian_space(theta) if frame == "space" else robot.jacobian_body(theta)
    else:
        raise ValueError(f'axes must be "home" or "current"; got {at!r}')
    return T_ref, A, np.array([(T_ref @ f)[:3, 3] for f in F[: robot.n]])


def _axis(ax, anchor, direction, length, *, line=True, arrow=True, label=None, zorder=2):
    """One screw axis as every picture draws it: a dashed line in the joints' blue through
    anchor along direction (omega-hat, or v-hat for a prismatic joint), an arrow along
    direction from anchor, and the label at the line's far end, clear of the joint and its
    neighbours. Returns (line, arrow, the line's two ends)."""
    c = COLOURS["axis"]
    ends = np.array([anchor - length * direction, anchor + length * direction])
    drawn = ax.plot(*ends.T, "--", color=c, lw=1.4, zorder=zorder)[0] if line else None
    head = _arrow(ax, anchor, 0.5 * length * direction, c, lw=2.4, zorder=zorder + 3) if arrow else None
    if label:
        ax.text(*ends[1], label, color=c, fontsize=11, zorder=6)
    return drawn, head, ends


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
    axes: str = "home",
    joints=None,
    show=SHOW_DEFAULT,
    ax=None,
    axis_length: float | None = None,
    figsize=(8, 7),
    labels: bool = True,
) -> RobotDrawing:
    """Draw a robot's skeleton at theta (default: home) with one screw axis per joint.

    axes="home" (default) draws the axes at home whatever theta is: the S_i in {s}
    (frame="space"), or the B_i in {b} (frame="body"), where {b} at home sits at the origin
    and the base is wherever M^{-1} puts it. axes="current" draws each axis where the joints
    inboard of it have carried it at theta: the columns of J_s(theta) in {s}, or of
    J_b(theta) in {b} at theta. At home the two are the same picture. draw_jacobian is the
    current one with the velocities the joints give the tool and the matrix beside it.

    The skeleton joins the joints' positions when the robot knows them (a URDF, a scene, or
    joint_frames_home); otherwise only the base and the tool are marked. joints selects
    which joints get axes (indices from 0; default all). show is any of "skeleton",
    "frames", "axes" (the axis lines), "omega" (the omega arrows at q_i), "construction"
    (q_i, -omega_i x q_i, h_i omega_i and v_i at the origin; with axes="current", J_s built
    column by column, J_si = (omega_i, -omega_i x q_i) at zero pitch, MR 5.1.1). Returns a
    RobotDrawing whose joints carry the vectors drawn, for checking by hand.
    MR 4.1.1, 4.1.3, 5.1; notes 4.1, 4.2, 5.1.
    """
    theta = np.zeros(robot.n) if theta is None else np.asarray(theta, float)
    T_ref, axes_mat, origins = _screw_axes(robot, theta, frame, axes)
    fig, ax = _axes(ax, figsize)
    known = robot.joint_frames_home is not None
    posed = [T_ref @ F for F in robot.frames(theta)]  # joint frames then the tool, at theta
    if known:
        skeleton = np.array([T_ref[:3, 3]] + [F[:3, 3] for F in posed])
    else:  # joint positions unknown: only the base and the tool, no made-up links
        skeleton = np.array([T_ref[:3, 3], posed[-1][:3, 3]])
    joints_sel = list(range(robot.n)) if joints is None else list(joints)
    reach = max(float(np.linalg.norm(p)) for p in skeleton) or 1.0
    L = axis_length if axis_length is not None else 0.35 * reach
    sub = "s" if frame == "space" else "b"

    if "skeleton" in show:
        if known:
            ax.plot(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2], "-", color=COLOURS["skeleton"], lw=3)
            ax.plot(skeleton[1:-1, 0], skeleton[1:-1, 1], skeleton[1:-1, 2], "o", color=COLOURS["joint"], ms=7)
        ax.plot(skeleton[-1:, 0], skeleton[-1:, 1], skeleton[-1:, 2], "s", color="black", ms=6)
    if "frames" in show:
        _triad(ax, np.eye(4), 0.25 * reach, f"{{{sub}}}")
        if frame == "body":
            _triad(ax, T_ref, 0.18 * reach, "{s}")
        else:
            _triad(ax, robot.M if axes == "home" else posed[-1], 0.18 * reach, "{b} at home" if axes == "home" else "{b}")

    drawing = RobotDrawing(fig, ax, frame, skeleton, axes=axes)
    extents = [skeleton]
    for i in joints_sel:
        omega, v = axes_mat[:3, i].copy(), axes_mat[3:, i].copy()
        kind = robot.joint_types[i]
        q_near, h, cross, pitch_term = _decompose(omega, v)
        q = origins[i].copy() if kind == "revolute" else q_near
        if kind == "revolute":  # the joint's own origin is on the axis; rebuild the pieces from it
            cross = -np.cross(omega, q)
            pitch_term = h * omega
        jd = JointDrawing(i, kind, omega, v, q, h, cross, pitch_term)
        direction = omega if kind == "revolute" else v / np.linalg.norm(v)
        anchor = q if kind == "revolute" else origins[i]
        k = i + 1
        if axes == "home":
            name = (r"$\mathcal{S}" if frame == "space" else r"$\mathcal{B}") + f"_{{{k}}}$"
        else:
            name = f"$J_{{{sub}{k}}}$"
        jd.axis_line, jd.omega_arrow, ends = _axis(
            ax,
            anchor,
            direction,
            L,
            line="axes" in show,
            arrow="omega" in show and kind == "revolute",
            label=name if labels and "axes" in show else None,
        )
        if "axes" in show:
            extents.append(ends)
        if "construction" in show:
            o = np.zeros(3)
            if kind == "revolute":
                (jd.q_line,) = ax.plot([0, q[0]], [0, q[1]], [0, q[2]], ":", color=COLOURS["q"], lw=1.5)
                ax.text(*(q / 2), f"$q_{{{k}}}$", color=COLOURS["q"], fontsize=10)
                if near_zero(h):  # v is the cross product itself: one arrow, labelled as both
                    jd.v_arrow = _arrow(ax, o, v, COLOURS["v"], lw=2.5)
                    if labels:
                        ax.text(*(v + 0.02 * reach), rf"$v_{{{k}}} = -\omega_{{{k}}} \times q_{{{k}}}$", color=COLOURS["v"], fontsize=10)
                else:
                    jd.cross_arrow = _arrow(ax, o, cross, COLOURS["cross"], lw=2.0)
                    jd.pitch_arrow = _arrow(ax, o + cross, pitch_term, COLOURS["pitch"], lw=2.0)
                    jd.v_arrow = _arrow(ax, o, v, COLOURS["v"], lw=2.5)
                    if labels:
                        ax.text(*(cross * 0.5 + 0.02 * reach), rf"$-\omega_{{{k}}} \times q_{{{k}}}$", color=COLOURS["cross"], fontsize=10)
                        ax.text(*(cross + 0.5 * pitch_term), rf"$h_{{{k}}}\omega_{{{k}}}$", color=COLOURS["pitch"], fontsize=10)
                        ax.text(*(v + 0.02 * reach), f"$v_{{{k}}}$", color=COLOURS["v"], fontsize=10)
                extents.append(np.array([cross, v]))
            else:
                jd.v_arrow = _arrow(ax, anchor, 0.6 * L * v, COLOURS["v"], lw=2.5)
                if labels:
                    ax.text(*(anchor + 0.6 * L * v), f"$v_{{{k}}}$", color=COLOURS["v"], fontsize=10)
        drawing.joints.append(jd)
    _set_equal(ax, np.vstack(extents))
    ax.view_init(elev=22, azim=-58)
    if axes == "home":
        ax.set_title(f"screw axes in {{{sub}}} at home" + ("" if not np.any(theta) else ", arm at θ"))
    else:
        ax.set_title(f"screw axes in {{{sub}}} at θ: the columns of $J_{sub}(\\theta)$")
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
        d.axis_line = ax.plot([0, v[0]], [0, v[1]], [0, v[2]], "--", color=COLOURS["axis"], lw=1.4)[0]
        d.v_arrow = _arrow(ax, o, v, COLOURS["v"], lw=2.5)
        pts = np.array([o, v])
    else:
        size = max(0.3, float(np.linalg.norm(q)), float(np.linalg.norm(v)))
        d.axis_line, d.omega_arrow, ends = _axis(ax, q, omega, 0.8 * size, label=r"$\hat\omega$" if labels else None)
        (d.q_line,) = ax.plot([0, q[0]], [0, q[1]], [0, q[2]], ":", color=COLOURS["q"], lw=1.5)
        if near_zero(h):
            d.v_arrow = _arrow(ax, o, v, COLOURS["v"], lw=2.5)
        else:
            d.cross_arrow = _arrow(ax, o, cross, COLOURS["cross"], lw=2.0)
            d.pitch_arrow = _arrow(ax, o + cross, pitch_term, COLOURS["pitch"], lw=2.0)
            d.v_arrow = _arrow(ax, o, v, COLOURS["v"], lw=2.5)
        pts = np.vstack([ends, [o, q, cross, v]])
        if labels:
            ax.text(*(q / 2), "$q$", color=COLOURS["q"])
            if near_zero(h):
                ax.text(*v, r"$v = -\hat\omega \times q$", color=COLOURS["v"])
            else:
                ax.text(*(cross * 0.5), r"$-\hat\omega \times q$", color=COLOURS["cross"])
                ax.text(*(cross + 0.5 * pitch_term), r"$h\hat\omega$", color=COLOURS["pitch"])
                ax.text(*v, "$v$", color=COLOURS["v"])
    _set_equal(ax, pts, pad=0.05)
    ax.set_title(f"screw axis (omega, v), pitch h = {h:.3g}" if np.isfinite(h) else "screw axis, infinite pitch")
    return d


def _slider_ranges(robot: Robot) -> list[tuple[float, float]]:
    """Each joint's slider range: its limits, else +-pi for a revolute joint and +-1 m for a slide."""
    if robot.joint_limits is not None:
        return [(float(lo), float(hi)) for lo, hi in robot.joint_limits]
    return [(-np.pi, np.pi) if k == "revolute" else (-1.0, 1.0) for k in robot.joint_types]


def explore(robot: Robot, *, draw=None, **kwargs):
    """Joint sliders in a notebook: redraws a drawing at the slider values, by default
    draw_robot; pass draw=viz.draw_jacobian or viz.draw_ellipsoid to watch the Jacobian or
    the ellipsoids change instead. kwargs go to the drawing (frame, show, force, ...).
    Ranges come from joint_limits when the robot has them. Needs ipywidgets."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display

    draw = draw_robot if draw is None else draw
    sliders = [
        widgets.FloatSlider(value=min(max(0.0, lo), hi), min=lo, max=hi, step=0.01, description=f"θ{i + 1}", readout_format=".2f")
        for i, (lo, hi) in enumerate(_slider_ranges(robot))
    ]
    out = widgets.Output()

    def redraw(*_):
        with out:
            out.clear_output(wait=True)
            d = draw(robot, [s.value for s in sliders], **kwargs)
            display(d.fig)  # into the Output widget; plt.show takes no figure
            plt.close(d.fig)

    for s in sliders:
        s.observe(redraw, names="value")
    display(widgets.VBox(sliders), out)
    redraw()
    return sliders
