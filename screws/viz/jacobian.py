"""Jacobians and manipulability ellipsoids, still and moving.

draw_jacobian draws a robot at theta with each column of its Jacobian as what it is: the
joint's screw axis now (J_si(theta) is S_i carried by the joints inboard of it) and the
velocity it gives the tool at unit rate. draw_ellipsoid draws the manipulability ellipsoid,
and the force ellipsoid, at the tool. animate_robot moves the robot through a change of
configuration at a constant rate, joint rates or tool twist, with both following and the
semi-axes charted; compare_paths plays the two kinds of constant rate side by side. Every
robot is drawn in 3D; planar=True draws a planar arm (every joint axis along z, every v in
the plane) flat, in x-hat_1 and x-hat_2, as the notes draw the 2R arm.

One colour per meaning, never per joint, and none of red, green and blue, which are the
frames' x, y and z: links orange and joints blue as in the notes, each joint's axis in the
joints' blue, the velocities the joints give the tool (and the manipulability ellipsoid, the
set of them) in VELOCITY, the force ellipsoid grey and dashed. Labels tell the joints apart.
Needs matplotlib (``screws[plot]``). MR 5.1, 5.4, 9.2; notes 5.1, 5.2, 5.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..kinematics import Ellipsoid, force_ellipsoid, manipulability_ellipsoid
from ..robot import Robot
from ..se3 import exp6, log6, se3_to_vec, transform_inv
from .animate import Animation, _finish, _fmt
from .static import COLOURS, _set_equal

__all__ = [
    "ColumnDrawing",
    "EllipsoidDrawing",
    "JacobianDrawing",
    "animate_robot",
    "compare_paths",
    "draw_ellipsoid",
    "draw_jacobian",
    "is_planar",
]

AXIS = COLOURS["axis"]  # a joint's axis, in the joints' blue
VELOCITY = "#8E44AD"  # the velocities the joints give the tool: not x's red, which is at the tool too
MANIPULABILITY, FORCE = VELOCITY, "#555555"
PATH = "#8C8C8C"
SHOW_JACOBIAN = ("skeleton", "frames", "axes", "columns")
SHOW_ELLIPSOID = ("skeleton", "frames", "ellipsoid")
SHOW_ANIMATION = ("skeleton", "frames", "axes", "ellipsoid", "path")
def is_planar(robot: Robot) -> bool:
    """True when the robot can be drawn flat with planar=True: every joint axis along z and
    every v in the xy plane (Robot.is_planar)."""
    return robot.is_planar()


@dataclass
class ColumnDrawing:
    """One column of the Jacobian as drawn, in the drawing's frame: the column S (of J_s,
    or J_b in the body frame), its axis line (q a point on it, the joint's own origin;
    s_hat; pitch h, inf for a prismatic joint), and the velocity it gives the tool point at
    unit rate, R J_bv[:, i] (the column itself, in 2D, of the notes' J(theta))."""

    index: int
    kind: str
    S: np.ndarray
    q: np.ndarray
    s_hat: np.ndarray
    h: float
    tool_velocity: np.ndarray
    axis_line: object = None
    arrow: object = None


@dataclass
class JacobianDrawing:
    fig: object
    ax: object
    frame: str
    theta: np.ndarray
    T_sb: np.ndarray
    J: np.ndarray  # the matrix printed: J_s, J_b, or the 2 x n tip Jacobian of a planar arm
    rank: int
    singular: bool
    planar: bool
    columns: list[ColumnDrawing] = field(default_factory=list)
    velocity: np.ndarray | None = None  # the tool point's velocity for dtheta, drawn head to tail
    arrow_scale: float = 1.0


@dataclass
class EllipsoidDrawing:
    fig: object
    ax: object
    theta: np.ndarray
    T_sb: np.ndarray
    kind: str
    planar: bool
    manipulability: Ellipsoid  # in the drawing's coordinates: {s}, or x-hat_1, x-hat_2 when planar
    force: Ellipsoid | None
    scale: float  # drawing metres per unit of semi-axis (m/s, rad/s)
    force_scale: float
    clipped: bool  # some semi-axis was longer than the plot and was cut short


@dataclass
class _Pose:
    """One configuration, in the drawing's coordinates (3D, or the first two for planar)."""

    theta: np.ndarray
    T: np.ndarray  # T_sb, always in {s}
    skeleton: np.ndarray
    joints: np.ndarray  # (n, d) each joint's origin
    lines: list  # per joint (S column in the drawing's frame, s_hat, h)
    C: np.ndarray  # (d, n) tool velocity per unit rate of each joint
    J: np.ndarray  # the matrix to print
    Js: np.ndarray
    Jb: np.ndarray
    tool: np.ndarray
    ref: np.ndarray  # {s} -> drawing coordinates
    manip: Ellipsoid | None = None
    force: Ellipsoid | None = None
    velocity: np.ndarray | None = None


class _RobotPainter:
    """Draws a robot at each configuration of a path with its Jacobian columns and
    ellipsoids, on 3D axes or, for a planar arm, 2D; one set of limits for the whole path."""

    def __init__(
        self,
        robot: Robot,
        thetas,
        *,
        frame="space",
        planar=False,
        kind="linear",
        show=SHOW_JACOBIAN,
        dthetas=None,
        scale=None,
        force_scale=None,
        labels=True,
        view=None,
    ):
        if frame not in ("space", "body"):
            raise ValueError(f'frame must be "space" or "body"; got {frame!r}')
        if kind not in ("linear", "angular"):
            raise ValueError(f'kind must be "linear" or "angular"; got {kind!r}')
        self.robot, self.frame, self.kind, self.show, self.labels = robot, frame, kind, tuple(show), labels
        self.planar = bool(planar)
        if self.planar and not robot.is_planar():
            raise ValueError(
                f"planar=True draws a planar arm flat, and {robot.name} is not one: its joint axes "
                "are not all along z with v in the xy plane"
            )
        if self.planar and kind == "angular":
            raise ValueError(
                "a planar arm turns its tool about z only: its angular ellipsoid is a segment, "
                "not an ellipse; draw it in 3D with planar=False"
            )
        self.d = 2 if self.planar else 3
        self.thetas = np.atleast_2d(np.asarray(thetas, float))
        self.dthetas = None if dthetas is None else np.atleast_2d(np.asarray(dthetas, float))
        self.known = robot.joint_frames_home is not None
        self.view = view or (22, -58)
        self.poses = [self._pose(k) for k in range(len(self.thetas))]
        base = np.vstack([p.skeleton for p in self.poses])
        self.reach = max(float(np.linalg.norm(base - base[:1], axis=1).max()), 1e-6)
        top = max((float(np.max(p.manip.radii)) for p in self.poses), default=0.0)
        self.scale = float(scale) if scale is not None else (0.3 * self.reach / top if top > 0 else 1.0)
        # the force ellipsoid's own units: its longest bounded semi-axis 0.3 of the reach in a
        # typical frame (the median), so the spikes near a singularity are cut short, not the rest
        tops = [float(np.max(r[np.isfinite(r)], initial=0.0)) for r in (p.force.radii for p in self.poses)]
        ftop = float(np.median(tops)) if tops else 0.0
        if force_scale is not None:
            self.force_scale = float(force_scale)
        else:
            self.force_scale = 0.3 * self.reach / ftop if ftop > 0 else self.scale
        self.cap = 0.9 * self.reach  # a drawn semi-axis longer than this is cut short and said so
        cmax = max(float(np.linalg.norm(p.C, axis=0).max()) for p in self.poses) or 1.0
        self.column_scale = 0.5 * self.reach / cmax
        if self.dthetas is not None:
            sums = [np.linalg.norm(np.cumsum(p.C * th, axis=1), axis=0).max() for p, th in zip(self.poses, self.dthetas)]
            self.velocity_scale = 0.6 * self.reach / (max(sums) or 1.0)
        self.limits = self._limits()

    # ----- numbers ---------------------------------------------------------------------

    def _pose(self, k) -> _Pose:
        robot, theta = self.robot, self.thetas[k]
        F = robot.frames(theta)
        T = F[-1]
        ref = np.eye(4) if self.frame == "space" else transform_inv(T)
        R = ref[:3, :3]
        d = self.d

        def pt(p):
            return (ref[:3, :3] @ p + ref[:3, 3])[:d]

        Js, Jb = robot.jacobian_space(theta), robot.jacobian_body(theta)
        joints = np.array([pt(f[:3, 3]) for f in F[:-1]])
        tool, origin = pt(T[:3, 3]), pt(np.zeros(3))
        skeleton = np.vstack([origin, joints, tool]) if self.known else np.array([origin, tool])
        J = Js if self.frame == "space" else Jb
        lines = []
        for i in range(robot.n):
            omega, v = Js[:3, i], Js[3:, i]
            if robot.joint_types[i] == "prismatic":
                lines.append((J[:, i], (R @ v)[:d] / np.linalg.norm(v), np.inf))
            else:
                lines.append((J[:, i], (R @ omega)[:d], float(omega @ v)))
        C = (R @ T[:3, :3] @ Jb[3:])[:d]
        block = C if self.kind == "linear" else (R @ Js[:3])[:d]
        pose = _Pose(theta, T, skeleton, joints, lines, C, C if self.planar else J, Js, Jb, tool, ref)
        pose.manip = manipulability_ellipsoid(block, center=tool)
        pose.force = force_ellipsoid(block, center=tool)
        if self.dthetas is not None:
            pose.velocity = C @ self.dthetas[k]
        return pose

    def _drawn(self, e: Ellipsoid, scale) -> tuple[np.ndarray, np.ndarray]:
        """(drawn semi-axes, which were cut short), inf drawn at the cap."""
        r = np.where(np.isinf(e.radii), np.inf, e.radii * scale)
        cut = r > self.cap
        return np.minimum(r, self.cap), cut

    def _limits(self) -> np.ndarray:
        pts, L = [], 0.35 * self.reach
        for k, p in enumerate(self.poses):
            pts.append(p.skeleton)
            if "frames" in self.show:  # the triads' tips, or an annotation arrow past the edge is dropped
                for T in (np.eye(4), p.T):
                    F = p.ref @ T
                    pts.append((F[:3, 3][:, None] + 0.15 * self.reach * F[:3, :3]).T[:, : self.d])
            if "axes" in self.show and not self.planar:
                for (_, s_hat, _), q in zip(p.lines, p.joints):
                    pts += [q[None] - L * s_hat, q[None] + L * s_hat]
            for e, sc, item in ((p.manip, self.scale, "ellipsoid"), (p.force, self.force_scale, "force")):
                if item in self.show:
                    r, _ = self._drawn(e, sc)
                    pts += [e.center + (e.axes * r).T, e.center - (e.axes * r).T]
            if "columns" in self.show:
                if self.dthetas is not None:
                    pts.append(p.tool + self.velocity_scale * np.cumsum(p.C * self.dthetas[k], axis=1).T)
                else:
                    pts.append(p.tool + self.column_scale * p.C.T)
            if "velocity" in self.show and p.velocity is not None:
                pts.append(p.tool + self.velocity_scale * p.velocity[None])
        return np.vstack(pts)

    # ----- drawing ---------------------------------------------------------------------

    def axes(self, fig, rect=None):
        if self.planar:
            return fig.add_axes(rect) if rect else fig.add_subplot()
        return fig.add_axes(rect, projection="3d") if rect else fig.add_subplot(projection="3d")

    def draw(self, ax, k, *, title=None, trace=None):
        p = self.poses[k]
        ax.cla()
        if not self.planar:
            ax.computed_zorder = False  # arrows and labels over the ellipsoid, always
        show = self.show
        if "path" in show and trace is not None:
            ax.plot(*trace.T, ":", color=PATH, lw=1.0)
            ax.plot(*trace[: k + 1].T, "-", color=PATH, lw=1.8)
        if "skeleton" in show:
            self._skeleton(ax, p)
        if "frames" in show:
            self._frames(ax, p)
        if "axes" in show:
            self._axes(ax, p)
        if "force" in show:
            self._ellipsoid(ax, p.force, self.force_scale, FORCE, dashed=True)
        if "ellipsoid" in show:
            self._ellipsoid(ax, p.manip, self.scale, MANIPULABILITY, dashed=False)
        if "columns" in show:
            self._columns(ax, p, None if self.dthetas is None else self.dthetas[k])
        if "velocity" in show and p.velocity is not None and "columns" not in show:
            label = (r"$\dot{p}$" if self.frame == "space" else "$v_b$") if self.labels else None
            self._arrow(ax, p.tool, self.velocity_scale * p.velocity, "black", 2.6, label)
        self._finish_axes(ax, title)

    def _finish_axes(self, ax, title):
        if self.planar:
            lo, hi = self.limits.min(axis=0), self.limits.max(axis=0)
            pad = 0.08 * float((hi - lo).max())
            ax.set_xlim(lo[0] - pad, hi[0] + pad)
            ax.set_ylim(lo[1] - pad, hi[1] + pad)
            ax.set_aspect("equal")
            ax.axis("off")
        else:
            _set_equal(ax, self.limits, pad=0.05 * self.reach)
            ax.set_box_aspect((1, 1, 1), zoom=1.05)
            ax.view_init(*self.view)
        if title:
            ax.set_title(title, fontsize=11)

    def _arrow(self, ax, o, vec, colour, lw, label=None, alpha=1.0):
        o, vec = np.asarray(o, float), np.asarray(vec, float)
        if np.linalg.norm(vec) < 1e-12:
            return None
        if self.planar:
            art = ax.annotate(
                "",
                xy=o + vec,
                xytext=o,
                arrowprops={"arrowstyle": "-|>", "color": colour, "lw": lw, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 12, "alpha": alpha},
                zorder=5,
                annotation_clip=False,
            )
        else:
            art = ax.quiver(*o, *vec, color=colour, linewidth=lw, arrow_length_ratio=0.15, zorder=5, alpha=alpha)
        if label:
            ax.text(*(o + 1.08 * vec), label, color=colour, fontsize=10, zorder=6)
        return art

    def _skeleton(self, ax, p):
        sk = p.skeleton
        if self.planar:
            b = sk[0]
            ax.plot([b[0] - 0.12 * self.reach, b[0] + 0.12 * self.reach], [b[1] - 0.03 * self.reach] * 2, color="black", lw=1.2)
        if self.known:
            ax.plot(*sk.T, "-", color=COLOURS["skeleton"], lw=3.5 if self.planar else 3, solid_capstyle="round", zorder=2)
            joints = sk[1:-1]
            if self.planar:
                ax.plot(*joints.T, "o", ms=9, mfc="white", mec=COLOURS["joint"], mew=1.6, zorder=3, ls="none")
            else:
                ax.plot(*joints.T, "o", color=COLOURS["joint"], ms=7, ls="none", zorder=3)
        ax.plot(*sk[-1:].T, "o" if self.planar else "s", color="black", ms=4 if self.planar else 6, zorder=4)

    def _frames(self, ax, p):
        length = 0.15 * self.reach
        for T, label in ((np.eye(4), "{s}"), (p.T, "{b}")):
            F = p.ref @ T
            o = F[:3, 3][: self.d]
            for k, c in enumerate("xyz"[: self.d]):
                self._arrow(ax, o, length * F[:3, k][: self.d], COLOURS[c], 1.4)
            below = -F[:3, 2] if not self.planar else np.array([-0.6, -1.0])  # under the triad
            ax.text(*(o + 0.35 * length * below), label, fontsize=9, ha="center", va="top")

    def _axes(self, ax, p):
        L = 0.35 * self.reach
        sub = "s" if self.frame == "space" else "b"
        colour = AXIS
        for i, ((_, s_hat, h), q) in enumerate(zip(p.lines, p.joints)):
            if self.planar:  # every axis points out of the page: a dot for +z, a cross for -z
                if self.robot.joint_types[i] == "revolute":
                    ax.plot(*q, "." if p.Js[2, i] > 0 else "x", color=colour, ms=6, zorder=4)
                continue
            a, b = q - L * s_hat, q + L * s_hat
            ax.plot(*np.array([a, b]).T, "--", color=colour, lw=1.4, zorder=2)
            if np.isfinite(h):
                self._arrow(ax, q, 0.5 * L * s_hat, colour, 2.4)
            if self.labels:
                ax.text(*b, f"$J_{{{sub}{i + 1}}}$", color=colour, fontsize=11, zorder=6)

    def _columns(self, ax, p, dtheta):
        tool = p.tool
        if dtheta is None:
            for i in range(self.robot.n):
                label = (f"$J_{{{i + 1}}}$" if self.planar else f"{i + 1}") if self.labels else None
                self._arrow(ax, tool, self.column_scale * p.C[:, i], VELOCITY, 2.0, label)
            return
        tail = tool.copy()
        for i in range(self.robot.n):
            step = self.velocity_scale * p.C[:, i] * dtheta[i]
            if self._arrow(ax, tail, step, VELOCITY, 2.0) is not None and self.labels:  # at the middle, to one side
                side = np.array([step[1], -step[0]]) if self.planar else np.zeros(3)
                side = 0.04 * self.reach * side / (np.linalg.norm(side) or 1.0)
                name = f"$J_{{{i + 1}}}\\dot\\theta_{{{i + 1}}}$" if self.planar else f"{i + 1}"
                ax.text(*(tail + 0.5 * step + side), name, color=VELOCITY, fontsize=10, ha="left", va="center", zorder=6)
            tail = tail + step
        label = (r"$\dot{p}$" if self.frame == "space" else "$v_b$") if self.labels else None
        self._arrow(ax, tool, tail - tool, "black", 2.6, label)

    def _ellipsoid(self, ax, e: Ellipsoid, scale, colour, dashed):
        r, cut = self._drawn(e, scale)
        unbounded = np.isinf(e.radii)
        ls = "--" if dashed else "-"
        finite = ~unbounded
        shown = Ellipsoid(e.center, e.axes, e.eigenvalues, np.where(finite, r, 0.0), e.kind)
        if self.planar:
            if finite.all():
                ax.plot(*shown.outline(160).T, ls, color=colour, lw=1.6, zorder=3)
            else:  # the bounded section: a segment along the finite axis
                u = e.axes[:, finite][:, 0] * r[finite][0] if finite.any() else np.zeros(2)
                ax.plot(*np.array([e.center - u, e.center + u]).T, ls, color=colour, lw=1.6, zorder=3)
        elif finite.all():
            X, Y, Z = shown.surface(28)
            if dashed:
                ax.plot_wireframe(X, Y, Z, color=colour, lw=0.4, alpha=0.4, rstride=4, cstride=4, zorder=3)
            else:
                ax.plot_surface(X, Y, Z, color=colour, alpha=0.22, linewidth=0, shade=True, zorder=3)
        else:  # the bounded cross-section through the finite axes
            t = np.linspace(0, 2 * np.pi, 120)
            A = e.axes[:, finite] * r[finite]
            ring = np.vstack([np.cos(t), np.sin(t)])[: A.shape[1]]
            pts = e.center + (A @ ring).T
            ax.plot(*pts.T, ls, color=colour, lw=1.4, zorder=3)
        for k in range(len(r)):  # the principal semi-axes
            u = e.axes[:, k] * r[k]
            style = ":" if (unbounded[k] or cut[k]) else ls
            if r[k] > 0:
                ax.plot(*np.array([e.center - u, e.center + u]).T, style, color=colour, lw=1.1, zorder=3)
            if (unbounded[k] or cut[k]) and self.labels:
                ax.text(*(e.center + 1.04 * u), "∞" if unbounded[k] else "…", color=colour, fontsize=11, zorder=6)


# ----- the matrix beside the robot ------------------------------------------------------


class _MatrixPanel:
    """J printed at (x, y) in figure coordinates."""

    LABEL = 7  # the name and "=", in character widths

    def __init__(self, fig, x, y, J, name, fontsize=10.5):
        m, n = J.shape
        self.prec = 3 if m <= 3 else 2
        w = self.prec + 5
        chars = self.LABEL + n * (w + 1)  # a monospace glyph is about 0.6 em wide
        fontsize = min(fontsize, (0.985 - x) * fig.get_figwidth() * 72 / (0.6 * chars))
        cw = 0.6 * fontsize / 72 / fig.get_figwidth()  # one character, in figure fractions
        fig.text(x, y, f"{name} =", fontsize=fontsize + 1, va="center", ha="left")
        x0, dx = x + self.LABEL * cw, (w + 1) * cw
        self.texts = []
        for j in range(n):
            t = fig.text(x0 + j * dx, y, "", family="monospace", fontsize=fontsize, va="center", ha="left")
            self.texts.append(t)
        self.w = w
        self.update(J)

    def update(self, J):
        for j, t in enumerate(self.texts):
            t.set_text("\n".join(_fmt(x, self.w, self.prec) for x in J[:, j]))


def _matrix_inches(J, fontsize=10.5) -> float:
    """The width a _MatrixPanel of J needs at fontsize, with a margin."""
    w = (3 if J.shape[0] <= 3 else 2) + 5
    return (_MatrixPanel.LABEL + J.shape[1] * (w + 1)) * 0.6 * fontsize / 72 + 0.35


def _joint_shades(n) -> list:
    """n shades of the joints' blue, light at the base to dark at the tool, for charts."""
    import matplotlib.colors as mcolors

    base = np.array(mcolors.to_rgb(AXIS))
    return [tuple(map(float, 1 - (1 - base) * f if f <= 1 else base * (2 - f))) for f in np.linspace(0.45, 1.45, n)]


def _rank(p: _Pose, planar) -> tuple[int, bool]:
    M = p.C if planar else p.Js
    r = int(np.linalg.matrix_rank(M, tol=1e-9))
    return r, r < min(M.shape)


def _units(robot, kind, force) -> str:
    if any(t != "revolute" for t in robot.joint_types):
        return ""  # mixed joints: the semi-axes mix units (notes 5.5)
    return {("linear", False): " m/s", ("angular", False): " rad/s", ("linear", True): " N", ("angular", True): " N·m"}[(kind, force)]


def _radii_text(e: Ellipsoid, units) -> str:
    return ", ".join(_num(x) for x in e.radii) + units


def _num(x) -> str:
    return "∞" if np.isinf(x) else f"{x:.3g}"


def _measures_text(e: Ellipsoid) -> str:
    mu1, mu2, mu3 = e.measures
    return f"μ1 = {_num(mu1)}, μ2 = {_num(mu2)}, μ3 = {_num(mu3)}"


def _theta_text(robot, theta) -> str:
    parts = [f"{np.rad2deg(t):.0f}°" if k == "revolute" else f"{t:.2f} m" for t, k in zip(theta, robot.joint_types)]
    return "θ = (" + ", ".join(parts) + ")"


# ----- still pictures -------------------------------------------------------------------


def draw_jacobian(
    robot: Robot,
    theta,
    *,
    frame: str = "space",
    dtheta=None,
    show=SHOW_JACOBIAN,
    matrix: bool = True,
    planar: bool = False,
    ax=None,
    figsize=None,
    labels: bool = True,
    view=None,
) -> JacobianDrawing:
    """The robot at theta with each column of its Jacobian drawn as what it is.

    Column i of J_s(theta) is joint i's screw axis now, S_i carried by the joints inboard of
    it; it is drawn as a dashed line through the joint in the joints' blue (with its
    omega-hat for a revolute joint), labelled Js_i. At the tool, an arrow per joint, numbered,
    gives the velocity of the tool point when that joint alone turns at unit rate,
    R_sb J_bv[:, i]. With dtheta the arrows are scaled by dtheta_i and laid head to tail,
    summing to the tool point's velocity p-dot. frame="body" draws in {b} and prints J_b;
    the lines are the same lines, seen from the tool. planar=True draws a planar arm flat:
    the arrows are then the columns J_i(theta) of the notes' 2 x n J(theta), which is the
    matrix printed, and the joint axes, all out of the page, are marked in the joint circles.
    The matrix is printed beside the robot (only when this function makes the figure). The
    title gives the rank, and says singular when it is below min(6, n) (min(2, n) in 2D).
    MR 5.1.1-5.1.2, 5.3; notes 5.1, 5.2, 5.4.
    """
    import matplotlib.pyplot as plt

    theta = np.asarray(theta, float)
    painter = _RobotPainter(
        robot, [theta], frame=frame, planar=planar, show=show, labels=labels, view=view,
        dthetas=None if dtheta is None else [np.asarray(dtheta, float)],
    )
    p = painter.poses[0]
    with_matrix = matrix and ax is None
    if ax is None:
        figsize = figsize or (7 + (_matrix_inches(p.J) if with_matrix else 0), 6.4)
        fig = plt.figure(figsize=figsize)
        split = min(7 / figsize[0], 0.7) if with_matrix else 1.0
        ax = painter.axes(fig, [0.0, 0.02, split, 0.94] if with_matrix else None)
    else:
        fig = ax.figure
    rank, singular = _rank(p, painter.planar)
    name = "J" if painter.planar else ("J_s" if frame == "space" else "J_b")
    title = f"{robot.name}: ${name}(\\theta)$, rank {rank}" + (", singular" if singular else "")
    painter.draw(ax, 0, title=title)
    if with_matrix:
        _MatrixPanel(fig, split, 0.5, p.J, f"${name}(\\theta)$")
        fig.text(split, 0.08, _theta_text(robot, theta), family="monospace", fontsize=10)
    columns = []
    for i, ((S, s_hat, h), q) in enumerate(zip(p.lines, p.joints)):
        cd = ColumnDrawing(i, robot.joint_types[i], np.asarray(S), q, s_hat, h, p.C[:, i])
        columns.append(cd)
    scale = painter.column_scale if dtheta is None else painter.velocity_scale
    return JacobianDrawing(fig, ax, frame, theta, p.T, p.J, rank, singular, painter.planar, columns, p.velocity, scale)


def draw_ellipsoid(
    robot: Robot,
    theta,
    *,
    kind: str = "linear",
    force: bool = False,
    show=None,
    scale=None,
    force_scale=None,
    planar: bool = False,
    ax=None,
    figsize=None,
    labels: bool = True,
    view=None,
) -> EllipsoidDrawing:
    """The robot at theta with its manipulability ellipsoid at the tool (in VELOCITY, its
    principal semi-axes drawn) and, with force=True, the force ellipsoid (grey, dashed).

    kind="linear" is the velocity of the tool point, from J_b's linear rows as MR advises,
    turned into {s} so it sits where the tool is; kind="angular" is the angular velocity,
    from J_s's angular rows. scale is drawing metres per unit of semi-axis (m/s, or rad/s);
    the default makes the longest semi-axis 0.3 of the robot's reach. force_scale has its own
    default, the same rule for the force ellipsoid; pass force_scale=scale to draw both at one
    scale, as the notes do. A force ellipsoid at a singularity is unbounded along the
    lost direction: its bounded section is drawn and that axis is dotted and marked ∞; a
    semi-axis longer than the plot is cut short and marked …. The title gives the semi-axes
    and mu1, mu2, mu3. For a chain mixing revolute and prismatic joints the semi-axes mix
    units and no unit is printed (notes 5.5). Every robot is drawn in 3D, where a planar
    arm's ellipsoid is flat and its mu are infinite; planar=True draws it flat, the ellipse of
    the notes' 2 x n J(theta). MR 5.4; notes 5.1, 5.5.
    """
    import matplotlib.pyplot as plt

    theta = np.asarray(theta, float)
    if show is None:
        show = SHOW_ELLIPSOID + (("force",) if force else ())
    elif force and "force" not in show:
        show = tuple(show) + ("force",)
    painter = _RobotPainter(
        robot, [theta], planar=planar, kind=kind, show=show, scale=scale, force_scale=force_scale, labels=labels, view=view
    )
    p = painter.poses[0]
    if ax is None:
        fig = plt.figure(figsize=figsize or (7, 6.4))
        ax = painter.axes(fig)
    else:
        fig = ax.figure
    units = _units(robot, kind, False)
    title = f"{_theta_text(robot, theta)}\nsemi-axes {_radii_text(p.manip, units)}; {_measures_text(p.manip)}"
    if force:
        title += f"\nforce semi-axes {_radii_text(p.force, _units(robot, kind, True))}"
    painter.draw(ax, 0, title=title)
    clipped = bool(painter._drawn(p.manip, painter.scale)[1].any())
    if force:
        clipped |= bool(painter._drawn(p.force, painter.force_scale)[1][np.isfinite(p.force.radii)].any())
    return EllipsoidDrawing(
        fig, ax, theta, p.T, kind, painter.planar, p.manip, p.force if force else None, painter.scale, painter.force_scale, clipped
    )


# ----- configuration changes ------------------------------------------------------------


def _screw_path(robot: Robot, theta_start, theta_end, s) -> tuple[np.ndarray, np.ndarray]:
    """The joint values that carry the tool along the constant-twist screw motion from
    T_sb(theta_start) to T_sb(theta_end), T(s) = T_start e^{[V_b] s}, by Newton-Raphson
    IK seeded from the previous frame, and the joint rates J_b^+ V_b that keep the twist
    V_b (per unit s) fixed. Returns (thetas, dthetas)."""
    T0, T1 = robot.fk(theta_start), robot.fk(theta_end)
    L = log6(transform_inv(T0) @ T1)
    V_b = se3_to_vec(L)
    thetas, dthetas, prev = [], [], np.asarray(theta_start, float)
    for k, sk in enumerate(s):
        T = T0 @ exp6(L * sk)
        r = robot.ik(T, prev, tol_omega=1e-10, tol_v=1e-10, max_iter=50)
        if not r.converged:
            why = (
                f"; with {robot.n} joints the tool cannot follow an arbitrary screw motion"
                if np.linalg.matrix_rank(robot.jacobian_body(prev), tol=1e-9) < 6
                else "; try more frames, or path='joint'"
            )
            raise ValueError(f"inverse kinematics lost the screw path at s = {sk:.3f} (frame {k}){why}")
        thetas.append(r.theta)
        dthetas.append(np.linalg.pinv(robot.jacobian_body(r.theta)) @ V_b)
        prev = r.theta
    return np.array(thetas), np.array(dthetas)


def _path(robot, theta_start, theta_end, path, frames):
    """(s, thetas, dthetas) for a change of configuration; dthetas are per unit s."""
    start = np.asarray(theta_start, float)
    if theta_end is None:
        if start.ndim != 2 or start.shape[1] != robot.n or len(start) < 2:
            raise ValueError("pass theta_start and theta_end, or one (N, n) path of N >= 2 configurations")
        s = np.linspace(0, 1, len(start))
        return s, start, np.gradient(start, s, axis=0)
    end = np.asarray(theta_end, float)
    s = np.linspace(0, 1, frames)
    if path == "joint":  # constant joint rates: a straight line in joint space
        return s, start + np.outer(s, end - start), np.tile(end - start, (frames, 1))
    if path == "screw":  # a constant twist of the tool; the joints follow
        thetas, dthetas = _screw_path(robot, start, end, s)
        return s, thetas, dthetas
    raise ValueError(f'path must be "joint" or "screw"; got {path!r}')


def _record(painter: _RobotPainter, s, dthetas) -> dict:
    P = painter.poses
    Js = np.array([p.Js for p in P])
    Jb = np.array([p.Jb for p in P])
    return {
        "s": s,
        "theta": painter.thetas,
        "dtheta": dthetas,
        "T_sb": np.array([p.T for p in P]),
        "J_s": Js,
        "J_b": Jb,
        "V_s": np.einsum("kij,kj->ki", Js, dthetas),
        "V_b": np.einsum("kij,kj->ki", Jb, dthetas),
        "tool": np.array([p.T[:3, 3] for p in P]),
        "radii": np.array([p.manip.radii for p in P]),
        "measures": np.array([p.manip.measures for p in P]),
    }


def animate_robot(
    robot: Robot,
    theta_start,
    theta_end=None,
    *,
    path: str = "joint",
    frames: int = 60,
    fps: int = 20,
    show=SHOW_ANIMATION,
    kind: str = "linear",
    force: bool = False,
    chart: bool = True,
    matrix: bool = False,
    planar: bool = False,
    scale=None,
    view=None,
    figsize=None,
) -> Animation:
    """The robot moving from theta_start to theta_end at a constant rate, its Jacobian and
    ellipsoid following, the ellipsoid's semi-axes charted below.

    path="joint" (default) moves every joint at a constant rate, a straight line in joint
    space; path="screw" moves the tool at a constant twist, the screw motion from
    T_sb(theta_start) to T_sb(theta_end), the joints following by inverse kinematics seeded
    from the frame before. Or pass one (N, n) array of configurations as theta_start, such
    as a joint_trajectory or an IKResult.history, and it is played as given. show takes the
    items of draw_jacobian and draw_ellipsoid plus "path" (the tool's path) and "velocity"
    (the tool point's velocity, d/ds of its position). The chart plots the semi-axes against
    s from 0 to 1, so a singularity shows as one going to zero. matrix prints J_s(theta)
    beside the robot (J with planar=True, which draws a planar arm flat). .data holds s,
    theta, dtheta (per unit s), T_sb, J_s, J_b, V_s, V_b, tool, radii and measures for every
    frame. MR 5.4, 9.2; notes 5.5.
    """
    import matplotlib.pyplot as plt

    if force and "force" not in show:
        show = tuple(show) + ("force",)
    s, thetas, dthetas = _path(robot, theta_start, theta_end, path, frames)
    painter = _RobotPainter(robot, thetas, planar=planar, kind=kind, show=show, dthetas=dthetas, scale=scale, view=view)
    data = _record(painter, s, dthetas)
    trace = np.array([p.tool for p in painter.poses])
    if figsize is None:
        figsize = (7.2 + (_matrix_inches(painter.poses[0].J) if matrix else 0), 7.6 if chart else 6.2)
    mw = min(7.2 / figsize[0], 0.7) if matrix else 1.0
    fig = plt.figure(figsize=figsize)
    top = (0.30 if painter.planar else 0.34) if chart else 0.08
    ax = painter.axes(fig, [0.0, top, mw, 0.96 - top])
    caption = fig.text(0.5 * mw, top - 0.03, "", ha="center", va="top", family="monospace", fontsize=9.5)
    cursor = None
    if chart:
        cax = fig.add_axes([0.11, 0.06, mw - 0.17, 0.13])
        units = _units(robot, kind, False)
        for k in range(data["radii"].shape[1]):
            cax.plot(s, data["radii"][:, k], color=MANIPULABILITY, lw=1.5, alpha=0.45 + 0.55 * k / max(1, data["radii"].shape[1] - 1))
        cax.set_xlim(0, 1)
        cax.set_ylim(0, 1.08 * max(float(data["radii"].max()), 1e-9))
        cax.set_xlabel("s, the fraction of the move", fontsize=9)
        cax.set_ylabel(f"semi-axes{units}", fontsize=9)
        cax.tick_params(labelsize=8)
        cursor = cax.axvline(0, color="black", lw=1.0)
    panel = None
    if matrix:
        name = r"$J(\theta)$" if painter.planar else r"$J_s(\theta)$"
        panel = _MatrixPanel(fig, mw, 0.62, painter.poses[0].J, name)
    how = {"joint": "constant joint rates", "screw": "constant twist of the tool"}
    title = f"{robot.name}: " + (how[path] if theta_end is not None else "a given path")

    def update(k):
        p = painter.poses[k]
        painter.draw(ax, k, title=title, trace=trace)
        rank, singular = _rank(p, painter.planar)
        caption.set_text(
            f"{_theta_text(robot, p.theta)}\n{_measures_text(p.manip)}"
            + (f"\nrank {'J' if painter.planar else 'J_s'} = {rank}, singular" if singular else "")
        )
        if cursor is not None:
            cursor.set_xdata([s[k], s[k]])
        if panel is not None:
            panel.update(p.J)

    return _finish(fig, update, len(s), fps, data)


def compare_paths(
    robot: Robot,
    theta_start,
    theta_end,
    *,
    frames: int = 60,
    fps: int = 20,
    show=("skeleton", "frames", "path", "velocity"),
    planar: bool = False,
    view=None,
    figsize=(12.5, 7.6),
) -> Animation:
    """The same change of configuration made two ways, side by side. Left, constant joint
    rates: theta-dot is fixed and the tool's twist V_b = J_b(theta) theta-dot changes, so
    the tool does not follow a screw. Right, a constant twist: the tool follows the one screw
    motion from start to end (a helix about its axis) and the joint rates change, found by
    inverse kinematics. Each panel charts the quantity that changes; the caption gives the
    one held fixed. .data["joint"] and .data["screw"] hold each side's numbers, as
    animate_robot's. MR 9.2, eqs. 9.7 and 9.14; notes 3.10, 9.2.
    """
    import matplotlib.pyplot as plt

    sj, thj, dj = _path(robot, theta_start, theta_end, "joint", frames)
    ss, ths, ds = _path(robot, theta_start, theta_end, "screw", frames)
    left = _RobotPainter(robot, thj, planar=planar, show=show, dthetas=dj, view=view)
    right = _RobotPainter(robot, ths, planar=planar, show=show, dthetas=ds, view=view)
    right.velocity_scale = left.velocity_scale = min(left.velocity_scale, right.velocity_scale)
    left.limits = right.limits = np.vstack([left._limits(), right._limits()])
    dl, dr = _record(left, sj, dj), _record(right, ss, ds)
    traces = [np.array([p.tool for p in pt.poses]) for pt in (left, right)]
    fig = plt.figure(figsize=figsize)
    axl = left.axes(fig, [0.0, 0.38, 0.5, 0.53])
    axr = right.axes(fig, [0.5, 0.38, 0.5, 0.53])
    cl = fig.add_axes([0.07, 0.11, 0.38, 0.14])
    cr = fig.add_axes([0.57, 0.11, 0.38, 0.14])
    names = [r"$\omega_1$", r"$\omega_2$", r"$\omega_3$", "$v_1$", "$v_2$", "$v_3$"]
    for j in range(6):  # a component along x-hat_b, y-hat_b or z-hat_b in that axis's colour; v dashed
        cl.plot(sj, dl["V_b"][:, j], "-" if j < 3 else "--", lw=1.4, label=names[j], color=COLOURS["xyz"[j % 3]])
    for i, colour in enumerate(_joint_shades(robot.n)):
        cr.plot(ss, ds[:, i], lw=1.4, label=f"$\\dot\\theta_{i + 1}$", color=colour)
    for c, ttl in ((cl, r"$\mathcal{V}_b(s)$, changing"), (cr, r"$\dot\theta(s)$, changing")):
        c.set_xlim(0, 1)
        c.set_title(ttl, fontsize=9.5)
        c.set_xlabel("s", fontsize=9)
        c.tick_params(labelsize=8)
        c.legend(fontsize=8, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.3), frameon=False, handlelength=1.2)
    curs = [cl.axvline(0, color="black", lw=1.0), cr.axvline(0, color="black", lw=1.0)]
    vb = dr["V_b"][0]
    fixed_l = "dθ/ds = (" + ", ".join(_fmt(x, 0, 2) for x in dj[0]) + "), fixed"
    fixed_r = "V_b = (" + ", ".join(_fmt(x, 0, 2) for x in vb) + ") per unit s, fixed"
    fig.text(0.25, 0.335, fixed_l, ha="center", va="top", family="monospace", fontsize=9)
    fig.text(0.75, 0.335, fixed_r, ha="center", va="top", family="monospace", fontsize=9)

    def update(k):
        left.draw(axl, k, title="constant joint rates\nthe tool's twist changes", trace=traces[0])
        right.draw(axr, k, title="constant twist of the tool\nthe joint rates change", trace=traces[1])
        for c in curs:
            c.set_xdata([sj[k], sj[k]])

    data = {"s": sj, "theta": thj, "T_sb": dl["T_sb"], "joint": dl, "screw": dr}
    return _finish(fig, update, frames, fps, data)
