"""Screw motions you can watch: a body moving under a constant twist, its screw axis and
pitch, the path of one point, and the velocities omega x p + v of a few points.

animate_twist plays a twist V in time; animate_screw sweeps a screw axis S over theta
with T(theta) = e^{[S] theta} T(0) printed alongside; compare_frames shows the same motion
twice, with the space twist's v_s at the {s} origin and the body twist's v_b at {b}, and
V_s = [Ad_{T_sb}] V_b under them. The presets door(), drawer() and screwdriver() are the
three motions of notes 3.10. Needs matplotlib (``screws[plot]``); MP4 also needs
``screws[video]``. MR 3.3.2-3.3.3; notes 3.5, 3.10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..motion import helix, point_velocity, screw_line, screw_motion
from ..se3 import adjoint
from .scenes import Scene, box
from .static import COLOURS, _set_equal

__all__ = ["Animation", "animate_screw", "animate_twist", "compare_frames"]

HOLD_START, HOLD_END = 8, 16  # frames held at either end, so a looping GIF can be read


@dataclass
class Animation:
    """A finished animation: the figure, the FuncAnimation, and the numbers drawn in each
    frame (theta, T_sb, and per entry point V_s, V_b, Ad, ...) for checking by hand."""

    fig: object
    anim: object
    fps: int
    data: dict = field(default_factory=dict)

    @property
    def thetas(self) -> np.ndarray:
        return self.data["theta"]

    @property
    def transforms(self) -> np.ndarray:
        return self.data["T_sb"]

    def save(self, path, fps: int | None = None, dpi: int | None = None) -> Path:
        """Write a .gif (Pillow, no extra needed) or .mp4 (ffmpeg from ``screws[video]``)."""
        from matplotlib import animation, rcParams

        path = Path(path)
        fps = fps or self.fps
        if path.suffix.lower() == ".gif":
            writer = animation.PillowWriter(fps=fps)
        elif path.suffix.lower() == ".mp4":
            import imageio_ffmpeg

            rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
        else:
            raise ValueError(f"save to .gif or .mp4, not {path.suffix!r}")
        self.anim.save(str(path), writer=writer, dpi=dpi)
        return path

    def _repr_html_(self):
        return self.anim.to_jshtml(fps=self.fps)


def _default_body() -> list:
    return [box([-0.15, -0.1, -0.025], [0.15, 0.1, 0.025], COLOURS["joint"])]


def _scene(src, T0, body, theta, points) -> Scene:
    """A Scene from a preset, or from a raw screw axis / twist and optional pieces."""
    if isinstance(src, Scene):
        return src
    S = np.asarray(src, float)
    T0 = np.eye(4) if T0 is None else np.asarray(T0, float)
    translation = bool(np.allclose(S[:3], 0))
    if theta is None:
        theta = (0.0, 1.0) if translation else (0.0, 2 * np.pi)
    body = _default_body() if body is None else list(body if isinstance(body, (list, tuple)) else [body])
    if points is None:
        points = np.array([[0.0, 0, 0], [0.15, 0.1, 0.025], [-0.15, 0.1, 0.025]])
    return Scene("screw", S, T0, body, tuple(theta), np.asarray(points, float))


def _fmt(x, w=6, p=2) -> str:
    x = 0.0 if abs(x) < 0.5 * 10**-p else float(x)
    return f"{x:{w}.{p}f}"


def _vec(V) -> str:
    return "(" + ", ".join(_fmt(x, 0, 2) for x in V[:3]) + ",  " + ", ".join(_fmt(x, 0, 2) for x in V[3:]) + ")"


def _matrix(M, w=6, p=2) -> str:
    return "\n".join("[" + " ".join(_fmt(x, w, p) for x in row) + " ]" for row in M)


def _rgba(c, alpha):
    from matplotlib.colors import to_rgba as _to_rgba

    return _to_rgba(c, alpha)


class _Painter:
    """Draws one scene at one configuration on a 3D axes; fixed limits for every frame."""

    def __init__(self, scene: Scene, thetas, arrow_scale: float):
        self.sc = scene
        self.thetas = np.asarray(thetas, float)
        self.k = arrow_scale
        self.T = screw_motion(scene.S, scene.T0, self.thetas)  # T_sb(theta)
        self.q, self.s_hat, self.h = screw_line(scene.S)
        p_trace = scene.T0[:3, :3] @ scene.points[scene.trace] + scene.T0[:3, 3]
        self.path = helix(scene.S, p_trace, self.thetas)
        pts = [self.path, np.zeros((1, 3))]
        for T in self.T[:: max(1, len(self.T) // 12)]:
            for m in scene.body:
                pts.append(m.transformed(T).vertices)
            P = self._points(T)
            pts.append(P + self.k * point_velocity(scene.S, P))
        pts += [m.vertices for m in scene.fixed]
        if scene.goal is not None:
            pts.append(scene.goal[None, :3, 3])
        allp = np.vstack(pts)
        self.size = float(np.ptp(allp, axis=0).max())
        if self.q is not None:  # the axis: the stretch of line beside everything drawn
            t = (allp - self.q) @ self.s_hat
            pad = 0.12 * (t.max() - t.min() + 1e-9)
            self.axis_ends = (self.q + (t.min() - pad) * self.s_hat, self.q + (t.max() + pad) * self.s_hat)
            allp = np.vstack([allp, self.axis_ends])
        self.limits = allp

    def _points(self, T):
        return self.sc.points @ T[:3, :3].T + T[:3, 3]

    def draw(self, ax, i, *, title=None, points=True, axis=True, b_triad=1.0):
        from matplotlib.colors import to_rgba
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        sc, T = self.sc, self.T[i]
        ax.cla()
        ax.computed_zorder = False  # arrows and labels over the body, always
        polys, colours = [], []
        meshes = list(sc.fixed) + [m.transformed(T) for m in sc.body]
        for m in meshes:
            for f, c in zip(m.faces, m.face_colours()):
                polys.append(m.vertices[f])
                colours.append(to_rgba(c, m.alpha))
        ax.add_collection3d(Poly3DCollection(polys, facecolors=colours, edgecolors=(0, 0, 0, 0.35), linewidths=0.4, zorder=1))
        L = 0.14 * self.size
        self._triad(ax, np.eye(4), L, "{s}", letters=True)
        if sc.goal is not None:
            self._triad(ax, sc.T0, L, "start", alpha=0.35)
            self._triad(ax, sc.goal, L, "end", alpha=0.35)
        self._triad(ax, T, L * b_triad, "{b}")
        if axis:
            self._axis(ax)
        ax.plot(*self.path.T, ":", color=COLOURS["pitch"], lw=1.0, alpha=0.6)
        ax.plot(*self.path[: i + 1].T, "-", color=COLOURS["pitch"], lw=2.0)
        if points:
            P = self._points(T)
            vel = point_velocity(sc.S, P)
            ax.scatter(*P.T, color="black", s=12, depthshade=False)
            for p, v in zip(P, vel):
                self._arrow(ax, p, self.k * v, COLOURS["v"], 2.0)
        _set_equal(ax, self.limits, pad=0.02 * self.size)
        ax.set_box_aspect((1, 1, 1), zoom=1.3)
        ax.view_init(*sc.view)
        ax.set_axis_off()  # the {s} triad carries the axes
        if title:
            ax.set_title(title, fontsize=12)

    def _axis(self, ax):
        c = COLOURS["axis"]
        if self.q is None:  # a translation has a direction but no line: draw v-hat beside the body
            base = self.limits.min(axis=0) + 0.1 * np.ptp(self.limits, axis=0)
            self._arrow(ax, base, 0.25 * self.size * self.s_hat, c, 2.5)
            ax.text(*(base + 0.27 * self.size * self.s_hat), "v̂   (h = ∞)", color=c, fontsize=10)
            return
        a, b = self.axis_ends
        ax.plot(*np.array([a, b]).T, "--", color=c, lw=1.5)
        self._arrow(ax, b - 0.2 * self.size * self.s_hat, 0.18 * self.size * self.s_hat, c, 2.5)
        ax.text(*b, f"  ŝ,  h = {self.h:.3g}", color=c, fontsize=11)

    @staticmethod
    def _arrow(ax, o, vec, colour, lw):
        if np.linalg.norm(vec) < 1e-9:
            return None
        return ax.quiver(*o, *vec, color=colour, linewidth=lw, arrow_length_ratio=0.18, zorder=5)

    def _triad(self, ax, T, length, label, letters=False, alpha=1.0):
        o = T[:3, 3]
        for k, c in enumerate("xyz"):
            self._arrow(ax, o, length * T[:3, k], _rgba(COLOURS[c], alpha), 1.6)
            if letters:
                ax.text(*(o + 1.12 * length * T[:3, k]), c, color=COLOURS[c], fontsize=9)
        ax.text(*(o - 0.45 * length * T[:3, 2]), label, fontsize=11, ha="center", va="top", alpha=max(alpha, 0.6))


def _frames(n_motion):
    return [0] * HOLD_START + list(range(n_motion)) + [n_motion - 1] * HOLD_END


def _finish(fig, update, n_motion, fps, data):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    anim = FuncAnimation(fig, update, frames=_frames(n_motion), interval=1000 / fps, blit=False)
    plt.close(fig)  # no second, static copy in a notebook; saving still works
    return Animation(fig, anim, fps, data)


def animate_twist(
    V,
    T0=None,
    *,
    body=None,
    points=None,
    duration: float = 4.0,
    fps: int = 20,
    arrow_scale: float | None = None,
    figsize=(6.4, 5.6),
) -> Animation:
    """A body moving under the constant twist V = (omega, v), written in {s}, for duration
    seconds: e^{[V] t} T0. Pass a Scene (door(), drawer(), screwdriver()) to play its motion
    in duration seconds, V = S thetadot. The screw axis is dashed with s-hat and the pitch,
    one point's path is traced (a helix; a circle at zero pitch), and the red arrows are
    the velocities omega x p + v of the marked points, drawn arrow_scale seconds long.
    MR 3.3.2; notes 3.10."""
    import matplotlib.pyplot as plt

    if isinstance(V, Scene):
        sc = V
        rate = (sc.theta[1] - sc.theta[0]) / duration
        V = sc.S * rate
    else:
        sc = _scene(V, T0, body, (0.0, duration), points)
        rate = 1.0
        V = sc.S
    n = max(2, round(duration * fps))
    ts = np.linspace(0, duration, n)
    k = (sc.arrow_scale if arrow_scale is None else arrow_scale) / rate
    painter = _Painter(Scene(**{**sc.__dict__, "S": V, "theta": (0.0, duration)}), ts, k)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    caption = fig.text(0.5, 0.03, "", ha="center", family="monospace", fontsize=9)
    name = sc.name if sc.name != "screw" else "a constant twist"

    def update(i):
        painter.draw(ax, i, title=f"{name}: V_s = {_vec(V)}")
        caption.set_text(f"t = {ts[i]:4.2f} s    arrows: ω × p + v, drawn {k:.2g} s long")

    return _finish(fig, update, n, fps, {"theta": ts, "T_sb": painter.T, "V_s": V})


def animate_screw(
    S,
    T0=None,
    *,
    body=None,
    points=None,
    theta=None,
    frames: int = 80,
    fps: int = 20,
    show_matrix: bool = True,
    arrow_scale: float | None = None,
    figsize=None,
) -> Animation:
    """The screw axis S = (omega, v), written in {s}, swept over theta: the body at
    T(theta) = e^{[S] theta} T(0). Beside it the line read off S (q = omega x v, s-hat and
    h = omega . v for a unit omega) and, with show_matrix, T(theta) itself as theta grows.
    S may be a Scene. MR 3.3.3.2, eq. 3.85; notes 3.10."""
    import matplotlib.pyplot as plt

    sc = _scene(S, T0, body, theta, points)
    th = np.linspace(sc.theta[0], sc.theta[1], frames)
    k = sc.arrow_scale if arrow_scale is None else arrow_scale
    painter = _Painter(sc, th, k)
    figsize = figsize or ((10.5, 5.6) if show_matrix else (6.4, 5.6))
    fig = plt.figure(figsize=figsize)
    if show_matrix:
        ax = fig.add_axes([0.0, 0.02, 0.58, 0.94], projection="3d")
        panel = fig.text(0.57, 0.5, "", family="monospace", fontsize=11.5, va="center")
    else:
        ax = fig.add_subplot(projection="3d")
        panel = None
    q, s_hat, h = painter.q, painter.s_hat, painter.h
    head = f"S = {_vec(sc.S)}\n"
    if q is None:
        head += f"ω = 0: no axis, a translation\nalong v̂ = ({_fmt(s_hat[0], 0)}, {_fmt(s_hat[1], 0)}, {_fmt(s_hat[2], 0)})\npitch h = ∞\n"
    else:
        head += (
            f"axis through q = ω × v = ({_fmt(q[0], 0)}, {_fmt(q[1], 0)}, {_fmt(q[2], 0)})\n"
            f"direction ŝ = ω = ({_fmt(s_hat[0], 0)}, {_fmt(s_hat[1], 0)}, {_fmt(s_hat[2], 0)})\n"
            f"pitch h = ωᵀv = {h:.3g}\n"
        )

    def update(i):
        painter.draw(ax, i, title=f"{sc.name}: {sc.note}" if sc.note else "T(θ) = e^[S]θ T(0)")
        if panel is not None:
            ang = f"{th[i]:5.2f} m" if sc.is_translation else f"{np.rad2deg(th[i]):6.1f}°"
            panel.set_text(
                head + f"\nθ = {ang}\n\nT(θ) = e^[S]θ T(0) =\n" + _matrix(painter.T[i], 7, 3)
            )

    return _finish(fig, update, frames, fps, {"theta": th, "T_sb": painter.T, "q": q, "s_hat": s_hat, "h": h})


def compare_frames(scene, *, frames: int = 80, fps: int = 20, view=None, figsize=(12, 8)) -> Animation:
    """One motion drawn twice, per unit rate (thetadot = 1). Left, the space twist V_s:
    its v_s is the velocity of the body point at the {s} origin, the body imagined to fill
    all of space; the grey marker is the point that was there at theta = 0, carried off
    by the motion while the point now at the origin keeps the same velocity. Right, the
    body twist V_b: v_b is the velocity of the point at {b}'s origin, in {b}'s axes, the
    same at every theta. Below: V_s = [Ad_{T_sb(theta)}] V_b, Ad changing, V_s not.
    view is (elev, azim); the default looks down more steeply than the scene's own view, so
    the {s} origin and the axis separate. MR 3.3.2, eq. 3.83; notes 3.10."""
    import matplotlib.pyplot as plt

    sc = scene if isinstance(scene, Scene) else _scene(scene, None, None, None, None)
    sc = Scene(**{**sc.__dict__, "view": view or (max(sc.view[0], 55), sc.view[1])})
    th = np.linspace(sc.theta[0], sc.theta[1], frames)
    painter = _Painter(sc, th, sc.arrow_scale)
    k = sc.arrow_scale
    S_b = sc.S_b
    Ads = np.array([adjoint(T) for T in painter.T])
    V_s = Ads @ S_b
    fig = plt.figure(figsize=figsize)
    axl = fig.add_axes([0.0, 0.3, 0.5, 0.65], projection="3d")
    axr = fig.add_axes([0.5, 0.3, 0.5, 0.65], projection="3d")
    eq = fig.text(0.5, 0.02, "", family="monospace", fontsize=11.5, ha="center", va="bottom")
    marker0 = np.zeros(3)  # the body point at the {s} origin at theta = 0, in {b} coordinates
    marker_b = (np.linalg.inv(sc.T0) @ np.append(marker0, 1))[:3]
    fwd = "rad" if not sc.is_translation else "m"

    def update(i):
        T = painter.T[i]
        painter.draw(axl, i, title="space twist V_s:\nv_s is the velocity of the body point at the {s} origin", points=False)
        m = T[:3, :3] @ marker_b + T[:3, 3]
        if painter.q is not None:  # the phantom arm: the axis to the marker, rigid with the body
            foot = painter.q + ((m - painter.q) @ painter.s_hat) * painter.s_hat
            axl.plot(*np.array([foot, m]).T, "--", color="grey", lw=1.2)
            foot0 = painter.q + ((0 - painter.q) @ painter.s_hat) * painter.s_hat
            axl.plot(*np.array([foot0, marker0]).T, ":", color=COLOURS["v"], lw=1.0)
        axl.scatter(*m, color="grey", s=25, depthshade=False)
        painter._arrow(axl, m, k * point_velocity(sc.S, m), "grey", 1.6)
        axl.scatter(0, 0, 0, color=COLOURS["v"], s=25, depthshade=False)
        painter._arrow(axl, marker0, k * V_s[i][3:], COLOURS["v"], 3.0)
        axl.text(*(k * V_s[i][3:] * 1.1 + [0, 0, 0.08 * painter.size]), "v_s", color=COLOURS["v"], fontsize=11, fontweight="bold")

        painter.draw(axr, i, title="body twist V_b:\nv_b is the velocity of the body point at the {b} origin", points=False, b_triad=1.6)
        vb_world = T[:3, :3] @ S_b[3:]
        painter._arrow(axr, T[:3, 3], k * vb_world, COLOURS["v"], 3.0)
        axr.text(*(T[:3, 3] + k * vb_world * 1.1 + [0, 0, 0.08 * painter.size]), "v_b", color=COLOURS["v"], fontsize=11, fontweight="bold")

        ang = f"{np.rad2deg(th[i]):5.1f}°" if fwd == "rad" else f"{th[i]:4.2f} m"
        left = ["V_s"] + [_fmt(x) for x in V_s[i]]
        mid = [f"[Ad_Tsb(θ)],  θ = {ang}"] + [" ".join(_fmt(x, 6, 2) for x in row) for row in Ads[i]]
        right = ["V_b"] + [_fmt(x) for x in S_b]
        rows = []
        for r in range(7):
            sep = " = " if r == 3 else "   "
            rows.append(f"{left[r]:>6}{sep}{mid[r]:^43}   {right[r]:>6}")
        eq.set_text("\n".join(rows) + f"\n(per unit rate θ̇ = 1; V_s and V_b stay fixed while the {sc.name} moves)")

    return _finish(
        fig,
        update,
        frames,
        fps,
        {"theta": th, "T_sb": painter.T, "Ad": Ads, "V_s": V_s, "V_b": np.tile(S_b, (frames, 1))},
    )

