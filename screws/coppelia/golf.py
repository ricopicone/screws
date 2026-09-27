"""A putting green for the UR5: a ball, a hole, a kinematic putter on the flange, and a putt.

The ball is the only dynamic body. The putter is a static, respondable tool rigidly attached
to the arm's last link, so it moves exactly where the joint trajectory puts it and pushes the
ball through contact. Everything else is kinematics from the rest of screws: the face pose
that addresses the ball, a straight stroke along the target line, IK for every waypoint.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..robot import Robot
from ..se3 import rp_to_transform, trans
from ..trajectory import cartesian_trajectory, joint_trajectory

__all__ = [
    "BALL_MASS",
    "BALL_RADIUS",
    "HOLE_RADIUS",
    "TURF_COLOUR",
    "TURF_THICKNESS",
    "Green",
    "PuttResult",
    "Putter",
    "address_pose",
    "attach_putter",
    "build_green",
    "green_mesh",
    "putt",
    "stroke_path",
]

BALL_RADIUS = 0.02135  # a regulation golf ball, 42.67 mm across
BALL_MASS = 0.0459  # kg
HOLE_RADIUS = 0.054  # a regulation cup, 108 mm across
TURF_THICKNESS = 0.05  # the turf slab the cup is cut through; deep enough to swallow the ball
TURF_COLOUR = (0.16, 0.55, 0.20)


def _shape(sim, h, *, static: bool, respondable: bool) -> None:
    sim.setObjectInt32Param(h, sim.shapeintparam_static, 1 if static else 0)
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1 if respondable else 0)


def _colour(sim, h, rgb) -> None:
    try:
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, list(rgb))
    except Exception:  # noqa: BLE001, S110 - colour is cosmetic
        pass


@dataclass
class Green:
    """The turf, the cup cut through it, the pin, and the ball."""

    scene: object
    ball: int
    turf: int
    cup_floor: int
    pin: tuple[int, ...]
    hole_position: np.ndarray  # (x, y, top) at the turf surface
    top: float  # height of the turf surface above the floor

    def ball_position(self) -> np.ndarray:
        return np.asarray(self.scene.sim.getObjectPosition(self.ball, self.scene.sim.handle_world), float)

    def ball_speed(self) -> float:
        v, _ = self.scene.sim.getObjectVelocity(self.ball)
        return float(np.linalg.norm(v))

    def distance_to_hole(self) -> float:
        """Horizontal distance from the ball's centre to the cup's centre."""
        return float(np.linalg.norm((self.ball_position() - self.hole_position)[:2]))

    def holed(self) -> bool:
        """True once the ball has dropped into the cup: inside the rim and below the surface."""
        p = self.ball_position()
        return self.distance_to_hole() <= HOLE_RADIUS and p[2] < self.top - BALL_RADIUS / 2

    def remove(self) -> None:
        self.scene.sim.removeObjects([self.ball, self.turf, self.cup_floor, *self.pin])


def green_mesh(*, size, thickness, center, hole, hole_radius, n: int = 48):
    """Vertices (V, 3) and triangles (F, 3) of a rectangular slab with a round hole through it.

    size is (sx, sy) and center (cx, cy) of the slab, whose bottom is at z = 0 and top at
    z = thickness; hole is (hx, hy). Outward normals, closed manifold, so it serves as a
    static respondable mesh.
    """
    sx, sy = float(size[0]), float(size[1])
    cx, cy = float(center[0]), float(center[1])
    hx, hy = float(hole[0]), float(hole[1])
    xmin, xmax, ymin, ymax = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2
    if not (xmin < hx - hole_radius and hx + hole_radius < xmax and ymin < hy - hole_radius and hy + hole_radius < ymax):
        raise ValueError("the hole must lie inside the slab")
    angles = 2 * np.pi * np.arange(n) / n
    inner = [(a, np.array([hx + hole_radius * np.cos(a), hy + hole_radius * np.sin(a)])) for a in angles]

    def ray_to_edge(a):
        d = np.array([np.cos(a), np.sin(a)])
        ts = []
        if d[0] > 1e-12:
            ts.append((xmax - hx) / d[0])
        if d[0] < -1e-12:
            ts.append((xmin - hx) / d[0])
        if d[1] > 1e-12:
            ts.append((ymax - hy) / d[1])
        if d[1] < -1e-12:
            ts.append((ymin - hy) / d[1])
        t = min(ts)
        return np.array([hx, hy]) + t * d

    outer = [(a, ray_to_edge(a)) for a in angles]
    for corner in ((xmax, ymax), (xmin, ymax), (xmin, ymin), (xmax, ymin)):
        a = float(np.arctan2(corner[1] - hy, corner[0] - hx)) % (2 * np.pi)
        if all(abs(a - b) > 1e-9 for b, _ in outer):
            outer.append((a, np.array(corner, float)))
    outer.sort(key=lambda ap: ap[0])
    ni, no = len(inner), len(outer)
    # vertex layout: top inner, top outer, bottom inner, bottom outer
    TI, TO, BI, BO = 0, ni, ni + no, ni + no + ni
    V = []
    for z in (thickness, 0.0):
        V += [[p[0], p[1], z] for _, p in inner]
        V += [[p[0], p[1], z] for _, p in outer]
    V = np.array(V)
    F = []
    # top: zip the two angle-sorted polygons; bottom: the same with reversed winding
    i = j = 0
    while i < ni or j < no:
        a_next = inner[(i + 1) % ni][0] + (2 * np.pi if i + 1 >= ni else 0.0) if i < ni else np.inf
        b_next = outer[(j + 1) % no][0] + (2 * np.pi if j + 1 >= no else 0.0) if j < no else np.inf
        if a_next <= b_next:
            F.append([TI + i % ni, TO + j % no, TI + (i + 1) % ni])
            i += 1
        else:
            F.append([TI + i % ni, TO + j % no, TO + (j + 1) % no])
            j += 1
    top_count = len(F)
    for tri in list(F[:top_count]):
        F.append([tri[0] + BI, tri[2] + BI, tri[1] + BI])
    for j in range(no):  # outer walls, outward
        k = (j + 1) % no
        F.append([BO + j, BO + k, TO + k])
        F.append([BO + j, TO + k, TO + j])
    for i in range(ni):  # hole walls, facing into the cup
        k = (i + 1) % ni
        F.append([TI + i, TI + k, BI + k])
        F.append([TI + i, BI + k, BI + i])
    return V, np.array(F, dtype=int)


def build_green(
    scene,
    *,
    ball_position,
    hole_position,
    size=(3.0, 3.0),
    thickness: float = TURF_THICKNESS,
    friction: float = 0.8,
    damping: float = 0.8,
    restitution: float = 0.2,
) -> Green:
    """Lay turf with a real cup at (x, y), a pin in it, and a ball at (x, y) on the surface.

    The turf is a static respondable slab of the given size (centred between the ball and
    the hole) and thickness, with the cup cut through it and a cup floor below. The ball's
    material (friction, Bullet's linear and angular damping, restitution) defaults roll 0.2
    to 0.35 m from face speeds of 0.3 to 0.5 m/s and stop within about 3.5 s.
    """
    sim = scene.sim
    bx, by = float(ball_position[0]), float(ball_position[1])
    hx, hy = float(hole_position[0]), float(hole_position[1])
    center = ((bx + hx) / 2, (by + hy) / 2)
    V, F = green_mesh(size=size, thickness=thickness, center=center, hole=(hx, hy), hole_radius=HOLE_RADIUS)
    turf = sim.createMeshShape(0, 0.0, [float(x) for x in V.reshape(-1)], [int(k) for k in F.reshape(-1)])
    _shape(sim, turf, static=True, respondable=True)
    _colour(sim, turf, TURF_COLOUR)
    cup_floor = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [2 * HOLE_RADIUS, 2 * HOLE_RADIUS, 0.004], 0)
    _shape(sim, cup_floor, static=True, respondable=True)
    sim.setObjectPosition(cup_floor, [hx, hy, 0.002], sim.handle_world)
    _colour(sim, cup_floor, (0.08, 0.08, 0.08))
    pin_height = 0.6
    pin = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [0.008, 0.008, pin_height], 0)
    _shape(sim, pin, static=True, respondable=False)
    sim.setObjectPosition(pin, [hx, hy, pin_height / 2], sim.handle_world)
    _colour(sim, pin, (0.95, 0.95, 0.9))
    flag = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [0.12, 0.003, 0.08], 0)
    _shape(sim, flag, static=True, respondable=False)
    sim.setObjectPosition(flag, [hx + 0.06, hy, pin_height - 0.04], sim.handle_world)
    _colour(sim, flag, (0.9, 0.1, 0.1))
    ball = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [2 * BALL_RADIUS] * 3, 0)
    sim.setShapeMass(ball, BALL_MASS)
    _shape(sim, ball, static=False, respondable=True)
    sim.setObjectPosition(ball, [bx, by, thickness + BALL_RADIUS], sim.handle_world)
    sim.setEngineFloatParam(sim.bullet_body_friction, ball, friction)
    sim.setEngineFloatParam(sim.bullet_body_restitution, ball, restitution)
    sim.setEngineFloatParam(sim.bullet_body_lineardamping, ball, damping)
    sim.setEngineFloatParam(sim.bullet_body_angulardamping, ball, damping)
    _colour(sim, ball, (0.95, 0.95, 0.95))
    return Green(scene, ball, turf, cup_floor, (pin, flag), np.array([hx, hy, thickness]), thickness)


@dataclass
class Putter:
    """A putter rigidly attached to the arm: a shaft along the tool's +z, a face whose outward
    normal is the tool's +x. T_tip_face is the face frame in the tip (last link) frame."""

    scene: object
    shaft: int
    face: int
    T_tip_face: np.ndarray
    face_size: tuple[float, float, float]

    def robot(self, robot: Robot) -> Robot:
        """The arm's Robot with M moved from the flange to the putter face."""
        return replace(robot, name=f"{robot.name}+putter", M=robot.M @ self.T_tip_face)

    def remove(self) -> None:
        self.scene.sim.removeObjects([self.shaft, self.face])


def attach_putter(
    scene,
    arm,
    *,
    shaft_length: float = 0.30,
    shaft_radius: float = 0.006,
    face_size=(0.02, 0.10, 0.03),
) -> Putter:
    """Build a putter on the arm's tip: static, respondable shapes parented to the last link.

    The face is a box of face_size (thickness along tool x, width along y, height along z)
    centred shaft_length along the tool's +z axis; its striking face is the +x side.
    """
    sim = scene.sim
    T_tip = arm.tip_frame()
    T_tip_face = trans([0.0, 0.0, shaft_length])
    shaft = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [2 * shaft_radius, 2 * shaft_radius, shaft_length], 0)
    _shape(sim, shaft, static=True, respondable=False)
    scene.set_frame(shaft, T_tip @ trans([0.0, 0.0, shaft_length / 2]))
    face = sim.createPrimitiveShape(sim.primitiveshape_cuboid, list(face_size), 0)
    _shape(sim, face, static=True, respondable=True)
    scene.set_frame(face, T_tip @ T_tip_face)
    _colour(sim, shaft, (0.6, 0.6, 0.6))
    _colour(sim, face, (0.8, 0.1, 0.1))
    for h in (shaft, face):
        sim.setObjectParent(h, arm.tip, True)
    return Putter(scene, shaft, face, T_tip_face, tuple(float(x) for x in face_size))


def _target_line(ball, hole) -> np.ndarray:
    d = np.asarray(hole, float) - np.asarray(ball, float)
    d[2] = 0.0
    n = np.linalg.norm(d)
    if n < 1e-9:
        raise ValueError("the ball is already on the hole")
    return d / n


def address_pose(ball, hole, *, back: float = 0.06) -> np.ndarray:
    """The face frame that addresses the ball: back metres behind it on the target line, at
    ball-centre height, face normal (tool x) along the line, shaft (tool z) pointing down."""
    ball = np.asarray(ball, float)
    x = _target_line(ball, hole)
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return rp_to_transform(np.column_stack([x, y, z]), ball - back * x)


def stroke_path(ball, hole, *, back: float = 0.06, through: float = 0.04, speed: float = 0.5, dt: float = 0.05):
    """Face poses, one per control step, for a straight stroke at constant speed from back
    metres behind the ball to through metres past its centre. The address pose itself is not
    included; the first pose is one step into the stroke."""
    T0 = address_pose(ball, hole, back=back)
    x = T0[:3, 0]
    length = back + through
    step = speed * dt  # exact face speed; the last pose is clamped to the stroke length
    n = max(1, int(np.ceil(length / step - 1e-9)))
    return [rp_to_transform(T0[:3, :3], T0[:3, 3] + min(k * step, length) * x) for k in range(1, n + 1)]


@dataclass
class PuttResult:
    holed: bool
    distance: float  # final horizontal distance from the ball to the hole, m
    ball_path: np.ndarray  # (N, 3) ball positions, one per step from the stroke onward
    face_path: np.ndarray  # (N, 3) putter-face positions (from the measured joints), same steps
    theta_final: np.ndarray


def _ik_or_raise(robot: Robot, T, seeds, what: str) -> np.ndarray:
    last = None
    for seed in seeds:
        res = robot.ik(T, seed, max_iter=50)
        last = res
        if res.converged and robot.within_limits(res.theta):
            return res.theta
    raise RuntimeError(
        f"inverse kinematics could not reach the {what} pose (errors {last.error_omega:.3g}, "
        f"{last.error_v:.3g}); move the ball or pass a better seed"
    )


def putt(
    scene,
    arm,
    robot_face: Robot,
    green: Green,
    *,
    back: float = 0.06,
    through: float = 0.06,
    speed: float = 0.5,
    lift: float = 0.12,
    approach_time: float = 2.0,
    settle_time: float = 5.0,
    seed=None,
) -> PuttResult:
    """Address the ball, stroke through it along the line to the hole, and watch it roll.

    robot_face is the arm's Robot with M at the putter face (Putter.robot). The arm must be in
    position mode and the simulation started. The approach goes in joint space to a pose lift
    metres above the address pose, descends straight down onto it, pauses, then strokes at a
    constant face speed. After the stroke the putter returns to the address pose and waits
    there, out of the shot, until the ball stops or settle_time has passed.

    The stock UR5 model's joints are limited to 90 deg/s, so face speeds above about 0.5 m/s
    lag the commanded stroke and strike weakly; 0.3 to 0.5 m/s rolls the ball 0.2 to 0.35 m
    on the default green.
    """
    dt = scene.dt
    ball = green.ball_position()
    hole = green.hole_position
    T_address = address_pose(ball, hole, back=back)
    T_lift = trans([0.0, 0.0, lift]) @ T_address
    theta_now = arm.theta()
    seeds = [theta_now] if seed is None else [np.asarray(seed, float), theta_now]
    theta_lift = _ik_or_raise(robot_face, T_lift, seeds, "lifted address")
    theta_address = _ik_or_raise(robot_face, T_address, [theta_lift], "address")
    ball_path, face_path = [], []

    def go(thetas, log=False):
        for th in thetas:
            arm.command(th)
            scene.step()
            if log:
                ball_path.append(green.ball_position())
                face_path.append(robot_face.fk(arm.theta())[:3, 3])

    n_approach = max(2, round(approach_time / dt))
    go(joint_trajectory(theta_now, theta_lift, approach_time, n_approach)[1:])
    theta = theta_lift
    descent = []
    for T in cartesian_trajectory(T_lift, T_address, approach_time / 2, max(2, n_approach // 2))[1:]:
        theta = _ik_or_raise(robot_face, T, [theta], "descent")
        descent.append(theta)
    go(descent)
    go([theta_address] * round(0.5 / dt))  # a still moment; an unsent target drifts

    theta = theta_address
    stroke = []
    for T in stroke_path(ball, hole, back=back, through=through, speed=speed, dt=dt):
        theta = _ik_or_raise(robot_face, T, [theta], "stroke")
        stroke.append(theta)
    go(stroke, log=True)
    n_back = max(2, round(1.0 / dt))
    go(joint_trajectory(theta, theta_address, 1.0, n_back)[1:], log=True)
    theta = theta_address
    for k in range(round(settle_time / dt)):
        go([theta], log=True)
        if k > round(0.5 / dt) and green.ball_speed() < 2e-3:
            break
    return PuttResult(
        green.holed(), green.distance_to_hole(), np.array(ball_path), np.array(face_path), theta
    )
