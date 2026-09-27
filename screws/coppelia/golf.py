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
from ..trajectory import joint_trajectory

__all__ = [
    "BALL_MASS",
    "BALL_RADIUS",
    "HOLE_RADIUS",
    "Green",
    "PuttResult",
    "Putter",
    "address_pose",
    "attach_putter",
    "build_green",
    "putt",
    "stroke_path",
]

BALL_RADIUS = 0.02135  # a regulation golf ball, 42.67 mm across
BALL_MASS = 0.0459  # kg
HOLE_RADIUS = 0.054  # a regulation cup, 108 mm across


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
    """The ball and the hole in a scene."""

    scene: object
    ball: int
    hole: int
    hole_position: np.ndarray  # (x, y, 0) in the world frame

    def ball_position(self) -> np.ndarray:
        return np.asarray(self.scene.sim.getObjectPosition(self.ball, self.scene.sim.handle_world), float)

    def ball_speed(self) -> float:
        v, _ = self.scene.sim.getObjectVelocity(self.ball)
        return float(np.linalg.norm(v))

    def distance_to_hole(self) -> float:
        """Horizontal distance from the ball's centre to the hole's centre."""
        return float(np.linalg.norm((self.ball_position() - self.hole_position)[:2]))

    def holed(self, *, speed_max: float = 0.3) -> bool:
        """True if the ball's centre is inside the cup and it is moving slower than speed_max."""
        return self.distance_to_hole() <= HOLE_RADIUS - BALL_RADIUS / 2 and self.ball_speed() <= speed_max

    def remove(self) -> None:
        self.scene.sim.removeObjects([self.ball, self.hole])


def build_green(
    scene,
    *,
    ball_position,
    hole_position,
    friction: float = 0.8,
    damping: float = 0.8,
    restitution: float = 0.2,
) -> Green:
    """Put a ball at (x, y) on the floor and a cup marker at (x, y).

    friction, damping (Bullet's linear and angular damping) and restitution are the ball's
    material; the defaults roll 0.2 m, 0.3 m and 0.7 m from face speeds of 0.3, 0.6 and
    1.0 m/s on CoppeliaSim 4.10's Bullet engine and stop within about 3.5 s.
    """
    sim = scene.sim
    bx, by = float(ball_position[0]), float(ball_position[1])
    hx, hy = float(hole_position[0]), float(hole_position[1])
    ball = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [2 * BALL_RADIUS] * 3, 0)
    sim.setShapeMass(ball, BALL_MASS)
    _shape(sim, ball, static=False, respondable=True)
    sim.setObjectPosition(ball, [bx, by, BALL_RADIUS], sim.handle_world)
    sim.setEngineFloatParam(sim.bullet_body_friction, ball, friction)
    sim.setEngineFloatParam(sim.bullet_body_restitution, ball, restitution)
    sim.setEngineFloatParam(sim.bullet_body_lineardamping, ball, damping)
    sim.setEngineFloatParam(sim.bullet_body_angulardamping, ball, damping)
    _colour(sim, ball, (0.95, 0.95, 0.95))
    hole = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [2 * HOLE_RADIUS, 2 * HOLE_RADIUS, 0.001], 0)
    _shape(sim, hole, static=True, respondable=False)
    sim.setObjectPosition(hole, [hx, hy, 0.0006], sim.handle_world)
    _colour(sim, hole, (0.05, 0.05, 0.05))
    return Green(scene, ball, hole, np.array([hx, hy, 0.0]))


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
    position mode and the simulation started. The approach goes to a pose lift metres above
    the address pose, descends, then strokes at constant face speed; the function returns when
    the ball stops or settle_time has passed after the stroke.

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

    def go(traj):
        for th in traj:
            arm.command(th)
            scene.step()

    n_approach = max(2, round(approach_time / dt))
    go(joint_trajectory(theta_now, theta_lift, approach_time, n_approach)[1:])
    go(joint_trajectory(theta_lift, theta_address, approach_time / 2, max(2, n_approach // 2))[1:])
    for _ in range(round(0.5 / dt)):  # a still moment before the stroke
        arm.command(theta_address)  # keep sending the target: an unsent target drifts
        scene.step()

    theta = theta_address
    path = []
    for T in stroke_path(ball, hole, back=back, through=through, speed=speed, dt=dt):
        theta = _ik_or_raise(robot_face, T, [theta], "stroke")
        arm.command(theta)
        scene.step()
        path.append(green.ball_position())
    steps_settle = round(settle_time / dt)
    for k in range(steps_settle):
        scene.step()
        path.append(green.ball_position())
        if k > int(0.5 / dt) and green.ball_speed() < 2e-3:
            break
    return PuttResult(green.holed(), green.distance_to_hole(), np.array(path), theta)
