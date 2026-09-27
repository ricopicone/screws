import numpy as np
import pytest

from screws import se3, so3
from screws.coppelia import Scene, golf
from tests.coppelia.fake_sim import two_joint_scene


def test_build_green_creates_turf_with_a_cup_and_a_ball_on_top():
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        green = golf.build_green(scene, ball_position=(0.5, 0.0), hole_position=(1.2, 0.1))
        ball = sim.objects[green.ball]
        assert ball.primitive[0] == sim.primitiveshape_spheroid
        assert np.isclose(ball.primitive[1][0], 2 * golf.BALL_RADIUS)
        assert not ball.static and ball.respondable and np.isclose(ball.mass, golf.BALL_MASS)
        assert ball.engine[sim.bullet_body_friction] == pytest.approx(0.8)
        # the ball starts on top of the turf, the hole position is on the turf surface
        assert np.allclose(green.ball_position(), [0.5, 0.0, green.top + golf.BALL_RADIUS])
        assert np.allclose(green.hole_position, [1.2, 0.1, green.top])
        turf = sim.objects[green.turf]
        assert turf.static and turf.respondable and hasattr(turf, "mesh")
        assert np.allclose(turf.color, golf.TURF_COLOUR)
        cup_floor = sim.objects[green.cup_floor]
        assert cup_floor.static and cup_floor.respondable
        assert np.isclose(green.distance_to_hole(), np.hypot(0.7, 0.1))
        assert not green.holed()
        handles = [green.ball, green.turf, green.cup_floor, *green.pin]
        green.remove()
        assert all(h not in sim.objects for h in handles)


def test_holed_means_the_ball_has_dropped_into_the_cup():
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        green = golf.build_green(scene, ball_position=(0.5, 0.0), hole_position=(1.2, 0.1))
        sim.setObjectPosition(green.ball, [1.21, 0.09, green.top + golf.BALL_RADIUS])
        assert not green.holed()  # sitting on the rim's plane does not count
        sim.setObjectPosition(green.ball, [1.21, 0.09, golf.BALL_RADIUS + 0.002])
        assert green.holed()


def test_green_mesh_is_a_closed_slab_with_a_round_hole():
    V, F = golf.green_mesh(size=(2.0, 1.0), thickness=0.03, center=(0.2, 0.1), hole=(0.5, 0.2), hole_radius=0.054, n=32)
    assert V.shape[1] == 3 and F.shape[1] == 3
    # closed manifold: every edge is shared by exactly two triangles
    edges = {}
    for tri in F:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges[(min(a, b), max(a, b))] = edges.get((min(a, b), max(a, b)), 0) + 1
    assert set(edges.values()) == {2}
    # top faces point up, bottom faces point down, and no top triangle covers the hole centre
    for tri in F:
        p = V[tri]
        nrm = np.cross(p[1] - p[0], p[2] - p[0])
        if np.allclose(p[:, 2], 0.03):
            assert nrm[2] > 0
            assert not _point_in_triangle((0.5, 0.2), p[:, :2])
        elif np.allclose(p[:, 2], 0.0):
            assert nrm[2] < 0
    assert V[:, 0].min() == pytest.approx(-0.8) and V[:, 0].max() == pytest.approx(1.2)
    assert V[:, 1].min() == pytest.approx(-0.4) and V[:, 1].max() == pytest.approx(0.6)


def _point_in_triangle(q, tri):
    (x1, y1), (x2, y2), (x3, y3) = tri
    d1 = (q[0] - x2) * (y1 - y2) - (x1 - x2) * (q[1] - y2)
    d2 = (q[0] - x3) * (y2 - y3) - (x2 - x3) * (q[1] - y3)
    d3 = (q[0] - x1) * (y3 - y1) - (x3 - x1) * (q[1] - y1)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def test_attach_putter_parents_a_kinematic_face_to_the_last_link_with_the_right_offset():
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        arm = scene.arm("/Arm")
        putter = golf.attach_putter(scene, arm, shaft_length=0.3)
        for h in (putter.shaft, putter.face):
            o = sim.objects[h]
            assert o.static and o.parent == arm.tip
        assert sim.objects[putter.face].respondable  # only the face strikes the ball
        assert not sim.objects[putter.shaft].respondable
        T_tip = arm.tip_frame()
        # the face centre sits shaft_length along the tip's +z axis
        face_expected = T_tip @ se3.trans([0, 0, 0.3])
        assert np.allclose(scene.frame(putter.face)[:3, 3], face_expected[:3, 3])
        assert np.allclose(putter.T_tip_face, se3.trans([0, 0, 0.3]))
        # the tool robot's M is the face frame at zero
        r = arm.robot()
        rt = putter.robot(r)
        assert np.allclose(rt.M, r.M @ putter.T_tip_face) and np.allclose(rt.S, r.S)
        putter.remove()
        assert putter.face not in sim.objects


def test_address_pose_geometry():
    ball = np.array([0.5, 0.0, golf.BALL_RADIUS])
    hole = np.array([1.2, 0.1, 0.0])
    T = golf.address_pose(ball, hole, back=0.06)
    d = np.array([0.7, 0.1, 0.0])
    d /= np.linalg.norm(d)
    assert np.allclose(T[:3, 0], d)  # face normal (tool x) points along the target line
    assert np.allclose(T[:3, 2], [0, 0, -1])  # shaft (tool z) points down
    assert np.allclose(T[:3, 3], ball - 0.06 * d)  # 6 cm behind the ball, at ball-centre height
    assert so3.is_so3(T[:3, :3])


def test_stroke_path_is_straight_through_the_ball_at_constant_speed():
    ball = np.array([0.5, 0.0, golf.BALL_RADIUS])
    hole = np.array([1.2, 0.0, 0.0])
    path = golf.stroke_path(ball, hole, back=0.06, through=0.04, speed=0.5, dt=0.05)
    assert path[0].shape == (4, 4)
    # 10 cm at 0.5 m/s is 0.2 s: 4 control steps of 0.05 s, one pose each
    assert len(path) == 4
    xs = [T[0, 3] for T in path]
    assert np.allclose(np.diff(xs), 0.025)
    assert np.isclose(xs[0], 0.5 - 0.06 + 0.025) and np.isclose(xs[-1], 0.5 + 0.04)
    assert all(np.allclose(T[:3, :3], path[0][:3, :3]) for T in path)
