import numpy as np
import pytest

from screws import se3, so3
from screws.coppelia import Scene, golf
from tests.coppelia.fake_sim import two_joint_scene


def test_build_green_creates_a_dynamic_ball_with_the_chosen_material():
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        green = golf.build_green(scene, ball_position=(0.5, 0.0), hole_position=(1.2, 0.1))
        ball = sim.objects[green.ball]
        assert ball.primitive[0] == sim.primitiveshape_spheroid
        assert np.isclose(ball.primitive[1][0], 2 * golf.BALL_RADIUS)
        assert not ball.static and ball.respondable and np.isclose(ball.mass, golf.BALL_MASS)
        assert ball.engine[sim.bullet_body_friction] == pytest.approx(0.8)
        assert np.allclose(green.ball_position(), [0.5, 0.0, golf.BALL_RADIUS])
        assert np.allclose(green.hole_position, [1.2, 0.1, 0.0])
        assert sim.objects[green.hole].static and not getattr(sim.objects[green.hole], "respondable", False)
        assert np.isclose(green.distance_to_hole(), np.hypot(0.7, 0.1))
        assert not green.holed()
        green.remove()
        assert green.ball not in sim.objects and green.hole not in sim.objects


def test_holed_needs_position_and_low_speed():
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        green = golf.build_green(scene, ball_position=(0.5, 0.0), hole_position=(1.2, 0.1))
        sim.setObjectPosition(green.ball, [1.21, 0.09, golf.BALL_RADIUS])
        assert green.holed()
        sim.objects[green.ball].velocity = np.array([0.5, 0, 0])
        assert not green.holed()  # rolling over the hole does not count
        assert green.holed(speed_max=1.0)


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
