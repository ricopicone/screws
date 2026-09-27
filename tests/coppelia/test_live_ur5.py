"""Tests against a running CoppeliaSim (4.9 or later; verified on 4.10.0) with the UR5 model
loaded at /UR5 (Model browser > robots > non-mobile > UR5, or sim.loadModel on UR5.ttm).

Run with: SCREWS_COPPELIASIM=1 uv run pytest -q -m coppelia
They are skipped otherwise (see tests/conftest.py).
"""

import numpy as np
import pytest

from screws import robots

pytestmark = pytest.mark.coppelia


@pytest.fixture
def scene():
    from screws.coppelia import Scene

    with Scene() as sc:
        yield sc


def test_connects_and_finds_six_joints(scene):
    arm = scene.arm("/UR5")
    assert arm.n == 6
    assert arm.joint_types() == ("revolute",) * 6


def test_robot_off_the_scene_matches_tip_frame(scene):
    arm = scene.arm("/UR5")
    r = arm.robot()
    rng = np.random.default_rng(0)
    for _ in range(5):
        th = rng.uniform(-1.5, 1.5, size=6)
        arm.teleport(th)
        assert np.allclose(r.fk(th), arm.tip_frame(), atol=1e-4)


def test_scene_robot_has_ur5_geometry(scene):
    # The model stands in a different zero pose from the textbook figure (rotated about z),
    # so compare what is pose-independent: the parallel-axis pattern and the link lengths.
    r = scene.arm("/UR5").robot(relative_to="base")
    w = r.S[:3].T
    q = np.array([np.cross(w[i], r.S[3:, i]) for i in range(6)])  # nearest point on each axis
    dot = lambda i, j: abs(float(w[i] @ w[j]))
    assert dot(0, 1) < 1e-6  # shoulder pan perpendicular to shoulder lift
    assert all(dot(1, k) > 1 - 1e-6 for k in (2, 3, 5))  # joints 2, 3, 4, 6 parallel
    assert dot(3, 4) < 1e-6 and dot(0, 4) > 1 - 1e-6  # wrist 2 parallel to joint 1
    u = robots.ur5()
    qu = np.array([np.cross(u.S[:3, i], u.S[3:, i]) for i in range(6)])
    # upper-arm and forearm lengths along the common normal of the parallel axes
    for a, b in ((1, 2), (2, 3)):
        d_scene = np.linalg.norm(np.cross(q[b] - q[a], w[a]))
        d_book = np.linalg.norm(np.cross(qu[b] - qu[a], u.S[:3, a]))
        assert np.isclose(d_scene, d_book, atol=2e-3), (a, b, d_scene, d_book)


def test_each_mode_steps_without_error(scene):
    arm = scene.arm("/UR5")
    scene.start()
    for mode, u in (("position", arm.theta()), ("velocity", np.zeros(6)), ("torque", np.zeros(6))):
        arm.mode(mode)
        arm.command(u)
        scene.step()
    assert scene.time > 0


def test_torque_sign_moves_joint_the_right_way(scene):
    arm = scene.arm("/UR5")
    arm.mode("torque")
    scene.start()
    before = arm.theta()[0]
    for _ in range(20):
        arm.command([+5.0, 0, 0, 0, 0, 0])
        scene.step()
    assert arm.theta()[0] > before


def test_scene_inertias_are_plausible(scene):
    arm = scene.arm("/UR5")
    r = arm.robot(inertias=True, relative_to="base")
    masses = [G[3, 3] for G in r.link_inertias]
    assert len(masses) == 6 and all(m > 0 for m in masses)
    assert 10 < sum(masses) < 30
    for G in r.link_inertias:
        assert np.all(np.linalg.eigvalsh(G[:3, :3]) > 0)
    g = r.gravity_forces(np.zeros(6))
    assert np.all(np.isfinite(g))


def test_records_a_movie_of_twenty_steps(scene, tmp_path):
    arm = scene.arm("/UR5")
    arm.mode("position")
    theta0 = arm.theta()
    with scene.record_video(tmp_path / "ur5.mp4", resolution=(320, 240)) as rec:
        scene.run(lambda t, th, dth: theta0 + 0.3 * np.sin(2 * t), duration=1.0, arm=arm)
    assert rec.frames.shape == (20, 240, 320, 3)
    assert rec.saved is not None and rec.saved.stat().st_size > 1000
    # the picture is not blank and changes as the arm moves
    assert rec.frames[0].std() > 5 and not np.array_equal(rec.frames[0], rec.frames[-1])


def test_putt_moves_the_ball_toward_the_hole(scene):
    from screws.coppelia import golf

    scene.set_time_step(0.01)  # a kinematic face must move a few mm per step to strike cleanly
    arm = scene.arm("/UR5")
    arm.teleport(np.zeros(6))
    putter = golf.attach_putter(scene, arm)
    rf = putter.robot(arm.robot())
    green = golf.build_green(scene, ball_position=(0.55, 0.30), hole_position=(0.85, 0.30))
    seed = np.array([-1.21, 0.18, 1.4, -0.01, -1.57, 0.36])
    arm.mode("kinematic")
    arm.teleport(seed)
    scene.start()
    try:
        res = golf.putt(scene, arm, rf, green, speed=0.3, seed=seed, settle_time=6.0)
    finally:
        scene.stop()
        green.remove()
        putter.remove()
    start, end = res.ball_path[0], res.ball_path[-1]
    assert end[0] - start[0] > 0.15  # rolled toward the hole along +x
    assert abs(end[1] - start[1]) < 0.05  # and stayed on the line
    assert res.distance < 0.25
    assert res.face_path[:, 2].min() > green.top + 0.002  # the putter never dips into the turf
