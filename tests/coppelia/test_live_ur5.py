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
