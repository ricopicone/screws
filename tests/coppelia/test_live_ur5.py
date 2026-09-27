"""Tests against a running CoppeliaSim 4.9 with the UR5 model loaded at /UR5.

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


def test_scene_robot_omegas_match_textbook_ur5(scene):
    # Same axis directions as the textbook's UR5 (up to the model's base placement).
    r = scene.arm("/UR5").robot()
    u = robots.ur5()
    assert np.allclose(np.abs(r.S[:3]), np.abs(u.S[:3]), atol=1e-6)


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
