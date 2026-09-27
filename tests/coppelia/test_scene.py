import numpy as np
import pytest

from screws.coppelia import Scene
from tests.coppelia.fake_sim import two_joint_scene


def test_scene_steps_time_and_reads_frames():
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        assert sim.stepping
        scene.start()
        t0 = scene.time
        scene.step()
        assert np.isclose(scene.time - t0, scene.dt) and scene.dt == 0.05
        assert np.allclose(scene.frame("/Arm/tip")[:3, 3], [0.3, 0, 0.5])
    assert not sim.running


def test_scene_stops_on_exception():
    sim = two_joint_scene()
    with pytest.raises(ZeroDivisionError), Scene(sim=sim) as scene:
        scene.start()
        1 / 0
    assert not sim.running


def test_missing_path_raises_with_path():
    with Scene(sim=two_joint_scene()) as scene, pytest.raises(LookupError, match="/Arm/nope"):
        scene.frame("/Arm/nope")


def test_show_frame_draws_three_lines():
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        scene.show_frame(np.eye(4), "goal")
    assert len(sim.drawings) == 3


def test_run_logs_every_step():
    with Scene(sim=two_joint_scene()) as scene:
        arm = scene.arm("/Arm")
        arm.mode("position")
        log = scene.run(lambda t, th, dth: [t, 0.0], duration=0.25, arm=arm)
    assert log.t.shape == (5,) and log.theta.shape == (5, 2) and log.T_sb.shape == (5, 4, 4)
    assert log.command.shape == (5, 2) and np.allclose(log.command[:, 0], log.t)
    assert np.allclose(log.t, [0, 0.05, 0.1, 0.15, 0.2])


def test_log_csv_and_mr_csv(tmp_path):
    with Scene(sim=two_joint_scene()) as scene:
        arm = scene.arm("/Arm")
        arm.mode("position")
        log = scene.run(lambda t, th, dth: [0.1, 0.2], duration=0.1, arm=arm)
    csv = tmp_path / "log.csv"
    log.to_csv(csv)
    lines = csv.read_text().splitlines()
    assert lines[0].startswith("t,theta_1,theta_2,dtheta_1")
    assert len(lines) == 3
    mr = tmp_path / "mr.csv"
    log.to_mr_csv(mr)
    rows = mr.read_text().splitlines()
    assert len(rows) == 2 and rows[1].count(",") == 1  # joint angles only, no header
