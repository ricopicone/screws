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
        raise ZeroDivisionError("controller crashed")
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


def test_exit_closes_the_socket_it_opened():
    sim = two_joint_scene()

    class Sock:
        closed = False

        def close(self):
            self.closed = True

    class Client:
        socket = Sock()

    sim._screws_client = Client()
    with Scene(sim=sim):
        pass
    assert sim._screws_client.socket.closed


def test_show_frame_replaces_a_named_triad_and_clear_removes_all():
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        scene.show_frame(np.eye(4), "goal")
        scene.show_frame(np.eye(4), "goal")
        scene.show_frame(np.eye(4), "other")
        assert len(sim.live_drawings()) == 6  # two triads, the first "goal" replaced
        scene.clear_frames()
        assert len(sim.live_drawings()) == 0


def test_run_reads_state_once_per_step():
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        arm = scene.arm("/Arm")
        arm.mode("position")
        sim.calls.clear()
        scene.run(lambda t, th, dth: th, duration=0.1, arm=arm)
    reads = [c for c in sim.calls if c[0] == "getJointPosition"]
    assert len(reads) == 2 * arm.n  # 2 steps x n joints, read once per step


def test_start_stops_a_stale_running_simulation_first():
    sim = two_joint_scene()
    sim.running = True  # left over from a crashed client
    sim.calls.clear()
    with Scene(sim=sim) as scene:
        scene.start()
        assert ("stopSimulation",) in sim.calls
        assert sim.calls.index(("stopSimulation",)) < sim.calls.index(("startSimulation",))
        assert sim.running


def test_exit_deregisters_the_stepping_client():
    sim = two_joint_scene()
    with Scene(sim=sim):
        assert sim.stepping
    assert not sim.stepping  # otherwise the server waits forever for this client's next step()


def test_log_plot_draws_four_axes_and_can_skip_show(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    with Scene(sim=two_joint_scene()) as scene:
        arm = scene.arm("/Arm")
        arm.mode("position")
        log = scene.run(lambda t, th, dth: [0.1, 0.2], duration=0.1, arm=arm)
    fig = log.plot()
    assert len(fig.axes) == 4 and shown == [True]
    fig2 = log.plot(show=False)
    assert len(fig2.axes) == 4 and shown == [True]
    plt.close("all")


def test_stop_waits_until_the_simulation_has_actually_stopped():
    sim = two_joint_scene()
    sim.stop_lag = 3  # the simulator reports "running" for three more polls after stopSimulation
    with Scene(sim=sim) as scene:
        scene.start()
        scene.stop()
        assert sim.getSimulationState() == sim.simulation_stopped
        assert sim.stop_lag == 0  # stop() polled until the state settled


def test_tracked_objects_are_removed_on_exit_even_after_an_exception():
    sim = two_joint_scene()
    with pytest.raises(ZeroDivisionError), Scene(sim=sim) as scene:
        h = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [0.04] * 3, 0)
        scene.track(h)
        assert h in sim.objects
        raise ZeroDivisionError("controller crashed")
    assert h not in sim.objects


def test_set_time_step_changes_dt_and_is_restored_on_exit():
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        assert scene.dt == 0.05
        scene.set_time_step(0.01)
        assert scene.dt == 0.01
        with pytest.raises(RuntimeError, match="running"):
            scene.start()
            scene.set_time_step(0.02)
    assert sim.getSimulationTimeStep() == 0.05  # put back when the Scene exits
