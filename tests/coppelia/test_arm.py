import numpy as np
import pytest

from screws.coppelia import Scene
from tests.coppelia.fake_sim import two_joint_scene


def test_arm_finds_joints_base_to_tip():
    with Scene(sim=two_joint_scene()) as scene:
        arm = scene.arm("/Arm")
        assert arm.n == 2 and arm.joint_names == ("j1", "j2")
        assert arm.tip_alias == "tip"


def test_arm_reads_and_commands_positions():
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        arm = scene.arm("/Arm")
        arm.mode("position")
        scene.start()
        arm.command([0.1, 0.2])
        scene.step()
        assert np.allclose(arm.theta(), [0.1, 0.2])
        assert np.allclose(arm.dtheta(), [2.0, 4.0])
        assert arm.tau().shape == (2,)


def test_wrong_mode_raises():
    with Scene(sim=two_joint_scene()) as scene:
        arm = scene.arm("/Arm")
        arm.mode("position")
        with pytest.raises(RuntimeError, match="torque"):
            arm.command_torques([1, 1])
        with pytest.raises(ValueError, match="mode"):
            arm.mode("impedance")


def test_torque_mode_sends_signed_torques():
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        arm = scene.arm("/Arm")
        arm.mode("torque")
        assert all(sim.ctrl_mode[h] == sim.jointdynctrl_force for h in arm.handles)
        arm.command([-2.0, 3.0])
        h1, h2 = arm.handles
        assert sim.target_force[h1] == -2.0 and sim.target_force[h2] == 3.0
        scene.start()
        scene.step()
        assert np.allclose(arm.tau(), [-2.0, 3.0])


def test_command_length_must_match_joint_count():
    with Scene(sim=two_joint_scene()) as scene:
        arm = scene.arm("/Arm")
        arm.mode("position")
        with pytest.raises(ValueError, match="2 joints"):
            arm.command([0.1])
        with pytest.raises(ValueError, match="2 joints"):
            arm.teleport([0.1, 0.2, 0.3])


def test_alias_lookup_uses_plain_aliases():
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        scene.arm("/Arm")
    assert sim.alias_options and all(opt == -1 for opt in sim.alias_options)


def test_tau_is_nan_when_the_joint_reports_no_force():
    sim = two_joint_scene()
    sim.force_errors = True
    with Scene(sim=sim) as scene:
        arm = scene.arm("/Arm")
        assert np.all(np.isnan(arm.tau()))


def test_teleport_is_kinematic_and_moves_tip():
    with Scene(sim=two_joint_scene()) as scene:
        arm = scene.arm("/Arm")
        arm.teleport([np.pi / 2, 0])
        assert np.allclose(arm.theta(), [np.pi / 2, 0])
        assert np.allclose(arm.tip_frame()[:3, 3], [0, 0.3, 0.5], atol=1e-9)


def test_robot_from_scene_derives_screw_axes_and_restores_pose():
    with Scene(sim=two_joint_scene()) as scene:
        arm = scene.arm("/Arm")
        arm.teleport([0.7, -0.4])
        r = arm.robot()
        assert np.allclose(arm.theta(), [0.7, -0.4])  # put back where it was
        assert r.name == "Arm" and r.joint_types == ("revolute", "revolute")
        assert np.allclose(r.S.T, [[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
        assert np.allclose(r.M[:3, 3], [0.3, 0, 0.5]) and len(r.joint_frames_home) == 2
        assert np.allclose(r.joint_limits, [[-3, 3], [-2, 2]])
        # the derived robot's FK agrees with the simulator's tip frame
        arm.teleport([0.3, 0.9])
        assert np.allclose(r.fk([0.3, 0.9]), arm.tip_frame())
        with pytest.raises(NotImplementedError, match="0.2"):
            arm.robot(inertias=True)


def test_robot_from_offset_tilted_and_prismatic_joints():
    from tests.coppelia.fake_sim import three_joint_scene

    with Scene(sim=three_joint_scene()) as scene:
        arm = scene.arm("/Rig")
        r = arm.robot()
        assert r.joint_types == ("revolute", "revolute", "prismatic")
        # j1: z axis through the base at (1, 0, 0): v = -omega x q = -(0,0,1)x(1,0,0) = (0,-1,0)
        assert np.allclose(r.S[:, 0], [0, 0, 1, 0, -1, 0])
        # j2: about world y through (1, 0, 0.5): v = -(0,1,0)x(1,0,0.5) = (-0.5, 0, 1)
        assert np.allclose(r.S[:, 1], [0, 1, 0, -0.5, 0, 1])
        # j3: prismatic along world x
        assert np.allclose(r.S[:, 2], [0, 0, 0, 1, 0, 0])
        assert np.allclose(r.M[:3, 3], [1.4, 0, 0.5])
        rng = np.random.default_rng(5)
        for _ in range(5):
            th = rng.uniform(-1, 1, size=3)
            arm.teleport(th)
            assert np.allclose(r.fk(th), arm.tip_frame(), atol=1e-9)


def test_tip_falls_back_to_last_joints_child():
    from tests.coppelia.fake_sim import three_joint_scene

    sim = three_joint_scene(with_tip=False)
    with Scene(sim=sim) as scene:
        arm = scene.arm("/Rig")
        assert arm.tip_alias == "slider"
        assert np.allclose(arm.tip_frame()[:3, 3], [1.4, 0, 0.5])
