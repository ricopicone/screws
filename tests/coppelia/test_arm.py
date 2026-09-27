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


def test_torque_mode_sets_force_and_velocity_sign():
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        arm = scene.arm("/Arm")
        arm.mode("torque")
        assert all(sim.ctrl_mode[h] == sim.jointdynctrl_force for h in arm.handles)
        arm.command([-2.0, 3.0])
        h1, h2 = arm.handles
        assert sim.target_force[h1] == 2.0 and sim.target_dq[h1] < 0
        assert sim.target_force[h2] == 3.0 and sim.target_dq[h2] > 0


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
