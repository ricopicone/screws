import numpy as np
import pytest

from screws.coppelia import Recorder, Scene
from tests.coppelia.fake_sim import two_joint_scene


def test_camera_is_created_placed_and_removed_on_exit():
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        cam = scene.camera(position=(1.5, -1.5, 1.0), look_at=(0, 0, 0.4), resolution=(64, 48))
        T = scene.frame(cam)
        assert np.allclose(T[:3, 3], [1.5, -1.5, 1.0])
        # the sensor looks along its +z axis: z points from the camera to the target
        z = T[:3, 2]
        toward = np.array([0, 0, 0.4]) - np.array([1.5, -1.5, 1.0])
        assert np.allclose(z, toward / np.linalg.norm(toward), atol=1e-9)
        assert abs(T[2, 0]) < 1e-9  # x axis stays horizontal, so the image is not rolled
        assert sim.objects[cam].resolution == (64, 48)
    assert cam not in sim.objects  # removed on exit


def test_existing_camera_path_is_reused_and_kept():
    sim = two_joint_scene()
    cam0 = sim.add("/Arm/cam", "visionsensor", np.eye(4))
    sim.objects[cam0].resolution = (32, 32)
    with Scene(sim=sim) as scene:
        assert scene.camera("/Arm/cam") == cam0
    assert cam0 in sim.objects


def test_recorder_captures_one_upright_frame_per_step(tmp_path):
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        arm = scene.arm("/Arm")
        arm.mode("position")
        with scene.record_video(tmp_path / "run.gif", resolution=(32, 24)) as rec:
            scene.run(lambda t, th, dth: th, duration=0.2, arm=arm)
        assert isinstance(rec, Recorder)
        assert rec.frames.shape == (4, 24, 32, 3) and rec.frames.dtype == np.uint8
        # upright: blue sky at the top row, brown ground at the bottom row
        assert rec.frames[0, 0, 0, 2] == 255 and rec.frames[0, -1, 0, 0] == 120
        assert rec.fps == pytest.approx(1 / sim.dt)
        assert not np.array_equal(rec.frames[0], rec.frames[1])  # frames follow the clock
    import imageio.v3 as iio

    back = iio.imread(tmp_path / "run.gif")
    assert back.shape[0] == 4


def test_every_and_manual_capture(tmp_path):
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        arm = scene.arm("/Arm")
        arm.mode("position")
        with scene.record_video(tmp_path / "run.mp4", every=2, resolution=(32, 24)) as rec:
            scene.run(lambda t, th, dth: th, duration=0.2, arm=arm)
            assert rec.frames.shape[0] == 2 and rec.fps == pytest.approx(1 / (2 * sim.dt))
            rec.capture()
            assert rec.frames.shape[0] == 3
    assert (tmp_path / "run.mp4").stat().st_size > 0


def test_unknown_extension_raises_before_the_run(tmp_path):
    with Scene(sim=two_joint_scene()) as scene, pytest.raises(ValueError, match="avi"):
        scene.record_video(tmp_path / "run.avi")
