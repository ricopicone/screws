import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from screws import robots, se3, viz


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_draw_robot_space_frame_draws_an_axis_per_joint():
    ur5 = robots.ur5()
    d = viz.draw_robot(ur5)
    assert d.fig is not None and d.ax.name == "3d"
    assert len(d.joints) == 6
    for i, j in enumerate(d.joints):
        assert np.allclose(j.omega, ur5.S[:3, i]) and np.allclose(j.v, ur5.S[3:, i])
        assert np.allclose(j.q, ur5.joint_frames_home[i][:3, 3])  # the joint's own home origin
        assert j.axis_line is not None and j.omega_arrow is not None
        assert j.v_arrow is None  # construction is off by default
    assert d.frame == "space"


def test_draw_robot_body_frame_uses_B_and_places_b_at_the_origin():
    ur5 = robots.ur5()
    d = viz.draw_robot(ur5, frame="body")
    for i, j in enumerate(d.joints):
        assert np.allclose(j.omega, ur5.B[:3, i]) and np.allclose(j.v, ur5.B[3:, i])
        # the joint's home origin, now expressed in {b}
        expected = (se3.transform_inv(ur5.M) @ ur5.joint_frames_home[i])[:3, 3]
        assert np.allclose(j.q, expected)
    assert np.allclose(d.skeleton[-1], 0)  # the tool sits at {b}'s origin, and the arm is at home
    assert np.allclose(d.skeleton[0], se3.transform_inv(ur5.M)[:3, 3])  # the base, seen from {b}


def test_construction_vectors_add_up_to_v():
    rrp = robots.rrp()
    d = viz.draw_robot(rrp, joints=[1], show=("skeleton", "axes", "omega", "construction"))
    j = d.joints[0]
    assert j.index == 1 and j.kind == "revolute"
    assert np.allclose(j.cross + j.pitch_term, j.v)
    assert np.allclose(j.cross, -np.cross(j.omega, j.q))
    assert np.isclose(j.pitch, 0.0) and np.allclose(j.pitch_term, 0)
    assert j.v_arrow is not None and j.q_line is not None
    assert j.cross_arrow is None and j.pitch_arrow is None  # zero pitch: v *is* the cross product, one arrow


def test_nonzero_pitch_draws_all_three_pieces():
    from screws import Robot

    S = se3.screw_axis(q=[0.3, 0.1, 0.2], s_hat=[0, 0, 1], h=0.05)
    r = Robot.from_screw_axes(np.eye(4), [S])
    d = viz.draw_robot(r, show=("axes", "omega", "construction"))
    j = d.joints[0]
    assert np.isclose(j.pitch, 0.05) and np.allclose(j.pitch_term, [0, 0, 0.05])
    assert j.cross_arrow is not None and j.pitch_arrow is not None and j.v_arrow is not None
    assert np.allclose(j.cross + j.pitch_term, j.v)


def test_prismatic_joint_draws_v_only():
    rrp = robots.rrp()
    d = viz.draw_robot(rrp, joints=[2], show=("axes", "omega", "construction"))
    j = d.joints[0]
    assert j.kind == "prismatic" and np.allclose(j.omega, 0)
    assert j.axis_line is not None and j.omega_arrow is None and j.v_arrow is not None


def test_posed_skeleton_matches_fk():
    ur5 = robots.ur5()
    theta = [0.3, -1.0, 1.2, -0.5, 0.8, 0.1]
    d = viz.draw_robot(ur5, theta=theta)
    assert np.allclose(d.skeleton[-1], ur5.fk(theta)[:3, 3])
    assert d.skeleton.shape == (8, 3)  # base, six joints, tool


def test_draw_screw_shows_the_pitch_term():
    S = se3.screw_axis(q=[0.3, 0, 0], s_hat=[0, 0, 1], h=0.05)
    d = viz.draw_screw(S)
    assert np.isclose(d.pitch, 0.05) and np.allclose(d.pitch_term, [0, 0, 0.05])
    assert np.allclose(d.q, [0.3, 0, 0])  # the point on the axis nearest the origin
    assert np.allclose(d.cross + d.pitch_term, S[3:])
    assert d.pitch_arrow is not None


def test_draw_into_a_given_axes_and_scale():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    d = viz.draw_robot(robots.ur5(), ax=ax, axis_length=0.5)
    assert d.ax is ax and d.fig is fig
    x0, x1 = d.joints[0].axis_line.get_data_3d()[2][:2]
    assert np.isclose(abs(x1 - x0), 1.0)  # a 0.5 m half-length either way along joint 1's z axis
