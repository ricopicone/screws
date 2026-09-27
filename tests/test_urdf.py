import pathlib

import numpy as np
import pytest

from screws import robots, urdf


def test_rrp_matches_notes():
    r = robots.rrp()
    assert r.name == "rrp"
    assert r.joint_types == ("revolute", "revolute", "prismatic")
    assert r.joint_names == ("joint1", "joint2", "joint3")
    assert np.allclose(r.M, [[0, 0, 1, 0.5], [0, 1, 0, 0], [-1, 0, 0, 0.4], [0, 0, 0, 1]], atol=1e-4)
    assert np.allclose(r.S.T, [[0, 0, 1, 0, 0, 0], [0, -1, 0, 0.4, 0, 0], [0, 0, 0, 1, 0, 0]])
    assert len(r.joint_frames_home) == 3
    assert np.allclose(r.joint_frames_home[1][:3, 3], [0, 0, 0.4])
    assert r.link_inertias is None and r.link_frames is None


def test_ur5_urdf_matches_textbook_table_to_3_decimals():
    u, t = robots.ur5(), robots.ur5(source="textbook")
    assert u.n == t.n == 6 and u.joint_types == ("revolute",) * 6
    assert np.allclose(u.S[:3], t.S[:3])  # omegas agree exactly
    assert np.abs(u.S[3:] - t.S[3:]).max() < 6e-4  # v differ in the 3rd decimal
    assert np.allclose(u.M[:3, 3], [0.8173, 0.1915, -0.0055], atol=5e-5)
    assert np.allclose(u.M[:3, :3], t.M[:3, :3], atol=1e-8)


def test_ur5_inertias_follow_mr_convention():
    u = robots.ur5()
    assert len(u.link_frames) == 7 and len(u.link_inertias) == 6
    G1 = u.link_inertias[0]
    assert np.allclose(np.diag(G1), [0.010267495893] * 2 + [0.00666] + [3.7] * 3)
    assert np.allclose(G1 - np.diag(np.diag(G1)), 0)
    prod = np.eye(4)
    for Mi in u.link_frames:
        prod = prod @ Mi
    assert np.allclose(prod, u.M)  # chain of link frames ends at {b}
    assert np.allclose(u.link_frames[0][:3, 3], [0, 0, 0.089159])
    # link 2's centre of mass sits 0.28 m along joint 2's local z
    assert np.allclose(u.link_frames[1][:3, 3], [0.28, 0.13585, 0], atol=1e-6)


def test_textbook_ur5_has_no_inertias_and_matches_notes_example():
    t = robots.ur5(source="textbook")
    assert t.link_inertias is None
    T = t.fk([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
    assert np.allclose(T[:3, 3], [0.392, 0.191, 0.419], atol=1e-3)
    assert np.allclose(T[:3, :3], [[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    with pytest.raises(ValueError, match="source"):
        robots.ur5(source="coppelia")


def test_ur5_textbook_configuration_matches_mr_example_4_5():
    u = robots.ur5()
    T = u.fk([0, -np.pi / 2, 0, 0, np.pi / 2, 0])
    assert np.allclose(T[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6)
    assert np.allclose(T[:3, 3], [0.095, 0.109, 0.988], atol=2e-3)


def test_from_urdf_string_and_robot_classmethod():
    from screws.robot import Robot

    xml = pathlib.Path(urdf.packaged("rrp.urdf")).read_text()
    assert urdf.load(xml).n == 3
    assert Robot.from_urdf(urdf.packaged("rrp.urdf")).n == 3


def test_ambiguous_chain_raises():
    xml = """<robot name="y">
    <joint name="a" type="fixed"><parent link="base"/><child link="l1"/><origin xyz="0 0 0" rpy="0 0 0"/></joint>
    <joint name="b" type="fixed"><parent link="base"/><child link="l2"/><origin xyz="0 0 0" rpy="0 0 0"/></joint>
    </robot>"""
    with pytest.raises(ValueError, match="l1.*l2|l2.*l1"):
        urdf.load(xml)
    assert urdf.load(xml, ee_link="l1").n == 0


def test_unknown_joint_type_raises():
    with pytest.raises(ValueError, match="floating"):
        urdf.load(
            '<robot name="z"><joint name="j" type="floating"><parent link="a"/>'
            '<child link="b"/></joint></robot>'
        )


def test_limits_are_read():
    xml = """<robot name="lim">
    <joint name="j1" type="revolute"><parent link="a"/><child link="b"/>
      <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-1.5" upper="2.5" effort="10" velocity="1"/></joint>
    </robot>"""
    r = urdf.load(xml)
    assert np.allclose(r.joint_limits, [[-1.5, 2.5]])


def test_axis_is_normalised_and_partial_limits_become_inf():
    xml = """<robot name="p">
    <joint name="j1" type="revolute"><parent link="a"/><child link="b"/>
      <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 2"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
    <joint name="j2" type="continuous"><parent link="b"/><child link="c"/>
      <origin xyz="0 0 0.3" rpy="0 0 0"/><axis xyz="0 3 0"/></joint>
    </robot>"""
    r = urdf.load(xml)
    assert np.allclose(r.S[:3, 0], [0, 0, 1]) and np.allclose(r.S[:3, 1], [0, 1, 0])
    assert np.allclose(r.joint_limits[0], [-1, 1])
    assert r.joint_limits[1, 0] == -np.inf and r.joint_limits[1, 1] == np.inf
    assert r.within_limits([0.5, 100.0]) and not r.within_limits([1.5, 0.0])


def test_no_joints_gives_a_readable_error():
    with pytest.raises(ValueError, match="no joints"):
        urdf.load('<robot name="empty"><link name="only"/></robot>')
