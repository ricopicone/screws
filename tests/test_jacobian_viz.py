import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import screws
from screws import robots, se3, viz

PI = np.pi
UR5_THETA = np.array([0.3, -1.1, 1.4, 0.2, 0.9, -0.5])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def rebuilt(c):
    """The screw axis read back off a drawn line: (s_hat, -s_hat x q + h s_hat), or (0, s_hat)."""
    if np.isinf(c.h):
        return np.concatenate([np.zeros(3), c.s_hat])
    return np.concatenate([c.s_hat, -np.cross(c.s_hat, c.q) + c.h * c.s_hat])


@pytest.mark.parametrize("frame", ["space", "body"])
@pytest.mark.parametrize("robot", [robots.ur5(), robots.rrp()], ids=["ur5", "rrp"])
def test_each_drawn_axis_line_is_its_jacobian_column(robot, frame):
    theta = UR5_THETA if robot.n == 6 else np.array([PI / 2, PI / 4, 0.2])
    d = viz.draw_jacobian(robot, theta, frame=frame)
    J = robot.jacobian_space(theta) if frame == "space" else robot.jacobian_body(theta)
    assert np.allclose(d.J, J) and not d.planar and d.ax.name == "3d"
    for i, c in enumerate(d.columns):
        assert np.allclose(c.S, J[:, i])
        assert np.allclose(rebuilt(c), J[:, i]), f"joint {i + 1}"


def test_tool_velocity_columns_sum_head_to_tail_to_p_dot():
    ur5 = robots.ur5()
    dtheta = np.array([0.5, -0.2, 0.3, 0.1, -0.4, 0.7])
    d = viz.draw_jacobian(ur5, UR5_THETA, dtheta=dtheta)
    T = ur5.fk(UR5_THETA)
    p_dot = T[:3, :3] @ ur5.jacobian_body(UR5_THETA)[3:] @ dtheta
    assert np.allclose(sum(c.tool_velocity * w for c, w in zip(d.columns, dtheta)), p_dot)
    assert np.allclose(d.velocity, p_dot)
    V_s = ur5.jacobian_space(UR5_THETA) @ dtheta  # p-dot = v_s + omega_s x p, MR 3.3.2
    assert np.allclose(p_dot, V_s[3:] + np.cross(V_s[:3], T[:3, 3]))
    body = viz.draw_jacobian(ur5, UR5_THETA, dtheta=dtheta, frame="body")
    assert np.allclose(body.velocity, ur5.jacobian_body(UR5_THETA)[3:] @ dtheta)  # v_b


def test_planar_2r_is_drawn_like_any_robot_and_flat_when_asked():
    arm = robots.planar_2r()
    assert viz.is_planar(arm) and not viz.is_planar(robots.ur5()) and not viz.is_planar(robots.rrp())
    plain = viz.draw_jacobian(arm, [0, PI / 4])
    assert not plain.planar and plain.ax.name == "3d" and np.allclose(plain.J, arm.jacobian_space([0, PI / 4]))
    d = viz.draw_jacobian(arm, [0, PI / 4], planar=True)
    assert d.planar and d.ax.name != "3d"
    assert np.allclose(d.J, [[-0.7071, -0.7071], [1.7071, 0.7071]], atol=1e-4)
    assert d.rank == 2 and not d.singular
    stretched = viz.draw_jacobian(arm, [0.3, 0.0], planar=True)
    assert stretched.rank == 1 and stretched.singular and "singular" in stretched.ax.get_title()
    with pytest.raises(ValueError, match="not one"):
        viz.draw_jacobian(robots.ur5(), UR5_THETA, planar=True)


def test_ur5_singular_pose_has_rank_five():
    """The notes' ur5-singular-poses (a): theta5 = 0 makes axes 2, 3, 4 and 6 parallel."""
    d = viz.draw_jacobian(robots.ur5("textbook"), np.deg2rad([0, -90, 90, 0, 0, 0]))
    assert d.rank == 5 and d.singular


def test_draw_ellipsoid_planar_matches_the_notes_and_the_core():
    arm = robots.planar_2r()
    d = viz.draw_ellipsoid(arm, [0, PI / 4], force=True, planar=True)
    assert d.planar and d.ax.name != "3d"
    assert np.allclose(d.manipulability.radii, [0.341, 2.073], atol=1e-3)
    core = arm.manipulability_ellipsoid([0, PI / 4], planar=True)
    assert np.allclose(d.manipulability.radii, core.radii) and np.allclose(d.manipulability.axes, core.axes)
    assert np.allclose(d.force.radii, [2.932, 0.482], atol=1e-3)
    assert np.allclose(d.manipulability.center, [1 + np.cos(PI / 4), np.sin(PI / 4)])
    reach = np.linalg.norm(d.manipulability.center)  # as far as the drawn arm extends from its base
    assert np.isclose(d.scale * d.manipulability.radii.max(), 0.3 * reach)  # the longest semi-axis 0.3 of it
    assert np.isclose(d.force_scale * d.force.radii.max(), 0.3 * reach) and not d.clipped  # its own units
    same = viz.draw_ellipsoid(arm, [0, PI / 4], force=True, force_scale=d.scale, planar=True)  # the notes' one scale
    assert same.force_scale == same.scale
    assert "μ1 = 6.08" in d.ax.get_title()
    flat = viz.draw_ellipsoid(arm, [0, PI / 4])  # in 3D: flat in z, so mu1 is infinite
    assert flat.ax.name == "3d" and np.allclose(sorted(flat.manipulability.radii), [0, 0.341, 2.073], atol=1e-3)
    assert "μ1 = ∞" in flat.ax.get_title()


def test_draw_ellipsoid_3d_is_the_robots_ellipsoid():
    ur5 = robots.ur5()
    d = viz.draw_ellipsoid(ur5, UR5_THETA, scale=0.5)
    e = ur5.manipulability_ellipsoid(UR5_THETA)
    assert d.ax.name == "3d" and d.scale == 0.5 and d.force is None
    assert np.allclose(d.manipulability.radii, e.radii) and np.allclose(d.manipulability.axes, e.axes)
    assert np.allclose(d.manipulability.center, e.center)
    ang = viz.draw_ellipsoid(ur5, UR5_THETA, kind="angular")
    assert np.allclose(ang.manipulability.radii, ur5.manipulability_ellipsoid(UR5_THETA, kind="angular").radii)


def test_singular_force_ellipse_runs_out_of_the_frame_marked_unbounded():
    d = viz.draw_ellipsoid(robots.planar_2r(), [0.3, 0.0], force=True, planar=True)
    assert np.isinf(d.force.radii).any() and d.manipulability.singular
    texts = [t for t in d.ax.texts if t.get_text() == "∞"]
    assert texts and d.clipped
    (x0, x1), (y0, y1) = d.ax.get_xlim(), d.ax.get_ylim()
    assert x1 - x0 < 10  # the plot was not stretched to hold an infinite ellipse
    x, y = texts[0].get_position()
    assert x0 < x < x1 and y0 < y < y1  # the mark is at the edge, inside the frame
    # what shows of it is its limit: two lines along the arm, 1/sqrt(lambda) either side
    outline = max(d.ax.lines, key=lambda line: len(line.get_xdata()))
    pts = np.column_stack([outline.get_xdata(), outline.get_ydata()])
    inside = pts[(pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)]
    along = d.force.axes[:, np.isinf(d.force.radii)][:, 0]
    across = (inside - d.force.center) @ np.array([-along[1], along[0]])
    width = d.force_scale * d.force.radii[np.isfinite(d.force.radii)][0]
    assert np.allclose(np.abs(across), width, rtol=5e-3)  # straight to the eye: far is 10 frames out
    # the same in 3D: the UR5 at a singular pose, linear force ellipsoid
    d3 = viz.draw_ellipsoid(robots.ur5("textbook"), np.deg2rad([0, -90, 0, 0, 30, 0]), force=True)
    assert d3.manipulability.singular == d3.force.singular


def test_planar_angular_ellipsoid_is_refused():
    with pytest.raises(ValueError, match="segment"):
        viz.draw_ellipsoid(robots.planar_2r(), [0, 1], kind="angular", planar=True)
    viz.draw_ellipsoid(robots.planar_2r(), [0, 1], kind="angular")  # fine in 3D


def test_draw_into_a_given_axes():
    fig, axes = plt.subplots(1, 2)
    a = viz.draw_ellipsoid(robots.planar_2r(), [0, PI / 4], ax=axes[0], force=True, planar=True)
    b = viz.draw_ellipsoid(robots.planar_2r(), [0, 3 * PI / 4], ax=axes[1], force=True, planar=True)
    assert a.fig is fig and b.ax is axes[1]


def test_animate_robot_constant_joint_rates(tmp_path):
    arm = robots.planar_2r()
    a = viz.animate_robot(arm, [0, PI / 2], [0.4, 0.05], frames=6, force=True, planar=True)
    th = a.data["theta"]
    assert np.allclose(th[0], [0, PI / 2]) and np.allclose(th[-1], [0.4, 0.05])
    assert np.allclose(a.data["dtheta"], [0.4, 0.05 - PI / 2])  # the same rates in every frame
    assert np.allclose(np.diff(th, axis=0), np.diff(th, axis=0)[0])
    for k in range(6):
        V_b = arm.jacobian_body(th[k]) @ a.data["dtheta"][k]
        assert np.allclose(a.data["V_b"][k], V_b)
        assert np.allclose(a.data["radii"][k], arm.manipulability_ellipsoid(th[k], planar=True).radii)
    assert a.data["radii"][-1].min() < a.data["radii"][0].min()  # approaching the stretched-out singularity
    assert a.save(tmp_path / "a.gif", dpi=30).stat().st_size > 1000


def test_animate_robot_screw_path_keeps_the_twist_constant(tmp_path):
    ur5 = robots.ur5()
    start = np.array([0.0, -1.2, 1.5, -1.6, -1.57, 0.0])
    end = np.array([0.6, -0.9, 1.1, -1.3, -1.57, 0.4])
    a = viz.animate_robot(ur5, start, end, path="screw", frames=8, show=("skeleton", "axes", "ellipsoid", "path", "velocity"))
    V_b = a.data["V_b"]
    assert np.allclose(V_b, V_b[0])
    T0, T1 = ur5.fk(start), ur5.fk(end)
    assert np.allclose(se3.vec_to_se3(V_b[0]), se3.log6(se3.transform_inv(T0) @ T1))
    # the rates are the ones the path takes: compare with finite differences of the IK
    fd = np.gradient(a.data["theta"], a.data["s"], axis=0)
    assert np.allclose(fd[1:-1], a.data["dtheta"][1:-1], atol=0.05 * np.abs(fd).max())
    assert np.allclose(a.data["T_sb"][0], T0) and np.allclose(a.data["T_sb"][-1], T1, atol=1e-8)
    # the joint rates are not constant
    assert not np.allclose(a.data["dtheta"], a.data["dtheta"][0], atol=1e-3)
    assert a.save(tmp_path / "s.gif", dpi=30).stat().st_size > 1000


def test_animate_robot_plays_a_given_path_with_the_matrix(tmp_path):
    rrp = robots.rrp()
    path = screws.joint_trajectory([0, 0.3, 0.1], [PI / 2, PI / 4, 0.3], 1.0, 5)
    a = viz.animate_robot(rrp, path, matrix=True, chart=False)
    assert np.allclose(a.data["theta"], path)
    assert a.save(tmp_path / "p.gif", dpi=30).stat().st_size > 1000
    with pytest.raises(ValueError, match="path"):
        viz.animate_robot(rrp, [0, 0, 0], [1, 1, 1], path="straight")
    with pytest.raises(ValueError):
        viz.animate_robot(rrp, [0, 0, 0])


def test_compare_paths(tmp_path):
    ur5 = robots.ur5()
    start = np.array([0.0, -1.2, 1.5, -1.6, -1.57, 0.0])
    end = np.array([0.6, -0.9, 1.1, -1.3, -1.57, 0.4])
    a = viz.compare_paths(ur5, start, end, frames=6)
    j, s = a.data["joint"], a.data["screw"]
    assert np.allclose(j["dtheta"], j["dtheta"][0]) and not np.allclose(j["V_b"], j["V_b"][0], atol=1e-3)
    assert np.allclose(j["T_sb"][-1], s["T_sb"][-1], atol=1e-8)  # the same end
    assert not np.allclose(j["tool"][3], s["tool"][3], atol=1e-3)  # by different routes
    assert a.save(tmp_path / "c.gif", dpi=30).stat().st_size > 1000


def test_a_two_joint_arm_cannot_follow_a_screw():
    with pytest.raises(ValueError, match="2 joints the tool cannot follow"):
        viz.compare_paths(robots.planar_2r(), [0, PI / 2], [PI / 2, PI / 3], frames=4)


def test_explore_slider_ranges_follow_limits_and_joint_kinds():
    from screws.viz.static import _slider_ranges

    rrp = robots.rrp()
    assert _slider_ranges(rrp.with_limits(None)) == [(-PI, PI), (-PI, PI), (-1.0, 1.0)]
    limited = robots.planar_2r().with_limits([[-1, 1], [0, 2.5]])
    assert _slider_ranges(limited) == [(-1.0, 1.0), (0.0, 2.5)]


def test_explore_redraws_any_drawing():
    pytest.importorskip("ipywidgets")
    pytest.importorskip("IPython")
    sliders = viz.explore(robots.planar_2r(), draw=viz.draw_ellipsoid, force=True)
    sliders[1].value = 0.5  # redraws without error
    assert len(sliders) == 2


def _segments_3d(ax):
    """Every line drawn on 3D axes, as (m, 2, 3) segments."""
    segs = []
    for line in ax.lines:
        x, y, z = line.get_data_3d()
        P = np.column_stack([x, y, z]).astype(float)
        segs += [P[i : i + 2] for i in range(len(P) - 1) if np.isfinite(P[i : i + 2]).all()]
    for c in ax.collections:
        segs += [np.asarray(s, float) for s in getattr(c, "_segments3d", [])]
    return np.array(segs)


def test_an_unbounded_force_ellipsoid_in_space_is_cut_at_the_frame():
    # the 2R arm drawn in 3D is flat in z, so its force ellipsoid is unbounded along z
    d = viz.draw_ellipsoid(robots.planar_2r(), [0, PI / 4], force=True)
    assert np.isinf(d.force.radii).sum() == 1 and d.clipped
    lo = np.array([d.ax.get_xlim()[0], d.ax.get_ylim()[0], d.ax.get_zlim()[0]])
    hi = np.array([d.ax.get_xlim()[1], d.ax.get_ylim()[1], d.ax.get_zlim()[1]])
    segs = _segments_3d(d.ax).reshape(-1, 3)
    assert ((segs >= lo - 1e-9) & (segs <= hi + 1e-9)).all()  # nothing drawn past the frame
    assert np.isclose(segs[:, 2].min(), lo[2]) and np.isclose(segs[:, 2].max(), hi[2])  # but up to it, along z
    assert "∞" in [t.get_text() for t in d.ax.texts]


def test_the_frame_follows_the_robot_not_the_force_ellipse_into_a_singularity():
    arm = robots.planar_2r()

    def span(anim):
        ax = anim.fig.axes[0]
        return np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim())

    plain = viz.animate_robot(arm, [0, 3 * PI / 4], [0, 0], planar=True, chart=False)
    forced = viz.animate_robot(arm, [0, 3 * PI / 4], [0, 0], force=True, planar=True, chart=False)
    assert np.all(np.array(span(forced)) < 1.3 * np.array(span(plain)))
