import numpy as np
import pytest

from screws import se3, so3
from screws import trajectory as tr


def test_time_scalings_match_mr():
    assert np.isclose(tr.cubic_time_scaling(2, 0.6), 0.216)
    assert np.isclose(tr.quintic_time_scaling(2, 0.6), 0.16308)
    assert tr.cubic_time_scaling(2, 0) == 0 and tr.cubic_time_scaling(2, 2) == 1
    assert tr.quintic_time_scaling(2, 0) == 0 and tr.quintic_time_scaling(2, 2) == 1


def test_joint_trajectory_matches_mr():
    start = [1, 0, 0, 1, 1, 0.2, 0, 1]
    end = [1.2, 0.5, 0.6, 1.1, 2, 2, 0.9, 1]
    traj = tr.joint_trajectory(start, end, 4, 6, scaling="cubic")
    assert traj.shape == (6, 8)
    assert np.allclose(traj[1], [1.0208, 0.052, 0.0624, 1.0104, 1.104, 0.3872, 0.0936, 1])
    assert np.allclose(traj[0], start) and np.allclose(traj[-1], end)
    q = tr.joint_trajectory(start, end, 4, 6)  # quintic by default
    assert np.allclose(q[0], start) and np.allclose(q[-1], end) and not np.allclose(q[1], traj[1])


def test_bad_scaling_and_single_row():
    with pytest.raises(ValueError, match="scaling"):
        tr.joint_trajectory([0], [1], 1, 5, scaling="linear")
    assert np.allclose(tr.joint_trajectory([0, 1], [2, 3], 1, 1), [[0, 1]])


X_START = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1.0]])
X_END = np.array([[0, 0, 1, 0.1], [1, 0, 0, 0], [0, 1, 0, 4.1], [0, 0, 0, 1.0]])


def test_screw_trajectory_matches_mr():
    traj = tr.screw_trajectory(X_START, X_END, 5, 4, scaling="cubic")
    assert len(traj) == 4 and traj[0].shape == (4, 4)
    assert np.allclose(traj[0], X_START) and np.allclose(traj[-1], X_END, atol=1e-6)
    assert np.allclose(traj[1][:3, 3], [0.441, 0.529, 1.601], atol=1e-3)
    assert np.allclose(traj[2][:3, 3], [-0.117, 0.473, 3.274], atol=1e-3)


def test_cartesian_trajectory_matches_mr_and_moves_straight():
    traj = tr.cartesian_trajectory(X_START, X_END, 5, 4, scaling="quintic")
    assert np.allclose(traj[1][:3, 3], [0.811, 0, 1.651], atol=1e-3)
    assert np.allclose(traj[2][:3, 3], [0.289, 0, 3.449], atol=1e-3)
    # positions lie on the straight segment between the endpoints
    for T in traj:
        d = T[:3, 3] - X_START[:3, 3]
        seg = X_END[:3, 3] - X_START[:3, 3]
        assert np.allclose(np.cross(d, seg), 0, atol=1e-9)
        assert so3.is_so3(T[:3, :3]) and se3.is_se3(T)
