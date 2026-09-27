import numpy as np
import pytest

from screws import robots, se3, so3
from screws import testing as st


def test_random_rotation_is_so3_and_seeded():
    rng = np.random.default_rng(0)
    R = st.random_rotation(rng)
    assert so3.is_so3(R)
    assert np.allclose(st.random_rotation(np.random.default_rng(0)), R)
    assert not np.allclose(st.random_rotation(np.random.default_rng(1)), R)


def test_random_transform_is_se3():
    T = st.random_transform(np.random.default_rng(2))
    assert se3.is_se3(T) and np.linalg.norm(T[:3, 3]) > 0


def test_random_theta_within_limits_and_shape():
    r = robots.ur5().with_limits([[-1, 1]] * 6)
    th = st.random_theta(r, np.random.default_rng(1))
    assert th.shape == (6,) and r.within_limits(th)
    assert st.random_theta(robots.rrp()).shape == (3,)  # no limits: [-pi, pi)


def test_assert_close_passes_and_names_index():
    st.assert_close(np.eye(3), np.eye(3) + 1e-12)
    with pytest.raises(AssertionError, match=r"\(1, 0\)"):
        st.assert_close(np.zeros((2, 2)), np.array([[0, 0], [1, 0]]), what="T")


def test_check_runs_reference_against_mine():
    st.check(lambda x: x * 2, lambda x: x + x, cases=[1.0, 2.0])
    with pytest.raises(AssertionError, match="case 1"):
        st.check(lambda x: x * 2, lambda x: x + 1 if x > 1 else x * 2, cases=[1.0, 2.0])


def test_check_unpacks_tuple_cases():
    st.check(lambda a, b: a + b, lambda a, b: b + a, cases=[(1.0, 2.0), (3.0, 4.0)])
