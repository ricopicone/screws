"""Helpers for checking your own implementation against the library's.

The course pattern: write the function yourself, then compare it with the screws version
on random inputs. Nothing here does robotics; it makes the comparison short to write.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from .se3 import rp_to_transform
from .so3 import exp3, vec_to_so3

__all__ = ["assert_close", "check", "random_rotation", "random_theta", "random_transform"]


def _rng(rng):
    return np.random.default_rng() if rng is None else rng


def random_rotation(rng=None) -> np.ndarray:
    """A random rotation matrix: the exponential of a random axis-angle with theta in [0, pi)."""
    rng = _rng(rng)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return exp3(vec_to_so3(axis * rng.uniform(0.0, np.pi)))


def random_transform(rng=None, *, scale: float = 1.0) -> np.ndarray:
    """A random homogeneous transformation: random rotation, position uniform in [-scale, scale]^3."""
    rng = _rng(rng)
    return rp_to_transform(random_rotation(rng), rng.uniform(-scale, scale, size=3))


def random_theta(robot, rng=None) -> np.ndarray:
    """A random joint vector for robot, within its joint_limits, or in [-pi, pi) without them."""
    rng = _rng(rng)
    if robot.joint_limits is None:
        return rng.uniform(-np.pi, np.pi, size=robot.n)
    lo, hi = robot.joint_limits[:, 0], robot.joint_limits[:, 1]
    return rng.uniform(lo, hi)


def assert_close(a, b, *, atol: float = 1e-9, what: str = "arrays") -> None:
    """Assert a and b agree to atol, or fail naming the largest discrepancy and where it is."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise AssertionError(f"{what} differ in shape: {a.shape} vs {b.shape}")
    diff = np.abs(a - b)
    if diff.size and diff.max() > atol:
        idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
        raise AssertionError(
            f"{what} differ by {diff.max():.3e} at index {tuple(int(i) for i in idx)}: "
            f"{a[idx]!r} vs {b[idx]!r} (atol {atol:g})"
        )


def check(mine: Callable, reference: Callable, cases: Iterable, *, atol: float = 1e-9) -> None:
    """Run mine and reference on each case and fail on the first disagreement.

    A tuple case is unpacked as positional arguments; anything else is the single argument.
    """
    for k, case in enumerate(cases):
        args = case if isinstance(case, tuple) else (case,)
        assert_close(mine(*args), reference(*args), atol=atol, what=f"case {k}: mine vs reference")
