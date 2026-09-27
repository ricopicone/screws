"""Robots that ship ready to use."""

from __future__ import annotations

import numpy as np

from ..robot import Robot
from ..se3 import revolute_axis
from ..urdf import load, packaged

__all__ = ["UR5_DIMENSIONS", "rrp", "ur5"]

#: The rounded UR5 dimensions of MR Figure 4.6 and the notes' UR5 example, in metres.
UR5_DIMENSIONS = dict(W1=0.109, W2=0.082, L1=0.425, L2=0.392, H1=0.089, H2=0.095)


def ur5(source: str = "urdf") -> Robot:
    """Universal Robots' UR5, a 6R arm.

    source="urdf" (default) loads the URDF the textbook prints in MR 4.2: the manufacturer's
    lengths to a tenth of a millimetre and the printed inertias, the closest published
    match to the CoppeliaSim model. source="textbook" builds the rounded table of MR
    Figure 4.6 and the notes' worked example (UR5_DIMENSIONS), with no inertias. The two
    differ in the third decimal of the v entries; the notes' problems ask why.
    MR 4.1.2 example 4.5, 4.2; notes 4.1, 4.3.
    """
    if source == "urdf":
        return load(packaged("ur5.urdf"))
    if source == "textbook":
        d = UR5_DIMENSIONS
        W1, W2, L1, L2, H1, H2 = (d[k] for k in ("W1", "W2", "L1", "L2", "H1", "H2"))
        M = np.array(
            [[-1, 0, 0, L1 + L2], [0, 0, 1, W1 + W2], [0, 1, 0, H1 - H2], [0, 0, 0, 1.0]]
        )
        axes = [
            revolute_axis([0, 0, 0], [0, 0, 1]),
            revolute_axis([0, 0, H1], [0, 1, 0]),
            revolute_axis([L1, 0, H1], [0, 1, 0]),
            revolute_axis([L1 + L2, 0, H1], [0, 1, 0]),
            revolute_axis([L1 + L2, W1, 0], [0, 0, -1]),
            revolute_axis([L1 + L2, 0, H1 - H2], [0, 1, 0]),
        ]
        return Robot.from_screw_axes(M, axes, name="ur5-textbook")
    raise ValueError(f'source must be "urdf" or "textbook"; got {source!r}')


def rrp() -> Robot:
    """The notes' RRP arm (section 4.3), the smallest robot with both joint types."""
    return load(packaged("rrp.urdf"))
