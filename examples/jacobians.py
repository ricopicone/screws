"""Jacobians and manipulability ellipsoids, as pictures and GIFs: the UR5's Jacobian columns,
the notes' 2R ellipses (section 5.5), the UR5's ellipsoids, the 2R arm stretching into its
singularity, the UR5 straightening its elbow, and constant joint rates beside a constant
twist. Writes into docs/media, or the directory given:
uv run python examples/jacobians.py [dir]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from screws import robots, viz

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "docs" / "media"
out.mkdir(parents=True, exist_ok=True)
PI = np.pi
ur5, arm = robots.ur5(), robots.planar_2r()
pose = np.array([0.3, -1.1, 1.4, 0.2, 0.9, -0.5])


def wrote(name):
    print("wrote", out / name)


# Each column of J_s is a joint's screw axis now; at the tool, the velocity it gives.
viz.draw_jacobian(ur5, pose).fig.savefig(out / "jacobian_ur5.png", dpi=80)
wrote("jacobian_ur5.png")

# The notes' figure of section 5.5: the 2R arm at two postures, both ellipses at one scale.
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, t2 in zip(axes, (PI / 4, 3 * PI / 4)):
    viz.draw_ellipsoid(arm, [0, t2], force=True, scale=0.3, force_scale=0.3, ax=ax)
fig.tight_layout()
fig.savefig(out / "ellipses_2r.png", dpi=80)
wrote("ellipses_2r.png")

viz.draw_ellipsoid(ur5, pose, force=True).fig.savefig(out / "ellipsoid_ur5.png", dpi=80)
wrote("ellipsoid_ur5.png")

# Moving: the 2R arm straightens at constant joint rates and its ellipse collapses to a segment.
show = ("skeleton", "frames", "ellipsoid", "force", "path")
viz.animate_robot(arm, [PI / 6, 2.4], [PI / 3, 0.0], show=show).save(out / "ellipse_2r.gif", dpi=70)
wrote("ellipse_2r.gif")

# The UR5 straightens its elbow (theta3 -> 0, a singularity), with J_s(theta) beside it.
start = np.array([0.0, -1.2, 1.5, -1.6, -1.57, 0.0])
viz.animate_robot(ur5, start, [0.4, -0.9, 0.0, -1.3, -1.57, 0.4], matrix=True).save(out / "ellipsoid_ur5.gif", dpi=70)
wrote("ellipsoid_ur5.gif")

# Constant joint rates beside a constant twist: the same start and end, two routes.
viz.compare_paths(ur5, start, [1.2, -0.9, 1.1, -1.3, -1.0, 0.4]).save(out / "paths_ur5.gif", dpi=70)
wrote("paths_ur5.gif")
