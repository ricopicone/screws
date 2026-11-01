"""The UR5's screw axes as pictures: joint 3's axis S_3 at home with v_3 built from q_3, and
the same axis at theta, J_s3, the third column of J_s(theta), built the same way. Writes into
docs/media, or the directory given: uv run python examples/screw_axes.py [dir]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from screws import robots, viz

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "docs" / "media"
out.mkdir(parents=True, exist_ok=True)
ur5 = robots.ur5()
build = ("skeleton", "frames", "axes", "omega", "construction")

viz.draw_robot(ur5, joints=[2], show=build).fig.savefig(out / "viz_ur5.png", dpi=65, bbox_inches="tight")
print("wrote", out / "viz_ur5.png")

theta = [0.3, -1.1, 1.4, 0.2, 0.9, -0.5]
d = viz.draw_robot(ur5, theta, axes="current", joints=[2], show=build)
d.fig.savefig(out / "axes_current_ur5.png", dpi=65, bbox_inches="tight")
print("wrote", out / "axes_current_ur5.png")
