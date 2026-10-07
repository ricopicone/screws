"""The three screw motions of the chapter 3 review, as GIFs: a door (zero pitch), a drawer
(infinite pitch) and a screwdriver (finite pitch), the door's space and body twists side by
side, and the screw motion that carries one frame to another. Writes into docs/media, or the
directory given: uv run python examples/screw_motions.py [dir]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from screws import se3, so3, viz

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "docs" / "media"
out.mkdir(parents=True, exist_ok=True)
for scene in (viz.door(), viz.drawer(), viz.screwdriver()):
    viz.animate_screw(scene).save(out / f"{scene.name}.gif", dpi=80)
    print("wrote", out / f"{scene.name}.gif")
viz.compare_frames(viz.door()).save(out / "door_frames.gif", dpi=80)
viz.animate_twist(viz.door(), duration=4).save(out / "door_twist.gif", dpi=80)
print("wrote", out / "door_frames.gif", out / "door_twist.gif")

# Any displacement is a screw motion (Chasles): find the one from {start} to {end} and watch it.
turn = so3.exp3(so3.vec_to_so3(np.deg2rad(160) * np.array([0.2, 0.3, 1.0]) / np.linalg.norm([0.2, 0.3, 1.0])))
T_start = se3.rp_to_transform(np.eye(3), [1.5, 0.5, 0.0])
T_end = se3.rp_to_transform(turn, [1.0, 1.6, 0.9])
viz.animate_screw(viz.screw_between(T_start, T_end, name="frame to frame")).save(out / "frame_to_frame.gif", dpi=80)
print("wrote", out / "frame_to_frame.gif")
