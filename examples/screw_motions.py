"""The three screw motions of the chapter 3 review, as GIFs: a door (zero pitch), a drawer
(infinite pitch) and a screwdriver (finite pitch), and the door's space and body twists side
by side. Writes into docs/media, or the directory given: uv run python examples/screw_motions.py [dir]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from screws import viz

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "docs" / "media"
out.mkdir(parents=True, exist_ok=True)
for scene in (viz.door(), viz.drawer(), viz.screwdriver()):
    viz.animate_screw(scene).save(out / f"{scene.name}.gif", dpi=80)
    print("wrote", out / f"{scene.name}.gif")
viz.compare_frames(viz.door()).save(out / "door_frames.gif", dpi=80)
viz.animate_twist(viz.door(), duration=4).save(out / "door_twist.gif", dpi=80)
print("wrote", out / "door_frames.gif", out / "door_twist.gif")
