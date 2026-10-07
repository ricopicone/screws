import matplotlib

matplotlib.use("Agg")
import subprocess
import sys

import numpy as np
import pytest

from screws import motion, viz

SCENES = [viz.door, viz.drawer, viz.screwdriver]


def test_presets_carry_the_notes_numbers():
    d = viz.door()
    assert np.allclose(d.S, [0, 0, 1, 0, -2, 0]) and np.allclose(d.S_b, [0, 0, 1, 0, 0.9, 0])
    assert np.allclose(d.T0[:3, 3], [2.9, 0, 1])
    assert viz.drawer().is_translation and motion.screw_line(viz.drawer().S)[2] == np.inf
    q, s, h = motion.screw_line(viz.screwdriver().S)
    assert np.allclose(s, [0, 0, -1]) and np.isclose(h, 0.01) and np.allclose(q, 0)


@pytest.mark.parametrize("make", SCENES)
def test_body_points_sit_on_the_body_at_home(make):
    sc = make()
    verts = np.vstack([m.vertices for m in sc.body])
    lo, hi = verts.min(axis=0) - 1e-9, verts.max(axis=0) + 1e-9
    assert np.all((sc.points >= lo) & (sc.points <= hi))


@pytest.mark.parametrize("make", SCENES)
@pytest.mark.parametrize("entry", ["twist", "screw", "frames"])
def test_every_entry_point_renders_every_scene(make, entry, tmp_path):
    fn = {"twist": lambda s: viz.animate_twist(s, duration=0.4, fps=5),
          "screw": lambda s: viz.animate_screw(s, frames=3),
          "frames": lambda s: viz.compare_frames(s, frames=3)}[entry]
    a = fn(make())
    out = a.save(tmp_path / "x.gif", dpi=30)  # after the figure was closed, as in a notebook
    assert out.stat().st_size > 1000


def test_animate_screw_ends_at_the_exponential():
    sc = viz.door()
    a = viz.animate_screw(sc, frames=5)
    assert np.allclose(a.transforms[-1], motion.screw_motion(sc.S, sc.T0, [sc.theta[1]])[0])
    assert np.allclose(a.data["q"], [2, 0, 0]) and a.data["h"] == 0


def test_compare_frames_space_twist_is_constant_and_equals_S_s():
    sc = viz.door()
    a = viz.compare_frames(sc, frames=6)
    assert np.allclose(a.data["V_s"], sc.S)
    assert not np.allclose(a.data["Ad"][0], a.data["Ad"][-1])  # the adjoint changes; V_s does not


def test_raw_twist_and_pure_translation(tmp_path):
    viz.animate_twist([0, 0, 0, 0.2, 0, 0], duration=0.4, fps=5).save(tmp_path / "t.gif", dpi=30)
    a = viz.animate_twist([0, 0, 1, 0, 0, 0.1], duration=0.4, fps=5)
    assert np.allclose(a.transforms[-1], motion.screw_motion([0, 0, 1, 0, 0, 0.1], np.eye(4), [0.4])[0])


def test_save_rejects_other_formats(tmp_path):
    with pytest.raises(ValueError):
        viz.animate_screw(viz.door(), frames=2).save(tmp_path / "x.png")


def test_import_screws_does_not_import_matplotlib():
    code = "import sys, screws; screws.viz.door(); print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
