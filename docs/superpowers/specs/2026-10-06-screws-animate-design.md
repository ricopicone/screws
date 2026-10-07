# screws 0.5: animated twists and screw axes — design

Task #784 (Rico, 2026-10-06). Students said twists, screw axes and wrenches feel too unsteady
to build on. The aim: a twist looks like a *motion* before it looks like a six-vector.
First deliverable: GIFs of the door, drawer and screwdriver for the chapter 3 review
(notes section 3.10, "Twists, screw axes and wrenches, again"), class on 2026-10-07.

## Scope (0.5.0)

In: `animate_twist`, `animate_screw` (with T(θ) printed), `compare_frames` (V_s beside V_b),
presets `door`, `drawer`, `screwdriver` matching the notes' numbers.
Out, later cycles: wrenches and frame-invariant power; robots (joints as screws, Jacobian
columns); a CoppeliaSim mode.

## Layout

- `screws/motion.py` — numpy only, core, flat-exported from `screws`:
  - `screw_line(S) -> (q, s_hat, h)`: q is the point on the axis nearest the origin
    (q = ω × v / ‖ω‖²), s_hat = ω/‖ω‖, h = ωᵀv/‖ω‖². When ω = 0: q = None,
    s_hat = v/‖v‖, h = inf.
  - `screw_motion(S, T0, thetas) -> (N, 4, 4)`: e^{[S]θ} T0 for each θ (S in the frame T0 is
    written in, i.e. the space form).
  - `point_velocity(V, p)`: ω × p + v (p in the twist's frame; p may be (N, 3)).
  - `helix(S, p, thetas) -> (N, 3)`: the path of point p under e^{[S]θ}.
- `screws/viz/` becomes a package. `viz/static.py` is the old `viz.py` unchanged;
  `viz/__init__.py` re-exports its names plus the new ones. matplotlib stays lazy, so
  `import screws` still needs numpy only.
- `viz/scenes.py` — `Scene` dataclass (name, S_s, T0 = T_sb(0), body mesh, theta range,
  marked points in body coordinates, the point whose helix is traced, notes) and the presets.
  Meshes are numpy (vertices, faces) built from boxes and prisms.
- `viz/animate.py` — the three entry points and `Animation` result.

## Presets (numbers from the notes)

- **door**: hinge line through q = (2, 0, 0), ŝ = +z, h = 0. {b} on the handle,
  T_sb(0) = (I, (2.9, 0, 1)). S_s = (0,0,1, 0,−2,0), S_b = (0,0,1, 0,0.9,0). Slab 1.1 wide,
  2 high, from the hinge along +x. θ from 0 to 100°.
- **drawer**: ω = 0, v̂ along +y out of a cabinet front at y = 0; slides 0.45 m. Equal
  arrows on all marked points; no axis line, only the direction v̂.
- **screwdriver**: ŝ = −z (driving down), q on the z-axis, pitch exaggerated (h = 0.01 m/rad,
  stated in the title) so the advance is visible; two turns.

## Entry points

All take `scene` (a Scene) or raw `S`/`V`, `T0`, `body` arguments, return an `Animation`
with `.anim` (FuncAnimation), `.fig`, `.save(path, fps=...)` (GIF via PillowWriter, MP4 via
ffmpeg from `screws[video]`), `_repr_html_` (to_jshtml), and the numbers drawn.

1. `animate_twist(V, T0, body, duration)` / `animate_twist(scene)`: body moving under a
   constant twist; screw axis dashed with ŝ and h labelled; the traced helix (circle at
   h = 0); velocity arrows ω × p + v at the marked body points, re-evaluated each frame.
2. `animate_screw(S, T0, body, theta=(0, θ_end))`: same, parameterised by θ, with a text
   panel giving q, h and the live T(θ) = e^{[S]θ} T(0).
3. `compare_frames(scene)`: two 3D panels of the same motion.
   Left "in {s}": phantom segment, rigid with the body, from the axis to the {s} origin;
   the arrow v_s at the {s} origin (constant); a marker on the body point that was at the
   origin at θ = 0, moving away — the point at the origin changes, its velocity does not.
   Right "in {b}": the {b} triad on the handle with v_b drawn there, turning with the body.
   Text under both: V_b (constant), [Ad_{T_sb(θ)}] (changing), and V_s = [Ad] V_b (constant).

## Testing

`tests/test_motion.py` (no matplotlib): door q and h; drawer h = inf and q None;
screwdriver helix advances 2πh per turn along ŝ; `screw_motion` equals the closed form
Trans(q) Rot(z, θ) Trans(−q) T0 from the notes; V_s = Ad(T_sb(θ)) V_b at sampled θ for each
preset; point_velocity equals finite differences of the moving point.
`tests/test_animate.py` (Agg): each entry point renders each preset to a GIF in tmp_path;
returned numbers match `motion`. Before release the GIFs are looked at by eye.

## Delivery

screws 0.5.0 on PyPI; `examples/screw_motions.py` writes `docs/media/{door,drawer,screwdriver,door_frames}.gif`;
README section. Embedding in notes 3.10 follows (include-py or committed GIFs).
