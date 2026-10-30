# screws 0.8: Jacobians and manipulability ellipsoids you can look at — design

Task #863 (Rico, 2026-10-29): "Add to the visualization module visualizations of Jacobians
and manipulability ellipsoids. A nice full feature set. Perhaps even animating how they
change through configurations (actually, visualizing configuration changes via constant
rate transformations would be useful anyway, regardless of the Jacobians and manipulability
ellipsoids)." Rico said "Go" without further questions; every call below not quoted from
him is the builder's, marked *(call)*.

Chapter 5 of the notes (velocity kinematics and statics, released at v0.47.3) already draws
these things by hand: `manipulability-and-force-ellipses.py` (the 2R arm's two ellipses),
`ur5-singular-poses.py` (the UR5 with each joint axis drawn where it is now),
`two-r-arm-jacobian-columns.tex` (the 2R arm's Jacobian columns as tip velocities). The
package should make each of those one call, for any robot, and then set them moving.

## Goals

1. The numbers first, numpy only: an `Ellipsoid` (centre, principal directions, semi-axes,
   the eigenvalues of A = JJᵀ, MR's three measures) for any Jacobian block, manipulability or
   force, and the tool's ellipsoids for a `Robot` at θ.
2. `viz.draw_jacobian`: the robot at θ with each column of the Jacobian drawn as what it is,
   the joint's screw axis *now* (J_si(θ) is S_i carried by the inboard joints) and the
   velocity it gives the tool at unit rate; with θ̇, those tool velocities head to tail
   summing to the tool's velocity. The matrix printed beside it, columns coloured to match.
3. `viz.draw_ellipsoid`: the robot at θ with the manipulability ellipsoid (and, optionally,
   the force ellipsoid) at the tool, principal semi-axes drawn, μ1, μ2, μ3 printed.
4. `viz.animate_robot`: a configuration change played as an `Animation`, the Jacobian
   columns and the ellipsoids following, a chart of the semi-axes underneath, so a
   singularity is watched approaching (an axis shrinking to nothing).
5. Constant-rate configuration changes, in both senses: constant joint rates (a straight
   line in joint space, the default) and a constant twist of the tool (the screw motion from
   T_sb(θ_start) to T_sb(θ_end), the joints following by IK). `viz.compare_paths` plays the
   two side by side: under constant θ̇ the twist changes; under a constant twist θ̇ changes.
   MR 9.2 makes the point in words; this makes it in a picture.
6. Planar arms drawn flat on request. Every robot is drawn in 3D, a planar one included;
   `planar=True` draws a robot whose joint axes are all along ẑ with every v in the plane in
   2D (x̂_1, x̂_2), matching the notes' 2R figures, and raises for any other robot. *(Rico,
   after the demo: "2D cases should probably act the same unless a special argument is
   passed." First built as automatic, by detection.)*

## Non-goals and anti-goals

- No change to the notes in this task. Using the new calls in chapter 5 (replacing the
  hand-written figure scripts, or a GIF in 5.5) is a follow-up for a notes session.
- No new MR notation. Labels use MR's: J_s, J_b, J_ω, J_v, A = JJᵀ, B = A⁻¹, λ_i, μ1–μ3.
  The one quantity MR does not name, the tool point's velocity in {s} coordinates,
  ṗ = R_sb J_bv θ̇, is described in words, not given a new symbol. *(call)*
- Anti-goal: a picture that claims more than the numbers. A force ellipsoid at a singularity
  is unbounded along the lost direction; it is drawn as the bounded cross-section with that
  direction marked unbounded, never as a big finite ellipsoid. A semi-axis clipped to fit the
  plot says so.
- Anti-goal: `import screws` gaining matplotlib. viz stays lazy.

## Core (numpy), `screws/kinematics.py`, flat-exported

- `Ellipsoid` (frozen dataclass): `center` (m,), `axes` (m×m, column i the unit direction of
  semi-axis i, an eigenvector of A, signed so its largest entry is positive, as the notes'
  ur5-ellipsoids program prints them), `radii` (m,), `eigenvalues` (m,), λ_i of A = JJᵀ in
  numpy's ascending order, `kind` ("manipulability" or "force"). Properties: `matrix`
  (A for manipulability, B = A⁻¹ for force; raises at a singular force ellipsoid),
  `measures` → (μ1, μ2, μ3) of that matrix, from the radii: μ1 = r_max/r_min, μ2 = μ1²,
  μ3 = ∏ r_i (for manipulability these are exactly `manipulability()`'s), `singular`.
  Methods `outline(n)` (m = 2: (n, 2) points) and `surface(n)` (m = 3: X, Y, Z grids, for
  bounded ellipsoids). An eigenvalue at or below 1e-12·λ_max is zero: radius 0 (velocity),
  inf (force). *(call: the threshold)*
- `manipulability_ellipsoid(J, center=None)`, `force_ellipsoid(J, center=None)`: J any m×n
  block. MR 5.4; notes 5.5.
- `Robot.manipulability_ellipsoid(theta, kind="linear", *, planar=False)`,
  `Robot.force_ellipsoid(...)`: centred at the tool (the origin of {b}) in {s} axes.
  "linear" uses J_b's linear rows, MR's advice (notes 5.5), turned into {s}: R_sb J_bv.
  "angular" uses J_s's angular rows (= R_sb J_bω). *(call: drawn in {s} so it sits where the
  tool is)* A planar arm's is 3D like any other (flat in z, μ1 = ∞); `planar=True` gives the
  ellipse in the plane from the x and y rows, the notes' 2 × n J(θ) (μ1 = 6.08 for the 2R
  arm at (0, π/4)), and raises for a robot that is not planar or for kind="angular".
  `Robot.is_planar()` is the test.
- `robots.planar_2r(L1=1.0, L2=1.0)`: the notes' 2R arm (sections 5.1 and 5.5) as a Robot
  with joint frames, so the notes' worked numbers are tests: at θ = (0, π/4), J =
  [[−0.7071, −0.7071], [1.7071, 0.7071]], semi-axes 0.341 and 2.073, force 2.932 and 0.482,
  μ1 = 6.08, μ2 = 36.9, μ3 = 0.707; at (0, 3π/4) semi-axes 0.659 and 1.073, μ1 = 1.63.

## Viz, `screws/viz/jacobian.py`, re-exported from `screws.viz`

One internal painter computes, for every configuration on a path, the joint frames, the
skeleton, J_s, J_b, T_sb, each joint's current axis line (q, ŝ, h from the column; anchored
at the joint's current origin for a revolute joint), the tool-velocity columns
c_i = R_sb J_bv[:, i], and the ellipsoids; it fixes the plot limits over the whole path so
an animation does not jump. The static functions are a one-frame painter.

Colours: one per meaning, never one per joint *(Rico, after the demo: the per-joint tab10
colours "walk over the RGB frame visualization convention", and the orange links / blue
joints convention made them hard to read; first built with `draw_robot`'s tab10 palette)*.
Red, green and blue are the frames' x̂, ŷ, ẑ and nothing else. Links #F78B11 and joints
#3B83CD as in the notes; each joint's axis (a column of J_s) in the joints' blue, as the notes'
ur5-singular-poses draws them. Velocity #8E44AD purple: the tool-velocity arrows, their
head-to-tail steps, and the manipulability ellipsoid (the notes use red, but red here is x̂,
and {b}'s x̂ starts at the tool where the arrows do). Force #555555 dashed. The tool's path
grey. ṗ black. The matrix black. Labels tell the joints apart: J_si on the axes, numbers (or
J_i, J_i θ̇_i in 2D, the notes' two-r-arm-jacobian-columns convention) on the arrows. Charts:
a twist's components in their axis's colour, ω solid and v dashed; joint rates in shades of
the joints' blue.

`show` items: `skeleton`, `frames` ({s}, and {b} at the tool), `axes` (the current joint
axes, the columns of J_s; not drawn in 2D, where they point out of the page), `columns`
(the c_i at the tool), `velocity` (ṗ = Σ c_i θ̇_i, needs θ̇), `ellipsoid`, `force`,
`path` (the tool's path, for animations).

- `draw_jacobian(robot, theta, *, frame="space", dtheta=None, show=..., matrix=True,
  planar=False, ax=None, figsize=None, labels=True) -> JacobianDrawing`. frame="body" draws
  in {b} coordinates and prints J_b; the lines are the same lines. In 2D the matrix is the
  2×n tip Jacobian, the notes' J(θ) of eq. two-r-jacobian. Title gives the rank and says
  "singular" when rank < min(6, n) (rank of the 2×n J in 2D). Returns the columns drawn
  (index, kind, S, q, ŝ, h, tool velocity, artists) for checking by hand.
- `draw_ellipsoid(robot, theta, *, kind="linear", force=False, show=None, scale=None,
  force_scale=None, planar=False, ax=None, figsize=None, labels=True) -> EllipsoidDrawing`.
  scale is drawing metres per unit of the ellipsoid (m/s, rad/s); default makes the longest
  semi-axis 0.3 of the robot's reach *(call)*. force_scale has its own default, the longest
  bounded force semi-axis 0.3 of the reach in the median frame *(call, changed after looking:
  at the notes' single scale the UR5's force ellipsoid, λ ≈ 0.1, swamped the arm)*; pass
  force_scale=scale for the notes' one-scale comparison. A planar arm's angular ellipsoid is one-dimensional;
  `kind="angular"` with planar drawing raises with that explanation.
- `animate_robot(robot, theta_start, theta_end=None, *, path="joint", frames=60, fps=20,
  show=..., kind="linear", force=False, dtheta_scale=None, chart=True, matrix=False,
  planar=False, view=None, figsize=None) -> Animation`. theta_start with theta_end: path
  "joint" (constant joint rates) or "screw" (constant tool twist, IK seeded from the previous
  frame; raises if IK loses the path). theta_start alone as an (N, n) array plays that path
  (a `joint_trajectory`, an `IKResult.history`). The chart plots the semi-axes against
  s ∈ [0, 1] with a cursor; the caption gives θ and μ1, μ2, μ3. `.data`: theta, T_sb, J_s,
  J_b, dtheta (per unit s), V_s, V_b, tool, radii, measures.
- `compare_paths(robot, theta_start, theta_end, *, frames=60, ...) -> Animation`: two
  panels, the same start and end. Left, constant joint rates: the tool's path curved, the
  twist V_b changing. Right, constant twist: the tool on its helix, the joint rates
  changing. Charts under each: θ̇_i(s) and V_b(s).
- `explore(robot, *, draw=None, **kwargs)`: sliders now redraw any of `draw_robot`,
  `draw_jacobian`, `draw_ellipsoid` (default `draw_robot`; `frame` and `show` still pass
  through, so old calls work). Slider ranges from joint_limits when the robot has them.

## Testing

Core (`tests/test_kinematics.py` or a new `tests/test_ellipsoid.py`, no matplotlib): the 2R
numbers above; `measures` equal `manipulability()`'s; force radii reciprocal with the same
axes; the volume product ∏ r_manip · ∏ r_force = 1; at θ2 = 0 the ellipse is singular
(radius 0, force inf); `outline` points satisfy xᵀA⁻¹x = 1; RRP J_bv's ellipsoid is diagonal
with semi-axes 0.7, 0.495, 1 (notes 5.5's second example); `Robot.*_ellipsoid` centre at the
tool and equal to R_sb times the J_bv ellipsoid; UR5 at θ_i = π/2 against the notes'
ur5-ellipsoids program.

Viz (`tests/test_jacobian_viz.py`, Agg): each drawn axis line rebuilds its column,
(ŝ, −ŝ × q + hŝ) = J_si, in both frames; the head-to-tail column arrows sum to R_sb J_bv θ̇;
2D is chosen for planar_2r and 3D for the UR5; the drawn ellipse's semi-axes equal the core
Ellipsoid's times the scale; the force ellipsoid at a singularity is marked unbounded, not
drawn finite; `animate_robot` and `compare_paths` render to GIF in tmp_path; constant joint
rates in "joint" (dtheta constant), constant V_b in "screw", both ending at the same tool
frame; a supplied (N, n) path is played as given. Before release every GIF is looked at.

## Delivery

screws 0.8.0: version in `_version.py`, `pyproject.toml`, the README version test.
`examples/jacobians.py` writes `docs/media/{jacobian_ur5.png, jacobian_2r_dtheta.png,
ellipses_2r.png, singular_2r.png, ellipsoid_ur5.png, ellipse_2r.gif, ellipsoid_ur5.gif,
paths_ur5.gif}`. The README section is the demo page Rico saw ("This demo is great. Make it
part of the documentation / readme"): each picture with the call that drew it under it, the
numbers block, and a table of the core calls.
Release is a v0.8.0 tag push (the GitHub Action publishes to PyPI), Rico's to make.

## Order of work

1. Core ellipsoids and `planar_2r`, tested against the notes. 2. The painter and
`draw_ellipsoid`. 3. `draw_jacobian`. 4. `animate_robot` (joint path, given path, screw
path). 5. `compare_paths`. 6. `explore`'s `draw=`. 7. Example, README, version, look at
every picture.
