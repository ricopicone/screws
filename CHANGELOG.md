# Changelog

Every release of `screws`, newest first. Versions are git tags, and a tag publishes to PyPI.

## Unreleased

### Added
- `CONTRIBUTING.md`, this changelog and `CITATION.cff`.
- CI on macOS and Windows as well as Linux.
- Project links and classifiers on PyPI.

### Fixed
- An animation of a body whose faces have unequal numbers of vertices (the screwdriver)
  could fail with infinite axis limits under matplotlib 3.11, which pads such faces with
  uninitialized memory and autoscaled on it. The `plot` extra now needs matplotlib 3.9.

## [0.9.0] - 2026-11-01

### Added
- `screws.symbolic`: the chapter functions in SymPy, under the same names and arguments
  (exponentials, adjoints, the product of exponentials, the space and body Jacobians, and
  the Newton-Euler dynamics: mass matrix, c(θ, θ̇), g(θ), forward dynamics), so the results
  are formulas in the joint variables. `SymbolicRobot`, and `Robot.symbolic()` to get one
  from any robot, its floats read as the decimals they print as.

### Changed
- SymPy is a dependency. `screws.symbolic` is imported the first time it is used, so
  `import screws` does not load SymPy.

## [0.8.0] - 2026-11-01

### Added
- `viz.draw_jacobian`: each column of J_s(θ) or J_b(θ) drawn as the joint's screw axis
  where the arm has carried it, with the tool velocities the joints give and the matrix.
- `viz.draw_ellipsoid` and `viz.animate_robot`: manipulability and force ellipsoids at the
  tool, still or while the robot moves, with their semi-axes charted.
- `screws.manipulability_ellipsoid`, `screws.force_ellipsoid` and the `Robot` methods of
  the same names, numpy only.
- `viz.draw_robot(axes="current")`, `planar=True` for a planar robot drawn flat, and
  `robots.planar_2r()`.

### Changed
- Every picture uses one colour per meaning rather than one per joint.

### Fixed
- `viz.explore` shows each redraw in its own output widget.

## [0.7.0] - 2026-10-18

### Added
- `Scene.release()` hands the clock back to CoppeliaSim between bursts of stepping, so its
  window stays live in a notebook.

## [0.6.0] - 2026-10-06

### Added
- `viz.screw_between`: the screw motion that carries one frame to another.

## [0.5.0] - 2026-10-06

### Added
- `screws.motion`: the geometry of a screw axis (its line, pitch, screw motion, point
  velocities and helix).
- `viz.animate_twist`, `viz.animate_screw` and `viz.compare_frames`, with door, drawer and
  screwdriver scenes.

## [0.4.1] - 2026-09-30

### Changed
- Skeletons are drawn only where the joint positions are known, and labels sit at the axis
  ends.
- `Robot.from_screw_axes` takes `joint_frames_home`.

## [0.4.0] - 2026-09-30

### Added
- `screws.viz`: a robot with its screw axes in the space or body frame, how v = −ω × q is
  built, and `viz.explore` joint sliders in a notebook.

## [0.3.0] - 2026-09-27

### Added
- `screws.coppelia.golf` and `examples/putt.py`: a putting green in CoppeliaSim and a UR5
  that putts on it.
- `arm.mode("kinematic")`.

### Changed
- The bridge removes a model's demo script when it takes the arm, and `Scene.stop` waits
  for the simulation to stop.

## [0.2.0] - 2026-09-27

### Added
- Dynamics (MR chapter 8): Newton-Euler inverse dynamics, the mass matrix, c(θ, θ̇), g(θ),
  end-effector forces, forward dynamics and Euler integration.
- Trajectories (MR chapter 9): time scalings, and joint, screw and Cartesian trajectories.
- `arm.robot(inertias=True)` reads the masses and inertias off a scene.
- `scene.record_video`: a movie of a run, one frame per simulation step.

## [0.1.0] - 2026-09-26

### Added
- Rotations and rigid-body motions (MR chapter 3), forward kinematics, Jacobians and
  numerical inverse kinematics with its iteration history (chapters 4 to 6), under
  snake_case names with MR's names kept as aliases. Every example in MR's docstrings runs
  as a test.
- `Robot`, a URDF loader, the UR5 (from the textbook's URDF and from MR Figure 4.6) and an
  RRP arm.
- `screws.testing`: random inputs and closeness checks, for checking your own
  implementation against the library's.
- `screws.coppelia`: `Scene`, `Arm` and `Log` over CoppeliaSim's ZMQ remote API.

[0.9.0]: https://github.com/ricopicone/screws/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/ricopicone/screws/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/ricopicone/screws/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/ricopicone/screws/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/ricopicone/screws/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/ricopicone/screws/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/ricopicone/screws/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ricopicone/screws/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ricopicone/screws/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ricopicone/screws/releases/tag/v0.1.0
