# screws

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23873253.svg)](https://doi.org/10.5281/zenodo.23873253)

Robotics in Python, built on screw theory: every rigid-body motion is a screw motion, the
exponential of a twist, and an open chain is a product of such exponentials. Kinematics,
Jacobians and dynamics all follow from that. The notation is that of Lynch and Park,
*Modern Robotics* (MR).

- **The mathematics.** Rotations and rigid-body motions with their exponentials and
  logarithms; twists, wrenches and adjoints; forward and inverse kinematics; space and body
  Jacobians; manipulability and force ellipsoids; Newton-Euler dynamics; trajectories.
- **Robots.** A `Robot` class that carries M, the screw axes and the link inertias, a URDF
  loader, and three robots ready to use: a UR5, an RRP arm and a planar 2R arm.
- **Formulas.** `screws.symbolic` does the same computations in SymPy, so the product of
  exponentials, the Jacobians and the equations of motion come out in the joint variables.
- **Pictures.** `screws.viz` draws a robot with its screw axes, animates screw motions, and
  shows Jacobians and manipulability ellipsoids, still or while the robot moves, with joint
  sliders in a notebook.
- **Simulation.** `screws.coppelia` drives CoppeliaSim: read a robot off a scene, command it
  by position, velocity or torque, log and plot the run, and record a movie.
- **Checks.** `screws.testing` compares your own implementation of a function with the
  library's.

`screws` started from the MR code library, and all of that library is still here: each
function under a snake_case name, with MR's name kept as an alias
(`screws.FKinSpace is screws.fk_space`). The tests run every example in MR's docstrings
through those aliases.

## Install

One or the other:

```
uv add screws                 # the mathematics only; numpy and SymPy are the dependencies
uv add "screws[coppelia]"     # the same, plus the CoppeliaSim ZMQ remote API client
```

Extras for the bridge: `screws[plot]` (matplotlib, for `Log.plot`) and `screws[video]`
(imageio and ffmpeg, for `Recorder`). `uv add "screws[coppelia,plot,video]"` takes all three.

## Four lines

```python
import screws

ur5 = screws.robots.ur5()                          # M, screw axes, inertias from the textbook's URDF
T = ur5.fk([0.3, -1.2, 0.8, -0.4, 1.1, 0.2])         # a reachable, non-singular pose
result = ur5.ik(T, theta0=[0.1, -1.4, 0.1, 0.1, 1.4, 0.1])   # result.theta, result.converged, result.history
```

## Documentation

The guide and the API reference are at <https://ricopicone.github.io/screws/>:

- [Kinematics and the Robot class](https://ricopicone.github.io/screws/guide/kinematics/), [dynamics and trajectories](https://ricopicone.github.io/screws/guide/dynamics/),
  and [checking your own code](https://ricopicone.github.io/screws/guide/checking/) against the library
- [Formulas](https://ricopicone.github.io/screws/guide/formulas/): the same computations in SymPy
- Pictures: [the screw axes](https://ricopicone.github.io/screws/guide/pictures/screw-axes/),
  [screw motions](https://ricopicone.github.io/screws/guide/pictures/screw-motions/), and
  [Jacobians and manipulability ellipsoids](https://ricopicone.github.io/screws/guide/pictures/jacobians/)
- [CoppeliaSim](https://ricopicone.github.io/screws/guide/coppelia/): driving the simulator, and a UR5 that putts
- [MR's names](https://ricopicone.github.io/screws/guide/names/) beside the `screws` names

## Help, contributing and citing

Questions and bug reports go in the [issues](https://github.com/ricopicone/screws/issues);
how to report one well, and how to propose a change, is in
[CONTRIBUTING.md](https://github.com/ricopicone/screws/blob/main/CONTRIBUTING.md). What each release changed is in
[CHANGELOG.md](https://github.com/ricopicone/screws/blob/main/CHANGELOG.md). To cite `screws`, use [CITATION.cff](https://github.com/ricopicone/screws/blob/main/CITATION.cff), or
GitHub's "Cite this repository" button, which reads it. Zenodo archives every release:
the DOI [10.5281/zenodo.23873253](https://doi.org/10.5281/zenodo.23873253) stands for `screws` as a whole and leads to
the latest version, and each version has a DOI of its own on that page.

## Licence

MIT. Portions derived from the Modern Robotics code library, copyright 2018 Huan Weng,
Bill Hunt, Jarvis Schultz and Mikhail Todes, MIT licence; see [`LICENSE`](https://github.com/ricopicone/screws/blob/main/LICENSE).
