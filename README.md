# screws

Screw-theory robotics in Python, after Lynch and Park, *Modern Robotics* (MR).
`screws` is the MR code library reorganised: the same mathematics under
snake_case names, MR's own names kept as aliases, a `Robot` class, a URDF
loader, robots that ship ready to use, and a bridge to CoppeliaSim.

## Install

One or the other:

```
uv add screws                 # the mathematics only; numpy is the sole dependency
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

The free functions are the reference implementation and read as the book does:
`screws.exp6(screws.vec_to_se3(S * theta))` is $e^{[\mathcal{S}]\theta}$,
`screws.fk_space(M, S, theta)` is the space form of the product of exponentials, and
`screws.FKinSpace` is the very same function under MR's name.

Screw-axis lists are **6xn, one axis per column**, as MR writes them. Nothing guesses the
orientation (a 6x6 array is ambiguous for a six-axis arm), so hand entry goes through a
sequence of 6-vectors: `screws.Robot.from_screw_axes(M, [S1, S2, ...])`.

## Check your own code against the library

```python
import numpy as np
import screws

def my_exp6(se3mat): ...                          # your implementation

rng = np.random.default_rng(0)
cases = [screws.vec_to_se3(rng.normal(size=6)) for _ in range(20)]
screws.testing.check(my_exp6, screws.exp6, cases)  # raises on the first disagreement
```

## Dynamics and trajectories

```python
import numpy as np
import screws

ur5 = screws.robots.ur5()                          # link frames and inertias from the URDF
theta = np.zeros(6)
tau_g = ur5.gravity_forces(theta)                  # joint torques that hold the arm still
M = ur5.mass_matrix(theta)                         # 6x6, symmetric positive definite
tau = ur5.inverse_dynamics(theta, np.zeros(6), np.ones(6) * 0.1)   # + optional F_tip
ddtheta = ur5.forward_dynamics(theta, np.zeros(6), tau)
theta_ref = screws.joint_trajectory(theta, theta + 0.5, T_final=2.0, N=41)   # 41 rows, quintic
```

`inverse_dynamics`, `mass_matrix`, `velocity_quadratic_forces`, `gravity_forces`,
`end_effector_forces`, `forward_dynamics`, `euler_step` and the two `*_trajectory` functions
are MR chapter 8 as free functions too, with the robot's `link_frames`, `link_inertias` and
`S` passed in. `joint_trajectory`, `screw_trajectory` and `cartesian_trajectory` take
`scaling="quintic"` (default) or `"cubic"`. A `Robot` without inertias raises
`MissingInertias` from every dynamics method.

## CoppeliaSim

Works with CoppeliaSim 4.9 or later through the ZMQ remote API (0.1 verified on 4.10.0).

```python
from screws.coppelia import Scene

with Scene() as scene:                 # connects to localhost:23000 in stepping mode
    arm = scene.arm("/UR5")            # the joints under that tree, base to tip
    robot = arm.robot()                # a screws.Robot read off the scene at zero
    arm.mode("position")
    log = scene.run(lambda t, theta, dtheta: theta_desired(t), duration=5.0, arm=arm)
log.plot()                             # four stacked time plots: theta, dtheta, tau, command
log.to_csv("run.csv")
```

**Record a movie.** One frame per simulation step, so the movie plays at simulated time
however slowly the controller ran:

```python
with Scene() as scene:
    arm = scene.arm("/UR5")
    arm.mode("position")
    with scene.record_video("run.mp4", position=(1.5, -1.5, 1.0), look_at=(0, 0, 0.4)):
        scene.run(controller, duration=5.0, arm=arm)
```

`record_video` creates a camera (or reuses one: `camera="/Camera"`), grabs its image before
every step (`every=2` halves the frame rate), and on exit writes `.mp4` or `.gif`.

**Read the scene's masses.** `arm.robot(inertias=True)` reads every dynamic shape's mass and
inertia and builds the MR link frames and spatial inertias, so a model-based controller can
be run against the simulator's own numbers, and against the textbook's URDF numbers, and the
difference explained. `relative_to="base"` makes {s} the model's base frame.

`Scene` owns the connection and the clock (`start`, `step`, `stop`, `time`, `dt`,
`frame`, `show_frame`). `Scene.run` records every step into a `Log` (`t`, `theta`, `dtheta`,
`tau`, `command`, `T_sb` as arrays); `Log.plot()` draws one time-series axis per joint quantity
and shows the figure, `Log.to_csv` writes it out. `Arm` reads (`theta`, `dtheta`, `tau`, `tip_frame`) and commands
in one of three modes (`position`, `velocity`, `torque`), or `teleport`s without physics
to animate an IK history. `Arm.robot()` derives M and the screw axes from the scene's
joint frames (omega is the joint's z axis, v = -omega x q).

## An example: putting

`examples/putt.py` builds a green in a running CoppeliaSim (a ball, a cup and a kinematic
putter on the UR5's flange), addresses the ball, strokes straight through it along the line
to the hole at a chosen face speed, and reports where the ball stopped:

```
uv run python examples/putt.py --speed 0.5 --video putt.mp4
```

The pieces are in `screws.coppelia.golf`: `build_green`, `attach_putter` (whose `.robot()`
moves `M` from the flange to the putter face), `address_pose`, `stroke_path` and `putt`.
Vary the speed, the hole position or the ball's friction and see what the physics does.

## Two UR5s

`screws.robots.ur5()` is built from the URDF the textbook prints in section 4.2, whose
lengths are the manufacturer's to a tenth of a millimetre (89.159, 135.85, 425, 119.7, 392.25,
93, 94.65 and 82.3 mm) and which carries the link inertias. `screws.robots.ur5(source="textbook")`
is the rounded table of MR Figure 4.6 ($W_1 = 109$, $W_2 = 82$, $L_1 = 425$, $L_2 = 392$,
$H_1 = 89$, $H_2 = 95$ mm) with no inertias. Use the first to match a simulator or a real arm;
use the second to reproduce the book's worked examples digit for digit. The two agree in every
axis direction and differ by under a millimetre in position.

## Names

| Modern Robotics | `screws` |
|---|---|
| `NearZero` | `near_zero` |
| `Normalize` | `normalize` |
| `RotInv` | `rot_inv` |
| `VecToso3` | `vec_to_so3` |
| `so3ToVec` | `so3_to_vec` |
| `AxisAng3` | `axis_angle3` |
| `MatrixExp3` | `exp3` |
| `MatrixLog3` | `log3` |
| `RpToTrans` | `rp_to_transform` |
| `TransToRp` | `transform_to_rp` |
| `TransInv` | `transform_inv` |
| `VecTose3` | `vec_to_se3` |
| `se3ToVec` | `se3_to_vec` |
| `Adjoint` | `adjoint` |
| `ScrewToAxis` | `screw_axis` |
| `AxisAng6` | `axis_angle6` |
| `MatrixExp6` | `exp6` |
| `MatrixLog6` | `log6` |
| `ProjectToSO3` | `project_so3` |
| `ProjectToSE3` | `project_se3` |
| `DistanceToSO3` | `distance_so3` |
| `DistanceToSE3` | `distance_se3` |
| `TestIfSO3` | `is_so3` |
| `TestIfSE3` | `is_se3` |
| `FKinBody` | `fk_body` |
| `FKinSpace` | `fk_space` |
| `JacobianBody` | `jacobian_body` |
| `JacobianSpace` | `jacobian_space` |
| `IKinBody` | `ik_body` |
| `IKinSpace` | `ik_space` |
| `ad` | `ad` |
| `InverseDynamics` | `inverse_dynamics` |
| `MassMatrix` | `mass_matrix` |
| `VelQuadraticForces` | `velocity_quadratic_forces` |
| `GravityForces` | `gravity_forces` |
| `EndEffectorForces` | `end_effector_forces` |
| `ForwardDynamics` | `forward_dynamics` |
| `EulerStep` | `euler_step` |
| `InverseDynamicsTrajectory` | `inverse_dynamics_trajectory` |
| `ForwardDynamicsTrajectory` | `forward_dynamics_trajectory` |
| `CubicTimeScaling` | `cubic_time_scaling` |
| `QuinticTimeScaling` | `quintic_time_scaling` |
| `JointTrajectory` | `joint_trajectory` |
| `ScrewTrajectory` | `screw_trajectory` |
| `CartesianTrajectory` | `cartesian_trajectory` |

Where a screws function's signature matches MR's, the alias **is** that function
(`screws.FKinSpace is screws.fk_space`). `IKinBody` and `IKinSpace` are thin wrappers that return
MR's `(thetalist, success)` tuple; the primaries return an `IKResult` with the iteration
history. No deprecation warnings, ever.

## Licence

MIT. Portions derived from the Modern Robotics code library, copyright 2018 Huan Weng,
Bill Hunt, Jarvis Schultz and Mikhail Todes, MIT licence; see `LICENSE`.
