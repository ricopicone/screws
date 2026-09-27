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

`Scene` owns the connection and the clock (`start`, `step`, `stop`, `time`, `dt`,
`frame`, `show_frame`). `Scene.run` records every step into a `Log` (`t`, `theta`, `dtheta`,
`tau`, `command`, `T_sb` as arrays); `Log.plot()` draws one time-series axis per joint quantity
and shows the figure, `Log.to_csv` writes it out. `Arm` reads (`theta`, `dtheta`, `tau`, `tip_frame`) and commands
in one of three modes (`position`, `velocity`, `torque`), or `teleport`s without physics
to animate an IK history. `Arm.robot()` derives M and the screw axes from the scene's
joint frames (omega is the joint's z axis, v = -omega x q). Scene inertias arrive in 0.2.

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

Where a screws function's signature matches MR's, the alias **is** that function
(`screws.FKinSpace is screws.fk_space`). `IKinBody` and `IKinSpace` are thin wrappers that return
MR's `(thetalist, success)` tuple; the primaries return an `IKResult` with the iteration
history. No deprecation warnings, ever.

## Licence

MIT. Portions derived from the Modern Robotics code library, copyright 2018 Huan Weng,
Bill Hunt, Jarvis Schultz and Mikhail Todes, MIT licence; see `LICENSE`.
