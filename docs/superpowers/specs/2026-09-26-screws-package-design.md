# `screws`: design of the MR library successor and CoppeliaSim bridge

Date: 2026-09-26
Task: memme-465565 #760 (spun out of #759, Sim labs design)
Status: draft for Rico's review. Nothing has been implemented; this
document is the deliverable of #760.

## Problem

ME 465 / MME 565 teaches Lynch and Park, *Modern Robotics* (MR), with
CoppeliaSim as the simulator. The textbook's Python library
(`modern_robotics` 1.1.1, MIT, last released Feb 2023) is a flat list of
fifty CamelCase functions with `Slist`/`thetalist` arguments, no robot
object, no URDF reader, and no connection to a live simulator. The course's
sim labs (Sim 1 to 3 and the capstone) drive a UR5 in CoppeliaSim through
the ZMQ remote API with real physics, so students need three things the MR
library does not give them: a robot they can hand around as one object, a
way to read and command the simulator from ordinary Python, and a reference
implementation they can check their own code against.

`screws` is that package. It keeps MR's mathematics and notation exactly,
renames and reorganises the code, and adds the simulator bridge.

## Decided before this spec (task #760)

- Name `screws`, `import screws as sc`. Free on PyPI as of 2026-09-26.
- Derived from the MR library under its MIT licence; their copyright notice
  and licence text stay in `LICENSE` and in the module header of every file
  that carries their code.
- New snake_case primary names for functions and arguments. MR's CamelCase
  names remain as aliases with no deprecation warnings, so book and code read
  side by side.
- Organised from the bottom up: a robot class with methods, not only a flat
  function list.
- CoppeliaSim bridge in `screws.coppelia`, an optional dependency, so the
  mathematics imports without the simulator.
- Sim 1 (Wed 2026-09-30) does not depend on `screws`. The package grows lab
  by lab from Sim 2 onward.

## Non-goals

- No Denavit-Hartenberg parameters, no analytic IK for specific arms
  (MR ch. 6.1 is done on paper in this course).
- No MR chapter 10 (motion planning), 12 (grasping), or 13 (wheeled mobile
  robots) in the first three releases. Sim 3 falls in the chapter 13 week;
  see the open questions.
- No URDF geometry: `<visual>`, `<collision>`, meshes, `xacro`, `<mimic>`,
  and `<transmission>` are ignored.
- No plotting inside the maths. `matplotlib` is an extra used only by
  `Log.plot()` in the bridge and by `simulate_control` when asked.
- No MATLAB or Mathematica ports.
- No CSV playback into MR's wiki scenes as a primary path. A `Log.to_mr_csv()`
  export is a cheap extra, because the scenes remain good viewers.

## Design

### 1. Package layout

```
screws/
  __init__.py       re-exports every public maths name and the MR aliases
  _version.py
  so3.py            rotations: MR 3.2
  se3.py            rigid-body motions, twists, screws, wrenches: MR 3.3, 3.4
  kinematics.py     forward kinematics, Jacobians, numerical IK: MR 4, 5, 6.2
  dynamics.py       Newton-Euler and its products: MR 8
  trajectory.py     time scalings and point-to-point trajectories: MR 9
  control.py        computed-torque control and its simulation: MR 11.4
  robot.py          the Robot class and IKResult
  urdf.py           URDF -> Robot
  robots/           robots that ship ready to use (ur5, the notes' RRP arm)
  testing.py        random configurations and closeness assertions
  aliases.py        the MR CamelCase names
  coppelia/         the CoppeliaSim bridge (optional dependency)
    __init__.py     Scene, Arm, Log
    _sim.py         the thin layer over the ZMQ client and its conversions
```

One module per MR chapter group, so a student who knows which chapter a
function belongs to knows which file to open. `__init__.py` flattens
everything, so `sc.exp6` and `sc.se3.exp6` are the same object.

Dependencies: `numpy` only. Extras: `screws[coppelia]` adds
`coppeliasim-zmqremoteapi-client==2.0.4` (which brings `pyzmq` and `cbor2`);
`screws[plot]` adds `matplotlib`. Python 3.10 or later. Built with `uv` and
`hatchling`, tested with `pytest`.

### 2. Conventions that hold everywhere

**Arrays.** Every function accepts anything `numpy.asarray` accepts and
returns `float64` `ndarray`s. Vectors are one-dimensional. A transformation
is 4x4, a rotation 3x3, a twist or screw axis or wrench a 6-vector in MR's
order (angular part first).

**Screw-axis lists are 6xn with one axis per column**, exactly as MR writes
$[\mathcal{S}_1 \cdots \mathcal{S}_n]$ and exactly as the Jacobian is
$6 \times n$. This is the one place the package refuses to be clever: with
$n = 6$ a 6x6 array could be either orientation, so **no function guesses**.
Where a student is entering axes by hand, the entry point takes a *sequence
of 6-vectors* (`Robot.from_screw_axes(M, [S1, S2, ...])`) and stacks them
itself. The stacked attribute is always `robot.S`, 6xn.

**Argument names.** A quantity MR writes as one letter keeps that letter:
`M`, `S`, `B`, `theta`, `T`, `R`, `p`, `g`, `tau`, `J`. Everything MR names
with a compound (`Slist`, `thetalist`, `Mlist`, `Glist`, `Ftip`, `eomg`,
`ev`, `Tf`) becomes a word or a clear compound: `S`, `theta`,
`link_frames`, `link_inertias`, `F_tip`, `tol_omega`, `tol_v`, `T_final`.
Derivatives are `dtheta`, `ddtheta`. The rule keeps the code as close to
the book as a snake_case library can be. (Open question 1 asks whether the
single letters should instead become words.)

**Exponentials take the bracketed matrix, as the book writes them.**
`exp3(so3mat)` computes $e^{[\omega]\theta}$ from the 3x3 matrix
$[\omega]\theta$ and `exp6(se3mat)` computes $e^{[\mathcal{S}]\theta}$ from the
4x4 matrix. This is MR's convention and the one the notes' formulas read
in. The reverse, `log3(R)` and `log6(T)`, return the bracketed matrices.
Two conveniences that MR does not have, `rot(omega_hat, theta)` and
`trans(p)`, build $\operatorname{Rot}(\hat\omega, \theta)$ and
$\operatorname{Trans}(p)$ directly because the notes use that notation
constantly; they are marked as additions.

**Docstrings cite both sources.** Every public function's docstring ends
with a line of the form `MR 4.1.1; notes 4.1 (Product of exponentials
formula)`. The notes' URL slug is the section id, so the line is a lookup
key in both directions.

**Aliases.** `screws.aliases` defines every MR name. Where our signature
and return value equal MR's, the alias is the same object
(`sc.FKinSpace is sc.fk_space` is `True`). Where ours differ (the IK
functions return a result object; trajectory functions take a string
instead of MR's `3`/`5`), the alias is a thin wrapper with MR's exact
signature and return, and its docstring says which primary it wraps. No
warnings, ever. The aliases exist so a student reading MR 4.4 can type what
the book says.

### 3. The mathematics, MR name by MR name

The table is the contract. Behaviour is MR's unless the "change" column
says otherwise.

| MR | `screws` | change |
|---|---|---|
| `NearZero` | `near_zero` | |
| `Normalize` | `normalize` | |
| `RotInv` | `rot_inv` | |
| `VecToso3` / `so3ToVec` | `vec_to_so3` / `so3_to_vec` | |
| `AxisAng3` | `axis_angle3` | returns `(omega_hat, theta)` |
| `MatrixExp3` / `MatrixLog3` | `exp3` / `log3` | |
| `RpToTrans` / `TransToRp` | `rp_to_transform` / `transform_to_rp` | |
| `TransInv` | `transform_inv` | |
| `VecTose3` / `se3ToVec` | `vec_to_se3` / `se3_to_vec` | |
| `Adjoint` | `adjoint` | |
| `ScrewToAxis` | `screw_axis(q, s_hat, h)` | plus `revolute_axis(q, s_hat)` and `prismatic_axis(s_hat)`, which MR lacks and the joint-frame derivation needs ($h = \infty$ breaks `ScrewToAxis`) |
| `AxisAng6` | `axis_angle6` | returns `(S, theta)` |
| `MatrixExp6` / `MatrixLog6` | `exp6` / `log6` | |
| `ProjectToSO3` / `ProjectToSE3` | `project_so3` / `project_se3` | |
| `DistanceToSO3` / `DistanceToSE3` | `distance_so3` / `distance_se3` | |
| `TestIfSO3` / `TestIfSE3` | `is_so3` / `is_se3` | |
| (none) | `rot`, `trans`, `rotation_from_rpy` | additions; `rotation_from_rpy` is URDF's fixed-axis roll-pitch-yaw, the notes' eq. rpy-to-rotation |
| `FKinSpace(M, Slist, thetalist)` | `fk_space(M, S, theta)` | |
| `FKinBody(M, Blist, thetalist)` | `fk_body(M, B, theta)` | |
| (none) | `joint_frames(M_joints, S, theta)` | addition: every joint frame $T_{0i}(\theta)$ from the zero-position joint frames `M_joints`, for stick figures and for animating a chain |
| `JacobianSpace` / `JacobianBody` | `jacobian_space(S, theta)` / `jacobian_body(B, theta)` | |
| (none) | `manipulability(J)` | addition: MR 5.4's three measures $\mu_1, \mu_2, \mu_3$ for the angular and linear blocks |
| `IKinBody(Blist, M, T, thetalist0, eomg, ev)` | `ik_body(M, B, T, theta0, *, tol_omega=1e-3, tol_v=1e-4, max_iter=20)` | returns `IKResult`; alias returns MR's `(thetalist, success)` |
| `IKinSpace` | `ik_space(M, S, T, theta0, *, ...)` | same |
| `ad` | `ad` | |
| `InverseDynamics(..., g, Ftip, Mlist, Glist, Slist)` | `inverse_dynamics(theta, dtheta, ddtheta, g, F_tip, link_frames, link_inertias, S)` | |
| `MassMatrix` | `mass_matrix(theta, link_frames, link_inertias, S)` | |
| `VelQuadraticForces` | `velocity_quadratic_forces(theta, dtheta, ...)` | |
| `GravityForces` | `gravity_forces(theta, g, ...)` | |
| `EndEffectorForces` | `end_effector_forces(theta, F_tip, ...)` | |
| `ForwardDynamics` | `forward_dynamics(theta, dtheta, tau, g, F_tip, ...)` | |
| `EulerStep` | `euler_step(theta, dtheta, ddtheta, dt)` | |
| `InverseDynamicsTrajectory` / `ForwardDynamicsTrajectory` | `inverse_dynamics_trajectory` / `forward_dynamics_trajectory` | trajectories are Nxn arrays, as in MR |
| `CubicTimeScaling(Tf, t)` / `QuinticTimeScaling` | `cubic_time_scaling(T_final, t)` / `quintic_time_scaling` | |
| `JointTrajectory(..., Tf, N, method)` | `joint_trajectory(theta_start, theta_end, T_final, N, scaling="quintic")` | `scaling` is `"cubic"` or `"quintic"`; alias maps MR's `3`/`5` |
| `ScrewTrajectory` / `CartesianTrajectory` | `screw_trajectory(X_start, X_end, T_final, N, scaling)` / `cartesian_trajectory(...)` | same |
| `ComputedTorque` | `computed_torque(theta, dtheta, e_int, g, link_frames, link_inertias, S, theta_d, dtheta_d, ddtheta_d, Kp, Ki, Kd)` | gains may be scalars or n-vectors (MR: scalars only) |
| `SimulateControl` | `simulate_control(...)` | `plot=False` by default; MR always plots |

`IKResult` is a frozen dataclass: `theta` (the final iterate), `converged`
(bool), `iterations` (int), `history` (an (iterations+1) x n array of every
iterate), `error_omega` and `error_v` (final norms). The history is what
Sim 2 animates: the lab "animating inverse-kinematics solutions" is
`for theta in result.history: arm.teleport(theta); scene.step()`.

### 4. The `Robot` class

```python
@dataclass(frozen=True)
class Robot:
    name: str
    M: ndarray                       # 4x4 home configuration of {b} in {s}
    S: ndarray                       # 6xn space-frame screw axes at home
    joint_types: tuple[str, ...]     # "revolute" | "prismatic"
    joint_names: tuple[str, ...]
    joint_limits: ndarray | None     # nx2, radians or metres
    joint_frames_home: tuple[ndarray, ...] | None  # T_{0i}(0), one per joint
    link_frames: tuple[ndarray, ...] | None   # M_{i-1,i}, n+1 of them (MR Mlist)
    link_inertias: tuple[ndarray, ...] | None # G_i, 6x6 each (MR Glist)
    gravity: ndarray = (0, 0, -9.81)

    n: int                           # property
    B: ndarray                       # property: [Ad_{M^{-1}}] S, cached

    fk(theta) -> ndarray             # T_sb(theta); space form
    frames(theta) -> list[ndarray]   # T_{0i}(theta) for i = 1..n, then T_sb
    jacobian_space(theta), jacobian_body(theta)
    ik(T, theta0, *, frame="body", **tol) -> IKResult
    within_limits(theta) -> bool
    inverse_dynamics(theta, dtheta, ddtheta, F_tip=None) -> tau
    mass_matrix(theta), velocity_quadratic_forces(theta, dtheta),
    gravity_forces(theta), end_effector_forces(theta, F_tip)
    forward_dynamics(theta, dtheta, tau, F_tip=None) -> ddtheta
    with_gravity(g), with_inertias(link_frames, link_inertias) -> Robot

    @classmethod from_screw_axes(cls, M, axes, *, name="robot", joint_types=None, ...)
    @classmethod from_urdf(cls, path, *, base_link=None, ee_link=None)
```

Frozen, so a robot can be shared between a controller and a logger without
either mutating it; the `with_*` methods return copies. The dynamics
methods raise `MissingInertias` with a one-line message if `link_inertias`
is `None`. `fk` is one method, not two: the space and body forms agree, and
a student who wants to see that calls the two free functions. Every method
is a one-line delegation to the free function in the chapter module, so the
class adds no mathematics of its own and the free functions stay the
reference implementation.

`from_screw_axes` takes the axes as a sequence of 6-vectors (see §2).
`joint_types` defaults to reading each axis: $\omega = 0$ means prismatic.
`joint_frames_home` is filled by the URDF loader and by the scene reader,
which both have the joint frames; `from_screw_axes` leaves it `None`, and
`frames(theta)` then falls back to a frame per joint with origin
$q_i = \omega_i \times v_i$ (the point on the axis nearest the origin) and
$\hat z$ along $\omega_i$, which is all a stick figure needs. The fallback
is documented as such, and a prismatic joint reuses the previous origin.

### 5. The URDF loader

`screws.urdf.load(path, *, base_link=None, ee_link=None) -> Robot`, and
`Robot.from_urdf` delegates to it. Parsing is `xml.etree` from the standard
library.

The chain: from `base_link` (default: the one link that is no joint's child)
to `ee_link` (default: the one link that is no joint's parent, or an error
naming the candidates if there are several). Joints of type `revolute` and
`continuous` are revolute; `prismatic` is prismatic; `fixed` joints
accumulate into the neighbouring transformation and add no column. Any
other type raises.

Kinematics follow the notes' section 4.3 exactly. Each joint's `<origin>`
is $T_{i-1,i}(0)$ with `rpy` converted by `rotation_from_rpy`. The
accumulated product gives $T_{0i}(0)$; the `<axis>` in the joint's own frame
gives $\mathcal{A}_i$; then $\mathcal{S}_i = [\mathrm{Ad}_{T_{0i}(0)}]
\mathcal{A}_i$ and $M$ is the product carried through the fixed joints to
the end-effector link. The loader records the joint `<limit>` bounds when
present.

Inertia follows MR 8.3's convention so that the result feeds
`inverse_dynamics` unchanged: link frame $\{i\}$ is placed at the link's
centre of mass with axes from the `<inertial><origin>`, so $\mathcal{G}_i =
\operatorname{diag}(\mathcal{I}_b, m I)$ with $\mathcal{I}_b$ read from
`<inertia>`, and `link_frames[i]` is the transformation from frame $\{i-1\}$
(the previous link's centre of mass) to $\{i\}$, with `link_frames[n]` the
end-effector frame relative to the last one. A link without `<inertial>`
gets zero mass. Fixed joints between links with inertia are not supported
in the first release (the UR5 does not need them) and raise.

### 6. Robots that ship

`screws.robots.ur5(source="urdf")` returns the UR5 built from the URDF the
textbook prints in MR 4.2: the manufacturer's lengths to a tenth of a
millimetre and the printed inertias, so it matches the CoppeliaSim model
as closely as any published data can. `ur5(source="textbook")` returns the
rounded version of MR Figure 4.6 and the notes' UR5 example ($W_1 = 109$,
$W_2 = 82$, $L_1 = 425$, $L_2 = 392$, $H_1 = 89$, $H_2 = 95$ mm), with no
inertias. The two differ in the third decimal of $v_i$, which is exactly
what the notes' problem "Forward kinematics, computed and checked" part
(c) asks students to account for, so both must exist and be named.

`screws.robots.rrp()` returns the notes' RRP arm, whose URDF the notes
print; it is the smallest robot with both joint types and the one the
loader's tests are written against.

The UR5 URDF text ships inside the package (`robots/ur5.urdf`) and
`ur5(source="urdf")` is literally `load(that file)`, so the loader is
exercised on every import of the robot.

### 7. Pedagogy: the library as the reference answer

The course's pattern is that students implement a function themselves and
check it against the library, and a solution that imports the library's
version for the graded part loses those marks. The package supports the
pattern in three ways and no more.

- **`screws.testing`**: `random_rotation()`, `random_transform()`,
  `random_theta(robot, rng=None)` (within limits when the robot has them),
  `assert_close(a, b, *, atol=1e-9)` with a message that prints the
  largest discrepancy and where it is, and `check(mine, reference, cases)`
  which runs both on each case and reports the first disagreement. That is
  the whole module.
- **Every free function is the reference, not the class.** A student's
  `my_exp6` is compared with `sc.exp6`, never with a method, so the check
  reads as one formula against another.
- **Stable, boring numerics.** The library does what the book says
  (MR's algorithms, MR's tolerances by default) so a correct student
  implementation agrees to rounding, and a disagreement means a mistake
  rather than a different algorithm. No fast paths that change results.

### 8. The CoppeliaSim bridge, `screws.coppelia`

Imports only when asked; `import screws` never touches `zmq`. The bridge
talks to CoppeliaSim 4.9 through the ZMQ remote API in stepping mode, so
the student's Python program decides when simulated time advances.

```python
from screws.coppelia import Scene

with Scene() as scene:                 # localhost:23000; sim.setStepping(True)
    arm = scene.arm("/UR5")            # joints in that tree, base to tip
    robot = arm.robot()                # a screws.Robot read off the scene
    arm.mode("position")               # or "velocity" or "torque"
    scene.start()
    for k in range(400):
        theta = arm.theta()
        arm.command(theta_desired[k])  # dispatches on the mode
        scene.step()
    scene.stop()
scene.log.plot()
```

**`Scene`** owns the connection and simulated time. Methods: `start()`,
`step()`, `stop()`, `time` and `dt` (from `sim.getSimulationTime` and
`sim.getSimulationTimeStep`), `frame(path) -> ndarray` (any object's 4x4
in the world frame, from `sim.getObjectMatrix`), `set_frame(path, T)`,
`show_frame(T, name)` (a drawn triad for a target pose, via
`sim.addDrawingObject`; a teaching aid and nothing more), `arm(path)`, and
`run(controller, *, duration, log=True)`, which loops
`controller(t, theta, dtheta) -> command` over `step()` and logs. The
context manager stops the simulation and closes the socket, so a crashed
controller does not leave CoppeliaSim running.

**`Arm`** is a list of joint handles under one tree, found with
`sim.getObjectsInTree(..., sim.sceneobject_joint)` and ordered base to tip
by depth, plus a tip: the tree's `ee`/`tip` dummy when there is one, else
the last joint's child. Methods: `theta()`, `dtheta()`, `tau()` (measured
joint forces), `command(u)` (dispatches on `mode`), `command_positions`,
`command_velocities`, `command_torques` (torque mode sets the joint's
dynamic control mode to force and applies the signed torque directly with
`sim.setJointTargetForce(h, tau, True)`; CoppeliaSim 4.3 and later apply
the value with its sign in force mode), `teleport(theta)`
(`sim.setJointPosition`, kinematic, for animating IK histories and
trajectories without physics), `tip_frame()`, `joint_frames()`, and
`robot(*, inertias=True) -> Robot`.

**`Arm.robot()` reads the robot off the scene.** With the arm at its zero
position: $M$ is the tip frame; for joint $i$ the scene's joint frame gives
$\hat\omega_i$ as its local $z$ axis and $q_i$ as its origin, so
$\mathcal{S}_i$ is `revolute_axis(q_i, z_i)` (or `prismatic_axis(z_i)`).
This is the derivation Sim 1 has students do by hand; from Sim 2 on the
library does it and the student checks. If the arm is not at zero when
asked, the method teleports it to zero, reads, and teleports back. With
`inertias=True` it also reads each link shape's mass and inertia
(`sim.getShapeMass`, `sim.getShapeInertia`, which also returns the
centre-of-mass frame) and builds MR-convention `link_frames` and
`link_inertias` as the URDF loader does. Reading joint limits from
`sim.getJointInterval` completes the `Robot`.

**`Log`** is what `run` and the context manager accumulate: arrays `t`,
`theta`, `dtheta`, `tau`, `command`, and `T_sb`, with `to_csv(path)`,
`to_mr_csv(path)` (joint angles only, MR's scene format), and `plot()`
(matplotlib, one axis per quantity). Logging is on by default and costs
one extra `getObjectMatrix` per step.

**Conversions** live in `_sim.py` and are unit-tested without a simulator:
CoppeliaSim's 12-float row-major matrix to and from 4x4, its `(x, y, z, qx,
qy, qz, qw)` pose to and from 4x4, and the 9-float inertia to 3x3. The ZMQ
client's `sim` object is injected, so tests pass a fake.

**Errors.** A failed connection raises `SimulatorNotRunning` with the one
sentence a student needs ("Start CoppeliaSim and open a scene, then run
this again"). A path that resolves to nothing raises with the path echoed.
Nothing in the bridge retries silently.

### 9. The inertia question from #759

Decision: **the bridge reads inertias from the scene, and the mismatch is
still the lesson.** `Arm.robot()` gives the controller the simulator's own
masses and inertias, so a computed-torque controller against the scene is
against the truth, which is what a capstone control checkpoint should
grade. The lesson is then explicit rather than accidental: the checkpoint
asks teams to run the same controller once with `ur5(source="urdf")`'s
textbook inertias and once with the scene's, and to explain the tracking
error. MR 11.4 already frames this as the model-error experiment
(`gtilde`, `Mtildelist`, `Gtildelist` in `SimulateControl`), and
`simulate_control` keeps that interface.

### 10. Testing

- **MR's own docstring examples are the alias test suite.** Every function
  in `modern_robotics/core.py` carries an "Example Input / Output" block.
  A generator script (`tools/harvest_mr_examples.py`, run once, output
  committed as `tests/mr_examples.py`) turns them into parametrised tests
  that call the alias and compare with `atol=1e-6`. `modern_robotics` is a
  dev dependency so the same tests also compare each alias with MR
  directly on random inputs.
- **Property tests** for the mathematics: `exp`/`log` round trips on
  random elements, `fk_space == fk_body`, `adjoint(T) @ adjoint(T_inv) = I`,
  `inverse_dynamics(forward_dynamics(...)) = tau`, `ik` converging to a
  configuration whose `fk` is the target.
- **Loader tests**: the notes' RRP URDF reproduces the notes' $M$ and
  $\mathcal{S}_i$; the UR5 URDF reproduces MR's Figure 4.6 table to three
  decimals and MR's printed inertias exactly; the two UR5 sources differ
  by at most `6e-4` in $v$ and not at all in $\omega$.
- **Bridge tests**: conversions and `Arm.robot()` run against a fake `sim`.
  Tests that need CoppeliaSim are marked `coppelia` and skipped unless
  `SCREWS_COPPELIASIM=1`; CI never runs them, the release checklist does,
  against the UR5 scene.

### 11. Documentation

A `README.md` with the four things a student does (install, import, load
the UR5, connect to the scene) and a table of MR name to `screws` name (the
table in §3, kept in sync by a test that reads `aliases.py`). No separate
docs site in the first releases; the notes are the textbook and each
function's docstring points at them.

### 12. Releases and the lab calendar

| version | by | for | contents |
|---|---|---|---|
| 0.1 | Fri 2026-10-09 | Sim 2 lab session Mon 10/12, due Wed 10/14 (week 8) | so3, se3, kinematics, `Robot`, `IKResult` with history, `urdf`, `robots.ur5`/`rrp`, `testing`, aliases, bridge (`Scene`, `Arm` without inertias, `Log`) |
| 0.2 | Mon 2026-10-26 | capstone assigned (week 10), Checkpoint 1 kinematics (week 12) | dynamics, trajectory, `Arm.robot(inertias=True)`, `show_frame` |
| 0.3 | Mon 2026-11-16 | Checkpoint 2 planning (week 14), Checkpoint 3 control and Sim 3 (week 15) | control, torque mode, `simulate_control` against the scene |

Published to PyPI by hand (the course's `uv`-managed repo pins the
version); the release checklist runs the `coppelia`-marked tests against
the UR5 scene first.

### 13. Repository

`~/screws`, GitHub `ricopicone/screws`, MIT. `pyproject.toml` with
`hatchling`; `uv` for everything; `ruff` and `pytest` in the dev group.
This spec lives at `docs/superpowers/specs/`, the implementation plan at
`docs/superpowers/plans/`.

## Open questions for Rico

1. **Single-letter arguments.** §2 keeps `M`, `S`, `B`, `theta`, `T` because
   the book does. The alternative is words everywhere (`home`,
   `screw_axes`, `body_axes`, `joint_angles`). The recommendation is the
   letters: they are what the notes' formulas say.
2. **Exponentials take matrices.** `exp6([S]θ)` reads as the book. The
   alternative, `exp6(S, theta)` taking the axis and the angle, is
   friendlier to type and to a student who has not yet met the bracket.
   The recommendation is the book's form as primary with `exp6(S, theta)`
   refused (no overloading), and `rot(omega_hat, theta)` as the friendly
   entry point for rotations.
3. **Sim 3 and chapter 13.** Sim 3 lands in the wheeled-mobile-robots week
   with nothing in MR's library to derive from. Either Sim 3 becomes the
   capstone's control lab (and the schedule row moves) or `screws` gains a
   small `mobile.py` (odometry, the four-mecanum-wheel model of MR 13.2)
   in 0.3. The recommendation is the former; the spec does not include
   `mobile.py`.
4. **Which UR5 is default.** §6 makes the URDF numbers the default because
   they match the simulator. If problem sets keep asking for the
   textbook's rounded table, `source="textbook"` will be typed often.
