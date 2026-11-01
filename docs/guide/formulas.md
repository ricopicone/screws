# Formulas: `screws.symbolic`

The same functions in SymPy, so the product of exponentials, the Jacobians and the
dynamics come out as formulas in the joint variables. `screws.symbolic` is imported the
first time it is used, so a plain `import screws` stays quick:

```python
import sympy as sp
import screws
import screws.symbolic as sym

L1, L2 = sp.symbols("L1 L2", positive=True)
arm = sym.SymbolicRobot.from_screw_axes(
    sym.trans([L1 + L2, 0, 0]),
    [sym.revolute_axis([0, 0, 0], [0, 0, 1]), sym.revolute_axis([L1, 0, 0], [0, 0, 1])],
)
arm.fk()              # p = (L1 cos(theta1) + L2 cos(theta1 + theta2), L1 sin(theta1) + L2 sin(theta1 + theta2), 0)
arm.jacobian_body()   # first column (0, 0, 1, L1 sin(theta2), L1 cos(theta2) + L2, 0)

ur5 = screws.robots.ur5("textbook").symbolic()             # 0.425 is read as 17/40
T = ur5.fk()                                                 # T_sb(theta1, ..., theta6), simplified
Js = sp.lambdify(ur5.theta, ur5.jacobian_space(), "numpy")   # back to numbers, fast
```

Every function keeps the name and the arguments of its numerical twin (`sym.fk_space(M, S,
theta)`, `sym.jacobian_body(B, theta)`, `sym.exp6(se3mat)`, `sym.adjoint(T)`, MR's names
too: `sym.FKinSpace is sym.fk_space`) and returns SymPy matrices, vectors as columns. Leave
out theta and the joint variables are the symbols `theta1 ... thetan` (`ur5.theta`;
`thetadot1 ...` and `thetaddot1 ...` in the dynamics). `simplify=False` skips the
simplification, which is most of the time on a six-joint arm.

`robot.symbolic()` reads a robot's floats as `sympy.nsimplify(x, rational=True)` does, as
the decimals they print as; where you mean pi/2, pass `sympy.pi / 2`, not `np.pi / 2`. A
URDF that rounds pi/2 leaves a few millionths where zeros were meant, and these never
cancel, so `symbolic()` warns; `decimals=` rounds first. With
`screws.robots.ur5().symbolic(decimals=6)`, `gravity_forces()` is the UR5's g(theta), three
short nonzero entries, in about twenty seconds (`sympy.N(g, 4)` shows it in decimals).
`simplify=` also takes a function to use instead, such as `sympy.trigsimp`.

The dynamics give MR 8.1's two-link equations of motion from the Newton-Euler recursion:

```python
m1, m2, g = sp.symbols("m1 m2 g", positive=True)
point_mass = [sp.diag(0, 0, 0, m, m, m) for m in (m1, m2)]
arm = arm.with_gravity([0, -g, 0]).with_inertias(
    [sym.trans([L1, 0, 0]), sym.trans([L2, 0, 0]), sp.eye(4)], point_mass
)
arm.mass_matrix()                  # [[L1**2*m1 + ... + 2*L1*L2*m2*cos(theta2) + ..., ...], ...]
arm.velocity_quadratic_forces()    # c(theta, thetadot)
arm.gravity_forces()               # g(theta)
arm.forward_dynamics()             # thetaddot in terms of tau1, tau2
```

Not symbolic, being numerical by nature: the logarithms, inverse kinematics, manipulability,
time stepping and trajectories.
