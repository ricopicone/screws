# Kinematics and the Robot class

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

## Two UR5s

`screws.robots.ur5()` is built from the URDF the textbook prints in section 4.2, whose
lengths are the manufacturer's to a tenth of a millimetre (89.159, 135.85, 425, 119.7, 392.25,
93, 94.65 and 82.3 mm) and which carries the link inertias. `screws.robots.ur5(source="textbook")`
is the rounded table of MR Figure 4.6 ($W_1 = 109$, $W_2 = 82$, $L_1 = 425$, $L_2 = 392$,
$H_1 = 89$, $H_2 = 95$ mm) with no inertias. Use the first to match a simulator or a real arm;
use the second to reproduce the book's worked examples digit for digit. The two agree in every
axis direction and differ by under a millimetre in position.
