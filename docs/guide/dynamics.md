# Dynamics and trajectories

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
