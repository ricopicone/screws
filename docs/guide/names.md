# MR's names

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
