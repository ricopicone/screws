"""The Modern Robotics library's CamelCase names, bound to their screws equivalents.

Where the signatures match, the alias is the very same function object (``FKinSpace is
fk_space``). Where screws changed the interface (the IK functions return an IKResult), the
alias is a thin wrapper with MR's exact signature and return. No deprecation warnings: the
names exist so a student can type what the book says.
"""

from __future__ import annotations

from . import dynamics, kinematics, se3, so3, trajectory

#: MR name -> screws primary name.
ALIASES: dict[str, str] = {
    # chapter 3
    "NearZero": "near_zero",
    "Normalize": "normalize",
    "RotInv": "rot_inv",
    "VecToso3": "vec_to_so3",
    "so3ToVec": "so3_to_vec",
    "AxisAng3": "axis_angle3",
    "MatrixExp3": "exp3",
    "MatrixLog3": "log3",
    "RpToTrans": "rp_to_transform",
    "TransToRp": "transform_to_rp",
    "TransInv": "transform_inv",
    "VecTose3": "vec_to_se3",
    "se3ToVec": "se3_to_vec",
    "Adjoint": "adjoint",
    "ScrewToAxis": "screw_axis",
    "AxisAng6": "axis_angle6",
    "MatrixExp6": "exp6",
    "MatrixLog6": "log6",
    "ProjectToSO3": "project_so3",
    "ProjectToSE3": "project_se3",
    "DistanceToSO3": "distance_so3",
    "DistanceToSE3": "distance_se3",
    "TestIfSO3": "is_so3",
    "TestIfSE3": "is_se3",
    # chapter 4
    "FKinBody": "fk_body",
    "FKinSpace": "fk_space",
    # chapter 5
    "JacobianBody": "jacobian_body",
    "JacobianSpace": "jacobian_space",
    # chapter 6
    "IKinBody": "ik_body",
    "IKinSpace": "ik_space",
    # chapter 8
    "ad": "ad",
    "InverseDynamics": "inverse_dynamics",
    "MassMatrix": "mass_matrix",
    "VelQuadraticForces": "velocity_quadratic_forces",
    "GravityForces": "gravity_forces",
    "EndEffectorForces": "end_effector_forces",
    "ForwardDynamics": "forward_dynamics",
    "EulerStep": "euler_step",
    "InverseDynamicsTrajectory": "inverse_dynamics_trajectory",
    "ForwardDynamicsTrajectory": "forward_dynamics_trajectory",
    # chapter 9
    "CubicTimeScaling": "cubic_time_scaling",
    "QuinticTimeScaling": "quintic_time_scaling",
    "JointTrajectory": "joint_trajectory",
    "ScrewTrajectory": "screw_trajectory",
    "CartesianTrajectory": "cartesian_trajectory",
}

# Identity aliases: the same object under MR's name.
NearZero = so3.near_zero
Normalize = so3.normalize
RotInv = so3.rot_inv
VecToso3 = so3.vec_to_so3
so3ToVec = so3.so3_to_vec
AxisAng3 = so3.axis_angle3
MatrixExp3 = so3.exp3
MatrixLog3 = so3.log3
RpToTrans = se3.rp_to_transform
TransToRp = se3.transform_to_rp
TransInv = se3.transform_inv
VecTose3 = se3.vec_to_se3
se3ToVec = se3.se3_to_vec
Adjoint = se3.adjoint
ScrewToAxis = se3.screw_axis
AxisAng6 = se3.axis_angle6
MatrixExp6 = se3.exp6
MatrixLog6 = se3.log6
ProjectToSO3 = so3.project_so3
ProjectToSE3 = se3.project_se3
DistanceToSO3 = so3.distance_so3
DistanceToSE3 = se3.distance_se3
TestIfSO3 = so3.is_so3
TestIfSE3 = se3.is_se3
FKinBody = kinematics.fk_body
FKinSpace = kinematics.fk_space
JacobianBody = kinematics.jacobian_body
JacobianSpace = kinematics.jacobian_space
ad = dynamics.ad
InverseDynamics = dynamics.inverse_dynamics
MassMatrix = dynamics.mass_matrix
VelQuadraticForces = dynamics.velocity_quadratic_forces
GravityForces = dynamics.gravity_forces
EndEffectorForces = dynamics.end_effector_forces
ForwardDynamics = dynamics.forward_dynamics
EulerStep = dynamics.euler_step
InverseDynamicsTrajectory = dynamics.inverse_dynamics_trajectory
ForwardDynamicsTrajectory = dynamics.forward_dynamics_trajectory
CubicTimeScaling = trajectory.cubic_time_scaling
QuinticTimeScaling = trajectory.quintic_time_scaling


def _scaling(method):
    if method == 3:
        return "cubic"
    if method == 5:
        return "quintic"
    raise ValueError(f"method must be 3 (cubic) or 5 (quintic); got {method!r}")


def JointTrajectory(thetastart, thetaend, Tf, N, method):
    """MR's JointTrajectory: method 3 is cubic, 5 is quintic. Wraps screws.trajectory.joint_trajectory."""
    return trajectory.joint_trajectory(thetastart, thetaend, Tf, N, _scaling(method))


def ScrewTrajectory(Xstart, Xend, Tf, N, method):
    """MR's ScrewTrajectory. Wraps screws.trajectory.screw_trajectory."""
    return trajectory.screw_trajectory(Xstart, Xend, Tf, N, _scaling(method))


def CartesianTrajectory(Xstart, Xend, Tf, N, method):
    """MR's CartesianTrajectory. Wraps screws.trajectory.cartesian_trajectory."""
    return trajectory.cartesian_trajectory(Xstart, Xend, Tf, N, _scaling(method))


def IKinBody(Blist, M, T, thetalist0, eomg, ev):
    """MR's IKinBody: returns (thetalist, success). Wraps screws.kinematics.ik_body."""
    r = kinematics.ik_body(M, Blist, T, thetalist0, tol_omega=eomg, tol_v=ev)
    return r.theta, r.converged


def IKinSpace(Slist, M, T, thetalist0, eomg, ev):
    """MR's IKinSpace: returns (thetalist, success). Wraps screws.kinematics.ik_space."""
    r = kinematics.ik_space(M, Slist, T, thetalist0, tol_omega=eomg, tol_v=ev)
    return r.theta, r.converged


__all__ = list(ALIASES) + ["ALIASES"]
