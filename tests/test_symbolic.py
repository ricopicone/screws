import inspect
import re
import subprocess
import sys
import warnings

import modern_robotics.core as mr_core
import numpy as np
import pytest
import sympy as sp

import screws
import screws.symbolic as sym
from tests.mr_examples import EXAMPLES
from tests.test_mr_examples import _NP, _compare, _evaluate_output, _printed_atol

L1, L2, m1, m2, g = sp.symbols("L1 L2 m1 m2 g", positive=True)
t1, t2 = sym.joint_symbols(2)
d1, d2 = sym.joint_symbols(2, "thetadot")


def _num(m) -> np.ndarray:
    # lambdify with common subexpressions: evalf walks an unsimplified tree, and on the
    # forward dynamics of MR's three-joint example that takes minutes
    return np.array(sp.lambdify([], sp.Matrix(m), "numpy", cse=True)(), dtype=float)


def _same(a, b) -> bool:
    return sp.simplify(sp.Matrix(a) - sp.Matrix(b)) == sp.zeros(*sp.Matrix(a).shape)


def _exact(robot):
    """The exact twin of a URDF robot, round-off and all (from_robot warns about it)."""
    with pytest.warns(UserWarning, match="decimals="):
        return robot.symbolic()


def _two_link():
    """The planar 2R arm of MR 8.1.1: point masses m1, m2 at the ends of links L1, L2, gravity -y."""
    axes = [sym.revolute_axis([0, 0, 0], [0, 0, 1]), sym.revolute_axis([L1, 0, 0], [0, 0, 1])]
    arm = sym.SymbolicRobot.from_screw_axes(sym.trans([L1 + L2, 0, 0]), axes, gravity=[0, -g, 0])
    G = [sp.diag(0, 0, 0, m, m, m) for m in (m1, m2)]
    return arm.with_inertias([sym.trans([L1, 0, 0]), sym.trans([L2, 0, 0]), sp.eye(4)], G)


# ----- reading numbers ----------------------------------------------------------------


def test_floats_are_read_as_the_numbers_they_print_as():
    assert sym.exact(0.425) == sp.Rational(17, 40)
    assert sym.exact(0.089159) == sp.Rational(89159, 1000000)
    assert sym.exact(-0.10915) == sp.Rational(-2183, 20000)
    assert sym.exact(8.393) == sp.Rational(8393, 1000)
    assert sym.exact(0.1 + 0.2) == sp.Rational(3, 10)
    assert sym.exact(1 / 3) == sp.Rational(1, 3)
    assert sym.exact(3) == 3 and isinstance(sym.exact(3), sp.Integer)


def test_floats_are_read_as_sympy_reads_them():
    for x in (0.425, 1 / 3, np.pi / 2, np.cos(np.pi / 4), np.cos(np.pi / 2), -3.67320510e-06):
        assert sym.exact(x) == sp.nsimplify(x, rational=True)
    assert sym.exact(np.pi / 2) != sp.pi / 2  # write sympy.pi / 2 for that
    assert sym.exact(np.cos(np.pi / 2)) != 0  # 6e-17 of round-off is kept; decimals= drops it


def test_decimals_rounds_first():
    rrp = screws.robots.rrp()  # its URDF writes pi/2 as 1.5708
    assert _exact(rrp).M[0, 0] != 0
    assert rrp.symbolic(decimals=4).M == sp.Matrix(
        [[0, 0, 1, sp.Rational(1, 2)], [0, 1, 0, 0], [-1, 0, 0, sp.Rational(2, 5)], [0, 0, 0, 1]]
    )
    assert sym.exact(np.array([0.123456, 2e-5]), decimals=4) == sp.Matrix([sp.Rational(247, 2000), 0])


def test_sympy_objects_pass_through_untouched():
    x = sp.Float(0.425)
    assert sym.exact(x) is x
    assert sym.exact(L1) is L1
    m = sym.exact(np.array([[0.5, L1], [0, 1]], dtype=object))
    assert m == sp.Matrix([[sp.Rational(1, 2), L1], [0, 1]])


def test_vectors_are_columns():
    assert sym.revolute_axis([0, 0, 0], [0, 0, 1]).shape == (6, 1)
    assert sym.exact([1, 2, 3]).shape == (3, 1)


# ----- rotations and rigid-body motions -------------------------------------------------


def test_rot_is_rodrigues_in_the_angle_itself():
    th = sp.Symbol("theta", real=True)
    R = sym.rot([0, 0, 1], th)
    assert R == sp.Matrix([[sp.cos(th), -sp.sin(th), 0], [sp.sin(th), sp.cos(th), 0], [0, 0, 1]])


def test_exp3_finds_the_angle_of_a_symbolic_unit_axis():
    a, th = sp.symbols("alpha theta", real=True)
    R = sym.rot([sp.cos(a), sp.sin(a), 0], th)
    assert not R.has(sp.Abs) and not any(p.exp == sp.S.Half for p in R.atoms(sp.Pow))
    assert _same(R.T * R, sp.eye(3))
    assert _same(R * sp.Matrix([sp.cos(a), sp.sin(a), 0]), [sp.cos(a), sp.sin(a), 0])


def test_exp3_and_exp6_agree_with_the_numbers():
    rng = np.random.default_rng(0)
    for _ in range(5):
        V = np.round(rng.normal(size=6), 3)
        se3 = screws.vec_to_se3(V)
        assert np.allclose(_num(sym.exp6(se3, simplify=False)), screws.exp6(se3))
        assert np.allclose(_num(sym.exp3(se3[:3, :3], simplify=False)), screws.exp3(se3[:3, :3]))


def test_exp6_of_a_symbolic_screw_motion_matches_the_numbers():
    th = sp.Symbol("theta", real=True)
    S = sym.screw_axis([1, 2, 0], [0, 0, 1], sp.Rational(1, 10))
    T = sym.exp6(sym.vec_to_se3(S * th))
    for value in (0.3, -1.2, 2.9):
        want = screws.exp6(screws.vec_to_se3(screws.screw_axis([1, 2, 0], [0, 0, 1], 0.1) * value))
        assert np.allclose(_num(T.subs(th, value)), want)


def test_exp6_of_a_translation():
    d = sp.Symbol("d", real=True)
    T = sym.exp6(sym.vec_to_se3(sym.prismatic_axis([0, 1, 0]) * d))
    assert T == sym.trans([0, d, 0])


def test_adjoint_and_inverse_round_trip():
    T = sym.rp_to_transform(sym.rot([0, 0, 1], t1), [L1, L2, 0])
    assert _same(sym.transform_inv(T) * T, sp.eye(4))
    assert _same(sym.adjoint(T) * sym.adjoint(sym.transform_inv(T)), sp.eye(6))
    V = sp.Matrix(sp.symbols("w1:4 v1:4", real=True))
    assert sym.se3_to_vec(sym.vec_to_se3(V)) == V


# ----- kinematics -------------------------------------------------------------------------


def test_two_link_arm_in_closed_form():
    arm = _two_link()
    c1, s1, c12, s12 = sp.cos(t1), sp.sin(t1), sp.cos(t1 + t2), sp.sin(t1 + t2)
    T = sp.Matrix(
        [[c12, -s12, 0, L1 * c1 + L2 * c12], [s12, c12, 0, L1 * s1 + L2 * s12], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    assert arm.fk() == T
    assert sym.fk_body(arm.M, arm.B) == T
    assert arm.jacobian_space() == sp.Matrix(
        [[0, 0], [0, 0], [1, 1], [0, L1 * s1], [0, -L1 * c1], [0, 0]]
    )
    assert arm.jacobian_body() == sp.Matrix(
        [[0, 0], [0, 0], [1, 1], [L1 * sp.sin(t2), 0], [L1 * sp.cos(t2) + L2, L2], [0, 0]]
    )


def test_an_axis_unit_only_to_round_off_leaves_no_root():
    T = _exact(screws.robots.ur5()).fk(simplify=False)  # its URDF writes pi/2 as 1.570796325
    assert not any(p.exp.is_Rational and not p.exp.is_integer for p in T.atoms(sp.Pow))


@pytest.mark.parametrize("robot", [screws.robots.ur5(), screws.robots.rrp()], ids=["ur5", "rrp"])
def test_kinematics_match_the_numbers(robot):
    sr = _exact(robot)
    exprs = {
        "fk": sr.fk(simplify=False),
        "fk_body": sym.fk_body(sr.M, sr.B, simplify=False),
        "Js": sr.jacobian_space(simplify=False),
        "Jb": sr.jacobian_body(simplify=False),
    }
    f = {k: sp.lambdify(sr.theta, e, "numpy") for k, e in exprs.items()}
    rng = np.random.default_rng(1)
    for _ in range(3):
        th = rng.uniform(-np.pi, np.pi, size=robot.n)
        assert np.allclose(f["fk"](*th), robot.fk(th))
        assert np.allclose(f["fk_body"](*th), robot.fk(th))
        assert np.allclose(f["Js"](*th), robot.jacobian_space(th))
        assert np.allclose(f["Jb"](*th), robot.jacobian_body(th))


def test_joint_frames_match_the_numbers():
    robot = screws.robots.ur5()
    sr = _exact(robot)
    frames = sym.joint_frames(robot.joint_frames_home, sr.S, simplify=False)
    th = np.linspace(-1, 1, 6)
    want = screws.joint_frames(robot.joint_frames_home, robot.S, th)
    for F, W in zip(frames, want):
        assert np.allclose(_num(F.subs(dict(zip(sr.theta, th)))), W)


def test_simplified_ur5_fk_is_compact_and_right():
    robot = screws.robots.ur5("textbook")
    T = robot.symbolic().fk()
    th = [0.3, -1.2, 0.8, -0.4, 1.1, 0.2]
    assert np.allclose(_num(T.subs(dict(zip(sym.joint_symbols(6), th)))), robot.fk(th))
    assert sp.count_ops(T) < sp.count_ops(robot.symbolic().fk(simplify=False)) / 2


def test_theta_may_be_given():
    arm = _two_link()
    T = arm.fk([sp.pi / 2, 0])
    assert T[:3, 3] == sp.Matrix([0, L1 + L2, 0])


# ----- dynamics ---------------------------------------------------------------------------


def test_round_off_in_a_robot_is_flagged_and_decimals_cures_it():
    with pytest.warns(UserWarning, match=r"ur5: -?3\.6e-09 .*decimals=6"):
        screws.robots.ur5().symbolic()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        screws.robots.ur5().symbolic(decimals=6)
        screws.robots.ur5("textbook").symbolic()


def test_simplify_may_be_a_function():
    arm = _two_link()
    assert arm.fk(simplify=sp.trigsimp) == arm.fk()
    assert arm.fk(simplify=sp.expand)[0, 3] == L1 * sp.cos(t1) + L2 * sp.cos(t1) * sp.cos(t2) - L2 * sp.sin(t1) * sp.sin(t2)


def test_two_link_dynamics_match_mr_8_1():
    arm = _two_link()
    c2 = sp.cos(t2)
    M = sp.Matrix(
        [
            [m1 * L1**2 + m2 * (L1**2 + 2 * L1 * L2 * c2 + L2**2), m2 * (L1 * L2 * c2 + L2**2)],
            [m2 * (L1 * L2 * c2 + L2**2), m2 * L2**2],
        ]
    )
    c = sp.Matrix(
        [-m2 * L1 * L2 * sp.sin(t2) * (2 * d1 * d2 + d2**2), m2 * L1 * L2 * d1**2 * sp.sin(t2)]
    )
    gr = sp.Matrix(
        [(m1 + m2) * L1 * g * sp.cos(t1) + m2 * g * L2 * sp.cos(t1 + t2), m2 * g * L2 * sp.cos(t1 + t2)]
    )
    assert _same(arm.mass_matrix(), M)
    assert _same(arm.velocity_quadratic_forces(), c)
    assert _same(arm.gravity_forces(), gr)
    dd = sp.Matrix(sym.joint_symbols(2, "thetaddot"))
    assert _same(arm.inverse_dynamics(), M * dd + c + gr)
    F = sp.Matrix(sp.symbols("m_x m_y m_z f_x f_y f_z", real=True))
    assert _same(arm.end_effector_forces(), arm.jacobian_body().T * F)


def test_two_link_forward_dynamics_inverts_inverse_dynamics():
    arm = _two_link()
    tau = sp.Matrix(sp.symbols("tau1:3", real=True))
    dd = arm.forward_dynamics()
    assert _same(arm.inverse_dynamics(ddtheta=list(dd)), tau)


def test_ur5_gravity_matches_the_numbers():
    robot = screws.robots.ur5()
    sr = _exact(robot)
    assert sr.gravity == sp.Matrix([0, 0, sp.Rational(-981, 100)])
    gq = sr.gravity_forces(simplify=False)
    th = np.linspace(-1.0, 1.5, 6)
    assert np.allclose(_num(gq.subs(dict(zip(sr.theta, th)))).ravel(), robot.gravity_forces(th))


# ----- MR's examples, through the symbolic twins ------------------------------------------

_TWINS = [e for e in EXAMPLES if e[0] in sym.__all__]


def test_every_twin_has_an_example():
    assert {e[0] for e in _TWINS} >= {"MatrixExp6", "FKinSpace", "JacobianBody", "InverseDynamics"}


@pytest.mark.parametrize("name,inp,outp", _TWINS, ids=[e[0] for e in _TWINS])
def test_mr_docstring_example(name, inp, outp):
    ns = dict(_NP)
    exec(inp, ns)  # noqa: S102 - MR docstring example input
    params = list(inspect.signature(getattr(mr_core, name)).parameters)
    args = [ns[p] for p in params if p in ns]
    fn = getattr(sym, name)
    kw = {"simplify": False} if "simplify" in inspect.signature(fn).parameters else {}
    want = _evaluate_output(outp)

    def fit(got, want):  # a symbolic vector is a column; MR's are flat
        got = _num(got)
        return got.ravel() if np.ndim(want) == 1 else got

    got = fn(*args, **kw)
    got = tuple(map(fit, got, want)) if isinstance(want, tuple) else fit(got, want)
    _compare(got, want, atol=_printed_atol(outp))


# ----- interface ----------------------------------------------------------------------------


def test_mr_names_are_the_same_functions():
    assert sym.FKinSpace is sym.fk_space
    assert sym.MatrixExp6 is sym.exp6
    assert sym.JacobianBody is sym.jacobian_body
    assert sym.ad is sym.__dict__["ad"] and sym.__all__.count("ad") == 1
    assert not hasattr(sym, "IKinBody")
    missing = [n for n in sym.__all__ if not hasattr(sym, n)]
    assert not missing, missing


def test_a_sequence_of_screw_axes_is_refused():
    axes = [sym.revolute_axis([0, 0, 0], [0, 0, 1]), sym.revolute_axis([L1, 0, 0], [0, 0, 1])]
    with pytest.raises(TypeError, match="hstack"):
        sym.fk_space(sp.eye(4), axes)
    with pytest.raises(ValueError, match="3 entries for 2 joints"):
        sym.fk_space(sp.eye(4), sp.Matrix.hstack(*axes), [0, 0, 0])
    with pytest.raises(TypeError, match="sequence of 6-vectors"):
        sym.SymbolicRobot.from_screw_axes(sp.eye(4), sp.Matrix.hstack(*axes))


def test_joint_types_are_read_off_the_axes():
    r = sym.SymbolicRobot.from_screw_axes(
        sp.eye(4), [sym.revolute_axis([0, 0, 0], [0, 0, 1]), sym.prismatic_axis([1, 0, 0])]
    )
    assert r.joint_types == ("revolute", "prismatic")
    assert r.gravity == sp.Matrix([0, 0, -g])


def test_dynamics_without_inertias_raise():
    with pytest.raises(screws.MissingInertias):
        screws.robots.rrp().symbolic(decimals=4).mass_matrix()


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)


def test_screws_symbolic_is_imported_on_first_use():
    r = _run(
        "import sys, screws; assert 'sympy' not in sys.modules; "
        "assert screws.symbolic.fk_space and 'sympy' in sys.modules; "
        "from screws import *; from screws import symbolic; assert symbolic is screws.symbolic"
    )
    assert r.returncode == 0, r.stderr
    with pytest.raises(AttributeError, match="no attribute 'nonesuch'"):
        screws.nonesuch  # noqa: B018


def test_import_screws_does_not_import_sympy():
    r = _run("import sys; sys.modules['sympy'] = None; import screws; screws.robots.ur5().fk([0] * 6)")
    assert r.returncode == 0, r.stderr


def test_the_formulas_page_runs():
    from pathlib import Path

    page = (Path(__file__).parents[1] / "docs" / "guide" / "formulas.md").read_text(encoding="utf-8")
    code = "\n".join(re.findall(r"```python\n(.*?)```", page, re.DOTALL))
    ns: dict = {}
    exec(code, ns)  # noqa: S102 - the documentation's own example
    assert sp.simplify(ns["arm"].mass_matrix()[1, 1] - L2**2 * m2) == 0
