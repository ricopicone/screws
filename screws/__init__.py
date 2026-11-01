"""screws: screw-theory robotics, after Lynch and Park, *Modern Robotics* (MR).

``import screws`` gives every chapter function flat (``screws.exp6``, ``screws.fk_space``),
MR's CamelCase names as aliases (``screws.FKinSpace is screws.fk_space``), the ``Robot`` class,
the ``robots`` that ship, the ``urdf`` loader and the ``testing`` helpers. The CoppeliaSim
bridge is ``screws.coppelia``, imported only on request. The SymPy twins of the chapter
functions are ``screws.symbolic``, imported the first time it is used: SymPy takes a tenth
of a second or more to load, and most scripts never need it.
"""

from typing import TYPE_CHECKING

from . import dynamics, kinematics, motion, robots, se3, so3, testing, trajectory, urdf, viz
from ._version import __version__
from .aliases import *
from .aliases import ALIASES
from .dynamics import *
from .kinematics import *
from .kinematics import IKResult
from .motion import *
from .robot import MissingInertias, Robot
from .se3 import *
from .so3 import *
from .trajectory import *

__all__ = (
    [
        "__version__",
        "ALIASES",
        "IKResult",
        "dynamics",
        "MissingInertias",
        "Robot",
        "kinematics",
        "motion",
        "robots",
        "se3",
        "so3",
        "testing",
        "trajectory",
        "urdf",
        "viz",
    ]
    + list(so3.__all__)
    + list(se3.__all__)
    + list(kinematics.__all__)
    + list(motion.__all__)
    + list(dynamics.__all__)
    + list(trajectory.__all__)
    + list(ALIASES)
)

# symbolic is left out of __all__ so that `from screws import *` does not load SymPy
if TYPE_CHECKING:
    from . import symbolic as symbolic


def __getattr__(name: str):
    if name == "symbolic":
        import importlib

        return importlib.import_module(".symbolic", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
