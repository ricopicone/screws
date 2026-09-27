"""screws: screw-theory robotics, after Lynch and Park, *Modern Robotics* (MR).

``import screws as sc`` gives every chapter function flat (``sc.exp6``, ``sc.fk_space``),
MR's CamelCase names as aliases (``sc.FKinSpace is sc.fk_space``), the ``Robot`` class,
the ``robots`` that ship, the ``urdf`` loader and the ``testing`` helpers. The CoppeliaSim
bridge is ``screws.coppelia``, imported only on request.
"""

from . import kinematics, robots, se3, so3, testing, urdf
from ._version import __version__
from .aliases import *
from .aliases import ALIASES
from .kinematics import *
from .kinematics import IKResult
from .robot import MissingInertias, Robot
from .se3 import *
from .so3 import *

__all__ = (
    [
        "__version__",
        "ALIASES",
        "IKResult",
        "MissingInertias",
        "Robot",
        "kinematics",
        "robots",
        "se3",
        "so3",
        "testing",
        "urdf",
    ]
    + list(so3.__all__)
    + list(se3.__all__)
    + list(kinematics.__all__)
    + list(ALIASES)
)
