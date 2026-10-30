"""Pictures and animations of screw axes. Everything here needs matplotlib
(``screws[plot]``), imported only when a function is called.

Static: draw_robot, draw_screw, explore (static.py). Moving: animate_twist,
animate_screw, compare_frames (animate.py), with the scenes door, drawer and screwdriver
(scenes.py) and the Mesh, box and prism to build your own body. Jacobians and ellipsoids:
draw_jacobian, draw_ellipsoid, animate_robot, compare_paths (jacobian.py).
"""

from .animate import Animation, animate_screw, animate_twist, compare_frames
from .jacobian import (
    ColumnDrawing,
    EllipsoidDrawing,
    JacobianDrawing,
    animate_robot,
    compare_paths,
    draw_ellipsoid,
    draw_jacobian,
    is_planar,
)
from .scenes import Mesh, Scene, box, door, drawer, prism, screw_between, screwdriver
from .static import (  # noqa: F401  COLOURS and SHOW_DEFAULT stay importable from screws.viz
    COLOURS,
    SHOW_DEFAULT,
    JointDrawing,
    RobotDrawing,
    ScrewDrawing,
    draw_robot,
    draw_screw,
    explore,
)

__all__ = [
    "Animation",
    "ColumnDrawing",
    "EllipsoidDrawing",
    "JacobianDrawing",
    "JointDrawing",
    "Mesh",
    "RobotDrawing",
    "Scene",
    "ScrewDrawing",
    "animate_robot",
    "animate_screw",
    "animate_twist",
    "box",
    "compare_frames",
    "compare_paths",
    "door",
    "draw_ellipsoid",
    "draw_jacobian",
    "draw_robot",
    "draw_screw",
    "drawer",
    "explore",
    "is_planar",
    "prism",
    "screw_between",
    "screwdriver",
]
