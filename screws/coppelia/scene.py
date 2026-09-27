"""Scene: the connection to CoppeliaSim and its simulated clock, in stepping mode."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import _sim
from .arm import Arm
from .log import Log

__all__ = ["Scene"]


class Scene:
    """A CoppeliaSim scene driven from Python one step at a time.

    ``with Scene() as scene:`` connects (or takes an injected ``sim``), switches on
    stepping, and on exit stops the simulation if this object started it. Time only
    advances when you call step().
    """

    def __init__(self, host="localhost", port=23000, *, sim=None, log: bool = True):
        self.sim = _sim.connect(host, port) if sim is None else sim
        self.sim.setStepping(True)
        self.started = False
        self.log = Log() if log else None

    def __enter__(self) -> Scene:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.started:
            self.stop()

    # ----- time -------------------------------------------------------------------------

    def start(self) -> None:
        self.sim.startSimulation()
        self.started = True

    def step(self) -> None:
        self.sim.step()

    def stop(self) -> None:
        self.sim.stopSimulation()
        self.started = False

    @property
    def time(self) -> float:
        return float(self.sim.getSimulationTime())

    @property
    def dt(self) -> float:
        return float(self.sim.getSimulationTimeStep())

    # ----- objects ----------------------------------------------------------------------

    def _handle(self, obj) -> int:
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        try:
            return int(self.sim.getObject(obj))
        except Exception as exc:
            raise LookupError(f"no object at {obj!r} in the scene") from exc

    def frame(self, obj) -> np.ndarray:
        """The 4x4 configuration of an object (path or handle) in the world frame."""
        return _sim.matrix12_to_transform(
            self.sim.getObjectMatrix(self._handle(obj), self.sim.handle_world)
        )

    def set_frame(self, obj, T) -> None:
        """Place an object (path or handle) at T in the world frame."""
        self.sim.setObjectMatrix(self._handle(obj), _sim.transform_to_matrix12(T), self.sim.handle_world)

    def show_frame(self, T, name: str = "frame", size: float = 0.1) -> None:
        """Draw a triad at T: red x, green y, blue z. A teaching aid for target poses."""
        T = np.asarray(T, dtype=float)
        o = T[:3, 3]
        for k, colour in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
            h = self.sim.addDrawingObject(self.sim.drawing_lines, 3, 0.0, -1, 2, colour)
            tip = o + size * T[:3, k]
            self.sim.addDrawingObjectItem(h, [*o, *tip])

    def arm(self, path: str) -> Arm:
        return Arm(self, path)

    # ----- running ----------------------------------------------------------------------

    def record(self, arm: Arm, command=None) -> None:
        if self.log is not None:
            self.log.record(self.time, arm.theta(), arm.dtheta(), arm.tau(), command, arm.tip_frame())

    def run(self, controller: Callable, *, duration: float, arm: Arm, log: bool = True) -> Log:
        """Loop controller(t, theta, dtheta) -> command over the simulation for duration seconds.

        Starts the simulation if needed, records before each step, returns the Log.
        """
        if not self.started:
            self.start()
        if log and self.log is None:
            self.log = Log()
        steps = int(round(duration / self.dt))
        for _ in range(steps):
            theta, dtheta = arm.theta(), arm.dtheta()
            u = controller(self.time, theta, dtheta)
            if u is not None:
                arm.command(u)
            if log:
                self.record(arm, u)
            self.step()
        return self.log
