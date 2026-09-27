"""Scene: the connection to CoppeliaSim and its simulated clock, in stepping mode."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .. import se3, so3
from . import _sim
from .arm import Arm
from .log import Log
from .video import Recorder

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
        self._triads: dict[str, list[int]] = {}
        self._recorders: list[Recorder] = []
        self._created_sensors: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.started:
            self.stop()
        if self._created_sensors:
            try:
                self.sim.removeObjects(list(self._created_sensors))
            except Exception:  # noqa: BLE001, S110 - the simulator may already be gone
                pass
            self._created_sensors.clear()
        # De-register as a stepping client: the server advances only when every registered
        # stepping client has called step(), so a client that leaves silently freezes the clock
        # for everyone who comes after it.
        try:
            self.sim.setStepping(False)
        except Exception:  # noqa: BLE001, S110 - the simulator may already be gone; nothing to do
            pass
        client = getattr(self.sim, "_screws_client", None)
        socket = getattr(client, "socket", None)
        if socket is not None:
            socket.close()

    # ----- time -------------------------------------------------------------------------

    def start(self) -> None:
        """Start the simulation. A simulation left running by a crashed client is stopped first,
        since in stepping mode it would wait forever for that client's next step()."""
        if self.sim.getSimulationState() != self.sim.simulation_stopped:
            self.stop()
            for _ in range(200):
                if self.sim.getSimulationState() == self.sim.simulation_stopped:
                    break
                time.sleep(0.05)
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
        """Draw a triad at T: red x, green y, blue z. Drawing the same name again moves it."""
        self._remove_triad(name)
        T = np.asarray(T, dtype=float)
        o = T[:3, 3]
        handles = []
        for k, colour in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
            h = self.sim.addDrawingObject(self.sim.drawing_lines, 3, 0.0, -1, 2, colour)
            tip = o + size * T[:3, k]
            self.sim.addDrawingObjectItem(h, [*o, *tip])
            handles.append(h)
        self._triads[name] = handles

    def clear_frames(self) -> None:
        """Remove every triad drawn by show_frame."""
        for name in list(self._triads):
            self._remove_triad(name)

    def _remove_triad(self, name: str) -> None:
        for h in self._triads.pop(name, []):
            self.sim.removeDrawingObject(h)

    def camera(
        self,
        path: str | None = None,
        *,
        position=(1.5, -1.5, 1.0),
        look_at=(0.0, 0.0, 0.4),
        resolution=(640, 480),
        fov_deg: float = 60.0,
    ) -> int:
        """A vision sensor to record from: the one at ``path``, or a new perspective sensor
        placed at ``position`` looking at ``look_at`` with the image's up along world z. A
        created sensor is removed when the Scene exits."""
        if path is not None:
            return self._handle(path)
        w, h = int(resolution[0]), int(resolution[1])
        handle = int(
            self.sim.createVisionSensor(
                1 + 2,  # explicit handling, perspective projection
                [w, h, 0, 0],
                [0.01, 10.0, float(np.deg2rad(fov_deg)), 0.1, 0, 0, 0, 0, 0, 0, 0],
            )
        )
        p = np.asarray(position, dtype=float)
        z = np.asarray(look_at, dtype=float) - p  # the sensor looks along its +z axis
        z = z / np.linalg.norm(z)
        up = np.array([0.0, 0.0, 1.0])
        if abs(float(up @ z)) > 0.999:  # looking straight up or down: any horizontal up
            up = np.array([0.0, 1.0, 0.0])
        y = up - float(up @ z) * z  # image up: world up made perpendicular to the view
        y = y / np.linalg.norm(y)
        x = np.cross(y, z)
        R = np.column_stack([x, y, z])
        assert so3.is_so3(R)
        self.sim.setObjectMatrix(handle, _sim.transform_to_matrix12(se3.rp_to_transform(R, p)), self.sim.handle_world)
        self._created_sensors.append(handle)
        return handle

    def record_video(self, path, *, camera=None, every: int = 1, **camera_kwargs) -> Recorder:
        """A Recorder context: ``with scene.record_video("run.mp4") as rec: scene.run(...)``.

        ``camera`` is a sensor handle or path; None creates one with ``camera_kwargs``
        (position, look_at, resolution, fov_deg). Frames are grabbed every ``every`` steps
        and the movie plays at simulated time. Needs screws[video] to save.
        """
        rec = Recorder(self, path, camera=-1, every=every)  # validates the extension first
        rec.camera = self.camera(**camera_kwargs) if camera is None else self._handle(camera)
        return rec

    def arm(self, path: str, joints=None) -> Arm:
        """The Arm under path; ``joints`` names a subset (aliases or paths) when a tool adds its own."""
        return Arm(self, path, joints)

    # ----- running ----------------------------------------------------------------------

    def record(self, arm: Arm, command=None, *, theta=None, dtheta=None) -> None:
        """Append one row to the log; pass theta/dtheta already read this step to avoid re-reading."""
        if self.log is not None:
            theta = arm.theta() if theta is None else theta
            dtheta = arm.dtheta() if dtheta is None else dtheta
            self.log.record(self.time, theta, dtheta, arm.tau(), command, arm.tip_frame())

    def run(self, controller: Callable, *, duration: float, arm: Arm, log: bool = True) -> Log:
        """Loop controller(t, theta, dtheta) -> command over the simulation for duration seconds.

        Starts the simulation if needed, records before each step, returns the Log.
        """
        if not self.started:
            self.start()
        if log and self.log is None:
            self.log = Log()
        steps = round(duration / self.dt)
        for _ in range(steps):
            theta, dtheta = arm.theta(), arm.dtheta()
            u = controller(self.time, theta, dtheta)
            if u is not None:
                arm.command(u)
            if log:
                self.record(arm, u, theta=theta, dtheta=dtheta)
            for rec in list(self._recorders):
                rec.tick()
            self.step()
        return self.log
