"""A stand-in for CoppeliaSim's ``sim`` object: the subset the bridge calls, with a scene
graph of joints and dummies, joint state kept in dicts, and time advanced by step()."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from screws import se3
from screws.coppelia import _sim


@dataclass
class _Obj:
    handle: int
    alias: str
    path: str
    parent: int  # -1 = world
    kind: str  # "joint" | "dummy" | "shape"
    T_world_zero: np.ndarray  # frame at zero joint position
    joint_type: str = "revolute"
    interval: tuple[float, float] | None = None
    children: list[int] = field(default_factory=list)


class FakeSim:
    handle_world = -1
    sceneobject_joint = 1
    sceneobject_dummy = 4
    sceneobject_shape = 0
    joint_revolute = 10
    joint_prismatic = 11
    jointintparam_dynctrlmode = 2030
    jointdynctrl_free = 0
    jointdynctrl_force = 1
    jointdynctrl_velocity = 2
    jointdynctrl_position = 4
    drawing_lines = 1

    def __init__(self, dt: float = 0.05):
        self.dt = dt
        self.time = 0.0
        self.running = False
        self.stepping = False
        self.objects: dict[int, _Obj] = {}
        self.by_path: dict[str, int] = {}
        self.q: dict[int, float] = {}
        self.dq: dict[int, float] = {}
        self.force: dict[int, float] = {}
        self.target_q: dict[int, float] = {}
        self.target_dq: dict[int, float] = {}
        self.target_force: dict[int, float] = {}
        self.ctrl_mode: dict[int, int] = {}
        self.drawings: list = []
        self.calls: list[tuple] = []

    # --- scene construction (test helper) ---
    def add(self, path, kind, T, parent=-1, joint_type="revolute", interval=None) -> int:
        h = 100 + len(self.objects)
        alias = path.rsplit("/", 1)[-1]
        self.objects[h] = _Obj(h, alias, path, parent, kind, np.asarray(T, float), joint_type, interval)
        self.by_path[path] = h
        if parent != -1:
            self.objects[parent].children.append(h)
        if kind == "joint":
            self.q[h] = self.dq[h] = self.force[h] = 0.0
            self.ctrl_mode[h] = self.jointdynctrl_position
        return h

    def _ancestors_joints(self, h):
        out = []
        p = self.objects[h].parent
        while p != -1:
            if self.objects[p].kind == "joint":
                out.append(p)
            p = self.objects[p].parent
        return out[::-1]

    def _world_frame(self, h):
        # zero-position frame moved by every ancestor joint's motion about its own z axis
        T = self.objects[h].T_world_zero
        for j in self._ancestors_joints(h):
            Fj = self.objects[j].T_world_zero
            motion = se3.exp6(se3.vec_to_se3(np.r_[0, 0, 1, 0, 0, 0] * self.q[j]))
            T = Fj @ motion @ se3.transform_inv(Fj) @ T
        return T

    # --- sim API ---
    def getObject(self, path, options=None):
        if path not in self.by_path:
            raise RuntimeError(f"Object does not exist. (in function 'sim.getObject') {path}")
        return self.by_path[path]

    def getObjectAlias(self, h, options=0):
        return self.objects[h].alias

    def getObjectParent(self, h):
        return self.objects[h].parent

    def getObjectType(self, h):
        return {"joint": self.sceneobject_joint, "dummy": self.sceneobject_dummy}.get(
            self.objects[h].kind, self.sceneobject_shape
        )

    def getJointType(self, h):
        return self.joint_revolute if self.objects[h].joint_type == "revolute" else self.joint_prismatic

    def getObjectsInTree(self, base, obj_type, options=0):
        out = []
        stack = [base]
        while stack:
            h = stack.pop(0)
            if h != base or not (options & 1):
                if obj_type == self.sceneobject_shape + 0 and self.objects[h].kind == "shape" or obj_type == self.sceneobject_joint and self.objects[h].kind == "joint" or obj_type == self.sceneobject_dummy and self.objects[h].kind == "dummy":
                    out.append(h)
            stack.extend(self.objects[h].children)
        return out

    def getObjectMatrix(self, h, rel=-1):
        assert rel == -1
        return _sim.transform_to_matrix12(self._world_frame(h))

    def setObjectMatrix(self, h, m, rel=-1):
        self.objects[h].T_world_zero = _sim.matrix12_to_transform(m)

    def getJointPosition(self, h):
        return self.q[h]

    def setJointPosition(self, h, v):
        self.q[h] = float(v)

    def getJointVelocity(self, h):
        return self.dq[h]

    def getJointForce(self, h):
        return self.force[h]

    def getJointInterval(self, h):
        iv = self.objects[h].interval
        if iv is None:
            return True, [0.0, 0.0]
        return False, [iv[0], iv[1] - iv[0]]

    def setJointTargetPosition(self, h, v):
        self.target_q[h] = float(v)

    def setJointTargetVelocity(self, h, v):
        self.target_dq[h] = float(v)

    def setJointTargetForce(self, h, v, signed=True):
        self.target_force[h] = float(v)

    def setObjectInt32Param(self, h, param, value):
        self.calls.append(("setObjectInt32Param", h, param, value))
        if param == self.jointintparam_dynctrlmode:
            self.ctrl_mode[h] = value

    def setStepping(self, enable=True):
        self.stepping = enable

    def startSimulation(self):
        self.running = True

    def stopSimulation(self):
        self.running = False

    def step(self):
        assert self.running and self.stepping
        for h in self.q:
            mode = self.ctrl_mode[h]
            if mode == self.jointdynctrl_position and h in self.target_q:
                self.dq[h] = (self.target_q[h] - self.q[h]) / self.dt
                self.q[h] = self.target_q[h]
            elif mode == self.jointdynctrl_velocity and h in self.target_dq:
                self.dq[h] = self.target_dq[h]
                self.q[h] += self.dq[h] * self.dt
            elif mode == self.jointdynctrl_force and h in self.target_force:
                self.force[h] = self.target_force[h] * np.sign(self.target_dq.get(h, 1.0))
        self.time += self.dt

    def getSimulationTime(self):
        return self.time

    def getSimulationTimeStep(self):
        return self.dt

    def addDrawingObject(self, *args):
        self.drawings.append(list(args))
        return 900 + len(self.drawings)

    def addDrawingObjectItem(self, handle, item):
        self.drawings[handle - 901].append(item)


def two_joint_scene() -> FakeSim:
    """/Arm with j1 (z axis at the origin), j2 (z axis at (0, 0, 0.5)) and a dummy tip at
    (0.3, 0, 0.5). Both joints revolute about z, so both screw axes are (0,0,1, 0,0,0)."""
    sim = FakeSim()
    base = sim.add("/Arm", "shape", np.eye(4))
    j1 = sim.add("/Arm/j1", "joint", np.eye(4), parent=base, interval=(-3.0, 3.0))
    l1 = sim.add("/Arm/link1", "shape", se3.trans([0, 0, 0.25]), parent=j1)
    j2 = sim.add("/Arm/j2", "joint", se3.trans([0, 0, 0.5]), parent=l1, interval=(-2.0, 2.0))
    l2 = sim.add("/Arm/link2", "shape", se3.trans([0.15, 0, 0.5]), parent=j2)
    sim.add("/Arm/tip", "dummy", se3.trans([0.3, 0, 0.5]), parent=l2)
    return sim
