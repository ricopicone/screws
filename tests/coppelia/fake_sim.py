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
        self.alias_options: list[int] = []
        self.force_errors = False

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
        # zero-position frame moved by every ancestor joint's motion about (revolute) or
        # along (prismatic) its own local z axis, outermost ancestor applied last
        T = self.objects[h].T_world_zero
        for j in self._ancestors_joints(h)[::-1]:
            Fj = self.objects[j].T_world_zero
            local = np.r_[0, 0, 1, 0, 0, 0] if self.objects[j].joint_type == "revolute" else np.r_[0, 0, 0, 0, 0, 1]
            motion = se3.exp6(se3.vec_to_se3(local * self.q[j]))
            T = Fj @ motion @ se3.transform_inv(Fj) @ T
        return T

    # --- sim API ---
    def getObject(self, path, options=None):
        if path not in self.by_path:
            raise RuntimeError(f"Object does not exist. (in function 'sim.getObject') {path}")
        return self.by_path[path]

    def getObjectAlias(self, h, options=-1):
        self.alias_options.append(options)
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
            kinds = {self.sceneobject_shape: "shape", self.sceneobject_joint: "joint",
                     self.sceneobject_dummy: "dummy"}
            if (h != base or not (options & 1)) and self.objects[h].kind == kinds.get(obj_type):
                out.append(h)
            if h == base or not (options & 2):  # bit 2: first children only
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
        if self.force_errors:
            raise RuntimeError("joint is not dynamically enabled")
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
                self.force[h] = self.target_force[h]  # signed, as CoppeliaSim 4.3+ applies it
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


def three_joint_scene(with_tip: bool = True) -> FakeSim:
    """/Rig, based at (1, 0, 0): j1 revolute about world z through the base; j2 revolute about
    world y through (1, 0, 0.5) (its local z is world y, so its frame is Rot(x, -90deg));
    j3 prismatic along world x at (1, 0, 0.5) (local z is world x, frame Rot(y, 90deg)); a
    'slider' link and, optionally, a tip dummy at (1.4, 0, 0.5)."""
    from screws import so3

    sim = FakeSim()
    base = sim.add("/Rig", "shape", se3.trans([1, 0, 0]))
    j1 = sim.add("/Rig/j1", "joint", se3.trans([1, 0, 0]), parent=base, interval=(-3.0, 3.0))
    l1 = sim.add("/Rig/post", "shape", se3.trans([1, 0, 0.25]), parent=j1)
    R2 = so3.rot([1, 0, 0], -np.pi / 2)  # local z -> world y
    j2 = sim.add("/Rig/j2", "joint", se3.rp_to_transform(R2, [1, 0, 0.5]), parent=l1, interval=(-2.0, 2.0))
    l2 = sim.add("/Rig/elbow", "shape", se3.trans([1, 0, 0.5]), parent=j2)
    R3 = so3.rot([0, 1, 0], np.pi / 2)  # local z -> world x
    j3 = sim.add("/Rig/j3", "joint", se3.rp_to_transform(R3, [1, 0, 0.5]), parent=l2,
                 joint_type="prismatic", interval=(-0.3, 0.3))
    l3 = sim.add("/Rig/slider", "shape", se3.trans([1.4, 0, 0.5]), parent=j3)
    if with_tip:
        sim.add("/Rig/tip", "dummy", se3.trans([1.4, 0, 0.5]), parent=l3)
    return sim
