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
    mass: float = 0.0
    inertia: np.ndarray | None = None  # 3x3 about the COM, in the shape's axes, kg m^2
    com: np.ndarray | None = None  # COM pose relative to the shape frame (4x4)
    static: bool = True


class FakeSim:
    handle_world = -1
    sceneobject_joint = 1
    sceneobject_dummy = 4
    sceneobject_shape = 0
    joint_revolute = 10
    joint_prismatic = 11
    jointmode_kinematic = 0
    jointmode_dynamic = 5
    jointintparam_dynctrlmode = 2030
    jointdynctrl_free = 0
    jointdynctrl_force = 1
    jointdynctrl_velocity = 2
    jointdynctrl_position = 4
    drawing_lines = 1
    shapeintparam_static = 3003
    objintparam_visibility_layer = 10
    shapeintparam_respondable = 3004
    colorcomponent_ambient_diffuse = 0
    primitiveshape_cuboid = 1
    primitiveshape_spheroid = 2
    primitiveshape_cylinder = 3
    bullet_body_friction = 6003
    bullet_body_restitution = 6001
    bullet_body_lineardamping = 6004
    bullet_body_angulardamping = 6005
    sceneobject_visionsensor = 9
    sceneobject_script = 13

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
    def add(self, path, kind, T, parent=-1, joint_type="revolute", interval=None,
            mass=0.0, inertia=None, com=None, static=True) -> int:
        h = 100 + len(self.objects)
        alias = path.rsplit("/", 1)[-1]
        self.objects[h] = _Obj(h, alias, path, parent, kind, np.asarray(T, float), joint_type, interval,
                               mass=mass, inertia=None if inertia is None else np.asarray(inertia, float),
                               com=np.eye(4) if com is None else np.asarray(com, float), static=static)
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

    def getJointMode(self, h):
        return getattr(self.objects[h], "joint_mode", self.jointmode_dynamic), 0

    def setJointMode(self, h, mode, options=0):
        self.objects[h].joint_mode = mode
        return 1

    def getJointType(self, h):
        return self.joint_revolute if self.objects[h].joint_type == "revolute" else self.joint_prismatic

    def getObjectsInTree(self, base, obj_type, options=0):
        out = []
        stack = [base]
        while stack:
            h = stack.pop(0)
            kinds = {self.sceneobject_shape: "shape", self.sceneobject_joint: "joint",
                     self.sceneobject_dummy: "dummy", self.sceneobject_script: "script"}
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
        self.calls.append(("getJointPosition", h))
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

    def getObjectInt32Param(self, h, param):
        if param == self.shapeintparam_static:
            return 1 if self.objects[h].static else 0
        if param == self.jointintparam_dynctrlmode:
            return self.ctrl_mode[h]
        raise KeyError(param)

    def getShapeMass(self, h):
        return self.objects[h].mass

    def getShapeInertia(self, h):
        o = self.objects[h]
        inertia = np.zeros((3, 3)) if o.inertia is None else o.inertia
        return [float(x) for x in inertia.reshape(-1)], _sim.transform_to_matrix12(o.com)

    def setObjectInt32Param(self, h, param, value):
        self.calls.append(("setObjectInt32Param", h, param, value))
        if param == self.jointintparam_dynctrlmode:
            self.ctrl_mode[h] = value
        elif param == self.shapeintparam_static:
            self.objects[h].static = bool(value)
        elif param == self.shapeintparam_respondable:
            self.objects[h].respondable = bool(value)
        elif param == self.objintparam_visibility_layer:
            self.objects[h].layer = int(value)

    def setStepping(self, enable=True):
        self.stepping = enable

    simulation_stopped = 0
    simulation_advancing_running = 17

    stop_lag = 0  # polls after stopSimulation during which the state still reads "running"

    def getSimulationState(self):
        if self.running:
            return self.simulation_advancing_running
        if self.stop_lag > 0:
            self.stop_lag -= 1
            return self.simulation_advancing_running
        return self.simulation_stopped

    def startSimulation(self):
        self.calls.append(("startSimulation",))
        self.running = True

    def stopSimulation(self):
        self.calls.append(("stopSimulation",))
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

    floatparam_simulation_time_step = 3

    def getFloatParam(self, param):
        if param == self.floatparam_simulation_time_step:
            return self.dt
        raise KeyError(param)

    def setFloatParam(self, param, value):
        if param == self.floatparam_simulation_time_step:
            if self.running:
                raise RuntimeError("cannot change the time step while the simulation is running")
            self.dt = float(value)
            return 1
        raise KeyError(param)

    def createPrimitiveShape(self, kind, sizes, options=0):
        h = self.add(f"/Shape{len(self.objects)}", "shape", np.eye(4), static=False)
        self.objects[h].primitive = (kind, list(sizes))
        self.objects[h].engine = {}
        self.objects[h].velocity = np.zeros(3)
        return h

    def createTexture(self, path, options=0, plane_sizes=None, scaling_uv=None, xy_g=None, fixed=None, resolution=None):
        h = self.add(f"/Plane{len(self.objects)}", "shape", np.eye(4), static=True)
        w, hh = (0.1, 0.1) if plane_sizes is None else (float(plane_sizes[0]), float(plane_sizes[1]))
        self.objects[h].texture = str(path)
        self.objects[h].plane = (w, hh)
        self.objects[h].respondable = False
        self.objects[h].engine = {}
        return h, 1000 + h, [64, 64]

    def createForceSensor(self, options, int_params, float_params):
        h = self.add(f"/forceSensor{len(self.objects)}", "forcesensor", np.eye(4))
        return h

    def createMeshShape(self, options, shading_angle, vertices, indices):
        h = self.add(f"/Mesh{len(self.objects)}", "shape", np.eye(4), static=True)
        self.objects[h].mesh = (np.asarray(vertices, float).reshape(-1, 3), np.asarray(indices, int).reshape(-1, 3))
        self.objects[h].engine = {}
        return h

    def setShapeMass(self, h, m):
        self.objects[h].mass = float(m)

    def setShapeColor(self, h, colorname, component, rgb):
        self.objects[h].color = list(rgb)

    def setEngineFloatParam(self, param, h, value):
        self.objects[h].engine[param] = float(value)
        return 1

    def getEngineFloatParam(self, param, h):
        return self.objects[h].engine.get(param, 0.5)

    def setObjectParent(self, h, parent, keep_in_place=True):
        o = self.objects[h]
        if o.parent in self.objects:
            self.objects[o.parent].children.remove(h)
        o.parent = parent
        if parent in self.objects:
            self.objects[parent].children.append(h)
        # T_world_zero stays the object's world frame at zero; keep_in_place is implied
        return 1

    def getObjectPosition(self, h, rel=-1):
        return [float(x) for x in self._world_frame(h)[:3, 3]]

    def setObjectPosition(self, h, pos, rel=-1):
        self.objects[h].T_world_zero[:3, 3] = np.asarray(pos, float)

    def setObjectPose(self, h, pose, rel=-1):
        self.objects[h].T_world_zero = _sim.pose7_to_transform(pose)

    def getObjectVelocity(self, h):
        return [float(x) for x in getattr(self.objects[h], "velocity", np.zeros(3))], [0.0, 0.0, 0.0]

    def createVisionSensor(self, options, int_params, float_params):
        h = self.add(f"/visionSensor{len(self.objects)}", "visionsensor", np.eye(4))
        self.objects[h].resolution = (int(int_params[0]), int(int_params[1]))
        self.calls.append(("createVisionSensor", options, list(int_params), list(float_params)))
        return h

    def getVisionSensorImg(self, h, options=0, pos=None, size=None):
        # Bottom-up RGB bytes, as CoppeliaSim returns them: the bottom row is "ground" (brown),
        # the top row is "sky" (blue), so an upright frame has blue at row 0.
        w, hh = self.objects[h].resolution
        img = np.zeros((hh, w, 3), dtype=np.uint8)
        img[:, :, 2] = np.linspace(0, 255, hh, dtype=np.uint8)[:, None]  # blue grows with row index
        img[0, :, 0] = 120  # bottom row (index 0 in sensor order) is brown-ish ground
        img[:, :, 1] = int(self.time * 1000) % 256  # changes every step
        return img.tobytes(), [w, hh]

    def handleVisionSensor(self, h):
        return 0

    def removeObjects(self, handles, delay=False):
        for h in handles:
            o = self.objects.pop(h)
            self.by_path.pop(o.path, None)
            if o.parent in self.objects:
                self.objects[o.parent].children.remove(h)
        self.calls.append(("removeObjects", list(handles)))

    def addDrawingObject(self, *args):
        self.drawings.append(list(args))
        return 900 + len(self.drawings)

    def addDrawingObjectItem(self, handle, item):
        self.drawings[handle - 901].append(item)

    def removeDrawingObject(self, handle):
        self.drawings[handle - 901] = None

    def live_drawings(self):
        return [d for d in self.drawings if d is not None]


def two_joint_scene() -> FakeSim:
    """/Arm with j1 (z axis at the origin), j2 (z axis at (0, 0, 0.5)) and a dummy tip at
    (0.3, 0, 0.5). Both joints revolute about z, so both screw axes are (0,0,1, 0,0,0)."""
    sim = FakeSim()
    base = sim.add("/Arm", "shape", np.eye(4))
    j1 = sim.add("/Arm/j1", "joint", np.eye(4), parent=base, interval=(-3.0, 3.0))
    l1 = sim.add("/Arm/link1", "shape", se3.trans([0, 0, 0.25]), parent=j1,
                 mass=2.0, inertia=np.diag([0.1, 0.1, 0.02]), static=False)
    j2 = sim.add("/Arm/j2", "joint", se3.trans([0, 0, 0.5]), parent=l1, interval=(-2.0, 2.0))
    # link 2 is a static visual shell (its mass must be ignored) plus two dynamic cubes
    l2 = sim.add("/Arm/link2", "shape", se3.trans([0.15, 0, 0.5]), parent=j2, mass=99.0, static=True)
    sim.add("/Arm/cubeA", "shape", se3.trans([0.1, 0, 0.5]), parent=j2,
            mass=1.0, inertia=np.diag([0.01, 0.01, 0.01]), static=False)
    sim.add("/Arm/cubeB", "shape", se3.trans([0.3, 0, 0.5]), parent=j2,
            mass=3.0, inertia=np.diag([0.02, 0.02, 0.02]), static=False)
    sim.add("/Arm/tip", "dummy", se3.trans([0.3, 0, 0.5]), parent=l2)
    sim.add("/Arm/Script", "script", np.eye(4), parent=base)  # a stock model's demo script
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
