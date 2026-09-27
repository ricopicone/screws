"""A putting green for the UR5: a ball, a hole, a kinematic putter on the flange, and a putt.

The ball is the only dynamic body. The putter is a static, respondable tool rigidly attached
to the arm's last link, so it moves exactly where the joint trajectory puts it and pushes the
ball through contact. Everything else is kinematics from the rest of screws: the face pose
that addresses the ball, a straight stroke along the target line, IK for every waypoint.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..robot import Robot
from ..se3 import rp_to_transform, trans
from ..so3 import exp3, rot, vec_to_so3
from ..trajectory import cartesian_trajectory, joint_trajectory, quintic_time_scaling

__all__ = [
    "BALL_MASS",
    "BALL_RADIUS",
    "HOLE_RADIUS",
    "INSERT_MARGIN",
    "TURF_COLOUR",
    "TURF_THICKNESS",
    "Green",
    "PuttResult",
    "Putter",
    "address_pose",
    "attach_putter",
    "build_green",
    "camera_forward",
    "camera_yaw",
    "green_pieces",
    "logo_image",
    "putt",
    "seal_image",
    "stroke_path",
    "surface_mesh",
    "swing_path",
    "text_image",
]

BALL_RADIUS = 0.02135  # a regulation golf ball, 42.67 mm across
BALL_MASS = 0.0459  # kg
HOLE_RADIUS = 0.054  # a regulation cup, 108 mm across
TURF_THICKNESS = 0.05  # the turf slab the cup is cut through; deep enough to swallow the ball
TURF_COLOUR = (0.16, 0.55, 0.20)
TURF_COLOUR_8BIT = tuple(round(255 * c) for c in TURF_COLOUR)
TEXTURE_SIZE = 1024  # every texture is a square of this size: CoppeliaSim renders non-square ones dark
SMU_RED = (0.729, 0.047, 0.184)  # Saint Martin's University PMS 200, #BA0C2F
SMU_RED_8BIT = (186, 12, 47)
INSERT_MARGIN = 0.10  # the convex-wedge insert around the cup extends this far beyond its rim


def _shape(sim, h, *, static: bool, respondable: bool) -> None:
    sim.setObjectInt32Param(h, sim.shapeintparam_static, 1 if static else 0)
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1 if respondable else 0)


def _colour(sim, h, rgb) -> None:
    try:
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, list(rgb))
    except Exception:  # noqa: BLE001, S110 - colour is cosmetic
        pass


@dataclass
class Green:
    """The turf, the cup cut through it, the pin, and the ball."""

    scene: object
    ball: int
    turf: list[int]  # the slabs and the wedge insert, all static convex pieces (hidden)
    surface: int  # the one smooth visual surface with the round hole (no physics)
    cup_floor: int
    pin: tuple[int, ...]
    hole_position: np.ndarray  # (x, y, top) at the turf surface
    top: float  # height of the turf surface above the floor
    seal: int | None = None
    logo: int | None = None

    def ball_position(self) -> np.ndarray:
        return np.asarray(self.scene.sim.getObjectPosition(self.ball, self.scene.sim.handle_world), float)

    def ball_speed(self) -> float:
        v, _ = self.scene.sim.getObjectVelocity(self.ball)
        return float(np.linalg.norm(v))

    def distance_to_hole(self) -> float:
        """Horizontal distance from the ball's centre to the cup's centre."""
        return float(np.linalg.norm((self.ball_position() - self.hole_position)[:2]))

    def holed(self) -> bool:
        """True once the ball has dropped into the cup: inside the rim and below the surface."""
        p = self.ball_position()
        return self.distance_to_hole() <= HOLE_RADIUS and p[2] < self.top - BALL_RADIUS / 2

    def handles(self) -> list[int]:
        extras = [h for h in (self.seal, self.logo) if h is not None]
        return [self.ball, *self.turf, self.surface, self.cup_floor, *self.pin, *extras]

    def remove(self) -> None:
        self.scene.untrack(*self.handles())
        self.scene.sim.removeObjects(self.handles())


def _font(size: int):
    from PIL import ImageFont

    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def camera_yaw(position, look_at) -> float:
    """The yaw (about z) whose x direction is a camera's horizontal right: what to pass as
    build_green's ``yaw`` so text laid on the green reads correctly from that camera."""
    d = np.asarray(look_at, float)[:2] - np.asarray(position, float)[:2]
    return float(np.arctan2(d[1], d[0]) - np.pi / 2)


def camera_forward(position, look_at) -> np.ndarray:
    """The horizontal unit vector pointing away from a camera: "up" in its image on the green."""
    d = np.asarray(look_at, float)[:2] - np.asarray(position, float)[:2]
    return d / np.linalg.norm(d)


def _square_canvas(art, background, size: int = TEXTURE_SIZE):
    """Scale an RGBA artwork to fit a size x size square and centre it on a background of that
    colour. CoppeliaSim renders non-square textures at about half brightness (measured on
    4.10), so every texture laid on the green is a square; the background matches the turf
    so the unused part of the square is invisible."""
    from PIL import Image

    w, h = art.size
    scale = size / max(w, h)
    inner = (max(1, round(w * scale)), max(1, round(h * scale)))
    art = art.resize(inner, Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), background)
    canvas.paste(art, ((size - inner[0]) // 2, (size - inner[1]) // 2), art)
    return canvas


def text_image(path, text: str, *, fg=(255, 255, 255), bg=SMU_RED_8BIT, aspect: float = 14 / 9, size: int = 512) -> Path:
    """Write a square PNG of bold text on a solid background (defaults: white on SMU red) for a
    flag plane of the given width:height aspect: the text is drawn at that aspect and the
    image stretched to a square, so it reads correctly once mapped onto the plane (a
    non-square texture would render dark)."""
    from PIL import Image, ImageDraw

    canvas = (size, max(1, round(size / aspect)))
    im = Image.new("RGB", canvas, bg)
    draw = ImageDraw.Draw(im)
    font_size = int(canvas[1] * 0.6)
    font = _font(font_size)
    while font_size > 8:
        box = draw.textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= canvas[0] * 0.85 and box[3] - box[1] <= canvas[1] * 0.75:
            break
        font_size = int(font_size * 0.9)
        font = _font(font_size)
    box = draw.textbbox((0, 0), text, font=font)
    x = (canvas[0] - (box[2] - box[0])) / 2 - box[0]
    y = (canvas[1] - (box[3] - box[1])) / 2 - box[1]
    draw.text((x, y), text, font=font, fill=fg)
    path = Path(path)
    im.resize((size, size), Image.LANCZOS).save(path)
    return path


def seal_image(path, artwork, *, disc=(255, 255, 255), background=TURF_COLOUR_8BIT, size: int = 512) -> Path:
    """Write a PNG of a logo (a PNG with transparency, or any image) on a white disc that blends
    into the turf outside the disc, for a flat plane laid on the green. The artwork's own
    colours are kept, as the brand guide asks. size must be a power of two."""
    from PIL import Image, ImageDraw

    art = Image.open(artwork).convert("RGBA")
    im = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(im)
    draw.ellipse((0, 0, size - 1, size - 1), fill=disc)
    inner = int(size * 0.86)
    art = art.resize((inner, inner), Image.LANCZOS)
    offset = (size - inner) // 2
    im.paste(art, (offset, offset), art)
    path = Path(path)
    im.save(path)
    return path


def logo_image(path, artwork, *, plate=None, size: int = TEXTURE_SIZE) -> tuple[Path, float]:
    """Write a square PNG of a logo (a PNG with transparency) for a plane laid flat on the
    green, and return (path, aspect): the artwork is stretched to fill the square (non-square
    textures render dark) and reads correctly on a plane of width:height = aspect, which is
    what build_green's logo_aspect takes. With plate=None the transparent parts take the
    turf colour, so a white or coloured logo looks painted on the grass; plate=(255, 255, 255)
    puts it on a white sign. Colours are kept.
    """
    from PIL import Image

    art = Image.open(artwork).convert("RGBA")
    background = TURF_COLOUR_8BIT if plate is None else plate
    if plate is not None:  # a visible plate: pad the artwork so the plate shows around it
        w, h = art.size
        m = round(0.06 * w)
        padded = Image.new("RGBA", (w + 2 * m, h + 2 * m), (*plate, 255))
        padded.paste(art, (m, m), art)
        art = padded
    aspect = art.size[0] / art.size[1]
    canvas = Image.new("RGB", (size, size), background)
    stretched = art.resize((size, size), Image.LANCZOS)
    canvas.paste(stretched, (0, 0), stretched)
    path = Path(path)
    canvas.save(path)
    return path, aspect


def surface_mesh(*, size, thickness, center, hole, hole_radius=HOLE_RADIUS, n: int = 96):
    """Vertices (V, 3) and triangles (F, 3) of a rectangular slab with a round hole through it:
    the turf as one smooth *visual* surface (the physics uses green_pieces underneath).

    size is (sx, sy) and center (cx, cy) of the slab, whose bottom is at z = 0 and top at
    z = thickness; hole is (hx, hy). Outward normals, closed manifold, so it serves as a
    static respondable mesh.
    """
    sx, sy = float(size[0]), float(size[1])
    cx, cy = float(center[0]), float(center[1])
    hx, hy = float(hole[0]), float(hole[1])
    xmin, xmax, ymin, ymax = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2
    if not (xmin < hx - hole_radius and hx + hole_radius < xmax and ymin < hy - hole_radius and hy + hole_radius < ymax):
        raise ValueError("the hole must lie inside the slab")
    angles = 2 * np.pi * np.arange(n) / n
    inner = [(a, np.array([hx + hole_radius * np.cos(a), hy + hole_radius * np.sin(a)])) for a in angles]

    def ray_to_edge(a):
        d = np.array([np.cos(a), np.sin(a)])
        ts = []
        if d[0] > 1e-12:
            ts.append((xmax - hx) / d[0])
        if d[0] < -1e-12:
            ts.append((xmin - hx) / d[0])
        if d[1] > 1e-12:
            ts.append((ymax - hy) / d[1])
        if d[1] < -1e-12:
            ts.append((ymin - hy) / d[1])
        t = min(ts)
        return np.array([hx, hy]) + t * d

    outer = [(a, ray_to_edge(a)) for a in angles]
    for corner in ((xmax, ymax), (xmin, ymax), (xmin, ymin), (xmax, ymin)):
        a = float(np.arctan2(corner[1] - hy, corner[0] - hx)) % (2 * np.pi)
        if all(abs(a - b) > 1e-9 for b, _ in outer):
            outer.append((a, np.array(corner, float)))
    outer.sort(key=lambda ap: ap[0])
    ni, no = len(inner), len(outer)
    # vertex layout: top inner, top outer, bottom inner, bottom outer
    TI, TO, BI, BO = 0, ni, ni + no, ni + no + ni
    V = []
    for z in (thickness, 0.0):
        V += [[p[0], p[1], z] for _, p in inner]
        V += [[p[0], p[1], z] for _, p in outer]
    V = np.array(V)
    F = []
    # top: zip the two angle-sorted polygons; bottom: the same with reversed winding
    i = j = 0
    while i < ni or j < no:
        a_next = inner[(i + 1) % ni][0] + (2 * np.pi if i + 1 >= ni else 0.0) if i < ni else np.inf
        b_next = outer[(j + 1) % no][0] + (2 * np.pi if j + 1 >= no else 0.0) if j < no else np.inf
        if a_next <= b_next:
            F.append([TI + i % ni, TO + j % no, TI + (i + 1) % ni])
            i += 1
        else:
            F.append([TI + i % ni, TO + j % no, TO + (j + 1) % no])
            j += 1
    top_count = len(F)
    for tri in list(F[:top_count]):
        F.append([tri[0] + BI, tri[2] + BI, tri[1] + BI])
    for j in range(no):  # outer walls, outward
        k = (j + 1) % no
        F.append([BO + j, BO + k, TO + k])
        F.append([BO + j, TO + k, TO + j])
    for i in range(ni):  # hole walls, facing into the cup
        k = (i + 1) % ni
        F.append([TI + i, TI + k, BI + k])
        F.append([TI + i, BI + k, BI + i])
    return V, np.array(F, dtype=int)


def _ray_to_square(hx, hy, a, xmin, xmax, ymin, ymax):
    d = np.array([np.cos(a), np.sin(a)])
    ts = []
    if d[0] > 1e-12:
        ts.append((xmax - hx) / d[0])
    if d[0] < -1e-12:
        ts.append((xmin - hx) / d[0])
    if d[1] > 1e-12:
        ts.append((ymax - hy) / d[1])
    if d[1] < -1e-12:
        ts.append((ymin - hy) / d[1])
    return np.array([hx, hy]) + min(ts) * d


def _prism(top_xy, z0, z1):
    """A convex prism over a convex counter-clockwise polygon: (vertices, triangles), outward."""
    k = len(top_xy)
    V = np.array([[x, y, z1] for x, y in top_xy] + [[x, y, z0] for x, y in top_xy])
    F = []
    for m in range(1, k - 1):  # top fan, counter-clockwise from above
        F.append([0, m, m + 1])
    for m in range(1, k - 1):  # bottom fan, reversed
        F.append([k, k + m + 1, k + m])
    for m in range(k):  # walls
        n = (m + 1) % k
        F.append([k + m, k + n, n])
        F.append([k + m, n, m])
    return V, np.array(F, dtype=int)


def green_pieces(*, size, thickness, center, hole, hole_radius=HOLE_RADIUS, n: int = 32):
    """The turf as convex pieces: four cuboid slabs and, in a square insert around the cup,
    n wedge prisms whose inner edges form the round hole.

    Returns {"slabs": [(cx, cy, sx, sy), ...], "wedges": [(V, F), ...]}. Slabs have their
    bottom at z = 0 and top at z = thickness; wedge tops sit 0.3 mm lower so a ball rolling
    off a slab never catches their edge. Convex pieces keep the physics engine happy (a
    non-convex mesh is flagged by CoppeliaSim and kicks a rolling ball at its internal edges).
    """
    sx, sy = float(size[0]), float(size[1])
    cx, cy = float(center[0]), float(center[1])
    hx, hy = float(hole[0]), float(hole[1])
    a = hole_radius + INSERT_MARGIN
    xmin, xmax, ymin, ymax = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2
    if not (xmin <= hx - a and hx + a <= xmax and ymin <= hy - a and hy + a <= ymax):
        raise ValueError("the cup and its insert must lie inside the turf")
    slabs = [
        ((xmin + (hx - a)) / 2, cy, (hx - a) - xmin, sy),  # left of the insert, full depth
        (((hx + a) + xmax) / 2, cy, xmax - (hx + a), sy),  # right of the insert, full depth
        (hx, (ymin + (hy - a)) / 2, 2 * a, (hy - a) - ymin),  # in front of the insert
        (hx, ((hy + a) + ymax) / 2, 2 * a, ymax - (hy + a)),  # behind the insert
    ]
    slabs = [sl for sl in slabs if sl[2] > 1e-9 and sl[3] > 1e-9]
    angles = 2 * np.pi * np.arange(n) / n
    inner = [np.array([hx + hole_radius * np.cos(t), hy + hole_radius * np.sin(t)]) for t in angles]
    corners = []
    for corner in ((hx + a, hy + a), (hx - a, hy + a), (hx - a, hy - a), (hx + a, hy - a)):
        corners.append((float(np.arctan2(corner[1] - hy, corner[0] - hx)) % (2 * np.pi), np.array(corner)))
    wedges = []
    for i in range(n):
        t0, t1 = angles[i], angles[(i + 1) % n] + (2 * np.pi if i + 1 >= n else 0.0)
        outer0 = _ray_to_square(hx, hy, t0, hx - a, hx + a, hy - a, hy + a)
        outer1 = _ray_to_square(hx, hy, t1, hx - a, hx + a, hy - a, hy + a)
        between = [c for ang, c in corners if t0 < ang < t1 or t0 < ang + 2 * np.pi < t1]
        between.sort(key=lambda c: (float(np.arctan2(c[1] - hy, c[0] - hx)) - t0) % (2 * np.pi))
        # counter-clockwise around the wedge: out along ray i, along the square, back in
        poly = [inner[(i + 1) % n], inner[i], outer0, *between, outer1]
        wedges.append(_prism(poly, 0.0, thickness - 0.0003))
    return {"slabs": slabs, "wedges": wedges}


def build_green(
    scene,
    *,
    ball_position,
    hole_position,
    size=(3.0, 3.0),
    thickness: float = TURF_THICKNESS,
    friction: float = 0.8,
    damping: float = 0.6,
    restitution: float = 0.2,
    flag_image=None,
    seal_image=None,
    seal_position=None,
    seal_size: float = 0.35,
    logo_image=None,
    logo_position=None,
    logo_width: float = 0.6,
    logo_aspect: float | None = None,
    yaw: float = 0.0,
) -> Green:
    """Lay turf with a real cup at (x, y), a pin in it, and a ball at (x, y) on the surface.

    The turf is a static respondable slab of the given size (centred between the ball and
    the hole) and thickness, with the cup cut through it and a cup floor below; see
    green_pieces for why it is built from convex pieces. The ball's material (friction,
    Bullet's linear and angular damping, restitution) defaults roll about 0.3 m from a
    0.3 m/s face on a dynamic putter.

    flag_image (a PNG, e.g. from text_image) textures the flag; seal_image (e.g. from
    seal_image) is laid flat on the turf as a seal_size square at seal_position (x, y),
    default: beside the line, on the camera side; logo_image (e.g. from logo_image) is laid
    flat logo_width wide at logo_position, default: just past the seal along the yawed
    direction, with height logo_width / logo_aspect (default: the file's own aspect; pass the
    aspect logo_image returns, since it writes a square). yaw (radians about z) turns both so their text reads along a camera's
    horizontal: pass the camera's right-hand direction, e.g. golf.camera_yaw(position, look_at).
    Everything built is removed when the Scene exits, or by Green.remove().
    """
    sim = scene.sim
    bx, by = float(ball_position[0]), float(ball_position[1])
    hx, hy = float(hole_position[0]), float(hole_position[1])
    center = ((bx + hx) / 2, (by + hy) / 2)
    pieces = green_pieces(size=size, thickness=thickness, center=center, hole=(hx, hy))
    turf = []
    for cx, cy, sx, sy in pieces["slabs"]:
        h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [sx, sy, thickness], 0)
        _shape(sim, h, static=True, respondable=True)
        sim.setObjectPosition(h, [cx, cy, thickness / 2], sim.handle_world)
        _colour(sim, h, TURF_COLOUR)
        turf.append(h)
    for V, F in pieces["wedges"]:
        h = sim.createMeshShape(0, 0.0, [float(x) for x in V.reshape(-1)], [int(k) for k in F.reshape(-1)])
        _shape(sim, h, static=True, respondable=True)
        _colour(sim, h, TURF_COLOUR)
        turf.append(h)
    for h in turf:  # the physics pieces are invisible; one smooth surface is what you see
        sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0)
    V, F = surface_mesh(size=size, thickness=thickness, center=center, hole=(hx, hy), hole_radius=HOLE_RADIUS)
    surface = sim.createMeshShape(0, 0.0, [float(x) for x in V.reshape(-1)], [int(k) for k in F.reshape(-1)])
    _shape(sim, surface, static=True, respondable=False)
    _colour(sim, surface, TURF_COLOUR)
    cup_floor = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [2 * HOLE_RADIUS, 2 * HOLE_RADIUS, 0.004], 0)
    _shape(sim, cup_floor, static=True, respondable=True)
    sim.setObjectPosition(cup_floor, [hx, hy, 0.002], sim.handle_world)
    _colour(sim, cup_floor, (0.08, 0.08, 0.08))
    pin_height = 0.6
    pin = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [0.008, 0.008, pin_height], 0)
    _shape(sim, pin, static=True, respondable=False)
    sim.setObjectPosition(pin, [hx, hy, pin_height / 2], sim.handle_world)
    _colour(sim, pin, (0.95, 0.95, 0.9))
    flag_w, flag_h = 0.14, 0.09
    if flag_image is not None:
        flag, _, _ = sim.createTexture(str(flag_image), 0, [flag_w, flag_h])
        _shape(sim, flag, static=True, respondable=False)
        _colour(sim, flag, (1.0, 1.0, 1.0))
        # a vertical plane: image x along world x, image up along world z, facing -y
        R = np.column_stack([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        scene.set_frame(flag, rp_to_transform(R, [hx + 0.004 + flag_w / 2, hy, pin_height - flag_h / 2 - 0.01]))
    else:
        flag = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [flag_w, 0.003, flag_h], 0)
        _shape(sim, flag, static=True, respondable=False)
        sim.setObjectPosition(flag, [hx + 0.004 + flag_w / 2, hy, pin_height - flag_h / 2 - 0.01], sim.handle_world)
        _colour(sim, flag, SMU_RED)
    R_yaw = rot([0.0, 0.0, 1.0], yaw)
    right = R_yaw[:2, 0]
    seal = None
    if seal_position is None:
        seal_position = (center[0] + 0.15, center[1] - 0.45)
    if seal_image is not None:
        seal, _, _ = sim.createTexture(str(seal_image), 0, [seal_size, seal_size])
        _shape(sim, seal, static=True, respondable=False)
        _colour(sim, seal, (1.0, 1.0, 1.0))  # textures modulate the shape colour
        scene.set_frame(seal, rp_to_transform(R_yaw, [float(seal_position[0]), float(seal_position[1]), thickness + 0.0005]))
    logo = None
    if logo_image is not None:
        if logo_aspect is None:
            from PIL import Image

            w, h = Image.open(logo_image).size
            logo_aspect = w / h
        logo_h = logo_width / logo_aspect
        if logo_position is None:
            logo_position = np.asarray(seal_position, float) + (seal_size / 2 + 0.08 + logo_width / 2) * right
        logo, _, _ = sim.createTexture(str(logo_image), 0, [logo_width, logo_h])
        _shape(sim, logo, static=True, respondable=False)
        _colour(sim, logo, (1.0, 1.0, 1.0))
        scene.set_frame(logo, rp_to_transform(R_yaw, [float(logo_position[0]), float(logo_position[1]), thickness + 0.0005]))
    ball = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [2 * BALL_RADIUS] * 3, 0)
    sim.setShapeMass(ball, BALL_MASS)
    _shape(sim, ball, static=False, respondable=True)
    sim.setObjectPosition(ball, [bx, by, thickness + BALL_RADIUS], sim.handle_world)
    sim.setEngineFloatParam(sim.bullet_body_friction, ball, friction)
    sim.setEngineFloatParam(sim.bullet_body_restitution, ball, restitution)
    sim.setEngineFloatParam(sim.bullet_body_lineardamping, ball, damping)
    sim.setEngineFloatParam(sim.bullet_body_angulardamping, ball, damping)
    _colour(sim, ball, (0.95, 0.95, 0.95))
    green = Green(scene, ball, turf, surface, cup_floor, (pin, flag), np.array([hx, hy, thickness]), thickness, seal, logo)
    scene.track(*green.handles())
    return green


@dataclass
class Putter:
    """A putter on the arm: a shaft along the tool's +z and a face whose outward normal is
    the tool's +x, both static shapes parented to the last link (drive the arm in kinematic
    mode). T_tip_face is the face frame in the tip (last link) frame."""

    scene: object
    shaft: int
    face: int
    T_tip_face: np.ndarray
    face_size: tuple[float, float, float]

    def robot(self, robot: Robot) -> Robot:
        """The arm's Robot with M moved from the flange to the putter face."""
        return replace(robot, name=f"{robot.name}+putter", M=robot.M @ self.T_tip_face)

    def remove(self) -> None:
        self.scene.untrack(self.shaft, self.face)
        self.scene.sim.removeObjects([self.shaft, self.face])


def attach_putter(
    scene,
    arm,
    *,
    shaft_length: float = 0.30,
    shaft_radius: float = 0.006,
    face_size=(0.02, 0.10, 0.02),
) -> Putter:
    """Build a putter on the arm's tip: static shapes parented to the last link.

    The face is a box of face_size (thickness along tool x, width along y, height along z)
    centred shaft_length along the tool's +z axis; its striking face is the +x side and it is
    respondable. The shaft is a visual only. Drive the arm with arm.mode("kinematic") for a
    putt: the tree is then static and the face is a rigid mover that strikes the ball exactly
    where the trajectory says (a respondable static shape on a *dynamic* tree is flagged by
    CoppeliaSim as unrealistic, and a dynamic face on a force sensor wobbles and lags).
    """
    sim = scene.sim
    T_tip = arm.tip_frame()
    T_tip_face = trans([0.0, 0.0, shaft_length])
    shaft = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [2 * shaft_radius, 2 * shaft_radius, shaft_length], 0)
    _shape(sim, shaft, static=True, respondable=False)
    scene.set_frame(shaft, T_tip @ trans([0.0, 0.0, shaft_length / 2]))
    _colour(sim, shaft, (0.6, 0.6, 0.6))
    face = sim.createPrimitiveShape(sim.primitiveshape_cuboid, list(face_size), 0)
    _shape(sim, face, static=True, respondable=True)
    scene.set_frame(face, T_tip @ T_tip_face)
    _colour(sim, face, (0.8, 0.1, 0.1))
    for h in (shaft, face):
        sim.setObjectParent(h, arm.tip, True)
    scene.track(shaft, face)
    return Putter(scene, shaft, face, T_tip_face, tuple(float(x) for x in face_size))


def _target_line(ball, hole) -> np.ndarray:
    d = np.asarray(hole, float) - np.asarray(ball, float)
    d[2] = 0.0
    n = np.linalg.norm(d)
    if n < 1e-9:
        raise ValueError("the ball is already on the hole")
    return d / n


def address_pose(ball, hole, *, back: float = 0.06) -> np.ndarray:
    """The face frame that addresses the ball: back metres behind it on the target line, at
    ball-centre height, face normal (tool x) along the line, shaft (tool z) pointing down."""
    ball = np.asarray(ball, float)
    x = _target_line(ball, hole)
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return rp_to_transform(np.column_stack([x, y, z]), ball - back * x)


def stroke_path(ball, hole, *, back: float = 0.06, through: float = 0.04, speed: float = 0.5, dt: float = 0.05):
    """Face poses, one per control step, for a straight stroke at constant speed from back
    metres behind the ball to through metres past its centre. The address pose itself is not
    included; the first pose is one step into the stroke."""
    T0 = address_pose(ball, hole, back=back)
    x = T0[:3, 0]
    length = back + through
    step = speed * dt  # exact face speed; the last pose is clamped to the stroke length
    n = max(1, int(np.ceil(length / step - 1e-9)))
    return [rp_to_transform(T0[:3, :3], T0[:3, 3] + min(k * step, length) * x) for k in range(1, n + 1)]


def swing_path(
    ball,
    hole,
    *,
    shaft_length: float,
    back_angle: float = 0.25,
    through_angle: float = 0.2,
    speed: float = 0.3,
    dt: float = 0.01,
    gap: float = 0.01,
    face_thickness: float = 0.02,
    back_time: float = 1.0,
    return_time: float = 1.0,
) -> dict:
    """A putting stroke as a pendulum swing of the putter about the wrist.

    The address pose puts the face gap metres behind the ball's surface. The pivot is the top
    of the shaft; the face swings in the vertical plane of the target line: back and up by
    back_angle (quintic timing over back_time), then down through the ball at a constant
    angular rate speed / shaft_length (so the face meets the ball at that speed), on to
    through_angle, and finally back to the address pose over return_time.

    Returns {"backswing": [...], "downswing": [...], "back": [...]} of face poses, one per dt.
    """
    T0 = address_pose(ball, hole, back=BALL_RADIUS + face_thickness / 2 + gap)
    R0, o0 = T0[:3, :3], T0[:3, 3]
    y = R0[:, 1]
    pivot = o0 + np.array([0.0, 0.0, shaft_length])

    def pose(phi):  # positive phi swings the face backward (-x) and up
        R = exp3(vec_to_so3(-y * phi))
        return rp_to_transform(R @ R0, pivot + R @ (o0 - pivot))

    n_back = max(1, round(back_time / dt))
    backswing = [pose(back_angle * quintic_time_scaling(back_time, k * dt)) for k in range(1, n_back + 1)]
    omega = speed / shaft_length
    n_down = max(1, int(np.ceil((back_angle + through_angle) / (omega * dt) - 1e-9)))
    downswing = [pose(max(back_angle - k * omega * dt, -through_angle)) for k in range(1, n_down + 1)]
    n_ret = max(1, round(return_time / dt))
    back = [pose(-through_angle * (1.0 - quintic_time_scaling(return_time, k * dt))) for k in range(1, n_ret + 1)]
    return {"backswing": backswing, "downswing": downswing, "back": back}


@dataclass
class PuttResult:
    holed: bool
    distance: float  # final horizontal distance from the ball to the hole, m
    ball_path: np.ndarray  # (N, 3) ball positions, one per step from the stroke onward
    face_path: np.ndarray  # (N, 3) putter-face positions (from the measured joints), same steps
    theta_final: np.ndarray


def _ik_or_raise(robot: Robot, T, seeds, what: str) -> np.ndarray:
    last = None
    for seed in seeds:
        res = robot.ik(T, seed, max_iter=50)
        last = res
        if res.converged and robot.within_limits(res.theta):
            return res.theta
    raise RuntimeError(
        f"inverse kinematics could not reach the {what} pose (errors {last.error_omega:.3g}, "
        f"{last.error_v:.3g}); move the ball or pass a better seed"
    )


def putt(
    scene,
    arm,
    robot_face: Robot,
    green: Green,
    *,
    speed: float = 0.3,
    back_angle: float = 0.25,
    through_angle: float = 0.2,
    gap: float = 0.01,
    lift: float = 0.12,
    approach_time: float = 2.0,
    settle_time: float = 5.0,
    seed=None,
    shaft_length: float = 0.30,
    face_thickness: float = 0.02,
) -> PuttResult:
    """Address the ball, swing the putter through it toward the hole, and watch it roll.

    robot_face is the arm's Robot with M at the putter face (Putter.robot). Put the arm in
    kinematic mode (arm.mode("kinematic")) and start the simulation first. The approach goes
    in joint space to a pose lift metres above the address pose and descends straight onto
    it; the stroke is swing_path's pendulum: backswing, a pause, a downswing meeting the ball
    at speed, follow-through, and a return to the address pose, where the arm waits until the
    ball stops or settle_time has passed.

    Use a 10 ms control step (scene.set_time_step(0.01)): at the default 50 ms the face jumps
    15 mm per step and the strike is no longer clean. 0.3 m/s rolls the ball about 0.3 m on
    the default green.
    """
    dt = scene.dt
    ball = green.ball_position()
    hole = green.hole_position
    T_address = address_pose(ball, hole, back=BALL_RADIUS + face_thickness / 2 + gap)
    T_lift = trans([0.0, 0.0, lift]) @ T_address
    theta_now = arm.theta()
    seeds = [theta_now] if seed is None else [np.asarray(seed, float), theta_now]
    theta_lift = _ik_or_raise(robot_face, T_lift, seeds, "lifted address")
    theta_address = _ik_or_raise(robot_face, T_address, [theta_lift], "address")
    ball_path, face_path = [], []

    def go(thetas, log=False):
        for th in thetas:
            arm.command(th)
            scene.step()
            if log:
                ball_path.append(green.ball_position())
                face_path.append(robot_face.fk(arm.theta())[:3, 3])

    def follow(poses, theta, what):
        out = []
        for T in poses:
            theta = _ik_or_raise(robot_face, T, [theta], what)
            out.append(theta)
        return out

    n_approach = max(2, round(approach_time / dt))
    go(joint_trajectory(theta_now, theta_lift, approach_time, n_approach)[1:])
    descent = follow(cartesian_trajectory(T_lift, T_address, approach_time / 2, max(2, n_approach // 2))[1:], theta_lift, "descent")
    go(descent)
    go([theta_address] * round(0.5 / dt))  # a still moment at address

    swing = swing_path(
        ball, hole, shaft_length=shaft_length, back_angle=back_angle, through_angle=through_angle,
        speed=speed, dt=dt, gap=gap, face_thickness=face_thickness,
    )
    backswing = follow(swing["backswing"], theta_address, "backswing")
    go(backswing, log=True)
    go([backswing[-1]] * round(0.3 / dt), log=True)  # the pause at the top
    downswing = follow(swing["downswing"], backswing[-1], "downswing")
    go(downswing, log=True)
    go([downswing[-1]] * round(0.3 / dt), log=True)  # hold the follow-through
    back = follow(swing["back"], downswing[-1], "return")
    go(back, log=True)
    theta = back[-1]
    for k in range(round(settle_time / dt)):
        go([theta], log=True)
        if k > round(0.5 / dt) and green.ball_speed() < 2e-3:
            break
    return PuttResult(
        green.holed(), green.distance_to_hole(), np.array(ball_path), np.array(face_path), theta
    )
