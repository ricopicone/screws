import numpy as np
import pytest

from screws import se3, so3
from screws.coppelia import Scene, golf
from tests.coppelia.fake_sim import two_joint_scene


def test_build_green_creates_turf_with_a_cup_and_a_ball_on_top():
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        green = golf.build_green(scene, ball_position=(0.5, 0.0), hole_position=(1.2, 0.1))
        ball = sim.objects[green.ball]
        assert ball.primitive[0] == sim.primitiveshape_spheroid
        assert np.isclose(ball.primitive[1][0], 2 * golf.BALL_RADIUS)
        assert not ball.static and ball.respondable and np.isclose(ball.mass, golf.BALL_MASS)
        assert ball.engine[sim.bullet_body_friction] == pytest.approx(0.8)
        # the ball starts on top of the turf, the hole position is on the turf surface
        assert np.allclose(green.ball_position(), [0.5, 0.0, green.top + golf.BALL_RADIUS])
        assert np.allclose(green.hole_position, [1.2, 0.1, green.top])
        assert len(green.turf) == 4 + 32
        slab, wedge = sim.objects[green.turf[0]], sim.objects[green.turf[-1]]
        assert slab.static and slab.respondable and slab.primitive[0] == sim.primitiveshape_cuboid
        assert wedge.static and wedge.respondable and hasattr(wedge, "mesh")
        assert np.allclose(slab.color, golf.TURF_COLOUR) and np.allclose(wedge.color, golf.TURF_COLOUR)
        cup_floor = sim.objects[green.cup_floor]
        assert cup_floor.static and cup_floor.respondable
        assert np.isclose(green.distance_to_hole(), np.hypot(0.7, 0.1))
        assert not green.holed()
        handles = [green.ball, *green.turf, green.cup_floor, *green.pin]
        green.remove()
        assert all(h not in sim.objects for h in handles)


def test_holed_means_the_ball_has_dropped_into_the_cup():
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        green = golf.build_green(scene, ball_position=(0.5, 0.0), hole_position=(1.2, 0.1))
        sim.setObjectPosition(green.ball, [1.21, 0.09, green.top + golf.BALL_RADIUS])
        assert not green.holed()  # sitting on the rim's plane does not count
        sim.setObjectPosition(green.ball, [1.21, 0.09, golf.BALL_RADIUS + 0.002])
        assert green.holed()


def test_green_pieces_are_convex_and_tile_the_slab_around_a_round_hole():
    pieces = golf.green_pieces(size=(2.0, 1.0), thickness=0.05, center=(0.2, 0.1), hole=(0.5, 0.2), n=32)
    slabs, wedges = pieces["slabs"], pieces["wedges"]
    assert len(slabs) == 4 and len(wedges) == 32
    a = golf.HOLE_RADIUS + golf.INSERT_MARGIN
    # the slabs cover everything outside the insert square around the hole
    slab_area = sum(sx * sy for (_, _, sx, sy) in slabs)
    assert np.isclose(slab_area, 2.0 * 1.0 - (2 * a) ** 2)
    for cx, cy, sx, sy in slabs:  # no slab reaches into the insert square
        assert cx + sx / 2 <= 0.5 - a + 1e-9 or cx - sx / 2 >= 0.5 + a - 1e-9 or cy + sy / 2 <= 0.2 - a + 1e-9 or cy - sy / 2 >= 0.2 + a - 1e-9
    wedge_area = 0.0
    for V, F in wedges:
        top = V[np.isclose(V[:, 2], V[:, 2].max())][:, :2]
        # every top vertex is on or outside the hole circle
        assert np.all(np.linalg.norm(top - [0.5, 0.2], axis=1) >= golf.HOLE_RADIUS - 1e-9)
        # the top polygon is convex (all turns the same way) and closed manifold triangles
        k = len(top)
        turns = [float(np.cross(np.r_[top[(m + 1) % k] - top[m], 0], np.r_[top[(m + 2) % k] - top[(m + 1) % k], 0])[2]) for m in range(k)]
        assert all(t >= -1e-12 for t in turns)
        wedge_area += 0.5 * abs(sum(top[m, 0] * top[(m + 1) % k, 1] - top[(m + 1) % k, 0] * top[m, 1] for m in range(k)))
        edges = {}
        for tri in F:
            for aa, bb in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edges[(min(aa, bb), max(aa, bb))] = edges.get((min(aa, bb), max(aa, bb)), 0) + 1
        assert set(edges.values()) == {2}
    polygon_hole = 0.5 * 32 * golf.HOLE_RADIUS**2 * np.sin(2 * np.pi / 32)
    assert np.isclose(wedge_area, (2 * a) ** 2 - polygon_hole, atol=1e-9)


def _point_in_triangle(q, tri):
    (x1, y1), (x2, y2), (x3, y3) = tri
    d1 = (q[0] - x2) * (y1 - y2) - (x1 - x2) * (q[1] - y2)
    d2 = (q[0] - x3) * (y2 - y3) - (x2 - x3) * (q[1] - y3)
    d3 = (q[0] - x1) * (y3 - y1) - (x3 - x1) * (q[1] - y1)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def test_attach_putter_parents_a_kinematic_face_to_the_last_link_with_the_right_offset():
    sim = two_joint_scene()
    with Scene(sim=sim) as scene:
        arm = scene.arm("/Arm")
        putter = golf.attach_putter(scene, arm, shaft_length=0.3)
        shaft, face = sim.objects[putter.shaft], sim.objects[putter.face]
        # both ride on the last link as static shapes: the arm is driven kinematically for a
        # putt, so the tree is static and the respondable face strikes the ball cleanly
        assert shaft.static and not shaft.respondable and shaft.parent == arm.tip
        assert face.static and face.respondable and face.parent == arm.tip
        assert np.allclose(scene.frame(putter.face)[:3, 3], face_expected_origin(arm, 0.3))
        T_tip = arm.tip_frame()
        # the face centre sits shaft_length along the tip's +z axis
        face_expected = T_tip @ se3.trans([0, 0, 0.3])
        assert np.allclose(scene.frame(putter.face)[:3, 3], face_expected[:3, 3])
        assert np.allclose(putter.T_tip_face, se3.trans([0, 0, 0.3]))
        assert putter.face_size[2] == pytest.approx(0.02)
        # the tool robot's M is the face frame at zero
        r = arm.robot()
        rt = putter.robot(r)
        assert np.allclose(rt.M, r.M @ putter.T_tip_face) and np.allclose(rt.S, r.S)
        putter.remove()
        assert putter.face not in sim.objects


def face_expected_origin(arm, shaft_length):
    return (arm.tip_frame() @ se3.trans([0, 0, shaft_length]))[:3, 3]


def test_address_pose_geometry():
    ball = np.array([0.5, 0.0, golf.BALL_RADIUS])
    hole = np.array([1.2, 0.1, 0.0])
    T = golf.address_pose(ball, hole, back=0.06)
    d = np.array([0.7, 0.1, 0.0])
    d /= np.linalg.norm(d)
    assert np.allclose(T[:3, 0], d)  # face normal (tool x) points along the target line
    assert np.allclose(T[:3, 2], [0, 0, -1])  # shaft (tool z) points down
    assert np.allclose(T[:3, 3], ball - 0.06 * d)  # 6 cm behind the ball, at ball-centre height
    assert so3.is_so3(T[:3, :3])


def test_stroke_path_is_straight_through_the_ball_at_constant_speed():
    ball = np.array([0.5, 0.0, golf.BALL_RADIUS])
    hole = np.array([1.2, 0.0, 0.0])
    path = golf.stroke_path(ball, hole, back=0.06, through=0.04, speed=0.5, dt=0.05)
    assert path[0].shape == (4, 4)
    # 10 cm at 0.5 m/s is 0.2 s: 4 control steps of 0.05 s, one pose each
    assert len(path) == 4
    xs = [T[0, 3] for T in path]
    assert np.allclose(np.diff(xs), 0.025)
    assert np.isclose(xs[0], 0.5 - 0.06 + 0.025) and np.isclose(xs[-1], 0.5 + 0.04)
    assert all(np.isclose(T[2, 3], golf.BALL_RADIUS) for T in path)
    assert all(np.allclose(T[:3, :3], path[0][:3, :3]) for T in path)


def test_text_image_and_seal_image_write_pngs_in_brand_colours(tmp_path):
    from PIL import Image

    flag = golf.text_image(tmp_path / "flag.png", "SMU")
    im = Image.open(flag).convert("RGB")
    assert im.size[0] > im.size[1]
    assert im.getpixel((3, 3)) == golf.SMU_RED_8BIT  # background is SMU red
    assert any(im.getpixel((x, im.size[1] // 2)) == (255, 255, 255) for x in range(im.size[0]))  # white letters

    src = tmp_path / "seal.png"
    art = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    for x in range(80, 120):
        for y in range(80, 120):
            art.putpixel((x, y), (0, 0, 0, 255))
    art.save(src)
    seal = golf.seal_image(tmp_path / "seal_disc.png", src)
    im = Image.open(seal).convert("RGB")
    assert im.size == (512, 512)
    assert im.getpixel((2, 2)) == golf.TURF_COLOUR_8BIT  # corners blend into the turf
    assert im.getpixel((256, 40)) == (255, 255, 255)  # a white disc
    assert im.getpixel((256, 256)) == (0, 0, 0)  # the artwork stays black


def test_build_green_places_a_seal_plane_and_a_textured_flag(tmp_path):
    sim = two_joint_scene()
    flag = tmp_path / "flag.png"
    flag.write_bytes(b"png")
    seal = tmp_path / "seal.png"
    seal.write_bytes(b"png")
    with Scene(sim=sim) as scene:
        green = golf.build_green(
            scene, ball_position=(0.5, 0.3), hole_position=(0.9, 0.3),
            flag_image=flag, seal_image=seal, seal_position=(0.9, -0.15), seal_size=0.35,
        )
        seal_plane = sim.objects[green.seal]
        assert seal_plane.texture == str(seal) and seal_plane.plane == (0.35, 0.35)
        T = scene.frame(green.seal)
        assert np.allclose(T[:3, 3], [0.9, -0.15, green.top + 0.0005]) and np.allclose(T[:3, 2], [0, 0, 1])
        flag_plane = sim.objects[green.pin[1]]
        assert flag_plane.texture == str(flag)
        Tf = scene.frame(green.pin[1])
        assert np.allclose(Tf[:3, 2], [0, -1, 0])  # the flag faces -y, toward the default camera
        assert np.allclose(Tf[:3, 1], [0, 0, 1])  # image up is world up
        assert Tf[2, 3] > 0.5 and Tf[0, 3] > 0.9
        handles = [green.seal, *green.pin]
        green.remove()
        assert all(h not in sim.objects for h in handles)


def test_build_green_without_branding_has_no_seal():
    with Scene(sim=two_joint_scene()) as scene:
        green = golf.build_green(scene, ball_position=(0.5, 0.3), hole_position=(0.9, 0.3))
        assert green.seal is None
