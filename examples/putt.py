"""Putt a golf ball into a hole with the UR5.

Start CoppeliaSim with the stock UR5 model at /UR5 (Model browser > robots > non-mobile),
then:  uv run python examples/putt.py [--speed 0.45] [--video putt.mp4]

Everything on the green is built by the script: a ball, a cup, and a putter on the flange.
The putter is a kinematic tool (a static, respondable box), so the only physics is the ball.
The arm addresses the ball behind it on the line to the hole, strokes straight through at a
constant face speed, and the script reports where the ball stopped.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from screws.coppelia import Scene, golf

SEED = np.array([-1.21, 0.18, 1.4, -0.01, -1.57, 0.36])  # elbow up, shaft down, in front of the base
# The Hal and Inge Marcus School of Engineering seal, if this machine has it; any PNG works.
DEFAULT_SEAL = Path(
    "/Users/picone/Library/CloudStorage/OneDrive-SaintMartin'sUniversity/SMU HIMSE - Documents/Logos/"
    "EngineeringSeal-black-transparent.png"
)
CAMERA = {"position": (2.1, -1.7, 1.15), "look_at": (0.5, 0.25, 0.25), "resolution": (1280, 720)}
CONTROL_DT = 0.01  # a 10 ms control step keeps the stroke smooth; the movie samples every 5th step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--speed", type=float, default=0.3, help="face speed at impact, m/s (0.25 to 0.4)")
    ap.add_argument("--ball", type=float, nargs=2, default=(0.55, 0.30), metavar=("X", "Y"))
    ap.add_argument("--hole", type=float, nargs=2, default=(0.85, 0.30), metavar=("X", "Y"))
    ap.add_argument("--video", default=None, help="record the putt to this .mp4 or .gif")
    ap.add_argument("--flag-text", default="SMU", help="text on the flag (empty for a plain flag)")
    ap.add_argument("--seal", default=str(DEFAULT_SEAL) if DEFAULT_SEAL.exists() else None,
                    help="a PNG laid on the green as a seal (default: the school seal, if present)")
    args = ap.parse_args()

    art = Path(tempfile.mkdtemp(prefix="screws-putt-"))
    flag_image = golf.text_image(art / "flag.png", args.flag_text) if args.flag_text else None
    seal = golf.seal_image(art / "seal.png", args.seal) if args.seal else None

    with Scene() as scene:
        scene.set_time_step(CONTROL_DT)
        arm = scene.arm("/UR5")
        arm.teleport(np.zeros(6))
        robot = arm.robot()  # M and the screw axes off the scene
        putter = golf.attach_putter(scene, arm)
        robot_face = putter.robot(robot)  # M moved to the putter face
        green = golf.build_green(
            scene, ball_position=args.ball, hole_position=args.hole, flag_image=flag_image, seal_image=seal
        )
        arm.mode("kinematic")
        arm.teleport(SEED)
        scene.start()
        try:
            if args.video:
                with scene.record_video(args.video, every=round(0.05 / CONTROL_DT), **CAMERA):
                    result = golf.putt(scene, arm, robot_face, green, speed=args.speed, seed=SEED)
            else:
                result = golf.putt(scene, arm, robot_face, green, speed=args.speed, seed=SEED)
        finally:
            scene.stop()  # the green and the putter are removed when the Scene exits
            time.sleep(0.5)
        verdict = "holed!" if result.holed else f"missed by {result.distance:.2f} m"
        print(f"face speed {args.speed} m/s: {verdict}; the ball rolled "
              f"{np.linalg.norm(result.ball_path[-1][:2] - result.ball_path[0][:2]):.2f} m")


if __name__ == "__main__":
    main()
