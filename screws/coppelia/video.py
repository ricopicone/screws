"""Recorder: a movie of a run, one frame per simulation step, at simulated-time playback."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

__all__ = ["Recorder"]

_EXTENSIONS = (".mp4", ".gif")


class Recorder:
    """Grabs a vision sensor's image before every step and writes an MP4 or GIF on exit.

    Use through ``scene.record_video(path, ...)``. ``frames`` is (N, H, W, 3) uint8, upright;
    ``fps`` is 1 / (every * dt), so the movie plays at simulated time however slowly the
    controller ran. ``capture()`` grabs a frame by hand; ``Scene.run`` grabs one every
    ``every`` steps.
    """

    def __init__(self, scene, path, camera: int, every: int = 1):
        path = Path(path)
        if path.suffix.lower() not in _EXTENSIONS:
            raise ValueError(f"cannot write {path.suffix!r}; use one of {_EXTENSIONS}")
        if every < 1:
            raise ValueError("every must be at least 1")
        self.scene = scene
        self.sim = scene.sim
        self.path = path
        self.camera = camera
        self.every = int(every)
        self._frames: list[np.ndarray] = []
        self._steps = 0
        self.saved: Path | None = None

    # ----- frames -----------------------------------------------------------------------

    def capture(self) -> np.ndarray:
        """Grab the camera's current image (upright, RGB uint8) and keep it."""
        try:
            self.sim.handleVisionSensor(self.camera)  # refresh an explicitly handled sensor
        except Exception:  # noqa: BLE001, S110 - an implicitly handled sensor is already fresh
            pass
        raw, (w, h) = self.sim.getVisionSensorImg(self.camera)
        img = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(int(h), int(w), 3)
        frame = np.flipud(img).copy()  # CoppeliaSim images are bottom-up
        self._frames.append(frame)
        return frame

    def tick(self) -> None:
        """Called by Scene.run before each step: capture every ``every``-th step."""
        if self._steps % self.every == 0:
            self.capture()
        self._steps += 1

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def frames(self) -> np.ndarray:
        return np.array(self._frames) if self._frames else np.zeros((0, 0, 0, 3), dtype=np.uint8)

    @property
    def fps(self) -> float:
        return 1.0 / (self.every * self.scene.dt)

    # ----- saving -----------------------------------------------------------------------

    def save(self, path=None) -> Path:
        """Encode the frames to ``path`` (default: the path given at creation). Needs screws[video]."""
        import imageio.v2 as imageio

        path = Path(path) if path is not None else self.path
        if path.suffix.lower() not in _EXTENSIONS:
            raise ValueError(f"cannot write {path.suffix!r}; use one of {_EXTENSIONS}")
        frames = list(self._frames)
        if not frames:
            raise ValueError("no frames were captured")
        if path.suffix.lower() == ".gif":
            imageio.mimwrite(path, frames, duration=1000.0 / self.fps, loop=0)
        else:
            imageio.mimwrite(path, frames, fps=self.fps, macro_block_size=1)
        if not path.exists() or path.stat().st_size == 0:
            raise OSError(f"the encoder wrote nothing to {path} (is ffmpeg available? are the frames even-sized?)")
        self.saved = path
        return path

    # ----- context ----------------------------------------------------------------------

    def __enter__(self):
        if self not in self.scene._recorders:
            self.scene._recorders.append(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self in self.scene._recorders:
            self.scene._recorders.remove(self)
        if not self._frames:
            return
        try:
            self.save()
        except Exception as save_error:
            if exc_type is None:
                raise
            warnings.warn(f"the movie was not saved ({save_error}); the run's own error follows", stacklevel=2)
