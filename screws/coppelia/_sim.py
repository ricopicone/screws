"""The thin layer over CoppeliaSim's ZMQ remote API: connecting, and converting between
CoppeliaSim's flat matrix, pose and inertia encodings and 4x4 / 3x3 numpy arrays.

Everything here is testable without a simulator: the bridge receives a ``sim`` object and
never imports the client except inside connect().
"""

from __future__ import annotations

import numpy as np

from ..se3 import rp_to_transform

__all__ = [
    "SimulatorNotRunning",
    "connect",
    "inertia9_to_matrix",
    "matrix12_to_transform",
    "pose7_to_transform",
    "transform_to_matrix12",
    "transform_to_pose7",
]

STUDENT_SENTENCE = "Start CoppeliaSim and open a scene, then run this again."


class SimulatorNotRunning(RuntimeError):
    """CoppeliaSim did not answer on the remote API port."""


def _new_client(host: str, port: int):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    return RemoteAPIClient(host, port)


def connect(host: str = "localhost", port: int = 23000, *, timeout_s: float = 5.0):
    """Connect to CoppeliaSim's ZMQ remote API and return its ``sim`` object.

    Raises SimulatorNotRunning, with the sentence a student needs, if nothing answers.
    """
    try:
        client = _new_client(host, port)
        client.timeout = timeout_s
        sim = client.require("sim")
        sim.getSimulationTime()
    except Exception as exc:  # zmq.Again, ConnectionRefusedError, ...: all mean "not running"
        raise SimulatorNotRunning(
            f"No CoppeliaSim answered at {host}:{port} ({type(exc).__name__}). {STUDENT_SENTENCE}"
        ) from exc
    client.timeout = 10 * 60
    sim._screws_client = client  # keep the client alive as long as sim is
    return sim


def matrix12_to_transform(m) -> np.ndarray:
    """CoppeliaSim's 12 floats (rows 0-2 of T, row-major) to a 4x4 transformation."""
    T = np.eye(4)
    T[:3, :] = np.asarray(m, dtype=float).reshape(3, 4)
    return T


def transform_to_matrix12(T) -> list[float]:
    """A 4x4 transformation to CoppeliaSim's 12 floats (rows 0-2, row-major)."""
    return [float(x) for x in np.asarray(T, dtype=float)[:3, :].reshape(-1)]


def pose7_to_transform(p) -> np.ndarray:
    """CoppeliaSim's pose (x, y, z, qx, qy, qz, qw) to a 4x4 transformation."""
    p = np.asarray(p, dtype=float)
    x, y, z, w = p[3:7] / np.linalg.norm(p[3:7])
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return rp_to_transform(R, p[:3])


def transform_to_pose7(T) -> list[float]:
    """A 4x4 transformation to CoppeliaSim's pose (x, y, z, qx, qy, qz, qw)."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return [float(v) for v in (*T[:3, 3], x, y, z, w)]


def inertia9_to_matrix(i) -> np.ndarray:
    """CoppeliaSim's 9 inertia floats (row-major) to a 3x3 matrix."""
    return np.asarray(i, dtype=float).reshape(3, 3)
