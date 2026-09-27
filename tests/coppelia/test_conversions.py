import numpy as np
import pytest

from screws import se3, so3
from screws.coppelia import _sim


def test_matrix12_roundtrip():
    T = se3.rp_to_transform(so3.rot([0, 0, 1], 0.4), [1, 2, 3])
    m = _sim.transform_to_matrix12(T)
    assert len(m) == 12 and m[3] == 1.0 and m[7] == 2.0 and m[11] == 3.0
    assert np.allclose(_sim.matrix12_to_transform(m), T)
    assert _sim.matrix12_to_transform(m).shape == (4, 4)


def test_pose7_roundtrip_quaternion_order():
    T = se3.rp_to_transform(so3.rot([1, 0, 0], np.pi / 2), [0, 0, 1])
    p = _sim.transform_to_pose7(T)
    assert np.allclose(p[:3], [0, 0, 1])
    assert np.allclose(p[3:], [np.sqrt(0.5), 0, 0, np.sqrt(0.5)])  # x y z w
    assert np.allclose(_sim.pose7_to_transform(p), T)
    rng = np.random.default_rng(3)
    for _ in range(10):
        R = so3.exp3(so3.vec_to_so3(rng.normal(size=3)))
        T = se3.rp_to_transform(R, rng.normal(size=3))
        assert np.allclose(_sim.pose7_to_transform(_sim.transform_to_pose7(T)), T)


def test_inertia9_to_matrix():
    I = _sim.inertia9_to_matrix([1, 2, 3, 2, 4, 5, 3, 5, 6])
    assert np.allclose(I, [[1, 2, 3], [2, 4, 5], [3, 5, 6]])


def test_connect_without_simulator_raises_student_sentence(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("zmq timeout")

    monkeypatch.setattr(_sim, "_new_client", boom)
    with pytest.raises(_sim.SimulatorNotRunning, match="Start CoppeliaSim"):
        _sim.connect()


def test_connect_returns_sim_from_client(monkeypatch):
    class Client:
        timeout = 600

        def require(self, name):
            assert name == "sim"
            return type("Sim", (), {"getSimulationTime": staticmethod(lambda: 0.0)})()

    monkeypatch.setattr(_sim, "_new_client", lambda host, port: Client())
    sim = _sim.connect()
    assert sim.getSimulationTime() == 0.0


def test_connect_to_a_closed_port_raises_promptly():
    import socket
    import time

    # find a port nobody listens on
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    t0 = time.monotonic()
    with pytest.raises(_sim.SimulatorNotRunning, match="Start CoppeliaSim"):
        _sim.connect("127.0.0.1", port, timeout_s=0.5)
    assert time.monotonic() - t0 < 5.0
