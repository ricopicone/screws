import sys


def test_import_is_light():
    for m in ("zmq", "cbor2", "matplotlib"):
        sys.modules.pop(m, None)
    import screws

    assert screws.__version__.startswith("0.1")
    assert "zmq" not in sys.modules
    assert "matplotlib" not in sys.modules
