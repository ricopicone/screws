import sys


def test_import_is_light():
    for m in ("zmq", "cbor2", "matplotlib"):
        sys.modules.pop(m, None)
    import screws

    assert screws.__version__[0].isdigit()  # the exact release pin lives in test_readme_table
    assert "zmq" not in sys.modules
    assert "matplotlib" not in sys.modules
