import sys


def test_import_is_light():
    for m in ("zmq", "cbor2", "matplotlib"):
        sys.modules.pop(m, None)
    import screws

    assert screws.__version__[0].isdigit()  # the exact release pin lives in test_readme_table
    assert "zmq" not in sys.modules
    assert "matplotlib" not in sys.modules


def test_every_name_in_all_is_exported():
    import screws

    missing = [n for n in screws.__all__ if not hasattr(screws, n)]
    assert not missing, missing
    assert screws.joint_trajectory is screws.trajectory.joint_trajectory
