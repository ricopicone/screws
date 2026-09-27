import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("SCREWS_COPPELIASIM") == "1":
        return
    skip = pytest.mark.skip(reason="needs a running CoppeliaSim; set SCREWS_COPPELIASIM=1")
    for item in items:
        if "coppelia" in item.keywords:
            item.add_marker(skip)
