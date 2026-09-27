# Releasing screws

1. `uv run ruff check .` and `uv run pytest -q` are clean.
2. With CoppeliaSim open on the UR5 scene, run the simulator tests:
   `SCREWS_COPPELIASIM=1 uv run pytest -q -m coppelia`.
3. Bump the version in `screws/_version.py` and `pyproject.toml` (they must agree;
   `tests/test_readme_table.py` pins the release number).
4. If `screws/aliases.py` changed, regenerate the README table with
   `uv run python tools/print_alias_table.py` and paste it under "Names".
5. If the `modern_robotics` dev pin changed, re-harvest the docstring examples:
   `uv run python tools/harvest_mr_examples.py`.
6. `uv build`, then publish by hand with twine (`uv publish` ignores `~/.pypirc`):
   `uvx twine upload dist/*`.
7. Tag: `git tag v<version> && git push --tags`.
8. Pin the new version in the course repo.
