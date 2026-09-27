import pathlib
import re

from screws.aliases import ALIASES

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def test_every_alias_is_in_the_readme_table():
    text = README.read_text()
    rows = {m.group(1): m.group(2) for m in re.finditer(r"^\| `(\w+)` \| `(\w+)` \|", text, re.MULTILINE)}
    missing = {k: v for k, v in ALIASES.items() if rows.get(k) != v}
    assert not missing, f"README table is out of date for: {missing}"


def test_version_is_release():
    import screws

    assert screws.__version__ == "0.3.0"
    assert 'version = "0.3.0"' in (README.parent / "pyproject.toml").read_text()
