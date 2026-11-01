import pathlib
import re

from screws.aliases import ALIASES

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"
NAMES = README.parent / "docs" / "guide" / "names.md"  # the docs page that holds the table


def test_every_alias_is_in_the_names_table():
    text = NAMES.read_text(encoding="utf-8")
    rows = {m.group(1): m.group(2) for m in re.finditer(r"^\| `(\w+)` \| `(\w+)` \|", text, re.MULTILINE)}
    missing = {k: v for k, v in ALIASES.items() if rows.get(k) != v}
    assert not missing, f"the table in docs/guide/names.md is out of date for: {missing}"


def test_version_is_release():
    import screws

    assert screws.__version__ == "0.9.1"
    assert 'version = "0.9.1"' in (README.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert "\nversion: 0.9.1\n" in (README.parent / "CITATION.cff").read_text(encoding="utf-8")
