"""The examples page runs as written: each Python block in order, in one namespace, and where
a plain block follows one, the block's last line prints exactly what the page shows."""

import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

PAGE = Path(__file__).parents[1] / "docs" / "examples.md"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_every_picture_is_there():
    page = PAGE.read_text(encoding="utf-8")
    pictures = re.findall(r"\]\((media/[^)]+)\)", page)
    assert len(pictures) == 9
    missing = [p for p in pictures if not (PAGE.parent / p).is_file()]
    assert not missing, missing


def test_the_examples_page_runs_and_prints_what_it_shows():
    page = PAGE.read_text(encoding="utf-8")
    blocks = re.findall(r"```(python)?\n(.*?)```", page, re.DOTALL)
    ns: dict = {}
    checked = 0
    for (lang, code), (next_lang, shown) in zip(blocks, blocks[1:] + [("", None)]):
        if lang != "python":
            continue
        body = ast.parse(code).body
        if next_lang == "" and shown is not None and not shown.startswith("uv run"):
            *body, last = body
            exec(compile(ast.Module(body, []), "examples.md", "exec"), ns)  # noqa: S102
            value = eval(compile(ast.Expression(last.value), "examples.md", "eval"), ns)
            assert repr(value) == shown.rstrip("\n")
            checked += 1
        else:
            exec(compile(ast.Module(body, []), "examples.md", "exec"), ns)  # noqa: S102 - the page's own code
    assert checked == 2
