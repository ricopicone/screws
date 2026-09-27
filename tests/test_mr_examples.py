"""Every MR docstring example, run through the screws alias of the same name."""

import inspect

import modern_robotics.core as mr_core
import numpy as np
import pytest

import screws as sc
from screws.aliases import ALIASES
from tests.mr_examples import EXAMPLES

_NP = {"np": np, "numpy": np}


def _evaluate_output(src: str):
    # MR's Output blocks are numpy literals, sometimes with the commas between rows missing.
    src = src.replace("]\n", "],\n").replace("],\n]", "]\n]").rstrip(",\n")
    try:
        return eval(src, dict(_NP))
    except SyntaxError:
        fixed = src.replace("]\n                  [", "],\n                  [")
        return eval(fixed, dict(_NP))


def _compare(got, want, atol=1e-6, rtol=1e-4):
    if isinstance(want, tuple):
        assert isinstance(got, tuple) and len(got) == len(want)
        for g, w in zip(got, want):
            _compare(g, w, atol, rtol)
    elif isinstance(want, (bool, np.bool_)):
        assert bool(got) == bool(want)
    else:
        assert np.allclose(
            np.asarray(got, dtype=float), np.asarray(want, dtype=float), atol=atol, rtol=rtol
        )


@pytest.mark.parametrize("name,inp,outp", EXAMPLES, ids=[e[0] for e in EXAMPLES])
def test_mr_docstring_example(name, inp, outp):
    if name not in ALIASES:
        pytest.skip(f"{name} arrives in a later screws release")
    ns = dict(_NP)
    exec(inp, ns)
    # Bind by MR's own parameter names: the example defines them, and the screws alias
    # takes the same arguments in the same order.
    params = list(inspect.signature(getattr(mr_core, name)).parameters)
    args = [ns[p] for p in params if p in ns]
    assert len(args) == len(params), f"{name}: could not bind {params} from {sorted(ns)}"
    _compare(getattr(sc, name)(*args), _evaluate_output(outp))
