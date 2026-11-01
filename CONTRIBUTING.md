# Contributing to screws

`screws` is maintained by Rico Picone at Saint Martin's University, where it is the software
behind the robotics course ME 465/MME 565. Questions, bug reports and changes are welcome
from anyone.

## Getting help

Open an issue at <https://github.com/ricopicone/screws/issues>. Say what you tried, what
you expected and what happened, and give `screws.__version__`, your Python version and your
operating system. For a question about the mathematics, name the section of *Modern
Robotics* (MR) it concerns. If you are a student in a course that uses `screws`, ask your
instructor first; an issue is for the package itself.

## Reporting a bug

An issue with a few lines that run and show the problem, the output, and what you expected.
For a number that disagrees with something, say what you compared it with: MR's code
library, a worked example in the book, or a hand calculation. For the CoppeliaSim bridge,
give the CoppeliaSim version too.

## Proposing a change

For anything beyond a small fix, open an issue first, so that the design is agreed before the
code is written. Then:

1. Fork the repository and branch from `main`.
2. `uv sync --all-extras` sets up everything, the test and lint tools included.
3. Make the change, with tests. `uv run pytest -q` and `uv run ruff check .` must pass. The
   tests that need a running CoppeliaSim are skipped unless you ask for them; see
   [docs/RELEASING.md](https://github.com/ricopicone/screws/blob/main/docs/RELEASING.md) for how.
4. Open a pull request that says what changed and why. CI runs the tests on Linux, macOS and
   Windows.

## Conventions

- **Notation is MR's.** Screw axes are S (space) and B (body), the home configuration is M,
  a twist is (ω, v), and a list of screw axes is 6×n, one axis per column. A symbol MR does
  not use is named as such in the docstring.
- **MR's names stay.** A function with an MR counterpart gets a snake_case name, and MR's
  name is bound to it in `screws/aliases.py`; the table on [MR's names](https://ricopicone.github.io/screws/guide/names/) is tested against
  that file.
- **The free functions are the reference.** `Robot` methods call them rather than repeat
  them, and a numerical function that SymPy can do gets a twin of the same name and
  arguments in `screws.symbolic`.
- **The core stays light.** `import screws` needs only numpy; SymPy loads when
  `screws.symbolic` is first used, and matplotlib, the CoppeliaSim client and imageio only in
  the modules that draw, simulate or record.
- **Pictures use one colour per meaning**, as [the screw axes' page](https://ricopicone.github.io/screws/guide/pictures/screw-axes/)
  sets out.

## Releases

Each release is listed in [CHANGELOG.md](https://github.com/ricopicone/screws/blob/main/CHANGELOG.md). How one is made is in
[docs/RELEASING.md](https://github.com/ricopicone/screws/blob/main/docs/RELEASING.md).
