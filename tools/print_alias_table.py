"""Print the MR-name to screws-name table as Markdown, for the README."""

from screws.aliases import ALIASES


def table() -> str:
    lines = ["| Modern Robotics | `screws` |", "|---|---|"]
    lines += [f"| `{mr}` | `{ours}` |" for mr, ours in ALIASES.items()]
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
