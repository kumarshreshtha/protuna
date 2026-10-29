"""README and documentation quickstart synchronization with the runnable example.

Run from the repository root::

    python scripts/sync_readme_example.py
    python scripts/sync_readme_example.py --check
"""

import argparse
import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_START = "<!-- quickstart-code-start -->"
_END = "<!-- quickstart-code-end -->"


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Fail if the README is stale."
    )
    args = parser.parse_args()
    path = _ROOT / "README.md"
    readme = path.read_text(encoding="utf-8")
    if readme.count(_START) != 1 or readme.count(_END) != 1:
        parser.error(
            "README.md must contain exactly one pair of quickstart code markers"
        )
    before, rest = readme.split(_START)
    _, after = rest.split(_END)
    source = (
        (_ROOT / "examples/getting_started.py").read_text(encoding="utf-8").rstrip()
    )
    module = ast.parse(source)
    description = ast.get_docstring(module)
    if description is None:
        parser.error("examples/getting_started.py must contain a module docstring")
    docstring_end = module.body[0].end_lineno
    source = "\n".join(source.splitlines()[docstring_end:]).lstrip("\n")
    updated = f"{before}{_START}\n{description}\n\n```py\n{source}\n```\n{_END}{after}"
    if updated == readme:
        return 0
    if args.check:
        print(
            "README example is stale; run: uv run python scripts/sync_readme_example.py"
        )
        return 1
    path.write_text(updated, encoding="utf-8")
    print("Updated README.md from examples/getting_started.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
