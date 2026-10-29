"""Synchronization of bundled skill examples with repository examples.

Run from the repository root::

    python scripts/sync_skill_examples.py
    python scripts/sync_skill_examples.py --check
"""

import argparse
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DESTINATION = _ROOT / "protuna-skills/skills/protuna-components/examples"
_EXAMPLES = (
    "biohub_adapter.py",
    "genetic_algorithm_sampler.py",
    "structure_constraint.py",
    "ucb_acquisition.py",
)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Fail if bundled examples are stale."
    )
    args = parser.parse_args()
    stale = False
    for name in _EXAMPLES:
        source = (_ROOT / "examples" / name).read_bytes()
        destination = _DESTINATION / name
        if destination.exists() and destination.read_bytes() == source:
            continue
        stale = True
        if args.check:
            print(f"Stale bundled example: {destination.relative_to(_ROOT)}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source)
            print(f"Updated {destination.relative_to(_ROOT)} from examples/{name}")
    if args.check and stale:
        print("Run: uv run python scripts/sync_skill_examples.py")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
