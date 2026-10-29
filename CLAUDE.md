# CLAUDE.md

Protuna is an optimization framework for AI-guided protein engineering campaigns. It proposes candidate sequences for the next round of wet-lab experiments.

## Architecture

The package lives entirely under `protuna/` . The main modules are:

- `protuna.optim`: The primary optimization framework lives here.
- `protuna.data`: Everything needed to interface with protein engineering data and pre-processed public benchmark datasets.
- `protuna.score`: Functional library of protein scoring functions.
- `protuna.inference`: Ray based distributed inference interface and common model adapters examples.
- `protuna.utils`: For utilities that don't have anything to do with the library's main functionalities (downloading files etc.)
- `constants.py`: All global constants go here
- `typing.py`: Global type annotations used throughout protuna

Tests and documentations live one level up in separate `tests/` and `docs/` directories respectively.
Test files are named `test_{module_name}.py` and live under `tests/unit` for unit tests, `tests/integration` for integration tests and
each mimics the directory structure of `protuna`.

## Commands

All commands run through **uv**:

```bash
uv sync                        # install deps + dev group into .venv
uv run pytest                  # run the test suite
uv run ruff format .           # format
uv run ruff check .            # lint
uv run ty check                # type check
uv pre-commit run --all-files  # run pre-commit hooks
```

Building docs:

```bash
cd docs && uv run make html          # one-off build into docs/_build
uv run sphinx-autobuild docs docs/_build   # live-reload while writing docs
```

## Code Style

Ruff is configured with strict settings (ruff.toml):

- Docstrings follow the Google convention.
- **Import the module, not names from it.** Write `from protuna import data` then `data.tokenizer.encode(...)`, not `from protuna.data import tokenizer`.
- isort is configured `force-single-line` with a Google-like profile. The only allowed exceptions (name imports) are typing-related modules — see `single-line-exclusions` in `ruff.toml` (`typing`, `protuna.typing`, `collections.abc`, `jaxtyping`, `protuna.models.typing`, `pandera.typing`).
- **Type annotations are required** everywhere except `self`/`cls` and `*args`/`**kwargs`.
- Tensor shapes are annotated with `jaxtyping` (e.g. `Bool[torch.Tensor, "l v"]`); General convention for annotating shapes:
    - 'b': a batch of sequences
    - 'l': sequence length
    - 'v': vocabulary
    - 'd': hidden dims of a model
    - 'g': a group of sequences
    - 'n': N sequences
- Test files relax docstring rules for modules (`test_*.py`: `D100` ignored) - no module level docstring is required.
- **Always run `uv run ty`, `uv run ruff format` and then `uv run ruff check` on python files you edit** before finishing, and fix any lint or typing errors you introduced.

Use the `documentation` skill when writing docstrings.
Use the `protuna-components` skill when writing new `protuna.optim` components.
Use the `protuna-pipeline` skill when writing a protuna pipeline.

## Parallel Agents Note

Multiple agents may be working simultaneously. If you see errors in files you did NOT edit, do not try to fix them, another agent is likely mid-edit.
