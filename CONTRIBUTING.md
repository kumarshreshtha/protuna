Thank you for your interest in contributing to Protuna! If you're a new contributor, please first read the [Issue and PR Workflow](#issues-and-pr-workflow) section.

## Issues and PR workflow

Before starting an implementation, search the [GitHub issues](https://github.com/kumarshreshtha/protuna/issues)
for an existing discussion. If there isn't one, open a feature request or bug issue.

- For a feature request, describe the use case, expected behavior and proposed approach.
- For a bug, include reproduction steps, expected and actual behavior, and relevant
  environment details or error output.

Discuss the implementation in the issue and agree with the maintainers on the scope,
API, behavior and validation before writing the change. Once those details are agreed,
implement the change and open a pull request that links to the issue.

## Environment setup

`.python-version` selects Python 3.12 for development.
We use uv to manage dependencies and run development commands. Follow the official [uv installation guide](https://docs.astral.sh/uv/getting-started/installation) to setup uv.

Fork the repository on GitHub, then clone your fork and create a branch for the agreed change:

```sh
git clone git@github.com:YOUR_USERNAME/protuna.git
cd protuna
git remote add upstream https://github.com/kumarshreshtha/protuna.git
git switch -c my-change
uv sync
uv run pre-commit install
```

`uv sync` creates `.venv`, installs Protuna in editable mode and installs the default
development dependency group, including testing, linting, typing and documentation tools.
Run commands from the repository root with `uv run`; activating `.venv` is optional.
Run `uv sync` again after dependency or lockfile changes.

The `pre-commit install` command sets up both the `pre-commit` and `commit-msg`
hooks, as configured in `.pre-commit-config.yaml`.

## Codebase structure

The package lives under [protuna/](https://github.com/kumarshreshtha/protuna/tree/main/protuna). The main modules and submodules are:

- [optim](https://github.com/kumarshreshtha/protuna/tree/main/protuna/optim): The primary module hosting all optimization framework interfaces.
  - [spaces](https://github.com/kumarshreshtha/protuna/tree/main/protuna/optim/spaces): Search and design spaces, mutation rules and subspace sampling.
  - [samplers](https://github.com/kumarshreshtha/protuna/tree/main/protuna/optim/samplers): Candidate sequence generation strategies.
  - [constraints](https://github.com/kumarshreshtha/protuna/tree/main/protuna/optim/constraints): Structural typing contracts for constraints and common adapters.
  - [predictors](https://github.com/kumarshreshtha/protuna/tree/main/protuna/optim/predictors): Structural typing contracts for predictors and common adapters .
  - [acquisitions](https://github.com/kumarshreshtha/protuna/tree/main/protuna/optim/acquisitions): Base acquisition class and batch BO algorithms adapted to discrete input spaces.
  - [pipeline.py](https://github.com/kumarshreshtha/protuna/blob/main/protuna/optim/pipeline.py): Orchestrator for a single campaign round.
  - [storage.py](https://github.com/kumarshreshtha/protuna/blob/main/protuna/optim/storage.py): Storage of generated candidates and their evaluations.
- [data](https://github.com/kumarshreshtha/protuna/tree/main/protuna/data): Protein data handling and common benchmark datasets.
  - [datasets](https://github.com/kumarshreshtha/protuna/tree/main/protuna/data/datasets): Dataset interfaces, catalog and benchmark loaders.
  - [msa](https://github.com/kumarshreshtha/protuna/tree/main/protuna/data/msa): Multiple sequence alignment handling, I/O and sampling.
  - [tokenizers.py](https://github.com/kumarshreshtha/protuna/blob/main/protuna/data/tokenizers.py): Protein sequence encoding and decoding.
- [inference](https://github.com/kumarshreshtha/protuna/tree/main/protuna/inference): Local and Ray-based distributed inference.
  - [interfaces](https://github.com/kumarshreshtha/protuna/tree/main/protuna/inference/interfaces): Model adapter interfaces.
  - [models](https://github.com/kumarshreshtha/protuna/tree/main/protuna/inference/models): ESM and DPLM adapters as examples of huggingface hub adapters.
  - [serve](https://github.com/kumarshreshtha/protuna/tree/main/protuna/inference/serve): Ray actors, worker pools and request routing.
  - [handlers.py](https://github.com/kumarshreshtha/protuna/blob/main/protuna/inference/handlers.py): Contract for request handlers.
- [score](https://github.com/kumarshreshtha/protuna/tree/main/protuna/score): Functional protein scoring APIs.
- [utils](https://github.com/kumarshreshtha/protuna/tree/main/protuna/utils): General purpose shared utilities not specific to protein engineering.
  - [data_structs.py](https://github.com/kumarshreshtha/protuna/blob/main/protuna/utils/data_structs.py): Reusable data structures.
  - [devices.py](https://github.com/kumarshreshtha/protuna/blob/main/protuna/utils/devices.py): Compute device utilities.
  - [download.py](https://github.com/kumarshreshtha/protuna/blob/main/protuna/utils/download.py): File download helpers.
  - [paths.py](https://github.com/kumarshreshtha/protuna/blob/main/protuna/utils/paths.py): Path and storage helpers.
- [constants.py](https://github.com/kumarshreshtha/protuna/blob/main/protuna/constants.py): Global constants.
- [typing.py](https://github.com/kumarshreshtha/protuna/blob/main/protuna/typing.py): Shared type annotations.

Tests live under [tests/](https://github.com/kumarshreshtha/protuna/tree/main/tests),
and documentation lives under [docs/](https://github.com/kumarshreshtha/protuna/tree/main/docs).

## Code style

The package lives under `protuna/`, tests under `tests/` and documentation under `docs/`.
Follow the existing module layout and keep changes focused on the agreed issue.

- We follow [google python style guidelines](https://google.github.io/styleguide/pyguide.html).
  If unfamilar, worth a quick glance before you start.
- Use Ruff for formatting and linting and ty for typechecking. Add the vscode plugins
  for ruff and ty, they'll guide you through the expected behavior.
- We prefer module level imports except for typing-related modules;
  You can read the full list of single-line import exclusions in `ruff.toml`.
- Annotate tensor shapes with [jaxtyping](https://docs.kidger.site/jaxtyping/),
  for example `Bool[torch.Tensor, "l v"]` for a 2d boolean tensor of shape `l x v`.
  Throughout the repo we use `b` for a batch of sequences, `l` for sequence length,
  `m` for sequence with only masked positions, `v` for vocabulary,
  `d` for hidden dimensions, `g` for a group of sequences and `n` for N sequences.
- Add or update tests for changed behavior. Unit and integration tests live in
  `tests/unit/` and `tests/integration/` and mirror the package layout.
  Test modules do not require a module-level docstring.

## Documentation conventions

Keep documentation in sync with changes to public behavior and APIs.

- Write Google-style docstrings compatible with reStructuredText and Sphinx.
  Start with a concise imperative summary, such as "Fetch rows from the dataset."
- Include simple, doctest-compatible usage examples in module docstrings and in
  docstrings for non-trivial public functions and classes.
- Use Sphinx cross-references such as `:func:` and `:class:` to link APIs. When a
  wrapper shares behavior with another API, cross-reference its documentation.
- Use equations when they clarify mathematical behavior. For methods based on
  papers, add the bibtex entry to `docs/references/references.bib` (or
  `docs/references/datasets.bib` for dataset sources) and cite them
  in the documentation.

Build the documentation from the repository root:

```sh
uv run make -C docs html
```

Open `docs/_build/html/index.html` to review the generated pages. For live preview:

```sh
uv run sphinx-autobuild docs docs/_build/html
```

## Validating changes

Before opening a pull request, run the type checker, formatter and linter, and fix any
errors:

```sh
uv run ty check
uv run ruff format protuna tests
uv run ruff check protuna tests
uv run pytest
uv run pytest tests/integration -m integration
uv run pre-commit run --all-files
```

The default pytest run includes unit tests and module doctests, and excludes
integration tests. If your change affects integrations, run the relevant tests
explicitly; these may require external datasets, model checkpoints or Ray workers:

Build and review the documentation when changing docstrings or documentation pages.

## Commit messages

Commit messages are linted by [commitlint](https://commitlint.js.org) against the
[Conventional Commits](https://www.conventionalcommits.org) specification, and a message that
does not conform will be rejected by the `commit-msg` hook. The expected shape is:

```
<type>(<optional scope>): <subject>
```

for example:

```sh
git commit -m "feat(optim): add simulated annealing sampler"
git commit -m "fix: correct search space mask indexing"
git commit -m "docs(score): document zero-shot scoring functions"
```

Common types are `feat`, `fix`, `docs`, `test`, `refactor`, `ci` and `chore`.
Keep the subject in lower case, in the imperative mood and without a trailing period.

## Open a pull request

After implementing the agreed change and completing the relevant checks, open a
pull request against the upstream repository. Link the issue (for example,
`Closes #123`), describe the resulting behavior and validation.

## Merge strategy

Pull requests are "merged" using **Squash and merge**, merge commits are not allowed.
Rebase your branch onto the latest upstream `main` when it needs updating;
do not merge `main` into your branch. Ensure the resulting commit messages follow
the [commit message conventions](#commit-messages).
