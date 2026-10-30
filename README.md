<p align="center">
    <img src="https://raw.githubusercontent.com/kumarshreshtha/protuna/main/docs/_static/protuna-wordmark-readme-dark.svg#gh-dark-mode-only" alt="Protuna" width="560">
    <img src="https://raw.githubusercontent.com/kumarshreshtha/protuna/main/docs/_static/protuna-wordmark-readme-light.svg#gh-light-mode-only" alt="Protuna" width="560">
</p>
<p align="center">
    <a href="https://github.com/kumarshreshtha/protuna/actions/workflows/ci.yml"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/kumarshreshtha/protuna/ci.yml?branch=main&amp;label=Tests"></a>
    <img alt="Version" src="https://img.shields.io/github/v/release/kumarshreshtha/protuna">
    <img alt="Language" src="https://img.shields.io/badge/python-3.12-yellow?logo=python&amp;logoColor=white">
    <a href="https://protuna.readthedocs.io/en/latest/"><img alt="Docs" src="https://img.shields.io/readthedocs/protuna/latest?label=Docs"></a>
    <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

Protuna is an optimization framework for iterative protein engineering campaigns. Starting from historical mutagenesis data or a single hit[^1], it proposes the batch of variants to test next.

## Features

- **Guided generation:** Steer protein language models towards your design objectives with reward-guided generation – Feynman–Kac steering, classifier guidance, and beam search.
- **Scalable screening:** Ray-based distributed streaming APIs for large-scale screening on local accelerators or clusters.
- **Modular campaigns:** Combine interchangeable samplers, predictors, constraints, and acquisition strategies.
- **Multiple design goals:** Optimize for multiple properties with a primary objective and inequality constraints on the others.

## Getting Started

### Installation

<!-- installation-start -->
Protuna requires Python 3.12 or higher.

```sh
pip install protuna
```
<!-- installation-end -->

### Building with AI Agents

The `protuna-skills` plugin gives AI coding agents the guidance and examples needed
to work with Protuna. Use it to build, and run optimization pipelines,
configure distributed inference, or extend the framework with custom components.

See the [plugin guide](https://github.com/kumarshreshtha/protuna/blob/main/protuna-skills/README.md) for installation instructions for
coding and science agents.

### Quickstart

<!-- quickstart-code-start -->
The following code builds and runs a simple `Pipeline`.

It starts by defining which mutations are allowed and how to sample narrower search
spaces. The pipeline then samples random protein sequences in those subspaces,
uses a multiple sequence alignment (MSA) to score them based on amino acid
frequencies at the mutated sites, and selects up to 20 of the highest-scoring
sequences.

```py
from protuna import data
from protuna import optim
from protuna.optim import acquisitions
from protuna.optim import predictors
from protuna.optim import rules
from protuna.optim import samplers
from protuna.optim import spaces

dataset = data.get_dataset("gb1", source=data.DatasetSource.FLIP)
tokens = data.tokenizer.encode(dataset.target_sequence)
msa = data.TokenizedMSA.from_file(
    dataset.get_msa_path(data.MSAProtocol.ColabFold), format="a3m"
)
design_space = spaces.DesignSpace.from_rules(
    rules=[
        # Positions are illustrative and use zero-based indexing.
        rules.AllowOnly(positions=list(range(100, 125))),
        rules.DenyResidues(amino_acids=["C"]),
        rules.FixSites(positions=[105, 120]),
        # Protect sites where the query residue is >= 95% conserved.
        rules.MSAConserved(tokenized_msa=msa, threshold=0.95),
    ],
    init_sequences=[tokens],
    # Explore up to 4 subspaces, each with at most 3 mutable sites.
    space_sampler=spaces.RandomMaskSampler(max_mutations=3, n_subspaces=4),
)
pipeline = optim.Pipeline(
    design_space=design_space,
    sampler=samplers.RandomSampler(max_samples=100),  # Propose up to 4x100 variants
    objective=predictors.PSSMZeroshotPredictor(tokens, msa, laplace_smoothing=True),
    acquisition=acquisitions.Greedy(budget=20),
)
output = pipeline.run()
# Up to 20 sequences
candidates = data.tokenizer.decode(list(output.selection.token_ids))
```
<!-- quickstart-code-end -->

See the [pipeline documentation](https://protuna.readthedocs.io/en/latest/concepts/pipeline.html) for configuration options or read the [concepts overview](https://protuna.readthedocs.io/en/latest/concepts/overview.html) to understand Protuna's workflow and core primitives.

### Storage

<!-- storage-start -->
Protuna stores cached downloads and default pipeline artifacts under `$PROTUNA_HOME/protuna`. If `PROTUNA_HOME` is unset, it uses `$XDG_CACHE_HOME/protuna`, falling back to `$HOME/.cache/protuna`.

You can configure pipeline runs to store artifacts in another folder or directly in cloud storage (AWS, Azure, or GCP) to support large exploration campaigns.
<!-- storage-end -->

## Concepts

<p align="center">
    <img src="https://raw.githubusercontent.com/kumarshreshtha/protuna/main/docs/_static/protuna_pipeline_dark.png#gh-dark-mode-only" alt="Protuna optimization pipeline" width="600">
    <img src="https://raw.githubusercontent.com/kumarshreshtha/protuna/main/docs/_static/protuna_pipeline_light.png#gh-light-mode-only" alt="Protuna optimization pipeline" width="600">
</p>

Protuna separates candidate generation from experimental batch selection. A `Pipeline` combines four steps:

1. **Defining the search region.** A `DesignSpace` holds starting sequences and their allowed substitutions and can sample smaller search spaces for independent exploration.
2. **Generating proposals.** A `Sampler` proposes sequences within each search space.
3. **Evaluating proposals.** A `Predictor` estimates the objective property. Hard constraints reject candidates; soft constraints provide feasibility probabilities.
4. **Selecting candidates.** An `Acquisition` selects candidates within your budget, using criteria such as predicted fitness, uncertainty, or diversity.

Selection is limited to the candidates proposed by the sampler. Protein language models provide a useful prior over plausible sequences, but their likelihoods need not align with a campaign’s objective. Protuna supports reward-guided generation to steer exploration towards desired properties, using inference-time steering, externally aligned generators, or both.

Optional recycling uses promising candidates as parents for the next generation cycle, allowing subsequent cycles to build on promising mutations.

Read the [optimization overview](https://protuna.readthedocs.io/en/latest/concepts/overview.html) for the broader campaign workflow. To extend Protuna, see the guides for custom [model adapters](https://protuna.readthedocs.io/en/latest/guides/writing_adapters.html), [samplers](https://protuna.readthedocs.io/en/latest/guides/writing_samplers.html), and [acquisitions](https://protuna.readthedocs.io/en/latest/guides/writing_acquisitions.html).

## Modules

<!-- modules-start -->
| Module | Description |
| --- | --- |
| [protuna.optim](https://protuna.readthedocs.io/en/latest/api/optim/index.html) | The primary optimization framework. It contains all the primitives needed to set up and run your protein optimization task. |
| [protuna.inference](https://protuna.readthedocs.io/en/latest/api/inference/index.html) | Distributed inference interface and a few illustrative model adapters. |
| [protuna.data](https://protuna.readthedocs.io/en/latest/api/data/index.html) | Common preprocessed public benchmark datasets, MSA handling, and tokenization modules. |
| [protuna.score](https://protuna.readthedocs.io/en/latest/api/score/index.html) | Functional library of protein scoring methods such as zero-shot variant effect prediction (VEP) and contact prediction using a categorical Jacobian. |
| protuna.train (coming soon) | Future releases will include trainers and recipes for few-shot property prediction models, fine-tuning, and preference alignment of generators. |
<!-- modules-end -->

## Upcoming Releases

See the [roadmap for upcoming releases](https://protuna.readthedocs.io/en/latest/upcoming.html).

## Contributing

See [CONTRIBUTING.md](https://github.com/kumarshreshtha/protuna/blob/main/CONTRIBUTING.md).

## Citation

If you use Protuna in your research, please cite this software:

```
@software{protuna2026,
    title = {{Protuna: An optimization framework for AI-guided protein engineering}},
    author = {Shreshtha, Kumar},
    url = {https://github.com/kumarshreshtha/protuna},
    year = {2026},
    version = {0.1.0}
}
```

[^1]: Successfully generating candidates from a single data point depends on how well your goals, the zero-shot proxies you use for scoring, and the sampler's likelihoods align.
