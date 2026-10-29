# Optimization Overview

Protuna combines candidate generation, property prediction, and experimental
batch selection in a modular pipeline. Starting from historical mutagenesis
data or a single hit, you can use it to explore allowed mutations and propose
candidates for successive rounds of experiments.

(optimization-pipeline-figure)=
````{container}
```{image} ../_static/protuna_pipeline_light.png
:alt: Protuna optimization pipeline
:class: only-light
:width: 70%
:align: center
```

```{image} ../_static/protuna_pipeline_dark.png
:alt: Protuna optimization pipeline
:class: only-dark
:width: 70%
:align: center
```
````

## Optimizing experimentally measured properties

Protein engineering requires searching a sequence space far larger than an
experimental budget can cover. Each experimental round must
therefore balance testing variants expected to improve performance with
exploring variants that could inform subsequent rounds.

Most campaigns have several design goals. An enzyme may need higher activity,
greater stability, and improved expression, with gains in one property coming
at the expense of another. This leads to a search for Pareto-optimal variants:
sequences for which improving one property requires giving up performance on
another.

The current pipeline represents these goals through a scalar objective and
constraints on other properties. For example, one run could favor higher
activity while requiring stability and expression above chosen thresholds.

Historical assay data can be used to train predictors for the properties of
interest. These predictors estimate how untested variants will perform and
provide uncertainty estimates to guide selection. Measurements
from each new experimental round can then improve the predictions used in the
next round.

For certain proteins and properties, zero-shot methods may provide a starting
point when no mutagenesis data is available. They score variants without
training on the campaign's assay data. For example, evolutionary scores
from a multiple sequence alignment or a pretrained protein language model can
serve as proxies for experimental fitness. Using these scores assumes that
evolutionary preferences are informative about the desired property; this
relationship depends on the protein and the assay.

## Generation and selection

Protuna separates **candidate generation** from **experimental batch selection**.
A sampler proposes sequences within an explicit mutation space, predictors and
constraints evaluate them, and an acquisition strategy selects a batch for
experimental testing. Depending on the strategy, selection can favor high
predicted fitness, explore uncertain regions, or encourage diversity.

An acquisition can only select from the candidates proposed by the sampler.
Protein language models provide a useful prior over plausible sequences, but
high model likelihood does not necessarily imply better performance on the
properties being optimized. Guiding generation towards those properties can
make promising candidates more common in the pool available for selection.

Protuna supports several ways to guide generation:

- **Inference-time steering.** Samplers such as `FKSteeringSampler` and
  `NestedSMCSampler`, or `ReMDMSampler` with a guidance model, use property
  scores to guide proposals from a pretrained generator.
- **Generator alignment.** A generator can be fine-tuned or preference-aligned
  externally to favor desirable sequences, then used with a compatible sampler.
- **Combining both.** Inference-time steering can further guide an aligned
  generator towards the goals of a particular round.

The [sampler reference](../api/optim/samplers/index.rst) describes the available
generation strategies. The [custom samplers guide](../guides/writing_samplers.md)
explains how to implement your own.

## Terminology

At a high level, the main components that make up a typical workflow are:

| Component | Role |
| --- | --- |
| [Design space](spaces.md) | The starting sequences and their allowed mutations, defined through mutation rules or explicit masks. |
| [Search space](spaces.md#sampling-subspaces) | A starting sequence and the allowed residues at each position. Smaller subspaces can be sampled from the design space for independent exploration. |
| Generator | A model that a sampler can use to propose residues, such as a protein language model. |
| [Sampler](../api/optim/samplers/index.rst) | Proposes candidate sequences within a supplied search space. |
| [Property predictor](../api/optim/predictors.rst) | Estimates a property of a candidate, optionally with uncertainty. It can supply objective scores, evaluate constraints, or guide generation. |
| [Constraint](../api/optim/constraints.rst) | Evaluates a requirement on a candidate, returning a pass/fail decision or a feasibility probability. |
| Objective | The score used to assess progress towards the design goal, supplied by a predictor. |
| [Acquisition](../api/optim/acquisitions.rst) | Selects candidates for experimental testing according to the budget and selection criteria. |

## Pipeline execution

The [figure](#optimization-pipeline-figure) above shows how these components fit
together in a {py:class}`~protuna.optim.pipeline.Pipeline`:

1. **Define the design space.** Start with `N` sequences and their allowed
   mutations. Optionally sample `M` smaller search spaces around each starting
   sequence, giving `N × M` independent spaces to explore.
2. **Generate and evaluate candidates.** A sampler proposing `K` sequences per
   space produces a pool of `N × M × K` candidates. Hard constraints filter out
   unacceptable designs; soft constraints estimate feasibility for the
   survivors, which are then scored on the objective.
3. **Optionally recycle promising variants.** A recycler selects candidates as
   starting sequences for another generation cycle, allowing further mutations
   to build on promising combinations. These cycles use the same models and
   receive no new experimental measurements.
4. **Select the experimental batch.** The acquisition selects from eligible
   candidates across all generation cycles. `Pipeline.run()` returns these
   candidates alongside the run history.

Independent search spaces and model evaluations allow generation and screening
to scale across local accelerators or Ray workers. The
[serving patterns guide](../guides/serving_patterns.md) shows how to distribute
these components. See [Pipeline](pipeline.md) for configuration and execution
details, including optional stages and recycling.

Experimental measurements close the campaign loop. Between rounds, you can
update property predictors, align generators, choose new starting sequences,
and revise the mutation rules or selection criteria. These updates currently
happen between pipeline runs. A planned higher-level `Campaign` object will
coordinate pipelines and model training across experimental rounds.
