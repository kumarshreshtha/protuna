# Pipeline

A {py:class}`~protuna.optim.pipeline.Pipeline` brings together candidate generation, evaluation, and selection.
It searches a design space, optionally recycles promising candidates, and returns sequences for the next lab batch.

One call to {py:meth}`run() <protuna.optim.pipeline.Pipeline.run>` performs a computational search and returns selected
protein variants and their run history. A run can contain several
generation cycles, each building on promising candidates discovered in the
previous cycle.

Between experimental rounds, lab measurements can be used to update samplers
and predictors, revise constraints and acquisition parameters, and choose new
starting sequences for the next pipeline. The [optimization overview](overview.md)
page describes this broader campaign loop; this page follows the objects and data
inside a pipeline.

```{note}
Future releases are planned to introduce a higher-level `Campaign` object to
coordinate {py:class}`~protuna.optim.pipeline.Pipeline`s across experimental rounds. It will combine `Trainer`s
for updating samplers and predictors with schedulers that adjust pipeline
parameters to balance exploration and exploitation within a budget of
experimental rounds.
```

## Building a pipeline

The simplest pipelines can be built with just a `design_space` and
a `sampler`. For example, the pipeline below
enumerates all single substitutions of the starting sequence `"MKTA"`:

```python
from protuna import data
from protuna import optim
from protuna.optim import samplers
from protuna.optim import spaces

design_space = spaces.DesignSpace([spaces.SearchSpace.from_rules("MKTA", [])])
pipeline = optim.Pipeline(
    design_space=design_space,
    sampler=samplers.SiteSaturationSampler(batch_size=1),
)
output = pipeline.run()
sequences = data.tokenizer.decode(list(output.selection.token_ids))

assert len(sequences) == 4 * 19
```

The result is a {py:class}`~protuna.optim.pipeline.PipelineOutput` with
`selection` containing the selected candidate batch and a `store` containing
the run history. With no objective or constraints, there is no evaluation stage;
with no acquisition supplied, the default
{py:class}`~protuna.optim.acquisitions.base.IdentityAcquisition` selects every
generated candidate.

The {py:class}`~protuna.optim.pipeline.Pipeline` constructor accepts these components:


| Argument | Accepted object | Default behavior |
| --- | --- | --- |
| `design_space` | A {py:class}`~protuna.optim.spaces.design.DesignSpace` containing starting {py:class}`~protuna.optim.spaces.search.SearchSpace`s and an optional subspace sampler. | **Required**. Defines where generation can search. |
| `sampler` | A {py:type}`~protuna.optim.samplers.base.SamplerLike` handler that turns a search space into a batch of candidate token sequences. | **Required**. Proposes candidates for each supplied space. |
| `objective` | A {py:type}`~protuna.optim.predictors.base.PredictorLike` handler that takes a {py:class}`~protuna.optim.predictors.base.PredictionRequest` and returns a {py:class}`~protuna.optim.predictors.base.Prediction`. | `None` skips objective evaluation. |
| `hard_constraints` | A mapping of names to {py:type}`~protuna.optim.constraints.base.HardConstraintLike` handlers, each returning a scalar boolean tensor for a candidate. | `None` applies no hard filters. |
| `soft_constraints` | A mapping of names to {py:type}`~protuna.optim.constraints.base.SoftConstraintLike` handlers, each returning a scalar feasibility probability. | `None` supplies no soft feasibility scores. |
| `acquisition` | An {py:class}`~protuna.optim.acquisitions.base.Acquisition` whose {py:meth}`select() <protuna.optim.acquisitions.base.Acquisition.select>` method chooses final candidate IDs. | {py:class}`~protuna.optim.acquisitions.base.IdentityAcquisition` selects all eligible candidates. |
| `recycler` | An {py:class}`~protuna.optim.acquisitions.base.Acquisition` used to select parents for another generation cycle. | `None`; required when `num_cycles > 1`. |

Four additional arguments control execution: `num_cycles=1` sets the number of
generation and evaluation cycles; `evaluated_mutants=None` optionally supplies
tokenized sequences to exclude from generated proposals;
`lock_mutations_on_recycle=False` controls whether previously mutated sites are
fixed when recycling; and `release_handlers_after_use=False` controls whether
components release their resources after each stage.

The `*Like` types describe [handler contracts](handlers.md). You can pass local
implementations or served pools with the same request and response types.
Acquisitions have a different interface: they consume a {py:class}`~protuna.optim.candidates.CandidateStream` and
return sequence IDs through {py:meth}`select() <protuna.optim.acquisitions.base.Acquisition.select>`.

The [spaces page](spaces.md) explains how to build the design space. Its
`space_sampler` chooses which subspaces to explore; the pipeline's `sampler`
generates sequences within them. Configure the number of proposals in the
sampler and the budget for the final batch of candidates in the acquisition.

Optional components must still be compatible with one another. For example,
{py:class}`Greedy(budget=96) <protuna.optim.acquisitions.greedy.Greedy>` requires objective means and favors larger values. Omitting
`objective` is appropriate when the acquisition needs no predictions or computes
its own scores; see [Owning the objective](../guides/writing_acquisitions.md#owning-the-objective)
for an acquisition that manages its own model. A sampler's internal guidance
does not automatically become the pipeline's stored objective predictions.

## Initialization

Constructing a {py:class}`~protuna.optim.pipeline.Pipeline` records whether the
acquisition or recycler requires objective standard deviations or posterior samples.
Those requirements determine the outputs requested from the objective during evaluation.
Models are loaded when their stages run, not during pipeline construction.

## `Pipeline.run()`

Each call to {py:meth}`run() <protuna.optim.pipeline.Pipeline.run>` registers the
design space's starting sequences. The starting sequences must be unique, all have
the same length, and be variants of the same wild-type sequence.

```{note}
The next release is planned to lift the shared sequence-length restriction,
allowing initial search spaces around proteins of different lengths, such as
homologs of a target protein.
```
By default, `num_cycles=1`: generation and evaluation run once, followed by final
selection. With multiple cycles, recycling supplies new parents between cycles.

```{mermaid}
flowchart TD
    initial["DesignSpace"]
    subgraph cycle["One generation cycle"]
        spaces["Sample search spaces"]
        sampler["Sampler"]
        hard["Hard constraints"]
        soft["Soft constraints"]
        objective["Objective"]
        spaces --> sampler --> hard --> soft --> objective
    end
    initial --> spaces
    sampler -. "candidates" .-> store[("CampaignStore")]
    hard -. "pass/fail" .-> store
    soft -. "probabilities" .-> store
    objective --> more{"Another cycle?"}
    objective -. "predictions" .-> store
    more -- "No" --> acquisition["acquisition.select()"]
    more -- "Yes" --> recycler["recycler.select()"]
    recycler --> spaces
    store -. "CandidateStream" .-> acquisition
    acquisition --> result["selected batch"]
    store -. "CandidateStream" .-> recycler

```

Solid arrows show stage order and the recycling loop; dotted arrows show
records written to the store and candidate streams read by selectors. Both
selectors receive only candidates that pass the hard constraints.

### Generating candidates

The pipeline gets an iterator over the narrowed search spaces from {py:meth}`design_space.sample_search_spaces() <protuna.optim.spaces.design.DesignSpace.sample_search_spaces>` and passes it to
{py:meth}`sampler.stream() <protuna.inference.handlers.Handler.stream>`. Without a subspace sampler, the starting spaces themselves are used as inputs to the sampler.

Each response from the {py:class}`~protuna.optim.samplers.base.Sampler` is a batch of tokenized candidate sequences. The pipeline
moves these to the CPU, removes any matches in `evaluated_mutants`, and records the
remaining proposals with their sequence, parent, ancestor, and cycle IDs.

The pipeline deduplicates all identical proposals across different spaces and cycles
within a run, and excludes all starting sequences. Only the first occurrence of a
proposal is stored and evaluated, retaining that occurrence's lineage.
Samplers must respect the supplied search space. Responses may include duplicate
candidates and the starting sequence; the pipeline removes these before storage
and evaluation. Invalid proposals raise a `ValueError`.

### Filtering and feasibility

Constraints run one at a time: first the hard constraints in mapping iteration
order, then the soft constraints in their mapping order.

A failed hard constraint excludes a candidate from every later constraint,
objective evaluation, recycling, and final selection. Placing inexpensive hard
filters first can therefore reduce the work passed to expensive models.

Soft constraints attach feasibility probabilities to the survivors. They do
not remove candidates by themselves: the acquisition and recycler decide how
to use those probabilities. For example, {py:class}`~protuna.optim.acquisitions.greedy.Greedy` multiplies the soft
probabilities and keeps candidates whose combined feasibility is strictly
greater than its `feasibility_threshold`.

### Evaluating the objective

If `objective` is supplied, the pipeline loads it and creates a
{py:class}`~protuna.optim.predictors.base.PredictionRequest` for each surviving candidate in the current cycle. Each
request contains the candidate's tokens and two flags:

- `return_std` is true if either the acquisition or recycler declares
  `requires_std=True`.
- `return_samples` is true if either declares `requires_samples=True`.

The returned {py:class}`~protuna.optim.predictors.base.Prediction` contains a scalar mean and, when requested and
supported, a standard deviation or samples of the objective. The store records
these alongside the candidate ID. Choose a predictor that can provide the
outputs required by your selectors (acquisitions and recyclers); the pipeline
requests those outputs but does not enforce that the predictor supplies them.

With `objective=None`, this stage is skipped entirely. Constraints still run,
and selectors can still use the candidate sequences, lineage, and feasibility
results.

### Recycling

When `num_cycles > 1`, a `recycler` is required. After each cycle except the last,
the pipeline calls {py:meth}`recycler.select() <protuna.optim.acquisitions.base.Acquisition.select>` on the eligible candidates from **that
cycle only**. Its selected sequences become starting points for the next cycle.
The recycler's budget controls how many parents serve as starting sequences
for the next cycle.

For each selected parent, the pipeline replaces its starting sequence with the
selected candidate and keeps the original allowed mutation mask. The next cycle
can therefore explore a different subspace around the new parent.
Parent IDs change at each generation; ancestor IDs keep the connection to the
original starting sequences.

By default, `lock_mutations_on_recycle=False`, so a later cycle can change or
revert an earlier substitution where the original space permits it. With
`lock_mutations_on_recycle=True`, sites where a recycled parent differs from
its original ancestor are fixed to the parent's current residues. Other sites
remain available under the ancestor's mask.

The pipeline uses the same configured sampler, constraints, and objective throughout;
it does not retrain these components or collect new measurements between cycles.

### Final selection

After the final cycle, the acquisition receives eligible candidates from
all generated cycles.

The candidate pool is exposed as a
{py:class}`~protuna.optim.candidates.CandidateStream`, yielding
{py:class}`~protuna.optim.candidates.CandidateBatch` objects. Each batch contains
tokens and lineage IDs, together with any objective means, standard deviations,
posterior samples, and named feasibility scores. This allows a selector to
inspect a large stored pool in batches.

See [Working with candidate streams](../guides/writing_acquisitions.md#working-with-candidate-streams)
for batch fields, tensor shapes, repeated iteration, and retrieval by candidate ID
when writing your own acquisition.

The acquisition returns candidate IDs. The pipeline retrieves these candidates in
the acquisition's returned ID order and includes the batch as `output.selection`
in the `PipelineOutput` dataclass. The run history is available as `output.store`. The number of sequences returned
by the acquisition depends on its budget configuration and available candidates;
a budget is not a guarantee that enough eligible proposals exist.

## Storage and resource lifetime

The default {py:class}`~protuna.optim.storage.CampaignStore` records starting
sequences, generated candidates and their lineage, hard-constraint verdicts,
soft feasibility scores, and objective predictions. Records are buffered and
written to Parquet as buffers fill; reads include both buffered and written
records. {py:meth}`run() <protuna.optim.pipeline.Pipeline.run>` calls {py:meth}`finalize() <protuna.optim.storage.CampaignStore.finalize>` before returning to flush the remaining
data.

Pass your own store to `pipeline.run(store=store)` to choose a local directory or cloud URI, buffer size, and
read chunk size. A custom
store must use the campaign's protein length and register the keys of
`soft_constraints` in `soft_constraint_names`: the store uses those names to
distinguish probabilities from hard verdicts. The pipeline handles this
registration when it creates the default store.

By default, `release_handlers_after_use=False` leaves loaded resources available
for reuse. Setting it to `True` calls {py:meth}`release() <protuna.inference.handlers.Handler.release>` after each stage uses a
component, including constraints, the objective, the recycler, and the final
acquisition. Components are loaded again when needed in a later cycle. For
served handlers, releasing resources does not shut down the worker pool; see
the [handler lifecycle](handlers.md#the-handler-protocol).

See the [Pipeline API reference](../api/optim/pipeline.rst) for the full
constructor signature and the [Storage API reference](../api/optim/storage.rst) for
retrieving stored results.
