# Acquisitions and recyclers

## Source reference

First read `protuna.optim.acquisitions.base` for the `Acquisition` abstract base class,
from which every acquisition must inherit.

Read `protuna.optim.acquisitions` to see the list of existing acquisition
functions in Protuna and to select the appropriate submodules for inspection.

Read `protuna.optim.candidates` for the `CandidateStream` protocol and the
`CandidateBatch` it yields, and `protuna.optim.storage` for the concrete
`CandidateStreamFromStorage` class that the pipeline uses.

Inspect the most similar existing acquisition before writing a new one:

- See [ucb_acquisition.py](../examples/ucb_acquisition.py) for a UCB implementation that maintains a top-k heap of the highest UCB values from streaming predictor scores.
- `ThompsonSampling`: A selector that uses a single pass over predictor posterior samples for all candidates to make its selection.
- `DPP` and `KMBBO`: Selectors that first shortlist candidates and then select a diverse subset from this shortlist.

See `protuna.optim.acquisitions.scores` for the `ScoreFunction` contract that performs
the shortlisting. Score functions themselves don't satisfy the acquisition contract.

## Contract

Inherit from `acquisitions.Acquisition` and implement:

```python
def select(self, stream: candidates.CandidateStream) -> list[int]: ...
```

The returned list contains the `seq_id` values of the selected candidates from the
input `CandidateStream`. The `CandidateStream` is guaranteed to contain unique
candidate sequences within a pipeline run. The pipeline preserves the returned
ID order in `PipelineOutput.selection`.

Set `requires_std` and/or `requires_samples` when stored objective uncertainty is
needed; both default to `False`. Set them on the class or in `__init__` before
constructing the pipeline. The pipeline combines the acquisition's and recycler's
flags at construction and requests those outputs from the objective predictor;
changing the flags afterward does not update its requests. Propagate requirements
from any scoring function or kernel used by the selector.

Call `super().__init__()` before assigning a handler attribute. An acquisition can
own an embedding model or other handler; inherited `load` and `release` propagate
to registered handlers and nested registries. The pipeline calls `load()` before
selection and `release()` afterward when `release_handlers_after_use=True`.
Direct calls to `select()` must arrange that lifecycle themselves. A raw model
is not automatically managed: wrap it in a handler or implement lifecycle methods
that also propagate to registered children. See [adapters](adapters.md) if dealing
with models inside the acquisition.

A recycler is just another `Acquisition` applied to the candidates generated
in the current cycle to pick candidates that become parents in the next cycle.
The final acquisition sees candidates from **all generated cycles**.
Neither pool includes the starting sequences (cycle zero). Selection may therefore
run repeatedly on a recycler instance: reset temporary heaps and shortlists per call,
and define any deliberate persistent state.

## Stream semantics

The pipeline's storage stream is reiterable: each `iter(stream)` starts a fresh
pass. Do not rely on row order or batch boundaries being stable across passes;
associate retained data by `seq_id`.

Storage-backed batches contain CPU tensors, regardless of the objective model's
device. Move tensors explicitly when a selector's model or computation needs a
different device.

The `stream.candidates_from_ids(ids)` lookup returns candidates in the requested
ID order. It does not reapply the stream's
cycle or hard-constraint filters. Unknown IDs are omitted, and duplicate IDs do
not yield duplicate rows. Use it to retrieve candidates already evaluated via the
iterator, for instance, when shortlisting candidates before making a final selection.

### Missing values

An objective field is `None` when all its values are missing in a storage batch;
partially missing values appear as `NaN`. Soft constraints with no results in a
batch are omitted from the mapping, which is `None` when none have results.
Validate required fields and constraint names, and define how to handle nonfinite
values.

In a normal pipeline execution, all candidates that pass a hard constraint
are evaluated by the soft constraints and the objective. Missing values,
if any, can therefore be attributed to a silent failure or bug in the objective or constraint.

## Dealing with soft constraints

Hard-constraint failures are already excluded by pipeline stream **iteration**.
Soft feasibility does not automatically filter candidates. Decide how you want
to deal with feasibility probabilities.

The `acquisitions.feasibility(batch)` helper function multiplies the available
per-constraint probabilities and returns ones when the mapping is absent or empty.
Interpreting that product as joint feasibility assumes independence.

## Reuse an existing selector when appropriate

If only utility scoring or shortlisting changes, consider implementing the public
`acquisitions.ScoreFunction` protocol and passing it as `quality` to
`acquisitions.DPP` or `score_function` to `acquisitions.KMBBO`.

## Joint posterior samples

For methods using joint samples, set `requires_samples=True` and consume
`objective_samples`. Column `j` represents the same posterior function draw
or ensemble member across candidates.

## Model-owned objectives

An acquisition may own a trained model and compute predictions from `token_ids`,
for example when it needs conditioning or fantasy-model operations unavailable
from stored predictions. Manage that model through the lifecycle above; keep
training outside selection unless the algorithm explicitly requires an update.

Pass `objective=None` to skip the pipeline's objective stage when separate stored
predictions are unnecessary. All stored objective fields then remain `None`;
both acquisition and recycler must work under that condition. Keeping
a pipeline objective is also valid when, for example, the recycler needs it.
