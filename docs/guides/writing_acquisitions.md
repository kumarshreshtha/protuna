# Custom Acquisitions

An acquisition selects the candidates to test in the lab from the sequences
proposed by a sampler. It can favor high predicted fitness, explore uncertain
regions, or choose a diverse experimental batch.

This guide explains the acquisition contract and implements an upper confidence
bound selector that balances predicted fitness and uncertainty while keeping
only a small selection in memory.

## Acquisition contract

All acquisitions subclass
{py:class}`~protuna.optim.acquisitions.base.Acquisition` and implement its
{py:meth}`~protuna.optim.acquisitions.base.Acquisition.select` method:

```py
def select(self, stream: candidates.CandidateStream) -> list[int]: ...
```

The input provides an iterable over the generated candidate pool. The output is
a list of the selected candidates' `seq_id` values. The pipeline retrieves these
selected sequences in the acquisition's output ordering.

Two flags declare which optional objective predictions the acquisition needs:

| Flag | Requested prediction | Candidate batch field |
| --- | --- | --- |
| `requires_std` | Predicted standard deviation | `objective_std` |
| `requires_samples` | Samples of the objective | `objective_samples` |

Both default to `False`. Set them on the class when the requirements are fixed,
or in `__init__` when they depend on the configuration. For example,
{py:class}`~protuna.optim.acquisitions.dpp.DPP` combines the requirements of its
quality function and diversity kernel, while
{py:class}`~protuna.optim.acquisitions.kmeans.KMBBO` takes them from its score
function.

{py:class}`~protuna.optim.pipeline.Pipeline` reads these flags at construction
and forwards them as `return_std` and `return_samples` in each
{py:class}`~protuna.optim.predictors.base.PredictionRequest` to the objective.
The objective predictor must therefore be compatible with the requested outputs.

The same acquisition interface can be used as a pipeline's `recycler`. Recycling
selects parents from the current generation cycle for further mutations, while the
final acquisition selects from eligible candidates across the campaign's generation cycles.
You can read more about this in the [optimization overview](../concepts/overview.md).

## Working with candidate streams

{py:class}`~protuna.optim.candidates.CandidateStream` is a reiterable source of
{py:class}`~protuna.optim.candidates.CandidateBatch` objects. Each call to
`iter(stream)` creates a fresh iterator over the pool, so an acquisition can
make multiple passes, for example, to estimate a normalization factor before
scoring.

All tensors in a {py:class}`~protuna.optim.candidates.CandidateBatch` share a
leading dimension `b`, with matching rows across fields:

| Field | Contents |
| --- | --- |
| `seq_id` | The identifiers to return from `select()`. |
| `token_ids` | Encoded protein sequences of shape `[b, l]`. |
| `parent_id`, `ancestor_id`, `cycle_id` | Immediate parents, initial ancestors, and generation cycles. |
| `objective_mean`, `objective_std` | Per-candidate predicted means and standard deviations, when available. |
| `objective_samples` | A tensor of shape `[b, n]`, containing `n` objective samples per candidate, when available. |
| `feasibility` | Soft constraint names mapped to per-candidate feasibility probabilities, when available. |

Optional evaluation fields are `None` when unavailable. The pipeline's storage
stream excludes candidates that failed hard constraints. Soft constraints remain
probabilities for the acquisition to use.

Acquisitions receive streaming candidates because a sampler may propose millions
of sequences. Avoid turning the entire stream into a list or concatenating every
batch unless the proposals are known to fit in memory.

For methods that compare candidates jointly, a common pattern is to use retrieval
and reranking: first retain a manageable shortlist, then fetch its full records with
{py:meth}`~protuna.optim.candidates.CandidateStream.candidates_from_ids`.
This returns a single batch in the requested ID order, which keeps candidate
records aligned with any scores retained during retrieval. DPP and KMBBO illustrate
this separation between shortlisting and final selection.

## Example: Upper confidence bound selection

Suppose a predictor estimates each candidate's fitness and uncertainty, and we
want to fill one 96-well plate. Selecting only candidates with the highest
predicted means can overlook promising candidates whose predictions are uncertain. An upper
confidence bound (UCB) adds an uncertainty bonus:

```{math}
\operatorname{UCB}(x) = \mu(x) + \sqrt{\beta}\,\sigma(x).
```

Here, larger objective values are better. `beta=0` ranks by predicted mean;
increasing `beta` gives uncertain candidates more weight. With `beta=4`, the
bonus is twice the predicted standard deviation.

Our selector keeps the candidates with the highest UCB scores among those whose
combined feasibility exceeds a configurable threshold. This separates the feasibility
decision from ranking. It scores candidates individually, so it does not
explicitly encourage diversity among the selected sequences.

The helper {py:func}`~protuna.optim.acquisitions.base.feasibility`, available as
`acquisitions.feasibility(batch)`, computes combined feasibility by multiplying
the soft-constraint probabilities for each candidate. Two constraint
probabilities of `0.8`, for example, give a combined value of `0.64`. Interpreting
this product as the probability of satisfying all constraints assumes the
constraint outcomes are independent for that candidate. Without soft constraints,
the helper returns ones.

The complete source code is available in
[ucb_acquisition.py](https://github.com/kumarshreshtha/protuna/blob/main/examples/ucb_acquisition.py).

### Setting up the acquisition

The constructor calls `super().__init__()` to initialize the resource registry,
then stores the selection budget, confidence weight, and feasibility threshold.
This implementation always requests standard deviations.

```{literalinclude} ../../examples/ucb_acquisition.py
:language: python
:start-at: import math
:end-before: "# init-end"
```

### Selecting candidates

The selection loop checks that predictions are present, filters by feasibility,
and calculates UCB for the remaining candidates.
[TopKHeap](https://github.com/kumarshreshtha/protuna/blob/main/protuna/utils/data_structs.py)
retains at most `budget`
`(score, seq_id)` pairs, evicting the pair with the lowest score whenever a better
candidate arrives.
Only those pairs survive between streamed batches. For a positive budget, the
selector retains `O(budget)` state. Processing `N` eligible candidates takes
`O(N log budget)` heap work.

```{literalinclude} ../../examples/ucb_acquisition.py
:language: python
:start-after: "# init-end"
:dedent: 4
```

The retained pairs are returned in descending order of score, with larger IDs
breaking exact score ties. An empty pool or a zero budget returns `[]`; if fewer
than `budget` candidates pass the threshold, all of them are returned.

### Using the acquisition

Our `UpperConfidenceBound` acquisition is now ready to be used in a pipeline.
It can be passed to the pipeline as either its `acquisition` or its `recycler`.
Here, `design_space` and `sampler` come from your campaign setup, and `predictor`
returns objective means and standard deviations:

```py
from protuna import optim

acquisition = UpperConfidenceBound(
    budget=96,
    beta=4.0,
    feasibility_threshold=0.5,
)
pipeline = optim.Pipeline(
    design_space=design_space,
    sampler=sampler,
    objective=predictor,
    acquisition=acquisition,
)
output = pipeline.run()
selection = output.selection
```

```{note}
An alternative to filtering by combined feasibility is to use a
{py:class}`~protuna.optim.constraints.inequality.HardUCBInequalityConstraint`
in the pipeline's `hard_constraints` mapping for each constraint previously
treated as a soft constraint. Each hard constraint accepts a candidate when its
predicted values satisfy `mean + sqrt(beta) * std > threshold`, so only
candidates that pass all these checks reach our acquisition.
This policy retains candidates whose uncertainty leaves room to exceed a lower
property bound. A soft feasibility threshold instead requires the combined
probability of satisfying the constraints to exceed a chosen value.
```

## Models and resource management

An acquisition can also use its own models, for example, to embed shortlisted
sequences. {py:class}`~protuna.optim.acquisitions.base.Acquisition` inherits
{py:class}`~protuna.inference.handlers.HandlerRegistryMixin`, which registers
handlers assigned as attributes. Calling `super().__init__()` before assigning
a handler allows that handler to be registered. The inherited `load()`
and `release()` methods then propagate to the registered handlers.

The pipeline calls `load()` before selection and calls `release()` afterward
when `release_handlers_after_use=True`. A raw model that does not implement the handler
interface needs an adapter or explicit resource management in the acquisition.
See [Custom Model Adapters](writing_adapters.md) for loading and release patterns.

## Using joint posterior samples

The interface supports both pointwise scores, such as UCB, and methods that
operate on samples from a joint posterior. For the latter, set
`requires_samples=True` and use `batch.objective_samples`.

Joint samples must preserve their meaning across candidates: column `j` must
represent the same sampled function evaluated at every candidate, including
across streamed batches. For an ensemble, it can be the prediction from the
same ensemble member. For a Bayesian neural network, it can be a fixed sampled
set of weights; for a GP, a coherent posterior function draw.
The predictor is responsible for this consistency.

## Owning the objective

Other methods need more than stored predictions. A look-ahead method such as
knowledge gradient may condition a model on hypothetical observations and
evaluate the resulting posterior. Such an acquisition needs access to the
trained model and its conditioning operations. For further reading on fantasy
models, see BoTorch's
[knowledge gradient tutorial](https://botorch.org/docs/v0.16.0/tutorials/one_shot_kg).

An acquisition can own the trained model and evaluate candidates directly from
their `token_ids`. Pass `objective=None` to `Pipeline` to skip the separate
objective stage. With no stored objective evaluations, `objective_mean`,
`objective_std`, and `objective_samples` remain `None`. The acquisition must
compute the predictions it needs and manage its model through the resource lifecycle
described above.

Any recycler used in this configuration must also work without stored objective
predictions, either by evaluating candidates itself or by selecting from other
available fields.
