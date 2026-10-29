---
name: protuna-pipeline
description: Construct, configure, troubleshoot, or run Protuna optimization pipelines from existing components. Use to construct pipeline components such as design spaces, explicit stores, acquisitions, and predictors, distribute handlers using Ray, and diagnose pipeline failures. Use the protuna-components skill instead when implementing a custom component.
---

# Construct and debug a Protuna pipeline

Read referenced `protuna` modules from the local checkout or the package installed in the project's Python environment.
Resolve Markdown links relative to the file containing them.

## Before the pipeline

- A `Pipeline` performs inference-time search on the design space. It does not train any models.
- If your campaign relies on custom property prediction models, fine-tuned or preference-aligned generators,
  or any other trained components, ensure they are trained externally before starting pipeline construction.

## Building the pipeline

There are seven steps to building a pipeline, some of which are optional:

1. Define the `DesignSpace` – specify starting sequences, allowed mutations, and how to sample narrowed search spaces.
2. Pick a `Sampler` – specify how to sample candidates from these search spaces.
3. Set up any hard constraints – define Boolean-valued constraints that all candidates must meet.
4. Set up any soft constraints – define constraints that return the probability of being met.
5. Set up the objective predictor – define the predictor for the primary objective.
6. Pick a compatible `Acquisition` function – specify how to select the final batch to test in the lab.
7. Optionally, pick a recycler – specify how to select parent sequences for the next cycle.

Only a `DesignSpace` and `Sampler` are always required. Other components are optional.
Use them as needed to meet the specific design goals of the project.

### Cycles and rounds

A *cycle* is one pass through candidate generation and computational evaluation. After each cycle, the pipeline can either:

- Start another cycle, using sequences selected from the current cycle as starting points.
- Proceed to acquisition, which selects the final candidates for wet-lab testing from the generated samples.

A *round* includes the full pipeline run—which may contain several cycles—followed by wet-lab testing of the selected candidates. The experimental results provide feedback for the next round.

A recycler is just another `Acquisition` applied to the candidates generated in the current cycle
to pick candidates that become parents in the next cycle.

The term *selector* refers to either the recycler or the acquisition when the distinction between
them doesn't matter.

Read the accompanying [examples](references/examples.md) that follow these steps
to build simple pipelines. Consult other supporting pages as needed:

- [Batching](references/batching.md): before choosing handler `batch_size`, serving
  `max_batch_size`, or nested model limits; these control different workloads.
- [Storage](references/storage.md): for writing explicit stores, configuring cloud
  storage, and inspecting records.
- [Serving](references/serving.md): when using Ray to distribute components, choosing
  which handler to distribute, allocating resources, or cleaning up pools.

## Source reference

Inspect the following modules to see the construction arguments. Only read
the modules relevant to the project. For instance, if the user already
specified the sampler they want to use, only inspect that sampler.

- `protuna.optim.pipeline` for `Pipeline` construction
- `protuna.optim.spaces` for `DesignSpace` construction
- `protuna.optim.samplers` for existing samplers
- `protuna.optim.constraints` for constraint contracts
- `protuna.optim.predictors` for predictor contracts
- `protuna.optim.acquisitions` for existing acquisition functions
- `protuna.optim.storage` for creating a custom candidate store

For packages, first inspect their `__init__.py` to locate the submodule of interest.


## Defining the DesignSpace

Follow these rules to write a design space:

- Use `data.tokenizer` tokens throughout.
- Start with a single wild-type hit or, when available, a few (4–5) high-fitness mutants as the
  starting sequences. All starting sequences must be the same length.
- Use rules to specify residues or regions that the campaign requires preserving.
- Built-in rules use zero-based sequence positions, so adapt accordingly if
  surrounding user code uses one-based biological residue numbering.
- If the rules aren't expressive enough for your design goals (rare), directly construct the
  search spaces with the boolean `allowed` mask.
- The order of rules in the `from_rules` class method matters. Rules transform the initially all-true
  `[L, 20]` mask in order. Later rules can override the mask of previous rules. For example,
  `DenyResidues(["C"])` followed by `Allow(positions=[5])` reopens cysteine at site 5; reversing
  the order keeps it excluded there, unless it is the starting residue.
- An empty rule list allows all standard residues at every site.
- `SearchSpace` always restores the starting residue at every site after all rules run.
- To verify that the `SearchSpace` reflects expectations, check `mutable_positions`, `available_aa_at_pos()`,
  and `size()` against the intended fixed sites and residue choices. `space.is_valid()` can be used
  to check if a particular mutant is accessible in this space.
- `DesignSpace.space_sampler` chooses which mutable sites to explore together.
  With `space_sampler=None`, each starting space is yielded unchanged once per
  sampling pass.
- If none of the built-in space samplers in `protuna.optim.spaces.subspaces` meet the requirements of
  the project and you need to construct a new space sampler, refer to the
  `protuna-components` skill.
- Subspace mutation limits do not cap total distance from the original sequence over
  multiple cycles. For that requirement, add a hard mutation-count constraint with
  the intended fixed reference.

## Preparing the objective and constraints

The objective must follow the `Predictor` handler contract.
Wrap your existing prediction model in an `InferenceModel` (`protuna.inference.handlers`)
and implement the forward method to respect the `Predictor` handler contract.
In `load` and `release`, configure how the predictor's resources are loaded and released.

Similarly, a constraint must follow the `HardConstraint` or `SoftConstraint` handler contract.

Read the `protuna-components` skill and its `predictors`, `constraints`, and `adapters` references
for more details on configuring predictors and constraints, and for general steps to follow when
writing model adapters.

An ensemble of multiple predictors can be wrapped in `predictors.EnsemblePredictor`. A predictor
that predicts in arbitrary units and needs to be mapped to the units of an assay result can
be wrapped in `predictors.CalibratedPredictor` with the corresponding assay data. This is useful
for ranking models used in constraints with thresholds expressed in the assay's measurement units.

The objective may also be set to `None` for trivial pipelines like a directed evolution simulator.
But more importantly, it can be set to `None` when the acquisition itself takes care of
evaluating the objective on the supplied candidates. If this is the case for the acquisition
you are using for both selectors, then you may set the objective to `None`.


## Compatibility between the objective and selectors

An objective must be able to fulfil the uncertainty requirements of both
selectors. Check that your objective `Predictor`
can supply the uncertainty measures your chosen selectors need. These requirements are indicated by
the acquisition's `requires_std` and `requires_samples` class attributes, which
default to `False`. They can be overridden as class attributes in
`Acquisition` subclasses or set in `__init__` when their values need to be determined
dynamically from the acquisition's inputs.

Inspect the definitions of both your acquisition and recycler to verify this.

## Supplying constraints

### Ordering hard constraints

Stages run in this order: candidate generation → hard constraints → soft constraints →
objective → recycling or final acquisition.

Put hard constraints in increasing cost order in the mapping: cheap sequence or
mutation-count checks before model-based checks. A failure excludes that candidate
from every later constraint, objective evaluation, recycling, and final selection.
Order similarly priced checks by likely rejection rate to avoid unnecessary work.

### Check compatibility between soft constraints and selectors

Soft constraints return feasibility probabilities; they never filter candidates
on their own. The acquisition is free to use the feasibility probabilities the
way it wants. Verify both selectors consume them as intended. For example,
`IdentityAcquisition` ignores them; `Greedy` requires their product to be
**strictly greater** than a `feasibility_threshold`.

### Naming constraints

Keep hard and soft names distinct. If you're passing an explicit `CampaignStore`
to `pipeline.run(store=store)`, pass `soft_constraint_names=tuple(soft_constraints)`
to the store constructor. Unregistered names are stored as hard constraints.
The automatically created store registers soft names for you.

## Sequence budgets and deduplication

- Estimate the upper bound on candidate volume per cycle from starting parents × subspaces per parent ×
  proposals per subspace.
- The recycler budget controls the number of parents for the next cycle and can multiply later
  inference work.
- The pipeline deduplicates candidates across all spaces and cycles. Distinct IDs will always
  store unique sequences. Starting sequences are always excluded from the candidate proposals.
  If the lab needs to validate the starting sequences, include them in addition to the proposals
  from pipeline run.

## Running the pipeline

Call `output = pipeline.run(store=store)` with an optional explicit store; omitting
it creates a new store for the run. The returned `PipelineOutput` dataclass contains
`output.selection`, a `CandidateBatch` in the final acquisition's returned ID order,
and `output.store`, the run history.

## Resource management

Start with `release_handlers_after_use=False` when models fit together. Use `True`
if stages must release models to coexist in memory, allowing for reload costs and
shared-model lifetimes. This setting releases handlers after successful stages but does not guarantee cleanup
on failure or shut down served pools. Clean up owned resources explicitly
when needed, following the serving reference.

## Practical tips

- Avoid recycling in early campaigns, especially if using an objective model
  trained only on small amounts of campaign data.
- When constructing a pipeline for multi-property optimization, you can set one
  property as the objective and others as hard or soft constraints based on the
  targets of the campaign. These objectives can also be rotated from one
  wet-lab round to the next, making the objective of the current round a constraint
  in the next, depending on the lab results.

## When debugging

Retain an explicit store when diagnostics matter. Use a fresh run identity for a
smoke run or retry; reopening an existing `task_id` is not a resume mechanism and
can collide with stored files.

When diagnosing a completed run, read the [storage](references/storage.md) page
to understand the stored Parquet formats and how to inspect them. They track the
lineage of all generated samples along with their constraint and objective scores.

When diagnosing a live run, enable pipeline logging and inspect the live store,
which includes buffered records; Parquet alone may be incomplete before finalization.
`iter_samples(cycle_id=..., exclude_failed=False)` shows proposals including failures;
`get_candidate_stream(cycle_id=...)` shows hard-filter survivors with available
predictions and feasibility. Compare these with per-constraint records joined by
`seq_id`. Missing downstream evaluations for hard failures are expected.
Read `protuna.optim.storage` for the storage implementation and
`protuna.optim.candidates` for `CandidateStream` and `CandidateBatch` abstractions.
