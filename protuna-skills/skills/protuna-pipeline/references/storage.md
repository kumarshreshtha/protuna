# Configure and inspect campaign storage

`CampaignStore` persists sequences, lineage, constraints, and objective predictions
as Parquet. It is not a model checkpoint or a resumable pipeline scheduler.

## Source reference

- Read `protuna.optim.storage` for the `CampaignStore` implementation.
- Read `protuna.optim.candidates` if working with candidate streams
  (`store.get_candidate_stream`).

## Location and schema

Pass `optim.CampaignStore(...)` to `pipeline.run(store=store)`. With no explicit store,
the pipeline creates one and registers its soft-constraint names automatically.
With a custom store, match `protein_length` to the starting sequence and set
`soft_constraint_names=tuple(soft_constraints)` yourself. Any unregistered name
is treated as a hard verdict, regardless of the handler's intended semantics.
Keep hard and soft constraint names distinct.

Records go under `<location>/pipelines/<task_id>/`. `task_id=None` generates a
fresh UUID; a chosen task ID should identify a fresh run. `location=None` uses
Protuna's cache directory: `PROTUNA_HOME/protuna`, otherwise
`XDG_CACHE_HOME/protuna`, otherwise `~/.cache/protuna`.

This fragment assumes `wt` and `soft_constraints` are the pipeline inputs:

```python
from protuna import optim

store = optim.CampaignStore(
    protein_length=len(wt),
    location="./campaign-results",
    max_size_mb=64,
    scan_size=4096,
    soft_constraint_names=tuple(soft_constraints),
)
```

The four file families are `samples_part*.parquet`, `objective_part*.parquet`,
`hard_constraints_part*.parquet`, and `soft_constraints_part*.parquet`. Families
with no rows need not have files. Samples include `seq_id`, `token_ids`,
`parent_id`, `ancestor_id`, and `cycle_id`. Starting sequences have cycle ID zero;
generated cycles start at one. Within a pipeline run, distinct IDs correspond to
unique sequences: the pipeline deduplicates proposals across all spaces and
cycles and excludes starting sequences and `evaluated_mutants` before storage.
Direct writes to `CampaignStore` do not enforce sequence uniqueness.

## Buffering and reading

- `max_size_mb` is an approximate, positive, finite budget in MiB **per buffer**, not a
  combined RAM cap. There are four independent buffers. The estimate reserves
  writing headroom but checks after appending a batch, so a large response can
  exceed it.
- `scan_size` is a positive row count controlling candidate/read chunks.
- Constraint and prediction tensors can remain on their original device until
  read or flushed. GPU memory may grow from buffered outputs even with bounded
  forward passes.
- Reads include buffered and written records. A successful `Pipeline.run()` calls
  `store.finalize()` before returning. Finalization is not guaranteed if an exception occurs. If
  retaining partial results is useful, explicitly finalize in the caller's
  cleanup without hiding the original error; partial records are not a completed
  candidate pool or a resume checkpoint.

Using a live `store`, inspect candidates as follows:

```python
for batch in store.get_candidate_stream():
    print(batch.seq_id, batch.objective_mean, batch.feasibility)

for seq_id, tokens in store.iter_samples(exclude_failed=False):
    print(seq_id, tokens.shape)
```

`get_candidate_stream(cycle_id=None)` includes all generated cycles and excludes
hard failures. Optional prediction fields can be absent when no objective ran.
`iter_samples()` excludes failures by default; use `exclude_failed=False` for
diagnostics. `iter_base_sequences()` retrieves the starting sequences; passing
cycle zero to the candidate APIs raises an error.

`get_samples_unordered(ids, columns=["seq_id", "token_ids", "parent_id"])` returns
requested fields in arbitrary order and omits unknown IDs. Include `seq_id` and
reorder explicitly if ranking matters.

## Cloud storage

Use a URI such as `s3://my-protuna-bucket/campaigns` for `location` and pass
Polars connection options, for example `storage_options={"aws_region":
"eu-west-1"}`. Use the user's configured bucket and credentials; the URI does not
create a bucket. Options are forwarded to Polars for both reads and writes.

`storage_options=None` uses Polars defaults. `credential_provider="auto"` infers
a provider; `credential_provider=None` disables that provider rather than meaning
automatic authentication. Use environment or role-based credentials appropriate
to the deployment, not secrets embedded in example code. Inspect the installed
Polars configuration for provider-specific details when needed.

Only the pipeline directly interacts with storage. In distributed workflows,
the driver orchestrates pipeline storage; inference replicas do not need cloud
storage credentials configured.

## Reopening is not resuming

Constructing a store with an existing `task_id` resets its in-memory sequence,
cycle, and part counters. Subsequent writes can collide with existing files and
IDs. Do not use that pattern to append or resume. Use fresh locations/task IDs for
new runs. For offline inspection, scan the persisted Parquet families directly
with Polars and reconstruct joins by `seq_id`; do not create a writer as a reader.
