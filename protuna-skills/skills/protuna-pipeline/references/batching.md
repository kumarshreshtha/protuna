# Batching and volume controls

The pipeline uses the `Handler.stream` interface for sampler, objective, and
constraint handlers. Each handler can run locally or be served.

These handlers can also contain nested handlers, such as model adapters, which
can themselves run locally or be served. Nested handlers are typically called
through their `Handler.__call__` interface.

## Source references

- Read `protuna.inference.handlers` to see how in-process handlers manage batches.
- Read `protuna.inference.serve.pool` if working with served handlers to see
  how `max_batch_size` works.

## How batching works

### Local handlers

For a local handler, `stream` calls `__call__` with batches of up to `batch_size`
requests. Set `batch_size` when constructing the handler; it defaults to `None`.
When `batch_size=None`, `stream` consumes the entire input iterator and passes
all requests to `__call__` in a single call.

### Served handlers

For a served handler, calls to either `stream` or `__call__` are processed by the
worker's `__call__` implementation in batches of up to `max_batch_size` requests.
Set `max_batch_size` when serving the handler with `serve_queue_pool` or
`serve_routed_pool`; it defaults to 1.

### Model inference

For an `InferenceModel`, `__call__` passes requests to `forward()` in chunks of up
to `batch_size` requests. When `batch_size=None`, it processes all requests it
receives in a single forward call.

## When to leave `batch_size=None`

For `LocalHandler.stream`, use `batch_size=None` only for finite inputs that are
known to be small, since it materializes the entire input iterator before
making a single batch call. Use a finite batch size for large or unbounded
streams.

When the **model itself is served**, its worker receives at most `max_batch_size`
model requests per call. Setting `model.batch_size=None` is reasonable if that
batch fits in memory. If both `batch_size` and `max_batch_size` are finite, the
forward batch size is at most the smaller of the two. Setting the model's batch
size above the serving limit does not increase the forward batch size. Actual
batches can be smaller because of routing, workload size, and `max_wait_ms`.

When **serving at the sampler level**, keep the nested model's `batch_size`
finite. A single sampler request can generate many model requests, so the
sampler's `max_batch_size` does not cap the model's forward batch size. Set a
finite model batch size to avoid GPU out-of-memory errors. The same principle
applies to nested handlers within predictors or constraints.
