# Handlers

A **handler** is an object that processes requests through a shared interface,
whether it runs locally or on distributed workers. In Protuna, samplers, constraints,
predictors, and the model adapters they use are all handlers. Each has a different
job, but the code calling it follows the same pattern: load its resources, submit
requests, collect responses, and release its resources when they are no longer needed.

This common interface separates an algorithm's role in a campaign from where
its computation happens. A sampler can run in the calling process or across a
pool of Ray workers. Either way, the pipeline submits search spaces and receives
candidate sequences through the same operations.

## Requests and responses

The {py:class}`~protuna.inference.handlers.Handler` protocol is parameterized by
two types: {py:class}`Handler[RequestT, ResponseT] <protuna.inference.handlers.Handler>`. These describe **one request** and
**one response**, even though a handler normally processes several requests at
once.

Some examples of handlers in Protuna and their request and response types are:

| Handler role | request | response |
| --- | --- | --- |
| Sampler | A {py:class}`~protuna.optim.spaces.search.SearchSpace` | A batch of candidate token sequences from that space |
| Predictor | A {py:class}`~protuna.optim.predictors.base.PredictionRequest` containing a candidate and output options | A {py:class}`~protuna.optim.predictors.base.Prediction` containing a mean and optional uncertainty or samples |
| Hard constraint | A candidate's token sequence | A scalar boolean tensor indicating whether it passes |
| Soft constraint | A candidate's token sequence | A scalar floating-point tensor giving its constraint score |

For example, calling a sampler with ten search spaces returns ten candidate
batches. The number of candidates within each batch may differ. The handler
contract concerns the correspondence between requests and responses; the
sampler determines what each response contains.

## The handler protocol

Every handler exposes five operations:

| Operation | Contract |
| --- | --- |
| {py:meth}`handler(requests) <protuna.inference.handlers.Handler.__call__>` | Resolve a batch, returning one response per request in input order. |
| {py:meth}`handler.stream(requests) <protuna.inference.handlers.Handler.stream>` | Consume an iterator of `(request_id, request)` pairs and yield `(request_id, request, response)` triples as work completes. Results may arrive out of input order. |
| {py:meth}`handler.load(device=None) <protuna.inference.handlers.Handler.load>` | Load the resources needed to process requests. Local implementations can use the requested device; served pools manage placement on their workers. |
| {py:meth}`handler.release() <protuna.inference.handlers.Handler.release>` | Release held GPU resources without terminating the service. |
| {py:meth}`handler.shutdown() <protuna.inference.handlers.Handler.shutdown>` | Terminate a distributed service. The default local implementation does nothing. |

Batch inputs and outputs use {py:class}`~protuna.typing.SequenceLike`, so they
can be lists, tuples, or compatible array-like containers. Streaming uses
iterators instead. The caller supplies the integer request IDs, which allow it
to associate results with their inputs regardless of arrival order.


## Local and distributed handlers

A **local handler** executes in the calling process. A **served handler** is a
pool object that forwards requests to replicas running in Ray workers and
collects their responses. Serving changes the execution mechanism while
preserving the request and response interface.

For example, a {py:class}`~protuna.optim.samplers.baselines.SiteSaturationSampler` can be used
locally:

```python
from protuna import inference
from protuna.optim import samplers
from protuna.optim import spaces

search_spaces = [
    spaces.SearchSpace.from_rules("AC", []),
    spaces.SearchSpace.from_rules("DE", []),
]
local_sampler = samplers.SiteSaturationSampler(batch_size=1)
```

Serving that sampler with {py:func}`~protuna.inference.serve.api.serve_routed_pool`
creates a second object implementing the same contract.
This example uses two CPU workers and starts Ray if it is not already running:

```python
served_sampler = inference.serve_routed_pool(
    local_sampler,
    num_replicas=2,
    max_batch_size=1,
    actor_options={"num_cpus": 1},
)
```

The consumer below uses exactly the same operations for both objects:

```python
try:
    for handler in (local_sampler, served_sampler):
        handler.load()
        try:
            candidate_batches = handler(search_spaces)
            for space_id, space, candidates in handler.stream(enumerate(search_spaces)):
                print(space_id, space.init_sequence.shape, candidates.shape)
        finally:
            handler.release()
finally:
    served_sampler.shutdown()
```

```{note}
Calling {py:meth}`shutdown() <protuna.inference.handlers.Handler.shutdown>` is optional when the process that created the pool is about
to exit. With the default actor lifetime, Ray automatically terminates the
pool's actors when their creator process exits. Explicit shutdown lets you
terminate the pool earlier, for example in a long-running script or notebook.
```

The batch call preserves the order of `search_spaces`. The streaming loop uses
`space_id` to identify each result. Neither operation needs to inspect whether
the handler is local or served. The pipeline uses this same pattern to generate
candidates, apply constraints, and evaluate its objective.

## What streaming means

For a {py:class}`~protuna.inference.handlers.LocalHandler`, the default {py:meth}`stream <protuna.inference.handlers.LocalHandler.stream>`
implementation is a sequential iterator over batch calls. With a finite
`batch_size`, it reads that many requests, calls the handler, yields the results
in input order, and repeats. Streaming by itself **does not** make local execution
concurrent. With `batch_size=None`, this implementation consumes the entire
input iterator before making a single call, so a finite batch size is needed
for long or unbounded inputs.

Served pools use the same streaming interface to keep several submitted batches
in flight at once. They pull more requests as completed work is collected,
limiting how much work is outstanding. The two serving strategies differ in how
they assign that work:

- A {py:class}`~protuna.inference.serve.pool.RoutedPool` sends each request to the
  replica selected by its router. The default router cycles through replicas.
- A {py:class}`~protuna.inference.serve.pool.QueuePool` places requests in a shared
  queue. Workers pull new batches as they become available, allowing faster
  workers to take more work.

Both pools currently yield completed submissions in completion order. Consumers
should use the request IDs rather than assume that streaming preserves global
input order. The [distributed serving architecture](distributed.md) page covers the
pools in more detail.

## Composing handlers

A handler can also consume other handlers. An {py:class}`~protuna.optim.samplers.mlm.MLMSampler`, for example,
receives search spaces from its caller and submits logits requests to a model
handler internally. The two interfaces describe different levels of the same
computation.

Because both levels follow the handler contract, the execution boundary can be
placed at either level:

| Composition | Where the work happens |
| --- | --- |
| Local sampler with a local model | Sampling and model inference run in the calling process. |
| Local sampler with a served model | Sampling runs locally and sends model requests to a worker pool. |
| Served sampler containing a local model | Each sampler replica contains its own model instance. |
| Served sampler containing a served model | Sampler replicas send model requests to a separate model pool. |

The same model pool can also be shared by a sampler and a predictor that accept
the same model interface. These choices change resource use and how much data
crosses process boundaries, while preserving the interface seen by each
consumer. The [distributed serving patterns guide](../guides/serving_patterns.md)
shows how to construct and choose between these arrangements.

Resource management follows the composition too. Components using
{py:class}`~protuna.inference.handlers.HandlerRegistryMixin` register handler
attributes automatically and forward
{py:meth}`load() <protuna.inference.handlers.HandlerRegistryMixin.load>` and
{py:meth}`release() <protuna.inference.handlers.HandlerRegistryMixin.release>` to them, including
through nested registries. This lets a sampler manage the resources of its
underlying model through the same calls whether that model is local or served.

## Reading `*Like` annotations

Handler type aliases ending in `Like` name a request/response contract without
requiring a specific implementation. For example,
{py:type}`~protuna.optim.samplers.base.SamplerLike` is defined as:

```python
from protuna.typing import BatchTokenIDs

type SamplerLike = inference.Handler[spaces.SearchSpace, BatchTokenIDs]
```

An argument annotated as {py:type}`~protuna.optim.samplers.base.SamplerLike` accepts a local sampler, a served sampler,
or another implementation of that handler contract. Similarly,
{py:type}`~protuna.optim.predictors.base.PredictorLike` describes a handler from
{py:class}`~protuna.optim.predictors.base.PredictionRequest` to {py:class}`~protuna.optim.predictors.base.Prediction`, and
{py:type}`~protuna.inference.interfaces.mlm.MLMForLogitsLike` describes a handler
from {py:class}`~protuna.inference.interfaces.mlm.MLMLogitsRequest` to a logits tensor.

These aliases specify the operations and data types a consumer needs. They do
not promise access to implementation-specific methods: a local sampler exposes
{py:meth}`sample() <protuna.optim.samplers.base.Sampler.sample>`, for example, but code accepting {py:type}`~protuna.optim.samplers.base.SamplerLike` uses {py:meth}`__call__ <protuna.inference.handlers.Handler.__call__>` or
{py:meth}`stream() <protuna.inference.handlers.Handler.stream>` so it can also work with a served pool.

For local implementations, {py:class}`~protuna.inference.handlers.LocalHandler`
provides the default streaming behavior,
{py:class}`~protuna.inference.handlers.InferenceModel` adds batching through a
{py:meth}`forward() <protuna.inference.handlers.InferenceModel.forward>` method, and
{py:class}`~protuna.inference.handlers.ModelFreeLocalHandler` provides no-op
{py:meth}`load() <protuna.inference.handlers.ModelFreeLocalHandler.load>` and
{py:meth}`release() <protuna.inference.handlers.ModelFreeLocalHandler.release>` methods for handlers without model resources.
