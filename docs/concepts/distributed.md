# Distributed Serving Architecture

```{note}
This page explains how distributed serving works internally. You do not need
these details to use Protuna on a cluster. For basic usage and configuration
examples, start with the [distributed serving patterns guide](../guides/serving_patterns.md).
```

A **served handler** distributes requests across replicas running in Ray workers.
Each replica holds a copy of the handler, and a **pool** assigns work to those
replicas and collects their responses. The pool implements the same
{py:class}`~protuna.inference.handlers.Handler` contract as the handler it serves:
the caller loads resources, submits requests, collects responses, and releases
resources through the same operations.

This means a pipeline can use a local sampler or a served sampler without
knowing how its requests reach a worker. The [handlers](handlers.md) page
describes that shared interface. Here we look at the two serving architectures
behind it: a **routed pool**, which chooses a replica for each request, and a
**queue pool**, whose replicas pull work from a shared queue.

## Terminology

| Term | Meaning |
| --- | --- |
| Request | One input to a handler, such as a candidate sequence or a search space. |
| Response | The result for one request, such as a constraint decision or a batch of sampled candidates. |
| Submission | A collection of requests sent to a pool together, with one reference for collecting all their responses. |
| Worker batch | Requests passed together to one call of a worker's handler. A submission may span several worker batches, and a worker batch may combine requests from several submissions. |
| Replica | One copy of the served handler, with its own local state and resources. |
| Worker | The Ray actor that holds a replica, forms its batches, and returns its responses. |
| Pool | The object used by the caller to distribute requests across workers and collect responses. |
| Broker | The additional Ray actor in a queue pool that holds the shared request queue and tracks pending responses. |

## Submitting and collecting work

{py:func}`~protuna.inference.serve.api.serve_routed_pool` and
{py:func}`~protuna.inference.serve.api.serve_queue_pool` create pools from a
handler and its replication configuration. Both start Ray if it is not already
running and create the workers that will hold the replicas. Calling
{py:meth}`load() <protuna.inference.serve.pool.ModelPool.load>` on the returned pool then loads the handler's resources on every worker.

```{note}
Protuna's automatic initialization is a bare {py:func}`ray.init()` call without arguments.
Use {py:func}`ray.init(runtime_env=...) <ray.init>` to configure dependencies and environment
variables that workers need, including `PROTUNA_HOME` and any API keys used by
your handlers before serving. See Ray's
[initialization options](https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html)
and [runtime environment guide](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#specifying-a-runtime-environment-per-job) for details.
```

Both pools inherit from {py:class}`~protuna.inference.serve.pool.ModelPool`, an
abstract base class that defines the contract for submitting and collecting work.
Each pool implements two operations:

- {py:meth}`submit(requests) <protuna.inference.serve.pool.ModelPool.submit>`
  dispatches a collection of requests and returns a reference without waiting
  for the handler responses. The reference identifies the whole submission.
- {py:meth}`resolve(reference) <protuna.inference.serve.pool.ModelPool.resolve>`
  accepts that reference, waits for the responses, and returns one response per
  request in the submission's input order.

The usual batch call,
{py:meth}`pool(requests) <protuna.inference.serve.pool.ModelPool.__call__>`,
performs both steps as
{py:meth}`pool.resolve <protuna.inference.serve.pool.ModelPool.resolve>`({py:meth}`pool.submit <protuna.inference.serve.pool.ModelPool.submit>`(requests)). Streaming separates the steps so several
submissions can be processed concurrently.

## Routed pools

A {py:class}`~protuna.inference.serve.pool.RoutedPool` assigns each request to a
specific replica before sending it. Its
{py:type}`~protuna.inference.serve.pool.Router` is a callable that receives the
request and the total number of replicas, then returns a replica index.

The pool groups requests by their selected worker. This batch call shows one
submission split across two workers, each holding a handler replica:

```{mermaid}
sequenceDiagram
    participant C as Caller
    participant P as Routed pool
    participant W0 as Push worker 0
    participant W1 as Push worker 1
    C->>P: pool(requests)
    P->>P: Route each request
    P->>W0: Submit selected requests
    P->>W1: Submit selected requests
    par Worker 0 processes its requests
        W0->>W0: handler(batch)
        W0-->>P: Responses
    and Worker 1 processes its requests
        W1->>W1: handler(batch)
        W1-->>P: Responses
    end
    P->>P: Collect both groups in input order
    P-->>C: All submission responses
```

The default {py:class}`~protuna.inference.serve.routing.RoundRobinRouter` cycles
through replicas. With three replicas, successive requests go to replicas
0, 1, 2, 0, and so on. This distributes request counts evenly, but does not
account for differences in request cost or worker speed. Work already assigned
to a busy replica stays with that replica.

A router can also use the content of a request. For example,
{py:class}`~protuna.inference.serve.routing.MaskAffinityRouter` sends candidates
with the same mutated positions relative to a wild-type sequence to the same
replica. For a masked marginal predictor, placing these candidates in the same
batch allows them to share a masked model input. Routing creates the opportunity
to share computation; it does not itself cache model outputs or guarantee that
related requests land in the same batch.

The [sharing models and optimizing routing](../guides/serving_patterns.md#sharing-models-and-optimizing-routing)
section of the serving patterns guide shows how to use this router with a
masked marginal predictor.

### Push workers

Each routed replica runs in a
{py:class}`~protuna.inference.serve.actors.PushWorker`. A worker adds incoming
requests to its pending collection and forms batches up to
`max_batch_size`. Requests from separate submissions can share a worker batch.

A full batch is dispatched immediately. For a partial batch, `max_wait_ms`
controls how long the worker waits for more requests. If that wait expires while
other batches are still running or queued for execution, the partial batch is
held until they finish or enough requests arrive to fill it.

The worker calls
{py:meth}`handler(batch) <protuna.inference.handlers.Handler.__call__>`
in a dedicated thread, so it can continue
accepting requests while the handler runs. Handler calls execute one at a time
within that worker. Each incoming request has a **future**, a placeholder that
becomes ready when its response is available. The worker waits for the futures
belonging to each submitted group and returns their responses in that group's
input order, regardless of when the futures became ready.

The pool's {py:meth}`resolve() <protuna.inference.serve.pool.RoutedPool.resolve>`
then waits for the selected workers and places each
response at its recorded position in the original submission. This preserves
the batch-call contract without requiring workers to execute requests in a
global order.

## Queue pools

A {py:class}`~protuna.inference.serve.pool.QueuePool` sends requests to a shared
{py:class}`~protuna.inference.serve.actors.Broker` instead of selecting a worker
for each request. The broker holds the request queue and tracks the responses
needed to complete each submission.

The pool forwards each submission to the broker. This batch call shows two
workers fetching a batch each from the same submission and reporting responses
independently:

```{mermaid}
sequenceDiagram
    participant C as Caller
    participant P as Queue pool
    participant B as Broker
    participant W0 as Pull worker 0
    participant W1 as Pull worker 1
    C->>P: pool(requests)
    P->>B: Submit requests
    B->>B: Enqueue requests and track futures
    par Worker 0 takes a batch
        W0->>B: Fetch batch
        B-->>W0: Requests with IDs
        W0->>W0: handler(batch)
        W0->>B: Responses with IDs
    and Worker 1 takes a batch
        W1->>B: Fetch batch
        B-->>W1: Requests with IDs
        W1->>W1: handler(batch)
        W1->>B: Responses with IDs
    end
    B->>B: All submission futures resolved
    B-->>P: Responses in input order
    P-->>C: All submission responses
```

Each replica runs in a {py:class}`~protuna.inference.serve.actors.PullWorker`.
Workers ask the broker for their next batch, so faster workers can take more
work over time. This is useful when replicas have different processing speeds
or requests have uneven costs. There is no router or request affinity: any
worker can receive the next queued request.

### Pull workers and the broker

When a worker asks for a batch, the broker waits for the first request to become
available. It then gathers more requests up to that worker's `max_batch_size`,
waiting at most `max_wait_ms` after the first request. As with push workers,
one worker batch can contain requests from several submissions.

The pull worker calls
{py:meth}`handler(batch) <protuna.inference.handlers.Handler.__call__>`
in a dedicated thread and sends the
responses back to the broker with their internal request identifiers. The
broker uses those identifiers to resolve each request's future. It waits for
all futures belonging to a submission and returns their responses in that
submission's input order.

Each pull worker also starts fetching its next batch while processing the
current one. This overlaps queue access with computation, but means some work
can already be reserved by a busy worker. Queue-based balancing applies to work
still in the queue; it does not move batches between workers after they have
been taken.

## Streaming and work in flight

Both architectures use the same
{py:meth}`stream() <protuna.inference.serve.pool.ModelPool.stream>` workflow.
The pool reads `(request_id, request)` pairs in chunks the size of the combined
batch capacity of its replicas:

```text
requests per chunk = sum(group.num_replicas × group.max_batch_size)
```

For example, three replicas with `max_batch_size=32` give chunks of up to 96
requests. With `inflight_batches=4`, the pool keeps at most four submitted chunks
outstanding. As a submission completes, it collects the responses, submits the
next chunk, and yields `(request_id, request, response)` triples for the completed
chunk. This bounds how far ahead streaming reads and lets computation overlap
across replicas.

The chunks describe how much work the pool submits, not which batches a worker
will execute. A routed pool divides each chunk according to its router; queue
workers draw from the shared queue using their own batch limits.

For a routed pool, completion means all selected workers have returned their
portion of the submission. For a queue pool, it means the broker has received
every response in that submission. Streaming yields one triple per request,
but only after collecting the complete submission.

## Configuring replicas and pools

There are two ways to configure handler replicas. You can pass a local `handler`
to the serving function with a common set of options that apply to every replica:

```python
from protuna import inference

pool = inference.serve_routed_pool(
    handler,
    num_replicas=2,
    max_batch_size=32,
    actor_options={"num_cpus": 1},
)
```

Or you can define replica groups to configure individual replicas or sets of
replicas together:

```python
group = inference.ReplicaGroup(
    handler,
    num_replicas=2,
    max_batch_size=32,
    actor_options={"num_cpus": 1},
)
pool = inference.serve_routed_pool(group)
```

These two examples are equivalent: both create two replicas with the same
batching and CPU resource settings. The same two forms apply to
{py:func}`~protuna.inference.serve.api.serve_queue_pool`.

Replica groups become useful when different replicas need different settings.
For example, on a heterogeneous GPU fleet, you can pass a list of groups to use
smaller batches on A100 GPUs and larger batches on H100 GPUs:

```python
pool = inference.serve_queue_pool(
    [
        inference.ReplicaGroup(
            handler,
            num_replicas=2,
            max_batch_size=128,
            actor_options={"num_gpus": 1, "accelerator_type": "A100-40G"},
        ),
        inference.ReplicaGroup(
            handler,
            num_replicas=2,
            max_batch_size=256,
            actor_options={"num_gpus": 1, "accelerator_type": "H100"},
        ),
    ],
)
```

This creates four replicas, each reserving one GPU of the specified type in the
Ray cluster. The queue pool lets faster replicas fetch more work as they finish
their batches.

A {py:class}`~protuna.inference.serve.pool.ReplicaGroup` pairs a handler with the
configuration of its replicas. All replicas within a group share the same
batching and resource settings. These settings belong to each group:

| Parameter | Role |
| --- | --- |
| `handler` | The handler copied to each replica in the group. |
| `num_replicas` | Number of copies to run. Defaults to 1 on a {py:class}`~protuna.inference.serve.pool.ReplicaGroup`. |
| `max_batch_size` | Maximum requests passed to the handler in one worker batch. Defaults to 1. |
| `max_wait_ms` | Time allowed to fill a partial batch, in milliseconds. Defaults to 10; this is a batching delay, not a request timeout. |
| `responses_to_cpu` | Move response tensors and supported structured responses to CPU before returning them across a Ray boundary. Defaults to `True`. |
| `actor_options` | Ray options for each worker, including CPU/GPU resources and placement requirements. Defaults to an empty dictionary on a {py:class}`~protuna.inference.serve.pool.ReplicaGroup`. |

```{note}
Protuna does not currently support
[Ray Direct Transport (RDT)](https://docs.ray.io/en/latest/ray-core/direct-transport/direct-transport.html#direct-transport).
Setting `responses_to_cpu=False` skips the explicit CPU transfer; it does not
enable RDT. Future releases will explore RDT for direct GPU transfers between
nested handlers and between handlers and the driver process.
```

Other settings apply to the whole pool:

| Parameter | Applies to | Role |
| --- | --- | --- |
| `inflight_batches` | Both pools | Maximum outstanding submissions per streaming call. Defaults to 4. |
| `router` | Routed pools | Selects a replica for each request. `None` uses round-robin routing. |
| `broker_options` | Queue pools | Ray actor options for the shared broker. `None` uses Ray defaults. |
| `max_concurrent_submissions` | Queue pools | Maximum concurrent broker submission calls awaiting responses. Defaults to 1000; it counts submissions, not individual requests or worker batches. |

The [distributed serving patterns guide](../guides/serving_patterns.md) shows
how to configure these settings and choose whether to distribute a model, a
sampler or predictor, or both levels.


## Fault tolerance and current limitations

Both worker types catch exceptions raised by a handler call and report them to
the affected submitters. A failure in a worker batch fails every request in that
batch; collecting a submission containing those requests raises the error. The
workers do not automatically retry failed handler calls.

Worker or broker failure is a different case. Protuna currently has no
application-level recovery for work lost with an actor. In particular, a queue
worker can fail after taking requests from the broker, leaving their responses
pending without requeueing them. The broker's queue and pending responses are
held in memory. There is also currently no pool-wide request timeout or cancellation
interface, and stopping consumption of a stream does not cancel work already submitted.

The queue pool currently depends on a single broker for requests and responses which
can become a bottleneck for a large number of workers.

Recovery from worker and broker failures, along with support for multiple
brokers in a queue pool, is planned for future releases.
