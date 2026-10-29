# Configure distributed serving

Start by inspecting `serve_routed_pool()` and `serve_queue_pool()` for the two ways to
distribute a handler in Protuna.

## Source reference

Inspect the `protuna.inference.serve` package to locate the submodule
containing the class or function you want to inspect.

- `protuna.inference.serve.api` for the user-facing API (usually sufficient to get started with basic serving)
- `protuna.inference.serve.routing` for built-in request routers
- `protuna.inference.serve.pool` for the model pools that distribute a handler
- `protuna.inference.serve.actors` for the underlying Ray actors that manage request and response routing

## Initialize Ray before building distributed components

Call `ray.init(...)` with the intended configuration **before the first serving
API call** and before assembling the pipeline. Both `serve_routed_pool` and
`serve_queue_pool` call `ray.init()` if Ray is not already initialized, but that
fallback uses Ray's defaults. Explicit initialization lets you select a cluster
and configure the environment inherited by remote actor processes.

Protuna does not provision or start a multi-machine Ray cluster. Start the head
and worker nodes before running the pipeline script, using your cluster manager
or Ray's CLI. For example, run `uv run ray start --head --port=6379` on the head
and `uv run ray start --address='<head-ip>:6379'` on each worker, then run the
script on a cluster node.

For example:

```python
import ray

ray.init(
    address="<head-ip>:6379",  # Replace with the existing cluster's address.
    runtime_env={
        "env_vars": {
            "HF_HOME": "/srv/model-cache",
        },
    },
)
# Create serving pools and the Pipeline after initialization.
```

## Routing and replica groups

Use `serve_routed_pool` with its default round-robin router for a homogeneous
fleet. Use `serve_queue_pool` when faster workers should pull more work, such as
on mixed hardware. Queue serving introduces a shared broker that can become a bottleneck.

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

For heterogeneous configurations, pass a list of `inference.ReplicaGroup` objects.
For example, you can pass a list of groups to use smaller batches on A100 GPUs and
larger batches on H100 GPUs:

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


## Sharing GPUs between pipeline stages

Budget reservations across **all live replicas**, even when their stages run
sequentially. With two available GPUs, a sampler with two replicas reserving
`num_gpus=1` consumes both GPU resources. An objective pool with another two
one-GPU replicas cannot also schedule. Idle actors retain their reservations;
`release()` and `Pipeline(release_handlers_after_use=True)` unload handlers but
do not return those reservations to Ray. Only shutting down the pool frees them.

Use a Ray placement group to place one sampler replica and one objective replica
on each GPU. Fractional GPU reservations alone do not control which replicas
share a device: both sampler replicas could land on one GPU and both objective
replicas on the other. Reserve two bundles, each containing one GPU,
then assign one replica from each pool to each bundle. Each replica requests half
of its bundle's GPU and one CPU.

For example, with Ray already initialized, two available GPUs, two available
logical CPUs, and already-constructed `sampler` and `objective` handlers
and a `design_space`:

```python
import ray
from ray.util import scheduling_strategies

from protuna import data
from protuna import inference
from protuna import optim

pg = ray.util.placement_group([{"GPU": 1, "CPU": 2}] * 2, strategy="PACK")
try:
    ray.get(pg.ready(), timeout=10)
    bundle_strategies = [
        scheduling_strategies.PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=index,
        )
        for index in range(2)
    ]
    sampler_pool = inference.serve_routed_pool(
        [
            inference.ReplicaGroup(
                sampler,
                num_replicas=1,
                actor_options={
                    "num_gpus": 0.5,
                    "num_cpus": 1,
                    "scheduling_strategy": strategy,
                },
            )
            for strategy in bundle_strategies
        ],
    )
    try:
        objective_pool = inference.serve_routed_pool(
            [
                inference.ReplicaGroup(
                    objective,
                    num_replicas=1,
                    actor_options={
                        "num_gpus": 0.5,
                        "num_cpus": 1,
                        "scheduling_strategy": strategy,
                    },
                )
                for strategy in bundle_strategies
            ],
        )
        try:
            pipeline = optim.Pipeline(
                design_space=design_space,
                sampler=sampler_pool,
                objective=objective_pool,
                release_handlers_after_use=True,
            )
            output = pipeline.run()
            sequences = data.tokenizer.decode(list(output.selection.token_ids))
        finally:
            objective_pool.shutdown()
    finally:
        sampler_pool.shutdown()
finally:
    ray.util.remove_placement_group(pg)
```

For three handlers sharing each GPU, assign one replica of each handler to each
bundle and set each replica's `num_gpus` to `1 / 3`.

## Complete CPU serving smoke check

This requires Ray but no GPU, model weights, or external cluster. It starts a local
Ray runtime only if none exists and shuts down only the runtime it started.

```python
import tempfile

import ray

from protuna import data
from protuna import inference
from protuna import optim
from protuna.optim import samplers
from protuna.optim import spaces

owns_ray = not ray.is_initialized()
if owns_ray:
    ray.init(num_cpus=2)
try:
    pool = inference.serve_routed_pool(
        samplers.SiteSaturationSampler(batch_size=None),
        num_replicas=2,
        max_batch_size=1,
        actor_options={"num_cpus": 1},
    )
    try:
        with tempfile.TemporaryDirectory() as output_dir:
            pipeline = optim.Pipeline(
                design_space=spaces.DesignSpace(
                    [spaces.SearchSpace.from_rules("MKTA", [])],
                    space_sampler=spaces.PartitionSampler(n_partitions=2),
                ),
                sampler=pool,
            )
            output = pipeline.run()
            sequences = data.tokenizer.decode(list(output.selection.token_ids))
            assert len(sequences) == len(set(sequences)) == 76
    finally:
        pool.shutdown()
finally:
    if owns_ray:
        ray.shutdown()
```

The two partitions provide two requests to distribute. `batch_size=None` on this
sampler is bounded by one space per worker call, and this example's candidate
response is small. See [batching](batching.md) before generalizing to model-based
samplers with expanded populations.

## Lifecycle and failures

`release()` unloads handler resources without stopping workers or freeing their
Ray resource reservations. `shutdown()` terminates the pool. The nested registry's
`load` and `release` calls propagate to shared handlers; releasing can unload a shared model
needed later.

If your script exits soon after the pipeline finishes, Ray automatically shuts down
its workers when the driver process exits. Otherwise, keep references to all pools
created during the run and shut each down explicitly when done. Shutting down an
outer pool does not shut down a separately created model pool used inside it.

A handler exception fails all requests in its worker batch; there is no automatic
retry. Actor failures have no application-level work recovery. Queue work taken
by a failed worker can remain pending indefinitely; there is no pool-wide request
timeout or cancellation API. Stopping stream consumption does not cancel work
already submitted. Capture the error and use deliberate cleanup and a fresh run
after correcting the cause; do not promise transparent fault-tolerant execution.
