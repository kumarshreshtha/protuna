# Distributed Serving Patterns

In this guide, we'll go through the different ways to distribute inference workloads in Protuna. The {py:mod}`~protuna.inference` API provides several ways to handle this.

## Serving models

We start by constructing a local handler. The handler we are using here is a {py:class}`~protuna.inference.models.dplm.DplmLogitsModel`, which wraps the [FastPLMs](https://github.com/Synthyra/FastPLMs/tree/main) implementation of the [DPLM](https://github.com/bytedance/dplm) 150M model in the {py:type}`~protuna.inference.interfaces.mlm.MLMForLogitsLike` interface.

```py
from protuna.inference import models

local_handler = models.DplmLogitsModel(models.DplmModels.DPLM_150M, batch_size=256)
```

Assuming that the Ray cluster is up and running, we can now create several replicas of the model on the cluster. The simplest way to do this is:

```py
from protuna import inference

served_handler = inference.serve_routed_pool(
    local_handler,
    num_replicas=5,
    max_batch_size=128,
    actor_options={"num_gpus": 1},
)
```

```{note}
**On batch size**

When constructing local handlers for models, it is recommended to pass the `batch_size` argument to the model constructor (all {py:class}`~protuna.inference.handlers.InferenceModel` instances accept it). Without the `batch_size` argument, the model will consume all requests it receives at once, which can lead to out-of-memory (OOM) errors.

When the handler is served, the serving API's `max_batch_size` argument ({py:func}`~protuna.inference.serve.api.serve_routed_pool` or {py:func}`~protuna.inference.serve.api.serve_queue_pool`) controls how many samples the handler receives at once, so in those cases the handler can safely have `batch_size=None` (the default). However, if `batch_size` is set, it acts as the final limit on batch size. Setting it higher than `max_batch_size` will have no effect, while setting it lower will further divide the received batch.

```

This returns a handler that replicates the underlying `local_handler` five times, with each replica reserving one GPU in the Ray cluster. The handler is a {py:class}`~protuna.inference.serve.pool.RoutedPool`, meaning it uses a {py:type}`~protuna.inference.serve.pool.Router` to decide where each new request goes. When a router isn't explicitly provided, it defaults to a {py:class}`~protuna.inference.serve.routing.RoundRobinRouter` that dispatches successive requests to replicas in cyclic order. This default works well with a homogeneous fleet of GPUs, i.e., when all GPUs available in your cluster are identical.

Let's say we're on a cluster with different available GPU types and we'd like to configure them separately. This can be done using a {py:class}`~protuna.inference.serve.pool.ReplicaGroup` as follows:

```py
served_handler = inference.serve_routed_pool(
    [
        inference.ReplicaGroup(
            local_handler,
            num_replicas=3,
            max_batch_size=128,
            actor_options={"num_gpus": 1, "accelerator_type": "A100-40G"},
        ),
        inference.ReplicaGroup(
            local_handler,
            num_replicas=2,
            max_batch_size=256,
            actor_options={"num_gpus": 0.5, "accelerator_type": "H100"},
        ),
    ],
)
```

As before, it replicates the `local_handler` five times but now with 2 different configurations (`max_batch_size` 128 and 256) and only reserving 4 GPUs (the H100 is shared between 2 replicas). However, for this heterogeneous fleet, the default {py:type}`~protuna.inference.serve.pool.Router` is not a good choice as it'll delegate the same amount of work to all replicas, thus starving the more powerful H100 GPUs. At this point, you can create a custom {py:type}`~protuna.inference.serve.pool.Router` to balance the load more intelligently or switch to a {py:class}`~protuna.inference.serve.pool.QueuePool` using {py:func}`inference.serve_queue_pool <protuna.inference.serve.api.serve_queue_pool>` with the same arguments as above. The queue pool uses a shared queue among all served handlers, and each handler pulls new batches from the queue as it finishes, allowing faster accelerators to pick up more work.

## Choosing what to distribute

The local {py:class}`~protuna.inference.models.dplm.DplmLogitsModel` and the pools returned by {py:func}`inference.serve_routed_pool <protuna.inference.serve.api.serve_routed_pool>` and {py:func}`inference.serve_queue_pool <protuna.inference.serve.api.serve_queue_pool>` implement the {py:type}`~protuna.inference.interfaces.mlm.MLMForLogitsLike` handler interface, so downstream code can use them interchangeably. These handlers can be passed to any {py:class}`~protuna.optim.samplers.base.Sampler` that requires a {py:type}`~protuna.inference.interfaces.mlm.MLMForLogitsLike` handler for sampling candidates.

You can therefore create a local sampler with a local model:

```py
from protuna.optim import samplers

sampler = samplers.MLMSampler(local_handler, max_samples=100)
```

or a local sampler with a served model:

```py
sampler = samplers.MLMSampler(served_handler, max_samples=100)
```

A {py:class}`~protuna.optim.samplers.base.Sampler` is itself a {py:class}`~protuna.inference.handlers.Handler`, so both of these setups can also be distributed:

```py
sampler = inference.serve_routed_pool(
    samplers.MLMSampler(local_handler, max_samples=100),
    num_replicas=5,
    max_batch_size=64,
    actor_options={"num_gpus": 1},
)
```

and

```py
sampler = inference.serve_routed_pool(
    samplers.MLMSampler(served_handler, max_samples=100),
    num_replicas=10,
    max_batch_size=64,
    actor_options={"num_cpus": 1},
)
```

In the first case, we've passed a local model to the {py:class}`~protuna.optim.samplers.mlm.MLMSampler` and created 5 GPU replicas of it. This is similar to our initial setup of 5 replicas, but in this case we've chosen to distribute at the level of the sampler instead of the underlying model. This is a better design choice for most samplers since it reduces the number of requests crossing process boundaries and uses smaller payloads – sampled candidates instead of model logits. This also doesn't affect GPU utilization much since the bulk of a sampler's compute load is often in the model forward pass.

In the second case, the sampler receives a model that's already distributed across five model replicas. The sampler is then itself replicated 10 times, meaning 10 sampler replicas could send requests to the 5 models served on the GPU. Such over-parallelization can [impact the intended speedup](https://docs.ray.io/en/latest/ray-core/patterns/too-fine-grained-tasks.html) so this should only be done if there's a good reason, and the setup should be validated with a benchmark.

```{tip}
**The `*Like` type annotation**

All objects that accept a {py:class}`~protuna.inference.handlers.Handler` can consume a local or served handler without exception. In Protuna, we often use the `*Like` type annotations to indicate this. For instance, {py:type}`~protuna.optim.samplers.base.SamplerLike` represents either a local {py:class}`~protuna.optim.samplers.base.Sampler` or a served sampler created with {py:func}`inference.serve_queue_pool <protuna.inference.serve.api.serve_queue_pool>` or {py:func}`inference.serve_routed_pool <protuna.inference.serve.api.serve_routed_pool>`. Anything that consumes a {py:type}`~protuna.optim.samplers.base.SamplerLike` can use either.

```

## Sharing models and optimizing routing

Like samplers, all predictors and constraints also implement the {py:class}`~protuna.inference.handlers.Handler` contract, so similar parallelization can be achieved for them or their underlying models. The same handler can also be shared between different objects. For instance, we can share the `served_handler` we created above between an {py:class}`~protuna.optim.samplers.mlm.MLMSampler` and a {py:class}`~protuna.optim.predictors.zeroshot.MaskedMarginalPredictor`:

```py
from protuna.data import tokenizer
from protuna.optim import predictors

wt_tokens = tokenizer("MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTY")

predictor = predictors.MaskedMarginalPredictor(
    wt_tokens=wt_tokens, model=served_handler, batch_size=128
)
```

With {py:class}`~protuna.optim.predictors.zeroshot.MaskedMarginalPredictor`, we can also take advantage of a different {py:type}`~protuna.inference.serve.pool.Router` to optimize computation. For most campaigns in the initial rounds (which is where one might choose to use such a zero-shot predictor), we'll typically limit ourselves to a few mutations from the initial wild-type sequence. To achieve this, we sample smaller subspaces of the initial {py:class}`~protuna.optim.spaces.search.SearchSpace` using a suitable {py:class}`~protuna.optim.spaces.subspaces.SearchSpaceSampler` and therefore need to evaluate several mutants sharing the same set of mutated positions. Within each batch processed by {py:class}`~protuna.optim.predictors.zeroshot.MaskedMarginalPredictor`, candidates with the same mutated positions share a single masked model input, so the model evaluates that input only once. This deduplication applies within a batch; model logits are not cached across batches. Routing candidates with the same mutated positions to the same replica increases the opportunity to share this computation. This is what a {py:class}`~protuna.inference.serve.routing.MaskAffinityRouter` enables.

We can set up a distributed {py:class}`~protuna.optim.predictors.zeroshot.MaskedMarginalPredictor` with a {py:class}`~protuna.inference.serve.routing.MaskAffinityRouter` in the following way:

```py
local_predictor = predictors.MaskedMarginalPredictor(
    wt_tokens=wt_tokens, model=local_handler
)
served_predictor = inference.serve_routed_pool(
    local_predictor,
    num_replicas=5,
    max_batch_size=4096,
    actor_options={"num_gpus": 1},
    router=inference.MaskAffinityRouter(wt_tokens),
)
```

Submitting candidates with the same mutated positions consecutively helps the {py:class}`~protuna.inference.serve.routing.MaskAffinityRouter` group them into the same replica's batches. Candidates grouped into the same predictor batch therefore share one masked model input.

## Distributing with placement groups

Since samplers, constraints, and predictors run sequentially in a pipeline, they can often share the same GPU resources. However, assigning fractional GPU resources to replicas alone does not guarantee the intended placement.

For example, suppose we want to distribute a sampler and an objective across two GPUs, with one replica of each handler sharing each GPU. We might try the following:

```py
served_sampler = inference.serve_routed_pool(
    sampler,
    num_replicas=2,
    max_batch_size=128,
    actor_options={"num_gpus": 0.5},
)

served_objective = inference.serve_routed_pool(
    objective,
    num_replicas=2,
    max_batch_size=256,
    actor_options={"num_gpus": 0.5},
)
```

This configuration allows each replica to share a GPU, but it does not specify which replicas should share one. Both sampler replicas could end up on one GPU, with both objective replicas on the other.

To control placement, we can define a Ray {py:class}`~ray.util.placement_group.PlacementGroup` with two resource bundles, each reserving one GPU and one CPU. We then assign one sampler replica and one objective replica to each bundle using a {py:class}`~ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy`:

```py
import ray
from ray.util import scheduling_strategies


pg = ray.util.placement_group([{"GPU": 1, "CPU": 1}] * 2, strategy="PACK")
ray.get(pg.ready(), timeout=10)

bundle1 = scheduling_strategies.PlacementGroupSchedulingStrategy(pg, 0)
bundle2 = scheduling_strategies.PlacementGroupSchedulingStrategy(pg, 1)

served_sampler = inference.serve_routed_pool(
    [
        inference.ReplicaGroup(
            sampler,
            num_replicas=1,
            max_batch_size=128,
            actor_options={
                "num_gpus": 0.5,
                "num_cpus": 0,
                "scheduling_strategy": bundle1,
            },
        ),
        inference.ReplicaGroup(
            sampler,
            num_replicas=1,
            max_batch_size=128,
            actor_options={
                "num_gpus": 0.5,
                "num_cpus": 0,
                "scheduling_strategy": bundle2,
            },
        ),
    ],
)

served_objective = inference.serve_routed_pool(
    [
        inference.ReplicaGroup(
            objective,
            num_replicas=1,
            max_batch_size=256,
            actor_options={
                "num_gpus": 0.5,
                "num_cpus": 0,
                "scheduling_strategy": bundle1,
            },
        ),
        inference.ReplicaGroup(
            objective,
            num_replicas=1,
            max_batch_size=256,
            actor_options={
                "num_gpus": 0.5,
                "num_cpus": 0,
                "scheduling_strategy": bundle2,
            },
        ),
    ],
)
```

Each GPU now hosts one sampler replica and one objective replica, so both handlers are distributed across the two GPUs. Each replica reserves half a GPU, allowing the sampler and objective actors to share the resources reserved by their bundle.

The same pattern extends to three handlers: keep the two placement group bundles, assign one replica of each handler to each bundle, and set `"num_gpus"` to `1 / 3` for each replica group.

For more details, see [Ray's placement group guide](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html).
