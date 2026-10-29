"""Core API for serving distributed handlers with ray."""

from collections.abc import Sequence
from typing import cast, overload

import ray

from protuna.inference import handlers
from protuna.inference.serve import pool
from protuna.inference.serve import routing
from protuna.typing import SequenceLike


# TODO: add multithread execution for local model
def execute_parallel[RequestT, ResponseT](
    models: Sequence[handlers.Handler[RequestT, ResponseT]],
    requests: SequenceLike[RequestT],
) -> list[SequenceLike[ResponseT]]:
    """Send the same requests to several handlers.

    When every handler is served all of them are submitted to
    before any responses are collected, so the pools run concurrently;
    otherwise the handlers are called one after another.

    Args:
        models: handlers to run.
        requests: Requests passed to every handler.

    Returns:
        One response sequence per handler, ordered as *models*.
    """
    if all(isinstance(model, pool.ModelPool) for model in models):
        refs_by_model = []
        for model in models:
            refs_by_model.append(model.submit(requests))  # ty: ignore[unresolved-attribute]
        return [
            model.resolve(ref)  # ty: ignore[unresolved-attribute]
            for model, ref in zip(models, refs_by_model, strict=True)
        ]
    return [model(requests) for model in models]


@overload
def serve_queue_pool[RequestT, ResponseT](
    target: handlers.Handler[RequestT, ResponseT],
    /,
    *,
    num_replicas: int,
    max_batch_size: int = 1,
    max_wait_ms: float = 10,
    responses_to_cpu: bool = True,
    actor_options: pool.RayActorOptions,
    broker_options: pool.RayActorOptions | None = None,
    max_concurrent_submissions: int = 1000,
    inflight_batches: int = 4,
    worker_ready_timeout: int = 60,
) -> pool.QueuePool[RequestT, ResponseT]: ...


@overload
def serve_queue_pool[RequestT, ResponseT](
    target: Sequence[pool.ReplicaGroup[RequestT, ResponseT]]
    | pool.ReplicaGroup[RequestT, ResponseT],
    /,
    *,
    broker_options: pool.RayActorOptions | None = None,
    max_concurrent_submissions: int = 1000,
    inflight_batches: int = 4,
    worker_ready_timeout: int = 60,
) -> pool.QueuePool[RequestT, ResponseT]: ...


def serve_queue_pool[RequestT, ResponseT](
    target: handlers.Handler[RequestT, ResponseT]
    | pool.ReplicaGroup[RequestT, ResponseT]
    | Sequence[pool.ReplicaGroup[RequestT, ResponseT]],
    /,
    *,
    num_replicas: int | None = None,
    max_batch_size: int = 1,
    max_wait_ms: float = 10,
    responses_to_cpu: bool = True,
    actor_options: pool.RayActorOptions | None = None,
    broker_options: pool.RayActorOptions | None = None,
    max_concurrent_submissions: int = 1000,
    inflight_batches: int = 4,
    worker_ready_timeout: int = 60,
) -> pool.QueuePool[RequestT, ResponseT]:
    """Serve a handler on Ray replicas that pull from a shared queue.

    Starts Ray if it is not already running. The returned pool is itself a
    :class:`protuna.optim.inference.handlers.Handler`, so call sites do not change.

    Args:
        target: The handler to serve, or one or more
            :class:`~protuna.inference.serve.pool.ReplicaGroup` instances
            describing how to serve it.
        num_replicas: Number of replicas to run. Bare handlers only.
        max_batch_size: Largest batch a replica forms. Bare handlers only.
        max_wait_ms: How long a replica waits for a batch to fill before firing
            a partial one. Bare handlers only.
        responses_to_cpu: Move tensors in responses to the host before they
            cross a Ray boundary. Bare handlers only.
        actor_options: Ray resources for each replica. Bare handlers only.
        broker_options: Ray actor options for the shared broker. ``None`` uses
            Ray defaults.
        max_concurrent_submissions: Maximum concurrent broker submission calls
            awaiting responses.
        inflight_batches: Maximum submitted slices kept in flight across the pool
            while streaming. Each slice holds up to the combined request capacity
            of all replicas.
        worker_ready_timeout: time in seconds to wait for the ray actors to be
            created.

    Returns:
        A :class:`~protuna.inference.serve.pool.QueuePool` that dispatches
        requests to available replicas.

    Raises:
        TypeError: If ``num_replicas`` or ``actor_options`` is passed alongside
            replica groups, or either is omitted for a bare handler.
        ValueError: If *target* is an empty sequence of groups.

    """
    if not ray.is_initialized():
        ray.init()
    groups: list[pool.ReplicaGroup[RequestT, ResponseT]]
    if isinstance(target, pool.ReplicaGroup | Sequence):
        if num_replicas is not None or actor_options is not None:
            raise TypeError(
                "num_replicas, max_batch_size and actor_options are configured on the"
                " ReplicaGroup itself; pass them only when serving a bare handler."
            )
        groups = (
            [target]
            if isinstance(target, pool.ReplicaGroup)
            else list(cast("Sequence[pool.ReplicaGroup[RequestT, ResponseT]]", target))
        )
        if not groups:
            raise ValueError("serve() needs at least one ReplicaGroup.")
    else:
        if num_replicas is None or actor_options is None:
            raise TypeError(
                "serving a bare handler requires num_replicas and actor_options."
            )
        groups = [
            pool.ReplicaGroup[RequestT, ResponseT](
                target,
                num_replicas=num_replicas,
                max_wait_ms=max_wait_ms,
                max_batch_size=max_batch_size,
                actor_options=actor_options,
                responses_to_cpu=responses_to_cpu,
            )
        ]
    return pool.QueuePool(
        groups,
        broker_options=broker_options,
        max_concurrent_submissions=max_concurrent_submissions,
        inflight_batches=inflight_batches,
        worker_ready_timeout=worker_ready_timeout,
    )


@overload
def serve_routed_pool[RequestT, ResponseT](
    target: handlers.Handler[RequestT, ResponseT],
    /,
    *,
    num_replicas: int,
    max_batch_size: int = 1,
    max_wait_ms: float = 10,
    responses_to_cpu: bool = True,
    actor_options: pool.RayActorOptions,
    router: pool.Router[RequestT] | None = None,
    inflight_batches: int = 4,
    worker_ready_timeout: int = 60,
) -> pool.RoutedPool[RequestT, ResponseT]: ...


@overload
def serve_routed_pool[RequestT, ResponseT](
    target: Sequence[pool.ReplicaGroup[RequestT, ResponseT]]
    | pool.ReplicaGroup[RequestT, ResponseT],
    /,
    *,
    router: pool.Router[RequestT] | None = None,
    inflight_batches: int = 4,
    worker_ready_timeout: int = 60,
) -> pool.RoutedPool[RequestT, ResponseT]: ...


def serve_routed_pool[RequestT, ResponseT](
    target: handlers.Handler[RequestT, ResponseT]
    | pool.ReplicaGroup[RequestT, ResponseT]
    | Sequence[pool.ReplicaGroup[RequestT, ResponseT]],
    /,
    *,
    num_replicas: int | None = None,
    max_batch_size: int = 1,
    max_wait_ms: float = 10,
    responses_to_cpu: bool = True,
    actor_options: pool.RayActorOptions | None = None,
    router: pool.Router[RequestT] | None = None,
    inflight_batches: int = 4,
    worker_ready_timeout: int = 60,
) -> pool.RoutedPool[RequestT, ResponseT]:
    """Serve a handler on Ray replicas that receive requests through a router.

    Starts Ray if it is not already running. The returned pool is itself a
    :class:`protuna.optim.inference.handlers.Handler`, so call sites do not change.

    Args:
        target: The handler to serve, or one or more
            :class:`~protuna.inference.serve.pool.ReplicaGroup` instances
            describing how to serve it.
        num_replicas: Number of replicas to run. Bare handlers only.
        max_batch_size: Largest batch a replica forms. Bare handlers only.
        max_wait_ms: How long a replica waits for a batch to fill before firing
            a partial one. Bare handlers only.
        responses_to_cpu: Move tensors in responses to the host before they
            cross a Ray boundary. Bare handlers only.
        actor_options: Ray resources for each replica. Bare handlers only.
        router: Pick the replica index for each request. ``None`` uses
            :class:`~protuna.inference.serve.routing.RoundRobinRouter`.
        inflight_batches: Maximum submitted slices kept in flight across the pool
            while streaming. Each slice holds up to the combined request capacity
            of all replicas.
        worker_ready_timeout: time in seconds to wait for the ray actors to be
            created.


    Returns:
        A :class:`~protuna.inference.serve.pool.RoutedPool` that dispatches
        each request to the selected replica.

    Raises:
        TypeError: If ``num_replicas`` or ``actor_options`` is passed alongside
            replica groups, or either is omitted for a bare handler.
        ValueError: If *target* is an empty sequence of groups.

    """
    if not ray.is_initialized():
        ray.init()
    groups: list[pool.ReplicaGroup[RequestT, ResponseT]]
    if isinstance(target, pool.ReplicaGroup | Sequence):
        if num_replicas is not None or actor_options is not None:
            raise TypeError(
                "num_replicas, max_batch_size and actor_options are configured on the"
                " ReplicaGroup itself; pass them only when serving a bare handler."
            )
        groups = (
            [target]
            if isinstance(target, pool.ReplicaGroup)
            else list(cast("Sequence[pool.ReplicaGroup[RequestT, ResponseT]]", target))
        )
        if not groups:
            raise ValueError("serve() needs at least one ReplicaGroup.")
    else:
        if num_replicas is None or actor_options is None:
            raise TypeError(
                "serving a bare handler requires num_replicas and actor_options."
            )
        groups = [
            pool.ReplicaGroup[RequestT, ResponseT](
                target,
                num_replicas=num_replicas,
                max_wait_ms=max_wait_ms,
                max_batch_size=max_batch_size,
                actor_options=actor_options,
                responses_to_cpu=responses_to_cpu,
            )
        ]
    if router is None:
        router = routing.RoundRobinRouter()
    return pool.RoutedPool(
        groups,
        router,
        inflight_batches=inflight_batches,
        worker_ready_timeout=worker_ready_timeout,
    )
