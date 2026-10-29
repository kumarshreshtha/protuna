from collections.abc import Callable

import pytest
import ray

from protuna.inference.serve import api
from protuna.inference.serve import pool as serving_pool
from protuna.inference.serve import routing
from tests.unit.inference import _helpers


@pytest.mark.parametrize("serve", [api.serve_queue_pool, api.serve_routed_pool])
def test_serve_rejects_replica_settings_alongside_a_replica_group(
    serve: Callable[..., serving_pool.ModelPool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replica settings belong to the group, so passing both is ambiguous."""
    monkeypatch.setattr(ray, "is_initialized", lambda: True)

    with pytest.raises(TypeError, match="configured on the ReplicaGroup itself"):
        serve(serving_pool.ReplicaGroup(_helpers._RecordingHandler()), num_replicas=2)


@pytest.mark.parametrize("serve", [api.serve_queue_pool, api.serve_routed_pool])
def test_serve_rejects_an_empty_group_sequence(
    serve: Callable[..., serving_pool.ModelPool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pool with no groups would have no workers to serve from."""
    monkeypatch.setattr(ray, "is_initialized", lambda: True)

    with pytest.raises(ValueError, match="needs at least one ReplicaGroup"):
        serve([])


@pytest.mark.parametrize("serve", [api.serve_queue_pool, api.serve_routed_pool])
def test_serve_rejects_a_bare_handler_without_replica_settings(
    serve: Callable[..., serving_pool.ModelPool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare handler carries no replica configuration of its own."""
    monkeypatch.setattr(ray, "is_initialized", lambda: True)

    with pytest.raises(TypeError, match="requires num_replicas and actor_options"):
        serve(_helpers._RecordingHandler(), num_replicas=1)


def test_serve_queue_pool_builds_a_queue_pool(
    serve_pool: Callable[..., serving_pool.ModelPool],
) -> None:
    """Create a shared-queue pool through its dedicated entry point."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._BatchSizeHandler(), actor_options=_helpers._TINY_ACTOR
        )
    )

    assert isinstance(pool, serving_pool.QueuePool)


def test_serve_routed_pool_builds_a_routed_pool_when_given_a_router(
    serve_pool: Callable[..., serving_pool.ModelPool],
) -> None:
    """Use the supplied router for routed dispatch."""
    router = routing.RoundRobinRouter()
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._BatchSizeHandler(), actor_options=_helpers._TINY_ACTOR
        ),
        pool_kind="routed",
        router=router,
    )

    assert isinstance(pool, serving_pool.RoutedPool)
    assert pool.router is router


@pytest.mark.parametrize("pool_kind", ["queue", "routed"])
def test_serve_wraps_a_bare_handler_in_a_single_group(
    pool_kind: str,
    serve_pool: Callable[..., serving_pool.ModelPool],
) -> None:
    """A bare handler is served with the replica settings passed alongside it."""
    pool = serve_pool(
        _helpers._BatchSizeHandler(),
        pool_kind=pool_kind,
        num_replicas=2,
        max_batch_size=3,
        actor_options=_helpers._TINY_ACTOR,
    )

    assert len(pool.groups) == 1
    assert pool.groups[0].num_replicas == 2
    assert pool.groups[0].max_batch_size == 3


def test_execute_parallel_calls_every_local_handler_with_the_same_requests() -> None:
    """Plain handlers are invoked directly, one result list per handler."""
    handlers = [_helpers._RecordingHandler(), _helpers._BatchSizeHandler()]

    results = api.execute_parallel(handlers, [1, 2])

    assert results == [[2, 4], [(1, 2), (2, 2)]]


def test_execute_parallel_fans_pools_out_before_resolving_any(
    serve_pool: Callable[..., serving_pool.ModelPool],
) -> None:
    """Pools are all submitted to first, so their work overlaps."""
    pools = [
        serve_pool(
            serving_pool.ReplicaGroup(
                _helpers._RecordingHandler(), actor_options=_helpers._TINY_ACTOR
            )
        )
        for _ in range(2)
    ]

    results = api.execute_parallel(pools, [1, 2])

    assert results == [[2, 4], [2, 4]]


def test_execute_parallel_collects_batches_from_different_pool_kinds(
    serve_pool: Callable[..., serving_pool.ModelPool],
) -> None:
    """Collect ordered response batches from both queue and routed pools."""
    pools = [
        serve_pool(
            serving_pool.ReplicaGroup(
                _helpers._RecordingHandler(), actor_options=_helpers._TINY_ACTOR
            )
        ),
        serve_pool(
            serving_pool.ReplicaGroup(
                _helpers._RecordingHandler(), actor_options=_helpers._TINY_ACTOR
            ),
            pool_kind="routed",
            router=routing.RoundRobinRouter(),
        ),
    ]

    results = api.execute_parallel(pools, [1, 2])

    assert results == [[2, 4], [2, 4]]


def test_serve_routed_pool_defaults_to_round_robin(
    serve_pool: Callable[..., serving_pool.ModelPool],
) -> None:
    """Select round-robin routing when no router is supplied."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._PidHandler(), num_replicas=2, actor_options=_helpers._TINY_ACTOR
        ),
        pool_kind="routed",
    )

    assert isinstance(pool, serving_pool.RoutedPool)
    assert isinstance(pool.router, routing.RoundRobinRouter)
    assert len(set(pool([1, 2, 3, 4]))) == 2


@pytest.mark.parametrize("pool_kind", ["queue", "routed"])
def test_serve_passes_the_inflight_batches_to_the_pool(
    serve_pool: Callable[..., serving_pool.ModelPool], pool_kind: str
) -> None:
    """Forward the batch limit independently of the pool's request capacity."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._RecordingHandler(),
            max_batch_size=3,
            num_replicas=2,
            actor_options=_helpers._TINY_ACTOR,
        ),
        pool_kind=pool_kind,
        inflight_batches=2,
    )

    assert pool.num_parallel == 6
    assert pool.max_inflight == 2


def test_serve_queue_pool_accepts_broker_configuration(
    serve_pool: Callable[..., serving_pool.ModelPool],
) -> None:
    """Serve requests with custom broker resources and submission concurrency."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._RecordingHandler(), actor_options=_helpers._TINY_ACTOR
        ),
        broker_options={"num_cpus": 0},
        max_concurrent_submissions=2,
    )

    assert pool([1, 2, 3, 4]) == [2, 4, 6, 8]
