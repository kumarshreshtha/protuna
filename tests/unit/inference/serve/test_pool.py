from collections.abc import Callable, Sequence
from unittest import mock

import pytest
import ray
import torch

from protuna.inference.serve import pool as serving_pool
from protuna.typing import SequenceLike
from tests.unit.inference import _helpers

# Long enough that a batch never fires on the timer while the test is still
# submitting, so batch sizes in the assertions are deterministic.
_MAX_WAIT_MS = 500.0


class _StubPool(serving_pool.ModelPool[int, int, ray.ObjectRef]):
    """Concrete `ModelPool` that never talks to Ray."""

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    def submit(self, requests: SequenceLike[int]) -> ray.ObjectRef:
        raise NotImplementedError

    def resolve(self, reference: ray.ObjectRef) -> list[int]:
        raise NotImplementedError

    def wait(
        self, references: list[ray.ObjectRef], *, num_returns: int = 1
    ) -> tuple[list[ray.ObjectRef], list[ray.ObjectRef]]:
        raise NotImplementedError


@pytest.mark.parametrize("inflight_batches", [1, 2, 4])
def test_model_pool_counts_parallel_requests_separately_from_inflight_batches(
    inflight_batches: int,
) -> None:
    """Keep the batch limit independent of the combined replica capacity."""
    groups = [
        serving_pool.ReplicaGroup(
            _helpers._RecordingHandler(), max_batch_size=4, num_replicas=2
        ),
        serving_pool.ReplicaGroup(
            _helpers._RecordingHandler(), max_batch_size=3, num_replicas=1
        ),
    ]

    pool = _StubPool(groups, inflight_batches=inflight_batches)

    assert pool.num_parallel == 11
    assert pool.max_inflight == inflight_batches


def test_replica_groups_do_not_share_actor_options() -> None:
    """Each group gets its own actor options mapping."""
    first = serving_pool.ReplicaGroup(_helpers._RecordingHandler())
    second = serving_pool.ReplicaGroup(_helpers._RecordingHandler())

    first.actor_options["num_cpus"] = 2.0

    assert second.actor_options == {}


def test_pool_returns_one_response_per_request_in_order(
    serve_pool: Callable[..., serving_pool.ModelPool],
    pool_kind: str,
) -> None:
    """Calling a pool preserves the order of the requests it was given."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._RecordingHandler(), actor_options=_helpers._TINY_ACTOR
        ),
        pool_kind=pool_kind,
    )

    assert pool([1, 2, 3]) == [2, 4, 6]


def test_pool_batches_requests_up_to_the_group_batch_size(
    serve_pool: Callable[..., serving_pool.ModelPool],
    pool_kind: str,
) -> None:
    """Concurrent requests are coalesced into batches the handler sees at once."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._BatchSizeHandler(),
            max_batch_size=2,
            max_wait_ms=_MAX_WAIT_MS,
            actor_options=_helpers._TINY_ACTOR,
        ),
        pool_kind=pool_kind,
    )

    assert pool([1, 2, 3, 4]) == [(1, 2), (2, 2), (3, 2), (4, 2)]


def test_pool_streams_every_request_with_its_response(
    serve_pool: Callable[..., serving_pool.ModelPool],
    pool_kind: str,
) -> None:
    """Streaming pairs each response back with the request that produced it."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._RecordingHandler(), actor_options=_helpers._TINY_ACTOR
        ),
        pool_kind=pool_kind,
    )

    streamed = pool.stream(iter([(0, 1), (1, 2), (2, 3)]))

    assert sorted(streamed) == [(0, 1, 2), (1, 2, 4), (2, 3, 6)]


def test_pool_streams_more_requests_than_fit_in_the_inflight_window(
    serve_pool: Callable[..., serving_pool.ModelPool],
    pool_kind: str,
) -> None:
    """Requests beyond the inflight window are submitted as earlier ones finish."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._RecordingHandler(), actor_options=_helpers._TINY_ACTOR
        ),
        pool_kind=pool_kind,
        inflight_batches=2,
    )
    assert pool.max_inflight == 2

    streamed = pool.stream(iter([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]))

    assert sorted(streamed) == [
        (0, 1, 2),
        (1, 2, 4),
        (2, 3, 6),
        (3, 4, 8),
        (4, 5, 10),
    ]


def test_pool_defers_loading_until_requested(
    serve_pool: Callable[..., serving_pool.ModelPool],
    pool_kind: str,
) -> None:
    """Keep a served handler unloaded until the caller explicitly loads it."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._LoadableHandler(), actor_options=_helpers._TINY_ACTOR
        ),
        pool_kind=pool_kind,
    )

    assert pool([1]) == ["unloaded"]


@pytest.mark.parametrize(
    "device",
    [None, torch.device("cuda:3"), [torch.device("cuda:0"), torch.device("cuda:1")]],
)
def test_pool_loads_the_handler_inside_its_worker(
    serve_pool: Callable[..., serving_pool.ModelPool],
    pool_kind: str,
    device: torch.device | Sequence[torch.device] | None,
) -> None:
    """Let remote handlers choose devices independently of the caller's device."""
    handler = _helpers._LoadableHandler()
    pool = serve_pool(
        serving_pool.ReplicaGroup(handler, actor_options=_helpers._TINY_ACTOR),
        pool_kind=pool_kind,
    )

    pool.load(device)

    assert pool([1]) == ["auto"]
    assert handler.device == "unloaded"


def test_pool_releases_the_handler_inside_its_worker(
    serve_pool: Callable[..., serving_pool.ModelPool],
    pool_kind: str,
) -> None:
    """Release the remote handler while keeping the pool alive."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._LoadableHandler(), actor_options=_helpers._TINY_ACTOR
        ),
        pool_kind=pool_kind,
    )
    pool.load()

    pool.release()

    assert pool([1]) == ["released"]


def test_pool_can_reload_after_release(
    serve_pool: Callable[..., serving_pool.ModelPool],
    pool_kind: str,
) -> None:
    """Reuse the same replicas for another loading cycle."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._LoadableHandler(), actor_options=_helpers._TINY_ACTOR
        ),
        pool_kind=pool_kind,
    )
    pool.load()
    pool.release()

    pool.load()

    assert pool([1]) == ["auto"]


def test_routed_pool_spreads_requests_across_replicas(
    serve_pool: Callable[..., serving_pool.ModelPool],
) -> None:
    """A round-robin router reaches every replica in the group."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._PidHandler(), num_replicas=2, actor_options=_helpers._TINY_ACTOR
        ),
        pool_kind="routed",
    )

    assert len(set(pool([1, 2, 3, 4]))) == 2


def test_pool_propagates_a_handler_failure(
    pool_kind: str,
    serve_pool: Callable[..., serving_pool.ModelPool],
) -> None:
    """An exception raised inside the handler surfaces to the caller."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._FailingHandler(), actor_options=_helpers._TINY_ACTOR
        ),
        pool_kind=pool_kind,
    )

    with pytest.raises(RuntimeError, match="handler exploded"):
        pool([1])


def test_pool_rejects_a_handler_that_drops_responses(
    pool_kind: str,
    serve_pool: Callable[..., serving_pool.ModelPool],
) -> None:
    """Returning fewer responses than requests is reported, not silently zipped."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._ShortHandler(),
            max_batch_size=2,
            max_wait_ms=_MAX_WAIT_MS,
            actor_options=_helpers._TINY_ACTOR,
        ),
        pool_kind=pool_kind,
    )

    with pytest.raises(RuntimeError, match="forward returned 1 for 2 batch"):
        pool([1, 2])


@pytest.mark.parametrize(
    ("requests", "expected"),
    [([], []), ([3], [6]), ([3, 1, 3, 2, 5], [6, 2, 6, 4, 10])],
)
def test_pool_submit_returns_one_ref_for_ordered_responses(
    serve_pool: Callable[..., serving_pool.ModelPool],
    pool_kind: str,
    requests: list[int],
    expected: list[int],
) -> None:
    """Resolve an entire submission through one ref, including empty submissions."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._RecordingHandler(),
            num_replicas=2,
            max_batch_size=2,
            actor_options=_helpers._TINY_ACTOR,
        ),
        pool_kind=pool_kind,
    )

    ref = pool.submit(requests)

    assert pool.resolve(ref) == expected


def test_pool_stream_submits_parallel_slices_with_bounded_lookahead(
    serve_pool: Callable[..., serving_pool.ModelPool],
    pool_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound pending slices and replenish one slice as another completes."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._RecordingHandler(),
            max_batch_size=2,
            actor_options=_helpers._TINY_ACTOR,
        ),
        pool_kind=pool_kind,
        inflight_batches=2,
    )
    submitted: list[list[int]] = []
    submit = pool.submit

    def record_submission(
        requests: SequenceLike[int],
    ) -> ray.ObjectRef | serving_pool._RoutedReference:
        submitted.append(list(requests))
        return submit(requests)

    monkeypatch.setattr(pool, "submit", record_submission)
    requests = iter([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    streamed = pool.stream(requests)

    first = next(streamed)

    assert submitted == [[1, 2], [3, 4], [5, 6]]
    assert next(requests) == (6, 7)
    assert sorted([first, *streamed]) == [
        (0, 1, 2),
        (1, 2, 4),
        (2, 3, 6),
        (3, 4, 8),
        (4, 5, 10),
        (5, 6, 12),
    ]


def test_pool_stream_handles_a_partial_final_slice(
    serve_pool: Callable[..., serving_pool.ModelPool], pool_kind: str
) -> None:
    """Return every response when the input does not fill the final slice."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._RecordingHandler(),
            max_batch_size=2,
            actor_options=_helpers._TINY_ACTOR,
        ),
        pool_kind=pool_kind,
        inflight_batches=1,
    )

    assert sorted(pool.stream(iter([(0, 1), (1, 2), (2, 3)]))) == [
        (0, 1, 2),
        (1, 2, 4),
        (2, 3, 6),
    ]


def test_pool_stream_accepts_empty_input(
    serve_pool: Callable[..., serving_pool.ModelPool], pool_kind: str
) -> None:
    """Finish without submitting work when the request stream is empty."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._RecordingHandler(),
            actor_options=_helpers._TINY_ACTOR,
        ),
        pool_kind=pool_kind,
    )

    assert list(pool.stream(iter([]))) == []


def test_pool_accepts_more_requests_than_actor_submission_concurrency(
    serve_pool: Callable[..., serving_pool.ModelPool], pool_kind: str
) -> None:
    """Queue all requests within one actor call without exhausting concurrency."""
    pool = serve_pool(
        serving_pool.ReplicaGroup(
            _helpers._RecordingHandler(),
            max_batch_size=32,
            actor_options=_helpers._TINY_ACTOR,
        ),
        pool_kind=pool_kind,
    )

    assert pool([1] * 1001) == [2] * 1001


@pytest.mark.parametrize(("num_returns", "expected_calls"), [(1, 2), (2, 3)])
def test_routed_wait_counts_complete_submissions(
    num_returns: int, expected_calls: int
) -> None:
    """Wait for every child of the requested number of submissions."""
    pool = object.__new__(serving_pool.RoutedPool)
    a = ray.ObjectRef.from_random()
    b = ray.ObjectRef.from_random()
    c = ray.ObjectRef.from_random()
    first = serving_pool._RoutedReference((a, b), ((0,), (1,)), 2)
    second = serving_pool._RoutedReference((c,), ((0,),), 1)
    expected = {1: ([second], [first]), 2: ([first, second], [])}
    with mock.patch.object(
        ray,
        "wait",
        autospec=True,
        side_effect=[([a], [b, c]), ([c], [b]), ([b], [])],
    ) as wait:
        result = pool.wait([first, second], num_returns=num_returns)

    assert result == expected[num_returns]
    assert wait.call_args_list[:2] == [
        mock.call([a, b, c], num_returns=1),
        mock.call([b, c], num_returns=1),
    ]
    assert wait.call_count == expected_calls


def test_routed_wait_returns_empty_submissions_without_waiting() -> None:
    """Treat an empty submission as immediately complete."""
    pool = object.__new__(serving_pool.RoutedPool)
    child = ray.ObjectRef.from_random()
    pending = serving_pool._RoutedReference((child,), ((0,),), 1)
    empty = serving_pool._RoutedReference((), (), 0)
    with mock.patch.object(ray, "wait", autospec=True) as wait:
        result = pool.wait([pending, empty])

    assert result == ([empty], [pending])
    wait.assert_not_called()


def test_routed_wait_accepts_no_submissions() -> None:
    """Return empty partitions when there are no submissions."""
    pool = object.__new__(serving_pool.RoutedPool)

    assert pool.wait([]) == ([], [])


@pytest.mark.parametrize("num_returns", [0, -1, 2])
def test_routed_wait_clamps_return_counts(num_returns: int) -> None:
    """Clamp return counts to the available submissions."""
    pool = object.__new__(serving_pool.RoutedPool)
    reference = serving_pool._RoutedReference((), (), 0)

    with mock.patch.object(ray, "wait", autospec=True) as wait:
        result = pool.wait([reference], num_returns=num_returns)

    assert result == ([reference], [])
    wait.assert_not_called()
