from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Sequence
import dataclasses
import threading
from typing import Any, NamedTuple

import pytest
import torch

from protuna.inference import handlers
from protuna.inference.interfaces import mlm
from protuna.inference.serve import actors
from protuna.optim.predictors import base
from protuna.typing import SequenceLike
from tests.unit.inference import _helpers

# Short enough that a partial batch fires promptly, long enough that it does not
# fire while a test is still submitting into the same batch.
_TIMER_MS = 20.0

# A batch timer that will not fire for the lifetime of a test, so a partial batch
# stays partial until something else fires it.
_NEVER_MS = 60_000.0

# Long enough for _TIMER_MS to elapse and for a dispatch to reach its worker
# thread, short enough to keep the suite quick.
_IDLE_S = 0.05

# Worker coroutines that never resolve should fail the test rather than hang it.
_ASYNC_TIMEOUT_S = 10.0


@dataclasses.dataclass(frozen=True)
class _FakeTensor:
    """`_Transferable` stand-in that records the device it was moved to."""

    device: str

    def to(self, device: torch.device) -> _FakeTensor:
        return _FakeTensor(str(device))


class _Pair(NamedTuple):
    """Named tuple used to check that `_to_cpu` rebuilds it as its own type."""

    left: _FakeTensor
    right: _FakeTensor


class _TensorHandler(handlers.ModelFreeLocalHandler[int, _FakeTensor]):
    """Answers with an off-host response, so host transfer is observable."""

    def __call__(self, requests: SequenceLike[int]) -> list[_FakeTensor]:
        return [_FakeTensor("cuda:0") for _ in requests]


class _BlockingHandler(handlers.ModelFreeLocalHandler[int, int]):
    """Blocks inside the call until released, holding a batch in flight."""

    def __init__(self):
        super().__init__()
        self.batches: list[list[int]] = []
        self._released = threading.Event()

    def release(self) -> None:
        self._released.set()

    def __call__(self, requests: SequenceLike[int]) -> list[int]:
        self.batches.append(list(requests))
        self._released.wait(timeout=_ASYNC_TIMEOUT_S)
        return [request * 2 for request in requests]


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Drives a worker coroutine on a private event loop, bounded by a timeout."""
    return asyncio.run(asyncio.wait_for(coro, _ASYNC_TIMEOUT_S))


def _submit[RequestT, ResponseT](
    worker: actors.PushWorker[RequestT, ResponseT], request: RequestT
) -> Coroutine[Any, Any, ResponseT]:
    """Calls a worker's `submit` directly, unwrapped from Ray's method descriptor.

    `_PushWorker.submit` is a plain coroutine function until Ray turns the class
    into an actor, so these tests drive it in-process, on their own event loop.
    """
    return worker.submit(request)  # ty: ignore[call-non-callable]


async def _submit_all[RequestT, ResponseT](
    worker: actors.PushWorker[RequestT, ResponseT], requests: SequenceLike[RequestT]
) -> list[ResponseT]:
    """Submits every request concurrently, as a pool's replicas would."""
    return await asyncio.gather(*[_submit(worker, request) for request in requests])


async def _submit_all_errors[RequestT](
    worker: actors.PushWorker[RequestT, Any], requests: SequenceLike[RequestT]
) -> list[BaseException]:
    """Submits every request concurrently and collects what each one raised."""
    return await asyncio.gather(
        *[_submit(worker, request) for request in requests], return_exceptions=True
    )


async def _fill_after_idle(
    worker: actors.PushWorker[int, int],
    handler: _helpers._RecordingHandler,
    held: Sequence[int],
    trigger: int,
) -> tuple[list[list[int]], list[int]]:
    """Leaves `held` requests pending for a while, then fills the batch.

    Returns the batches the handler had seen while only `held` was pending, and
    the responses to every request once the batch fired.
    """
    pending = asyncio.gather(*[_submit(worker, request) for request in held])
    await asyncio.sleep(_IDLE_S)
    while_pending = list(handler.batches)
    filled = await _submit(worker, trigger)
    return while_pending, [*await pending, filled]


async def _fill_with_one_cancelled(
    worker: actors.PushWorker[int, int], kept: int, cancelled: int, trigger: int
) -> list[int]:
    """Cancels one pending request, then fills the batch with `trigger`."""
    kept_task = asyncio.create_task(_submit(worker, kept))
    cancelled_task = asyncio.create_task(_submit(worker, cancelled))
    await asyncio.sleep(0)
    cancelled_task.cancel()
    await asyncio.sleep(0)
    filled = await _submit(worker, trigger)
    return [await kept_task, filled]


async def _cancel_before_the_timer(
    worker: actors.PushWorker[int, int], request: int
) -> None:
    """Cancels the only pending request and waits for the batch timer to fire."""
    task = asyncio.create_task(_submit(worker, request))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(_IDLE_S)


async def _submit_while_blocked(
    worker: actors.PushWorker[int, int],
    handler: _BlockingHandler,
    requests: SequenceLike[int],
) -> tuple[int, list[int]]:
    """Submits every request while the handler is blocked, then releases it.

    Returns how many batches the worker had dispatched while it was blocked, and
    the responses to every request once it was let go.
    """
    pending = asyncio.gather(*[_submit(worker, request) for request in requests])
    await asyncio.sleep(_IDLE_S)
    dispatched = len(worker._task_refs)
    handler.release()
    return dispatched, await pending


def test_to_cpu_moves_a__Transferable_response() -> None:
    """A response that implements `to` is moved to the host."""
    assert actors._to_cpu(_FakeTensor("cuda:0")) == _FakeTensor("cpu")


def test_to_cpu_moves_a_real_tensor() -> None:
    """A torch tensor comes back on the CPU device."""
    assert actors._to_cpu(torch.zeros(2)).device.type == "cpu"


def test_to_cpu_recurses_into_a_mapping_and_returns_a_dict() -> None:
    """Mapping values are moved and the mapping is rebuilt as a plain dict."""
    moved = actors._to_cpu({"logits": _FakeTensor("cuda:1")})

    assert moved == {"logits": _FakeTensor("cpu")}


def test_to_cpu_recurses_into_a_list() -> None:
    """Every element of a sequence response is moved."""
    moved = actors._to_cpu([_FakeTensor("cuda:0"), _FakeTensor("cuda:1")])

    assert moved == [_FakeTensor("cpu"), _FakeTensor("cpu")]


def test_to_cpu_recurses_into_nested_containers() -> None:
    """_Transferables nested several containers deep are still moved."""
    moved = actors._to_cpu([{"a": [_FakeTensor("cuda:0")]}])

    assert moved == [{"a": [_FakeTensor("cpu")]}]


def test_to_cpu_rebuilds_a_namedtuple_as_its_own_type() -> None:
    """Named tuples keep their type, since they take fields positionally."""
    moved = actors._to_cpu(_Pair(_FakeTensor("cuda:0"), _FakeTensor("cuda:1")))

    assert moved == _Pair(_FakeTensor("cpu"), _FakeTensor("cpu"))


def test_to_cpu_keeps_a_plain_tuple_a_tuple() -> None:
    """A plain tuple response is rebuilt as a tuple, not a list."""
    moved = actors._to_cpu((_FakeTensor("cuda:0"),))

    assert moved == (_FakeTensor("cpu"),)


@pytest.mark.parametrize("response", ["ACDEF", b"ACDEF", 7, None])
def test_to_cpu_passes_through_responses_it_cannot_move(response: object) -> None:
    """Strings, bytes and non-_Transferable scalars are returned untouched."""
    assert actors._to_cpu(response) is response


def test_to_cpu_does_not_mutate_its_argument() -> None:
    """Containers are rebuilt, so the caller's response is left alone."""
    response = {"logits": _FakeTensor("cuda:0")}

    actors._to_cpu(response)

    assert response == {"logits": _FakeTensor("cuda:0")}


def test_push_worker_fires_a_full_batch_without_waiting_for_the_timer() -> None:
    """A batch that reaches its size goes to the handler immediately."""
    handler = _helpers._RecordingHandler()
    worker = actors.PushWorker(handler, max_batch_size=3, max_wait_ms=_NEVER_MS)

    responses = _run(_submit_all(worker, [1, 2, 3]))

    assert responses == [2, 4, 6]
    assert handler.batches == [[1, 2, 3]]


def test_push_worker_cancels_the_timer_when_a_batch_fills_early() -> None:
    """The pending timer is dropped once the batch fires on size."""
    worker = actors.PushWorker(
        _helpers._RecordingHandler(), max_batch_size=2, max_wait_ms=_NEVER_MS
    )

    _run(_submit_all(worker, [1, 2]))

    assert worker._timer is None


def test_push_worker_fires_a_partial_batch_when_the_timer_expires() -> None:
    """Fewer requests than the batch size still reach the handler, on the timer."""
    handler = _helpers._RecordingHandler()
    worker = actors.PushWorker(handler, max_batch_size=8, max_wait_ms=_TIMER_MS)

    responses = _run(_submit_all(worker, [1, 2, 3]))

    assert responses == [2, 4, 6]
    assert handler.batches == [[1, 2, 3]]


def test_push_worker_holds_a_partial_batch_until_it_fills() -> None:
    """Nothing reaches the handler while a partial batch waits out its timer."""
    handler = _helpers._RecordingHandler()
    worker = actors.PushWorker(handler, max_batch_size=3, max_wait_ms=_NEVER_MS)

    while_pending, responses = _run(_fill_after_idle(worker, handler, [1, 2], 3))

    assert while_pending == []
    assert responses == [2, 4, 6]


def test_push_worker_splits_requests_beyond_the_batch_size() -> None:
    """More requests than fit in one batch are dispatched as several."""
    handler = _helpers._RecordingHandler()
    worker = actors.PushWorker(handler, max_batch_size=2, max_wait_ms=_TIMER_MS)

    responses = _run(_submit_all(worker, [1, 2, 3, 4, 5]))

    assert responses == [2, 4, 6, 8, 10]
    assert handler.batches == [[1, 2], [3, 4], [5]]


@pytest.mark.parametrize(
    ("requests", "expected_dispatched", "expected_batches", "expected_responses"),
    [
        ([1, 2, 3], 1, [[1, 2], [3]], [2, 4, 6]),
        ([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]], [2, 4, 6, 8, 10]),
    ],
)
def test_push_worker_defers_an_expired_partial_batch_until_the_executor_is_idle(
    requests: list[int],
    expected_dispatched: int,
    expected_batches: list[list[int]],
    expected_responses: list[int],
) -> None:
    """Defer partial batches while full batches are running or queued."""
    handler = _BlockingHandler()
    worker = actors.PushWorker(handler, max_batch_size=2, max_wait_ms=_TIMER_MS)

    dispatched, responses = _run(_submit_while_blocked(worker, handler, requests))

    assert dispatched == expected_dispatched
    assert responses == expected_responses
    assert handler.batches == expected_batches


def test_push_worker_keeps_accumulating_requests_after_a_busy_timeout() -> None:
    """Include later arrivals in the partial batch deferred by a busy executor."""
    handler = _BlockingHandler()
    worker = actors.PushWorker(handler, max_batch_size=3, max_wait_ms=_TIMER_MS)

    async def scenario() -> tuple[int, list[int]]:
        pending = asyncio.gather(*[_submit(worker, r) for r in [1, 2, 3, 4]])
        await asyncio.sleep(_IDLE_S)
        later = asyncio.create_task(_submit(worker, 5))
        await asyncio.sleep(_IDLE_S)
        dispatched = len(worker._task_refs)
        handler.release()
        return dispatched, [*await pending, await later]

    dispatched, responses = _run(scenario())

    assert dispatched == 1
    assert responses == [2, 4, 6, 8, 10]
    assert handler.batches == [[1, 2, 3], [4, 5]]


def test_push_worker_drops_a_cancelled_request_from_the_batch() -> None:
    """A caller that went away is left out of the batch its request was in."""
    handler = _helpers._RecordingHandler()
    worker = actors.PushWorker(handler, max_batch_size=3, max_wait_ms=_NEVER_MS)

    responses = _run(_fill_with_one_cancelled(worker, kept=1, cancelled=2, trigger=3))

    assert responses == [2, 6]
    assert handler.batches == [[1, 3]]


def test_push_worker_skips_a_batch_of_only_cancelled_requests() -> None:
    """The timer firing on nothing but cancelled requests calls no handler."""
    handler = _helpers._RecordingHandler()
    worker = actors.PushWorker(handler, max_batch_size=3, max_wait_ms=_TIMER_MS)

    _run(_cancel_before_the_timer(worker, 1))

    assert handler.batches == []


def test_push_worker_defers_loading_until_requested() -> None:
    """Leave loading to the caller after constructing a worker."""
    handler = _helpers._LoadableHandler()

    actors.PushWorker(handler, max_batch_size=1)

    assert handler.device == "unloaded"


def test_push_worker_loads_the_handler_without_an_explicit_device() -> None:
    """Let the handler choose its device when the worker loads it."""
    handler = _helpers._LoadableHandler()
    worker = actors.PushWorker(handler, max_batch_size=1)

    worker.load()  # ty: ignore[call-non-callable]

    assert handler.device == "auto"


def test_push_worker_releases_the_handler() -> None:
    """Forward an explicit release to the handler."""
    handler = _helpers._LoadableHandler()
    worker = actors.PushWorker(handler, max_batch_size=1)
    worker.load()  # ty: ignore[call-non-callable]

    worker.release()  # ty: ignore[call-non-callable]

    assert handler.device == "released"


@pytest.mark.parametrize(
    ("responses_to_cpu", "expected"), [(True, "cpu"), (False, "cuda:0")]
)
def test_push_worker_moves_responses_to_the_host_when_asked(
    responses_to_cpu: bool, expected: str
) -> None:
    """Responses cross the Ray boundary on the host only when configured to."""
    worker = actors.PushWorker(
        _TensorHandler(), max_batch_size=1, responses_to_cpu=responses_to_cpu
    )

    responses = _run(_submit_all(worker, [1]))

    assert responses == [_FakeTensor(expected)]


def test_push_worker_fails_every_request_in_a_batch_when_the_handler_raises() -> None:
    """One handler failure is reported to each caller whose request was batched."""
    worker = actors.PushWorker(
        _helpers._FailingHandler(), max_batch_size=2, max_wait_ms=_TIMER_MS
    )

    errors = _run(_submit_all_errors(worker, [1, 2]))

    assert [str(error) for error in errors] == ["handler exploded", "handler exploded"]


def test_push_worker_rejects_a_handler_that_drops_responses() -> None:
    """Returning fewer responses than requests fails the batch, not the zip."""
    worker = actors.PushWorker(
        _helpers._ShortHandler(), max_batch_size=2, max_wait_ms=_TIMER_MS
    )

    errors = _run(_submit_all_errors(worker, [1, 2]))

    assert [str(error) for error in errors] == [
        "forward returned 1 for 2 batch",
        "forward returned 1 for 2 batch",
    ]


def test_broker_returns_responses_directly_in_submission_order() -> None:
    """Return responses in submission order when batches finish out of order."""

    async def scenario() -> list[int]:
        broker = actors.Broker[int, int]()
        first = asyncio.create_task(broker.submit(1))  # ty: ignore[call-non-callable]
        second = asyncio.create_task(broker.submit(2))  # ty: ignore[call-non-callable]
        batch = await broker.get_batch(2, 0.1)  # ty: ignore[call-non-callable]
        assert batch == [(0, 1), (1, 2)]
        broker.put_results([1], [4])  # ty: ignore[call-non-callable]
        broker.put_results([0], [2])  # ty: ignore[call-non-callable]
        return list(await asyncio.gather(first, second))

    assert _run(scenario()) == [2, 4]


def test_broker_propagates_a_batch_exception_to_every_submission() -> None:
    """Fail every pending submission when its batch fails."""

    async def scenario() -> list[object]:
        broker = actors.Broker[int, int]()
        first = asyncio.create_task(broker.submit(1))  # ty: ignore[call-non-callable]
        second = asyncio.create_task(broker.submit(2))  # ty: ignore[call-non-callable]
        await broker.get_batch(2, 0.1)  # ty: ignore[call-non-callable]
        broker.put_exception([0, 1], RuntimeError("batch failed"))  # ty: ignore[call-non-callable]
        return list(await asyncio.gather(first, second, return_exceptions=True))

    errors = _run(scenario())

    assert [str(error) for error in errors] == ["batch failed", "batch failed"]


@pytest.mark.parametrize(
    ("responses", "expected_shapes"),
    [
        (
            [
                base.Prediction(torch.tensor(1.0), torch.tensor(0.1)),
                base.Prediction(torch.tensor(2.0)),
            ],
            [torch.Size([3])],
        ),
        (
            [
                mlm.MLMOutput(
                    logits=torch.ones(2, 3), hidden_states=(torch.ones(2, 3),)
                ),
                mlm.MLMOutput(logits=torch.zeros(2, 3)),
            ],
            [torch.Size([3, 2, 3])],
        ),
    ],
)
def test_to_cpu_batch_transfers_structured_responses_together(
    responses: Sequence[base.Prediction] | Sequence[mlm.MLMOutput],
    expected_shapes: list[torch.Size],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use batched transfers instead of each response's individual transfer."""
    transferred_shapes: list[torch.Size] = []

    def record_transfer(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        assert device == torch.device("cpu")
        transferred_shapes.append(tensor.shape)
        return tensor

    monkeypatch.setattr(torch.Tensor, "to", record_transfer)

    moved = actors._to_cpu_batch(responses)

    assert transferred_shapes == expected_shapes
    assert len(moved) == 2
    assert moved[0] is not responses[0]


def test_to_cpu_batch_accepts_empty_input() -> None:
    """Do not try to stack an empty response batch."""
    assert actors._to_cpu_batch([]) == []


@pytest.mark.parametrize(
    ("requests", "expected_batches", "expected_responses"),
    [([], [], []), ([1, 2, 3, 4, 5], [[1, 2], [3, 4], [5]], [2, 4, 6, 8, 10])],
)
def test_push_worker_accepts_transport_batches_larger_than_model_batches(
    requests: list[int],
    expected_batches: list[list[int]],
    expected_responses: list[int],
) -> None:
    """Split a single transport submission into bounded handler calls."""
    handler = _helpers._RecordingHandler()
    worker = actors.PushWorker(handler, max_batch_size=2, max_wait_ms=_TIMER_MS)

    responses = _run(worker.submit_batch(requests))  # ty: ignore[call-non-callable]

    assert responses == expected_responses
    assert handler.batches == expected_batches


def test_broker_submit_batch_preserves_order_across_worker_results() -> None:
    """Regroup responses from separate worker batches in submission order."""

    async def scenario() -> list[int]:
        broker = actors.Broker[int, int]()
        pending = asyncio.create_task(broker.submit_batch([3, 1, 2]))  # ty: ignore[call-non-callable]
        batch = await broker.get_batch(3, 0.1)  # ty: ignore[call-non-callable]
        assert batch == [(0, 3), (1, 1), (2, 2)]
        broker.put_results([2], [4])  # ty: ignore[call-non-callable]
        broker.put_results([0, 1], [6, 2])  # ty: ignore[call-non-callable]
        return await pending

    assert _run(scenario()) == [6, 2, 4]


@pytest.mark.parametrize("responses_to_cpu", [False, True])
def test_push_worker_dispatches_tensor_batch_responses(responses_to_cpu: bool) -> None:
    """Resolve each submission with its row of a tensor response batch."""
    worker = actors.PushWorker(
        _helpers._TensorBatchHandler(),
        max_batch_size=2,
        responses_to_cpu=responses_to_cpu,
    )

    responses = _run(_submit_all(worker, [1, 2]))

    torch.testing.assert_close(torch.stack(responses), torch.tensor([2, 4]))


def test_to_cpu_batch_transfers_an_entire_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transfer tensor batches without splitting them into individual responses."""
    transferred_shapes: list[torch.Size] = []

    def record_transfer(tensor: torch.Tensor, device: str) -> torch.Tensor:
        assert device == "cpu"
        transferred_shapes.append(tensor.shape)
        return tensor

    monkeypatch.setattr(torch.Tensor, "to", record_transfer)
    responses = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    moved = actors._to_cpu_batch(responses)

    assert moved is responses
    assert transferred_shapes == [torch.Size([2, 2])]
