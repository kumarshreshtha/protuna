"""Ray actors that manage the transfer and processing of requests and responses."""

import asyncio
from collections.abc import Mapping, Sequence
import concurrent.futures
import itertools
import logging
from typing import cast, Protocol, runtime_checkable, Self

import ray
import torch

from protuna.inference import handlers
from protuna.typing import SequenceLike
from protuna.utils import devices

logger = logging.getLogger(__name__)


@runtime_checkable
class _Transferable(Protocol):
    """Responses that can be moved between devices."""

    def to(self, device: torch.device) -> Self:
        """Return this response placed on *device*."""
        ...


@runtime_checkable
class _BatchTransferable(Protocol):
    """Responses that support transferring a batch together."""

    @classmethod
    def to_device_batch(
        cls, responses: Sequence[Self], device: torch.device
    ) -> Sequence[Self]:
        """Return responses on the destination device in input order.

        Args:
            responses: Responses of this type to transfer.
            device: Destination device.
        """
        ...


def _to_cpu[ResponseT](response: ResponseT) -> ResponseT:
    if isinstance(response, _Transferable):
        return response.to(torch.device("cpu"))
    if isinstance(response, Mapping):
        return cast("ResponseT", {k: _to_cpu(v) for k, v in response.items()})
    if not isinstance(response, Sequence) or isinstance(response, str | bytes):
        return response
    moved = [_to_cpu(item) for item in response]
    if isinstance(response, tuple):
        # namedtuples take their fields positionally rather than as an iterable
        rebuilt = (
            type(response)(*moved) if hasattr(response, "_fields") else tuple(moved)
        )
        return cast("ResponseT", rebuilt)
    return cast("ResponseT", moved)


def _to_cpu_batch[ResponseT](
    responses: SequenceLike[ResponseT],
) -> SequenceLike[ResponseT]:
    if len(responses) == 0:
        return []
    if isinstance(responses, torch.Tensor):
        return cast("SequenceLike[ResponseT]", responses.to("cpu"))
    first = responses[0]
    if isinstance(first, _BatchTransferable) and all(
        type(response) is type(first) for response in responses
    ):
        return cast(
            "Sequence[ResponseT]",
            first.to_device_batch(
                cast("Sequence[_BatchTransferable]", responses), torch.device("cpu")
            ),
        )
    if all(isinstance(response, torch.Tensor) for response in responses):
        return cast(
            "Sequence[ResponseT]",
            devices.to_device_batch(
                cast("Sequence[torch.Tensor]", responses), torch.device("cpu")
            ),
        )
    return _to_cpu(responses)


class PushWorker[RequestT, ResponseT]:
    """Dynamically batch submitted requests and process them with the handler.

    The worker processes batches as they fill or when the wait limit expires. Each
    submission gets it's own future which is resolved when the results from the
    handler arrive.
    """

    def __init__(
        self,
        handler: handlers.Handler[RequestT, ResponseT],
        *,
        max_batch_size: int,
        max_wait_ms: float = 10,
        responses_to_cpu: bool = True,
    ):
        """Configure request batching.

        Args:
            handler: Handler that processes each batch.
            max_batch_size: Maximum requests in a batch.
            max_wait_ms: Milliseconds to wait before processing a partial batch.
            responses_to_cpu: Move response tensors to the CPU before returning them.
        """
        super().__init__()
        self.handler = handler
        self.pending: list[tuple[RequestT, asyncio.Future[ResponseT]]] = []
        self.max_wait = max_wait_ms / 1000
        self._max_batch_size = max_batch_size
        self._responses_to_cpu = responses_to_cpu
        self._timer = None
        self._task_refs: set[asyncio.Task[None]] = set()
        self._thread_executor = None
        self._processing = 0
        self._flush_due = False
        logger.debug(
            "PushWorker configured with handler=%s, max_batch_size=%s, "
            "max_wait_ms=%s, responses_to_cpu=%s",
            type(handler).__name__,
            max_batch_size,
            max_wait_ms,
            responses_to_cpu,
        )

    @ray.method
    def load(self) -> None:
        """Load the handler on its default device."""
        logger.debug("PushWorker loading handler %s", type(self.handler).__name__)
        self.handler.load()
        logger.debug("PushWorker handler loaded")

    @ray.method
    def release(self) -> None:
        """Release the resources held by the handler."""
        logger.debug("PushWorker releasing handler %s", type(self.handler).__name__)
        self.handler.release()
        logger.debug("PushWorker handler released")

    @ray.method
    async def submit(self, request: RequestT) -> ResponseT:
        """Queue a request and await its response.

        Args:
            request: Request to include in the next available batch.

        Returns:
            The handler response for the request.
        """
        return await self._enqueue(request)

    @ray.method
    async def submit_batch(self, requests: SequenceLike[RequestT]) -> list[ResponseT]:
        """Queue requests together and return their responses in input order.

        Args:
            requests: Requests to enqueue for processing.

        Returns:
            Handler responses in input order.
        """
        return list(await asyncio.gather(*(self._enqueue(r) for r in requests)))

    def _enqueue(self, request: RequestT) -> asyncio.Future[ResponseT]:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.pending.append((request, future))
        if len(self.pending) >= self._max_batch_size:
            self._process_batch()
        elif self._timer is None and not self._flush_due:
            self._timer = loop.call_later(self.max_wait, self._process_if_pending)
        return future

    def _process_if_pending(self) -> None:
        self._timer = None
        self._flush_due = bool(self.pending)
        if self._flush_due and self._processing == 0:
            self._process_batch()
        elif self._flush_due:
            logger.debug(
                "PushWorker deferring %s pending requests "
                "while %s batches are in flight",
                len(self.pending),
                self._processing,
            )

    def _process_batch(self) -> None:
        self._flush_due = False
        if self._timer is not None:  # batch filled before max_wait
            self._timer.cancel()
            self._timer = None
        batch, self.pending = self.pending, []
        batch = [(r, f) for r, f in batch if not f.cancelled()]
        if not batch:
            return
        logger.debug(
            "PushWorker dispatching batch of %s/%s requests",
            len(batch),
            self._max_batch_size,
        )
        self._processing += 1
        task = asyncio.get_running_loop().create_task(self._dispatch(batch))
        self._task_refs.add(task)
        task.add_done_callback(self._task_refs.discard)

    async def _dispatch(
        self, batch: list[tuple[RequestT, asyncio.Future[ResponseT]]]
    ) -> None:
        if self._thread_executor is None:
            self._thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        requests = [r for r, _ in batch]
        try:
            outputs = await asyncio.get_running_loop().run_in_executor(
                self._thread_executor, self.handler, requests
            )
            if len(outputs) != len(requests):
                raise RuntimeError(
                    f"forward returned {len(outputs)} for {len(requests)} batch"
                )
        except Exception as e:
            logger.exception(
                "PushWorker failed to process batch of %s requests", len(batch)
            )
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)
        else:
            if self._responses_to_cpu:
                outputs = _to_cpu_batch(outputs)
            for (_, future), result in zip(batch, outputs, strict=True):
                if not future.done():
                    future.set_result(result)
            logger.debug("PushWorker completed batch of %s requests", len(batch))
        finally:
            self._processing -= 1
            if self._processing == 0 and self._flush_due:
                self._process_batch()


class Broker[RequestT, ResponseT]:
    """Coordinate a shared request queue for :class:`PullWorker` instances.

    Workers retrieve batches as they become available to process them. The broker
    routes returned responses or exceptions back to the waiting submitters.
    """

    def __init__(self):
        """Initialize an empty request queue and pending response collection."""
        self._futures: dict[int, asyncio.Future[ResponseT]] = {}
        self.q: asyncio.Queue[tuple[int, RequestT]] = asyncio.Queue()
        self._counter = itertools.count()
        self._loop = asyncio.get_running_loop()
        logger.debug("Broker initialized")

    @ray.method
    async def get_batch(
        self, max_batch_size: int, max_wait: float
    ) -> list[tuple[int, RequestT]]:
        """Wait for a request, then collect a batch within the time limit.

        Args:
            max_batch_size: Maximum requests to return.
            max_wait: Seconds to wait for more requests after receiving the first.

        Returns:
            Pairs of request identifiers and requests in queue order.
        """
        # block until the first batch element is available
        batch = [await self.q.get()]
        loop = asyncio.get_running_loop()
        end_time = loop.time() + max_wait
        while len(batch) < max_batch_size:
            timeout = end_time - loop.time()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self.q.get(), timeout))
            except TimeoutError:
                break
        logger.debug(
            "Returning batch of %s/%s requests; %s requests remain queued",
            len(batch),
            max_batch_size,
            self.q.qsize(),
        )
        return batch

    @ray.method
    def put_exception(self, ref_ids: Sequence[int], exception: Exception) -> None:
        """Fail each pending request in a batch with the supplied exception.

        Args:
            ref_ids: Request identifiers returned by :meth:`get_batch`.
            exception: Exception to raise for each pending request.
        """
        for rid in ref_ids:
            self._futures.pop(rid).set_exception(exception)

    @ray.method
    def put_results(
        self, ref_ids: Sequence[int], responses: SequenceLike[ResponseT]
    ) -> None:
        """Resolve pending requests with their corresponding responses.

        Args:
            ref_ids: Request identifiers returned by :meth:`get_batch`.
            responses: Responses in the same order as the identifiers.
        """
        for rid, response in zip(ref_ids, responses, strict=True):
            self._futures.pop(rid).set_result(response)
        logger.debug(
            "Resolved %s requests; %s remain pending", len(ref_ids), len(self._futures)
        )

    def _enqueue(self, request: RequestT) -> asyncio.Future[ResponseT]:
        future = asyncio.get_running_loop().create_future()
        ref_id = next(self._counter)
        self._futures[ref_id] = future
        self.q.put_nowait((ref_id, request))
        logger.debug("Broker queued request %s; queue size=%s", ref_id, self.q.qsize())
        return future

    async def _submit(self, request: RequestT) -> ResponseT:
        return await self._enqueue(request)

    async def _submit_batch(self, requests: SequenceLike[RequestT]) -> list[ResponseT]:
        return list(await asyncio.gather(*(self._enqueue(r) for r in requests)))

    @ray.method(concurrency_group="pool")
    async def submit(self, request: RequestT) -> ResponseT:
        """Enqueue a request and await its response.

        Args:
            request: Request to make available to pull workers.

        Returns:
            The worker's handler response for the request.
        """
        pending = asyncio.run_coroutine_threadsafe(self._submit(request), self._loop)
        return await asyncio.wrap_future(pending)

    @ray.method(concurrency_group="pool")
    async def submit_batch(self, requests: SequenceLike[RequestT]) -> list[ResponseT]:
        """Queue requests together and return their responses in input order.

        Args:
            requests: Requests to make available to pull workers.

        Returns:
            Worker handler responses in input order.
        """
        pending = asyncio.run_coroutine_threadsafe(
            self._submit_batch(requests), self._loop
        )
        return await asyncio.wrap_future(pending)


class PullWorker[RequestT, ResponseT]:
    """Process batches by pulling from a shared queue managed by :class:`Broker`.

    The processing loop pulls batches from a :class:`Broker` and sends responses
    or exceptions back to it. Multiple workers can consume the same queue, taking
    on work as they become available, until asked to stop.
    """

    def __init__(
        self,
        handler: handlers.Handler[RequestT, ResponseT],
        broker: Broker[RequestT, ResponseT],
        *,
        max_batch_size: int,
        max_wait_ms: float = 10,
        responses_to_cpu: bool = True,
    ):
        """Configure the handler and shared request broker.

        Args:
            handler: Handler that processes each batch.
            broker: Ray broker actor supplying requests and accepting results.
            max_batch_size: Maximum requests to process in one batch.
            max_wait_ms: Milliseconds to wait for more requests after the first.
            responses_to_cpu: Move response tensors to the CPU before returning
                them to the broker.
        """
        self.handler = handler
        self._broker = broker
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000
        self._responses_to_cpu = responses_to_cpu
        self._stop = False
        self._thread_executor = None
        logger.debug(
            "PullWorker configured with handler=%s, max_batch_size=%s, "
            "max_wait_ms=%s, responses_to_cpu=%s",
            type(handler).__name__,
            max_batch_size,
            max_wait_ms,
            responses_to_cpu,
        )

    async def _resolve_batch(
        self,
    ) -> tuple[tuple[int, ...], tuple[RequestT]]:
        batch = await self._broker.get_batch.remote(self.max_batch_size, self.max_wait)
        rids, requests = zip(*batch, strict=True)
        return rids, requests

    @ray.method
    async def run(self) -> None:
        """Process broker batches until a stop request is observed."""
        logger.info("PullWorker starting processing loop")
        if self._thread_executor is None:
            self._thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        next_batch = asyncio.create_task(self._resolve_batch())
        while not self._stop:
            rids, requests = await next_batch
            next_batch = asyncio.create_task(self._resolve_batch())
            logger.debug(
                "PullWorker processing batch of %s/%s requests",
                len(requests),
                self.max_batch_size,
            )
            try:
                responses = await asyncio.get_running_loop().run_in_executor(
                    self._thread_executor, self.handler, requests
                )
                if len(responses) != len(requests):
                    raise RuntimeError(
                        f"forward returned {len(responses)} for {len(requests)} batch"
                    )
            except Exception as e:
                logger.exception(
                    "PullWorker failed to process batch of %s requests", len(requests)
                )
                self._broker.put_exception.remote(rids, e)
            else:
                if self._responses_to_cpu:
                    responses = _to_cpu_batch(responses)
                self._broker.put_results.remote(rids, responses)
                logger.debug(
                    "PullWorker dispatched results for %s requests", len(requests)
                )
        logger.info("PullWorker processing loop stopped")

    @ray.method
    def load(self) -> None:
        """Load the handler on its default device."""
        logger.debug("PullWorker loading handler %s", type(self.handler).__name__)
        self.handler.load()
        logger.debug("PullWorker handler loaded")

    @ray.method
    def release(self) -> None:
        """Release the resources held by the handler."""
        logger.debug("PullWorker releasing handler %s", type(self.handler).__name__)
        self.handler.release()
        logger.debug("PullWorker handler released")

    @ray.method
    def stop(self) -> None:
        """Request that the processing loop exit after its current iteration."""
        self._stop = True
        logger.debug("PullWorker stop requested")
