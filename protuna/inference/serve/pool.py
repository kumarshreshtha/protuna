"""Inference handlers for distributed serving."""

import abc
from collections.abc import Callable, Hashable, Iterator, Sequence
import dataclasses
import itertools
from typing import (
    Any,
    Literal,
    NotRequired,
    TypedDict,
)

import ray
from ray.util import scheduling_strategies
import torch

from protuna.inference import handlers
from protuna.inference.serve import actors
from protuna.typing import SequenceLike


class RayActorOptions(TypedDict):
    """Options for configuring the ray actor's resources.

    For option details, see Ray's official
    `ActorClass.options docs <https://docs.ray.io/en/latest/ray-core/api/doc/ray.actor.ActorClass.options.html#ray.actor.ActorClass.options>`_.
    """

    num_cpus: NotRequired[float]
    num_gpus: NotRequired[float]
    resources: NotRequired[dict[str, float]]
    label_selector: NotRequired[dict[str, str]]
    fallback_strategy: NotRequired[list[dict[str, Any]]]
    accelerator_type: NotRequired[str]
    memory: NotRequired[int]
    object_store_memory: NotRequired[int]
    max_restarts: NotRequired[int]
    max_task_retries: NotRequired[int]
    max_pending_calls: NotRequired[int]
    max_concurrency: NotRequired[int]
    allow_out_of_order_execution: NotRequired[bool]
    name: NotRequired[str]
    namespace: NotRequired[str]
    lifetime: NotRequired[Literal["detached"] | None]
    runtime_env: NotRequired[dict[str, Any]]
    scheduling_strategy: NotRequired[
        str
        | scheduling_strategies.NodeAffinitySchedulingStrategy
        | scheduling_strategies.PlacementGroupSchedulingStrategy
    ]
    enable_task_events: NotRequired[bool]


@dataclasses.dataclass
class ReplicaGroup[RequestT, ResponseT]:
    """A handler and it's replication configuration.

    Use one or more groups with :func:`~protuna.inference.serve.api.serve_queue_pool`
    or :func:`~protuna.inference.serve.api.serve_routed_pool`
    to run the same handler under different batching or resource configurations.

    Args:
        handler: The handler to replicate.
        max_batch_size: Largest batch a replica hands to the handler.
        num_replicas: Number of copies of *handler* to run.
        max_wait_ms: How long a replica waits for a batch to fill before
            firing a partial one.
        responses_to_cpu: Move tensors in responses to the host before they
            cross a Ray boundary.
        actor_options: Ray resources for each replica; see
            :class:`RayActorOptions`.
    """

    handler: handlers.Handler[RequestT, ResponseT]
    max_batch_size: int = 1
    num_replicas: int = 1
    max_wait_ms: float = 10.0
    responses_to_cpu: bool = True
    actor_options: RayActorOptions = dataclasses.field(default_factory=dict)


class ModelPool[RequestT, ResponseT, ReferenceT: Hashable](
    abc.ABC, handlers.Handler[RequestT, ResponseT]
):
    """Base class for distributing inference requests across a pool of Ray replicas.

    Requests are submitted asynchronously (:meth:`submit`) and collected later
    with :meth:`resolve`, which lets :meth:`stream` keep a bounded number of them
    in flight across the whole pool. Build a model pool using
    :func:`~protuna.inference.serve.api.serve_queue_pool` or
    :func:`~protuna.inference.serve.api.serve_routed_pool`.
    """

    def __init__(
        self,
        groups: Sequence[ReplicaGroup[RequestT, ResponseT]],
        inflight_batches: int = 4,
    ):
        """Configure the inflight requests in streaming mode and initialize the groups.

        Args:
            groups: Replica groups the pool dispatches to.
            inflight_batches: Maximum submitted slices kept in flight across the
                pool while streaming. Each slice holds up to ``num_parallel`` requests.
        """
        self.groups = groups
        self.num_parallel = sum(
            [group.num_replicas * group.max_batch_size for group in groups]
        )
        self.max_inflight = inflight_batches

    @abc.abstractmethod
    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Load all replicas, letting served handlers manage their own devices.

        Args:
            device: Accepted for compatibility with
                :class:`protuna.optim.inference.handlers.Handler` but ignored; served
                handlers manage their own devices.
        """
        ...

    @abc.abstractmethod
    def release(self) -> None:
        """Release GPU resources held by the handler."""
        ...

    @abc.abstractmethod
    def submit(self, requests: SequenceLike[RequestT]) -> ReferenceT:
        """Dispatch requests and return one ref to their responses in input order.

        Args:
            requests: Requests to dispatch.

        Returns:
            A reference to the submitted batch, collected with :meth:`resolve`.
        """
        ...

    @abc.abstractmethod
    def resolve(self, reference: ReferenceT) -> list[ResponseT]:
        """Resolve the submitted reference into a batch of responses.

        Args:
            reference: A batch reference returned by :meth:`submit`.

        Returns:
            Responses in the original request order.
        """
        ...

    @abc.abstractmethod
    def wait(
        self, references: list[ReferenceT], *, num_returns: int = 1
    ) -> tuple[list[ReferenceT], list[ReferenceT]]:
        """Wait for submitted batches to become ready.

        Args:
            references: Batch references returned by :meth:`submit`.
            num_returns: Number of ready references to wait for.

        Returns:
            A pair of lists containing ready and remaining references.
        """

    def stream(
        self, requests: Iterator[tuple[int, RequestT]]
    ) -> Iterator[tuple[int, RequestT, ResponseT]]:
        """Stream batches of request in completion order.

        Batches of ``num_parallel`` requests are submitted at once, with at most
        ``max_inflight`` batches in flight. Responses within each completed batch are
        yielded in input order.

        Args:
            requests: Pairs of request IDs and requests to process.

        Yields:
            Tuples of request ID, request, and response.
        """
        batches = iter(itertools.batched(requests, self.num_parallel))
        refs_to_requests = {}
        for batch in itertools.islice(batches, self.max_inflight):
            refs_to_requests[self.submit([r for _, r in batch])] = batch
        pending = list(refs_to_requests)
        while pending:
            done, pending = self.wait(pending, num_returns=1)
            for ref in done:
                batch = refs_to_requests.pop(ref)
                responses = self.resolve(ref)
                next_batch = next(batches, None)
                if next_batch is not None:
                    new_ref = self.submit([r for _, r in next_batch])
                    pending.append(new_ref)
                    refs_to_requests[new_ref] = next_batch
                for (request_id, request), response in zip(
                    batch, responses, strict=True
                ):
                    yield request_id, request, response

    def __call__(self, requests: SequenceLike[RequestT]) -> list[ResponseT]:
        """Submit requests and block until every response is ready.

        Args:
            requests: Requests to process.

        Returns:
            Responses in the original request order.
        """
        return self.resolve(self.submit(requests))


type Router[RequestT] = Callable[[RequestT, int], int]


@dataclasses.dataclass(frozen=True)
class _RoutedReference:
    """Group worker references with the positions of their responses."""

    refs: tuple[ray.ObjectRef, ...]
    order: tuple[tuple[int, ...], ...]
    size: int


class RoutedPool[RequestT, ResponseT](ModelPool[RequestT, ResponseT, _RoutedReference]):
    """Pool that pushes each request to the replica its router picks."""

    def __init__(
        self,
        groups: Sequence[ReplicaGroup[RequestT, ResponseT]],
        router: Router[RequestT],
        *,
        inflight_batches: int = 4,
        worker_ready_timeout: int = 60,
    ):
        """Initialize the pool and start its replicas.

        Args:
            groups: Replica groups the pool dispatches to.
            router: Picks the replica index for each request.
            inflight_batches: Maximum submitted slices kept in flight across the
                pool while streaming.
            worker_ready_timeout: time in seconds to wait for the ray actors to be
                created.
        """
        super().__init__(groups=groups, inflight_batches=inflight_batches)
        self._workers = [
            ray.remote(actors.PushWorker)
            .options(**group.actor_options)
            .remote(
                group.handler,
                max_wait_ms=group.max_wait_ms,
                max_batch_size=group.max_batch_size,
                responses_to_cpu=group.responses_to_cpu,
            )
            for group in groups
            for _ in range(group.num_replicas)
        ]
        ray.get(
            [worker.__ray_ready__.remote() for worker in self._workers],  # ty: ignore[unresolved-attribute]
            timeout=worker_ready_timeout,
        )
        self.router = router

    def submit(self, requests: SequenceLike[RequestT]) -> _RoutedReference:
        """Dispatch requests to their selected replicas and resolvable reference.

        Args:
            requests: Requests to route to replicas.

        Returns:
            A batch reference for collecting responses with :meth:`resolve`.
        """
        routed: dict[int, list[tuple[int, RequestT]]] = {}
        for index, request in enumerate(requests):
            worker_id = self.router(request, len(self._workers))
            routed.setdefault(worker_id, []).append((index, request))
        refs = tuple(
            self._workers[worker_id].submit_batch.remote([r for _, r in batch])
            for worker_id, batch in routed.items()
        )
        indices = tuple(tuple(index for index, _ in batch) for batch in routed.values())
        return _RoutedReference(refs, indices, size=len(requests))

    def resolve(self, reference: _RoutedReference) -> list[ResponseT]:
        """Collect worker responses in the original request order.

        Args:
            reference: A batch reference returned by :meth:`submit`.

        Returns:
            Responses for the submitted batch.
        """
        responses = [None] * reference.size
        results = ray.get(list(reference.refs))
        for worker_result, worker_order in zip(results, reference.order, strict=True):
            for result, order in zip(worker_result, worker_order, strict=True):
                responses[order] = result
        return responses  # ty: ignore[invalid-return-type]

    def wait(
        self, references: list[_RoutedReference], *, num_returns: int = 1
    ) -> tuple[list[_RoutedReference], list[_RoutedReference]]:
        """Wait for complete submissions and return completed and pending refs.

        Args:
            references: Batch references returned by :meth:`submit`.
            num_returns: Number of complete submissions to wait for, clamped to
                between one and the number of references when nonempty.

        Returns:
            A pair of lists containing selected ready and remaining references,
            or two empty lists if no references were supplied.
        """
        if not references:
            return [], []
        num_returns = min(max(num_returns, 1), len(references))
        remaining = list(ref for reference in references for ref in reference.refs)
        pending_children = set(remaining)
        while True:
            ready = [
                reference
                for reference in references
                if pending_children.isdisjoint(reference.refs)
            ]
            if len(ready) >= num_returns:
                done = ready[:num_returns]
                done_set = set(done)
                return done, [r for r in references if r not in done_set]

            completed, remaining = ray.wait(remaining, num_returns=1)
            pending_children.difference_update(completed)

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Load all replicas, letting served handlers manage their own devices.

        Args:
            device: Accepted for compatibility with
                :class:`protuna.optim.inference.handlers.Handler` but ignored; served
                handlers manage their own devices.
        """
        rids = [worker.load.remote() for worker in self._workers]
        ray.get(rids)

    def release(self) -> None:
        """Release model resources on every replica and wait for completion."""
        rids = [worker.release.remote() for worker in self._workers]
        ray.get(rids)

    def shutdown(self) -> None:
        """Terminate every replica in the pool."""
        for worker in self._workers:
            worker.__ray_terminate__.remote()  # ty: ignore[unresolved-attribute]


class QueuePool[RequestT, ResponseT](ModelPool[RequestT, ResponseT, ray.ObjectRef]):
    """Pool whose replicas pull requests from a shared queue.

    Work is balanced by availability — whichever replica is free takes the next
    request.
    """

    def __init__(
        self,
        groups: Sequence[ReplicaGroup[RequestT, ResponseT]],
        *,
        broker_options: RayActorOptions | None = None,
        max_concurrent_submissions: int = 1000,
        inflight_batches: int = 4,
        worker_ready_timeout: int = 60,
    ):
        """Initialize the pool and start its replicas.

        Args:
            groups: Replica groups the pool dispatches to.
            broker_options: Ray actor options for the shared broker. ``None``
                uses Ray defaults.
            max_concurrent_submissions: Maximum concurrent broker submission
                calls awaiting responses.
            inflight_batches: Maximum submitted slices kept in flight across the
                pool while streaming.
            worker_ready_timeout: time in seconds to wait for the ray actors to be
                created.
        """
        super().__init__(groups=groups, inflight_batches=inflight_batches)
        broker_options = {} if broker_options is None else broker_options
        # ty: ignore[no-matching-overload]
        self._broker: ray.actor.ActorHandle[actors.Broker] = (
            ray.remote(concurrency_groups={"pool": max_concurrent_submissions})(
                actors.Broker
            )
            .options(**broker_options)
            .remote()
        )
        self._workers = [
            ray.remote(actors.PullWorker)
            .options(**group.actor_options)
            .remote(
                group.handler,
                broker=self._broker,
                max_batch_size=group.max_batch_size,
                max_wait_ms=group.max_wait_ms,
                responses_to_cpu=group.responses_to_cpu,
            )
            for group in groups
            for _ in range(group.num_replicas)
        ]
        ray.get(self._broker.__ray_ready__.remote(), timeout=worker_ready_timeout)
        ray.get(
            [worker.__ray_ready__.remote() for worker in self._workers],  # ty: ignore[unresolved-attribute]
            timeout=worker_ready_timeout,
        )
        # start the workers
        [worker.run.remote() for worker in self._workers]

    def submit(self, requests: SequenceLike[RequestT]) -> ray.ObjectRef:
        """Enqueue requests together and return a response ref.

        Args:
            requests: Requests to enqueue as a batch.

        Returns:
            A Ray reference to the batch responses in input order.
        """
        return self._broker.submit_batch.remote(requests)

    def resolve(self, reference: ray.ObjectRef) -> list[ResponseT]:
        """Resolve a submitted batch through :func:`ray.get`.

        Args:
            reference: A batch reference returned by :meth:`submit`.

        Returns:
            Responses in the original request order.
        """
        return ray.get(reference)

    def wait(
        self, references: list[ray.ObjectRef], *, num_returns: int = 1
    ) -> tuple[list[ray.ObjectRef], list[ray.ObjectRef]]:
        """Wait for submitted batches through :func:`ray.wait`.

        Args:
            references: Batch references returned by :meth:`submit`.
            num_returns: Number of ready references to wait for.

        Returns:
            A pair of lists containing ready and remaining references.
        """
        return ray.wait(references, num_returns=num_returns)

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Load all replicas, letting served handlers manage their own devices.

        Args:
            device: Accepted for compatibility with
                :class:`protuna.optim.inference.handlers.Handler` but ignored; served
                handlers manage their own devices.
        """
        rids = [worker.load.remote() for worker in self._workers]
        ray.get(rids)

    def release(self) -> None:
        """Release model resources on every replica and wait for completion."""
        rids = [worker.release.remote() for worker in self._workers]
        ray.get(rids)

    def shutdown(self) -> None:
        """Stop and terminate every replica, then the queue broker."""
        for worker in self._workers:
            worker.stop.remote()
            worker.__ray_terminate__.remote()  # ty: ignore[unresolved-attribute]
        self._broker.__ray_terminate__.remote()
