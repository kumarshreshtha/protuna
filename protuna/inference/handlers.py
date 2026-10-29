"""Shared request-handling contract for local and distributed execution.

:class:`Handler` exposes the same batch, streaming, and lifecycle operations
whether requests run in the calling process or through a distributed service.
Application code can use either through this contract without knowing whether
the handler is served or how requests are executed.

:class:`LocalHandler` provides the base for in-process execution, with streaming
built from batch calls. :class:`ModelFreeLocalHandler` extends it with no-op
resource loading and release for handlers that do not hold model resources.
:class:`InferenceModel` extends :class:`LocalHandler` with batched forward passes.
:class:`HandlerRegistryMixin` propagates loading and release to handler attributes
and nested registries.
"""

import abc
from collections.abc import Iterator, Sequence
import itertools
from typing import Any, final, Protocol, runtime_checkable

import torch

from protuna.typing import SequenceLike


@runtime_checkable
class Handler[RequestT, ResponseT](Protocol):
    """A common interface for local or distributed request handling.

    Batch calls return one response per request in input order; :meth:`stream`
    returns request IDs, requests, and responses in completion order.
    :meth:`load`, :meth:`release`, and :meth:`shutdown` expose lifecycle
    operations through the same contract.

    The protocol supports runtime ``isinstance`` checks based on the presence
    of its operations; these checks do not validate method signatures.

    Application components can depend on this protocol to use local handlers
    and served, distributed handlers interchangeably. Callers do not need to
    detect whether a handler is served or manage its execution mechanism.
    """

    def __call__(self, requests: SequenceLike[RequestT]) -> SequenceLike[ResponseT]:
        """Resolve a batch of requests.

        Args:
            requests: Requests to resolve as a batch.

        Returns:
            One response per request, in input order.
        """
        ...

    def stream(
        self, requests: Iterator[tuple[int, RequestT]]
    ) -> Iterator[tuple[int, RequestT, ResponseT]]:
        """Resolve ``(request_id, request)`` pairs as they complete.

        Requests are pulled lazily and only a bounded number are ever in
        flight, so an arbitrarily long iterator is fine.

        Args:
            requests: Iterator of request IDs paired with requests to resolve.

        Yields:
            ``(request_id, request, response)`` triples, in completion order.
        """
        ...

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Load the resources needed to resolve requests.

        Args:
            device: Device or devices on which to load resources. ``None`` leaves
                device selection to the implementation.
        """
        ...

    def release(self) -> None:
        """Release GPU resources held by the handler."""
        ...

    def shutdown(self) -> None:
        """Gracefully terminate the handler if non-local."""
        ...


class LocalHandler[RequestT, ResponseT](Handler[RequestT, ResponseT], abc.ABC):
    """Base class for handlers that run locally in the calling process.

    :meth:`stream` consumes the request iterator in chunks of ``batch_size``.
    """

    batch_size: int | None = None

    def __init__(self, *, batch_size: int | None = None):
        """Initialize the local handler's streaming batch size.

        Args:
            batch_size: Requests passed to :meth:`__call__` at a time while
                streaming. ``None`` sends everything pending in one call.
        """
        self.batch_size = batch_size

    def stream(
        self, requests: Iterator[tuple[int, RequestT]]
    ) -> Iterator[tuple[int, RequestT, ResponseT]]:
        """Resolve requests in chunks of ``batch_size``.

        Args:
            requests: Iterator of request IDs paired with requests to resolve.

        Yields:
            ``(request_id, request, response)`` triples, in input order.
        """
        if self.batch_size is None:
            pending = list(requests)
            if not pending:
                return
            rids, requests_ = zip(*pending, strict=True)
            yield from zip(rids, requests_, self(requests_), strict=True)
        else:
            for batch in itertools.batched(requests, self.batch_size):
                batch_requests = [r for _, r in batch]
                batch_request_ids = [rid for rid, _ in batch]
                yield from zip(
                    batch_request_ids, batch_requests, self(batch_requests), strict=True
                )

    @abc.abstractmethod
    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Load local resources. See :meth:`Handler.load`.

        Args:
            device: Device or devices on which to load resources. ``None`` leaves
                device selection to the implementation.
        """
        ...

    @abc.abstractmethod
    def release(self) -> None:
        """Release GPU resources held by the handler."""
        ...

    @abc.abstractmethod
    def __call__(self, requests: SequenceLike[RequestT]) -> SequenceLike[ResponseT]:
        """Resolve a batch of requests.

        Args:
            requests: Requests to resolve as a batch.

        Returns:
            One response per request, in input order.
        """
        ...

    def shutdown(self) -> None:
        """Release the resources the handler holds; a no-op by default."""
        return None


class InferenceModel[RequestT, ResponseT](LocalHandler[RequestT, ResponseT], abc.ABC):
    """:class:`LocalHandler` that resolves calls in batches with :meth:`forward`."""

    @abc.abstractmethod
    def forward(self, requests: SequenceLike[RequestT]) -> SequenceLike[ResponseT]:
        """Process one batch of requests, in order.

        Args:
            requests: Requests to resolve in one forward pass.

        Returns:
            One response per request, in input order.
        """
        ...

    @final
    def __call__(self, requests: SequenceLike[RequestT]) -> list[ResponseT]:
        """Process requests in forward passes of up to ``batch_size`` requests.

        Args:
            requests: Requests to resolve. When ``batch_size`` is ``None``, all
                requests are passed to :meth:`forward` together.

        Returns:
            One response per request, in input order.
        """
        batch_size = len(requests) if self.batch_size is None else self.batch_size
        responses = []
        for batch in itertools.batched(requests, batch_size):
            responses.extend(self.forward(batch))
        return responses


class ModelFreeLocalHandler[RequestT, ResponseT](
    LocalHandler[RequestT, ResponseT], abc.ABC
):
    """Handle requests locally without model resources to load or release."""

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Accept a load request without allocating resources.

        Args:
            device: Unused device specification, accepted for compatibility with
                :meth:`Handler.load`.
        """
        return

    def release(self) -> None:
        """Release GPU resources held by the handler."""
        return


class HandlerRegistryMixin:
    """Register handler and other registry attributes for shared resource management.

    Assigning a :class:`Handler` or another :class:`HandlerRegistryMixin` to an
    attribute registers it automatically. Reassigning or deleting the attribute updates
    its registration. :meth:`load` and :meth:`release` propagate to registered handlers
    and recurse through nested registries.

    Use :meth:`register` to register objects explicitly.
    """

    def __init__(self, **kwargs):
        """Initialize the shared resource registry.

        Args:
            **kwargs: Keyword arguments forwarded to the next initializer in
                the method resolution order.
        """
        super().__init__(**kwargs)
        self._registry: dict[str, Handler | HandlerRegistryMixin] = {}

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Load resources for registered handlers and nested registries.

        Args:
            device: Device or devices forwarded to each registered object's
                :meth:`load` method. ``None`` leaves device selection to each
                handler.
        """
        for component in self._registry.values():
            component.load(device)

    def release(self) -> None:
        """Release resources held by all registered handlers and nested registries."""
        for component in self._registry.values():
            component.release()

    def register(self, name: str, component: "Handler | HandlerRegistryMixin") -> None:
        """Register a handler or nested registry for resource management.

        Args:
            name: Registration name, replacing any entry with the same name.
            component: Handler or nested registry whose resources this registry
                will load and release.
        """
        registry = self.__dict__.get("_registry")
        if registry is None:
            raise AttributeError(
                "cannot register component before super().__init__() call"
            )
        registry[name] = component

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        """Assign an attribute and update its resource registration."""
        if isinstance(value, (Handler, HandlerRegistryMixin)):
            self.register(name, value)
        elif (registry := self.__dict__.get("_registry")) is not None:
            registry.pop(name, None)
        return super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """Delete an attribute and remove its resource registration."""
        if (registry := self.__dict__.get("_registry")) is not None:
            registry.pop(name, None)
        return super().__delattr__(name)
