"""Shared execution layer for local and distributed inference.

The :class:`~handlers.Handler` protocol defines a common contract for interacting with
batches of requests, streaming results, and loading and releasing resources. Both local
handlers and Ray-backed pools expose this interface, so algorithms can use the
same requests and responses whether computation runs in the calling process or
across multiple workers.

The optimization framework builds on these contracts: samplers, constraints and
predictors can themselves be served as handlers, as well as consume model handlers
internally.

Examples:
    An in-process handler only has to implement ``__call__``:

    >>> from protuna.typing import SequenceLike
    >>> from protuna.inference import handlers, serve
    >>> class Doubler(handlers.ModelFreeLocalHandler[int, int]):
    ...     def __call__(self, requests: SequenceLike[int]) -> list[int]:
    ...         return [request * 2 for request in requests]
    >>> Doubler()([1, 2, 3])
    [2, 4, 6]

    Serving it returns a handler with the same interface, backed by two Ray
    replicas:

    >>> pool = serve.serve_queue_pool(
    ...     Doubler(), num_replicas=2, max_batch_size=8, actor_options={"num_cpus": 1}
    ... )  # doctest: +SKIP
    >>> pool([1, 2, 3])  # doctest: +SKIP
    [2, 4, 6]
    >>> pool.shutdown()  # doctest: +SKIP
"""

from .handlers import Handler as Handler
from .handlers import HandlerRegistryMixin as HandlerRegistryMixin
from .handlers import InferenceModel as InferenceModel
from .handlers import LocalHandler as LocalHandler
from .handlers import ModelFreeLocalHandler as ModelFreeLocalHandler
from .interfaces import MaskedLMLike as MaskedLMLike
from .interfaces import MLMForLogitsLike as MLMForLogitsLike
from .interfaces import MLMLogitsRequest as MLMLogitsRequest
from .interfaces import MLMOutput as MLMOutput
from .interfaces import MLMOutputConfig as MLMOutputConfig
from .interfaces import MLMRequest as MLMRequest
from .serve import execute_parallel as execute_parallel
from .serve import MaskAffinityRouter as MaskAffinityRouter
from .serve import RayActorOptions as RayActorOptions
from .serve import ReplicaGroup as ReplicaGroup
from .serve import RoundRobinRouter as RoundRobinRouter
from .serve import serve_queue_pool as serve_queue_pool
from .serve import serve_routed_pool as serve_routed_pool
