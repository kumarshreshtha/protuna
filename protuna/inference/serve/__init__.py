"""API and utilities for distributed serving of handlers.

:func:`~.api.serve_queue_pool` takes any handler and spreads it over a pool of Ray
replicas that batch incoming requests dynamically. The pool is itself a
:class:`~protuna.inference.handlers.Handler`, so moving a workload from
local to a cluster does not change how it is called.

Replication and resources are described by :class:`~pool.ReplicaGroup`. Several
groups can be passed to run one handler under different configurations.

Use :func:`~api.serve_queue_pool` to let available replicas pull requests from a shared
queue, or :func:`~api.serve_routed_pool` to select a replica for each request.
Routed pools use round-robin dispatch unless given a :py:type:`~pool.Router`.

:func:`~api.execute_parallel` sends a set of requests to several handlers at once.
"""

from .api import execute_parallel as execute_parallel
from .api import serve_queue_pool as serve_queue_pool
from .api import serve_routed_pool as serve_routed_pool
from .pool import RayActorOptions as RayActorOptions
from .pool import ReplicaGroup as ReplicaGroup
from .routing import MaskAffinityRouter as MaskAffinityRouter
from .routing import RoundRobinRouter as RoundRobinRouter
