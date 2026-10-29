"""Routers to distribute requests across replicas in a routed pool distribution."""

import hashlib
import random

from protuna.inference import interfaces
from protuna.typing import TokenIDs


class RoundRobinRouter[RequestT]:
    """Distribute successive requests across replicas in cyclic order.

    See :class:`MaskAffinityRouter` for one that routes on the content of the request.
    """

    def __init__(self):
        """Inits the router."""
        self._last_replica_id = -1

    def __call__(self, request: RequestT, num_replicas: int) -> int:
        """Return the index of the replica *request* should go to."""
        replica_id = (self._last_replica_id + 1) % num_replicas
        self._last_replica_id = replica_id
        return replica_id


class MaskAffinityRouter:
    """Send all requests sharing the same mutations to the same replica.

    Two requests differing from the wild type at the same positions are always
    routed together, so whatever a replica caches for a variant can be reused.
    Replicas are picked at random the first time a mutation set is seen.
    """

    def __init__(self, wt_token_ids: TokenIDs):
        """Inits the router.

        Args:
            wt_token_ids: Wild type sequence of shape ``(L,)`` that requests
                are compared against.
        """
        self._wt = wt_token_ids
        self._assigned: dict[bytes, int] = {}

    def __call__(
        self,
        request: interfaces.MLMRequest | interfaces.MLMLogitsRequest,
        num_replicas: int,
    ) -> int:
        """Return the index of the replica *request* should go to."""
        wt = self._wt.to(request.token_ids.device)
        mask = (request.token_ids != wt).cpu().contiguous().numpy()
        key = hashlib.blake2b(memoryview(mask), digest_size=16).digest()
        if key not in self._assigned:
            self._assigned[key] = random.randrange(num_replicas)
        return self._assigned[key]
