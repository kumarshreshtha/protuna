from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from protuna import inference
from protuna.inference import serve
from protuna.inference.interfaces import mlm
from protuna.inference.serve import routing

# Wild-type used by every router test; mutants below are expressed relative to
# these token ids.
_WT_TOKEN_IDS = (5, 7, 9, 11)


class _CannedRandrange:
    """Stand-in for `random.randrange` handing out canned replica ids.

    Records the ``stop`` argument of every call in :attr:`calls` so tests can
    assert both how often a fresh assignment was drawn and what replica count
    it was drawn against.
    """

    def __init__(self, *replica_ids: int):
        self._replica_ids = iter(replica_ids)
        self.calls: list[int] = []

    def __call__(self, stop: int) -> int:
        self.calls.append(stop)
        return next(self._replica_ids)


@pytest.fixture
def router() -> routing.MaskAffinityRouter:
    """Router keyed on the wild-type sequence shared by these tests."""
    return routing.MaskAffinityRouter(torch.tensor(_WT_TOKEN_IDS))


@pytest.fixture
def canned_randrange(monkeypatch: pytest.MonkeyPatch) -> _CannedRandrange:
    """Replaces the router's source of randomness with canned replica ids."""
    randrange = _CannedRandrange(3, 1, 0)
    monkeypatch.setattr(routing.random, "randrange", randrange)
    return randrange


def _request(token_ids: Sequence[int]) -> mlm.MLMRequest:
    return mlm.MLMRequest(token_ids=torch.tensor(token_ids))


def test_repeated_request_reuses_its_first_assignment(
    router: routing.MaskAffinityRouter, canned_randrange: _CannedRandrange
) -> None:
    """An already-seen mutation pattern is never re-drawn."""
    request = _request([5, 7, 2, 11])

    replica_ids = [router(request, 4), router(request, 4)]

    assert replica_ids == [3, 3]
    assert canned_randrange.calls == [4]


def test_same_mutated_sites_with_different_residues_share_a_replica(
    router: routing.MaskAffinityRouter, canned_randrange: _CannedRandrange
) -> None:
    """Affinity is keyed on which sites differ, not on the residues there."""
    first = router(_request([5, 7, 2, 11]), 4)
    second = router(_request([5, 7, 20, 11]), 4)

    assert (first, second) == (3, 3)
    assert canned_randrange.calls == [4]


def test_different_mutated_sites_are_assigned_independently(
    router: routing.MaskAffinityRouter, canned_randrange: _CannedRandrange
) -> None:
    """Each new mutation pattern draws its own replica."""
    first = router(_request([5, 7, 2, 11]), 4)
    second = router(_request([2, 7, 9, 11]), 4)

    assert (first, second) == (3, 1)
    assert canned_randrange.calls == [4, 4]


def test_wildtype_is_a_pattern_of_its_own(
    router: routing.MaskAffinityRouter, canned_randrange: _CannedRandrange
) -> None:
    """An unmutated request does not share the assignment of a mutant."""
    mutant = router(_request([5, 7, 2, 11]), 4)
    wildtype = router(_request(_WT_TOKEN_IDS), 4)

    assert (mutant, wildtype) == (3, 1)


def test_number_of_mutated_sites_distinguishes_patterns(
    router: routing.MaskAffinityRouter, canned_randrange: _CannedRandrange
) -> None:
    """A double mutant is routed separately from its single-mutant subset."""
    single = router(_request([5, 7, 2, 11]), 4)
    double = router(_request([5, 7, 2, 2]), 4)

    assert (single, double) == (3, 1)


def test_each_router_draws_its_own_assignment(
    canned_randrange: _CannedRandrange,
) -> None:
    """Assignments are per-router state, so a second router re-draws.

    A shared cache would make the second router reuse the first draw and
    return ``(3, 3)`` without consulting `randrange` a second time.
    """
    request = _request([5, 7, 2, 11])

    first = routing.MaskAffinityRouter(torch.tensor(_WT_TOKEN_IDS))(request, 4)
    second = routing.MaskAffinityRouter(torch.tensor(_WT_TOKEN_IDS))(request, 4)

    assert canned_randrange.calls == [4, 4]
    assert (first, second) == (3, 1)


@pytest.mark.parametrize("num_replicas", [1, 2, 8])
def test_assignment_falls_inside_the_replica_count(num_replicas: int) -> None:
    """Unpatched, the drawn replica is a valid index for the given count."""
    router = routing.MaskAffinityRouter(torch.tensor(_WT_TOKEN_IDS))

    replica_id = router(_request([5, 7, 2, 11]), num_replicas)

    assert 0 <= replica_id < num_replicas


def test_round_robin_router_cycles_through_the_replicas() -> None:
    """Consecutive requests land on successive replicas and wrap around."""
    router = routing.RoundRobinRouter()

    assigned = [router(request, 3) for request in range(5)]

    assert assigned == [0, 1, 2, 0, 1]


def test_mask_affinity_router_is_exported_from_serving_and_inference() -> None:
    """The serving and top-level exports resolve to the moved router."""
    assert serve.MaskAffinityRouter is routing.MaskAffinityRouter
    assert inference.MaskAffinityRouter is routing.MaskAffinityRouter


def test_logits_request_shares_affinity_with_mlm_request(
    router: routing.MaskAffinityRouter, canned_randrange: _CannedRandrange
) -> None:
    """Both request interfaces share assignments for the same mutated sites."""
    request = _request([5, 7, 2, 11])
    logits_request = mlm.MLMLogitsRequest(token_ids=torch.tensor([5, 7, 20, 11]))

    first = router(request, 4)
    second = router(logits_request, 4)

    assert (first, second) == (3, 3)
    assert canned_randrange.calls == [4]
