from __future__ import annotations

import pytest

from protuna.optim.samplers import base
from protuna.optim.samplers import baselines
from tests.unit.optim.samplers import _helpers


def test_random_sampler_returns_starting_sequence_when_nothing_can_mutate() -> None:
    """An immutable space yields repeated copies of its starting sequence."""
    batch = baselines.RandomSampler(max_samples=5).sample([_helpers._space("ACD", {})])

    assert _helpers._decode(batch[0]) == ["ACD"] * 5


def test_random_sampler_keeps_fixed_positions_at_the_wild_type_residue() -> None:
    """Positions outside the search space are never mutated."""
    space = _helpers._space("ACD", {1: "CEF"})

    batch = baselines.RandomSampler(max_samples=50).sample([space])[0]

    assert _helpers._residues_at(batch, 0) == {"A"}
    assert _helpers._residues_at(batch, 2) == {"D"}


def test_random_sampler_only_proposes_allowed_residues() -> None:
    """Mutable positions only take residues the search space allows."""
    space = _helpers._space("ACD", {1: "CEF"})

    batch = baselines.RandomSampler(max_samples=50).sample([space])[0]

    assert _helpers._residues_at(batch, 1) <= {"C", "E", "F"}


def test_random_sampler_returns_the_requested_number_of_samples() -> None:
    """Keep all draws even when the space is smaller than the request."""
    space = _helpers._space("ACD", {1: "CEF"})

    batch = baselines.RandomSampler(max_samples=7).sample([space])[0]

    assert batch.shape[0] == 7
    assert batch.shape[1] == 3


def test_site_saturation_sampler_enumerates_every_allowed_single_mutant() -> None:
    """Every allowed residue at every mutable site yields one single mutant."""
    space = _helpers._space("ACD", {0: "ACD", 2: "DE"})

    batch = baselines.SiteSaturationSampler().sample([space])[0]

    assert _helpers._decode(batch) == ["CCD", "DCD", "ACE"]


def test_brute_force_sampler_enumerates_every_combination_except_the_init() -> None:
    """The full cartesian product is returned with the wild type removed."""
    space = _helpers._space("ACD", {0: "AC", 2: "DE"})

    batch = baselines.BruteForceSampler().sample([space])[0]

    assert _helpers._decode(batch) == ["ACE", "CCD", "CCE"]


@pytest.mark.parametrize(
    "enumerating_sampler",
    [baselines.SiteSaturationSampler(), baselines.BruteForceSampler()],
)
def test_enumerating_samplers_return_one_batch_per_space(
    enumerating_sampler: base.Sampler,
) -> None:
    """Each requested space gets its own batch of proposals."""
    spaces = [_helpers._space("ACD", {0: "AC"}), _helpers._space("ACD", {2: "DE"})]

    batches = enumerating_sampler.sample(spaces)

    assert [_helpers._decode(batch) for batch in batches] == [["CCD"], ["ACE"]]
