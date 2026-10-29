from __future__ import annotations

from typing import Literal

import pytest
import torch

from protuna.optim.samplers import beam
from tests.unit.optim.samplers import _helpers


def test_mlm_beam_sampler_returns_empty_batch_for_immutable_space() -> None:
    """Return an empty token batch without requesting model inference."""
    space = _helpers._space("ACD", {})
    generator = _helpers._FakeMLM({"W": 1.0})

    batch = beam.MLMBeamSampler(generator, max_samples=2).sample([space])[0]

    assert batch.shape == (0, 3)
    assert batch.dtype == space.init_sequence.dtype
    assert batch.device == space.init_sequence.device
    assert generator.requests == []


@pytest.mark.parametrize(
    ("first_allowed", "second_allowed", "expected"),
    [({}, {0: "AW"}, [[], ["ACD", "WCD"]]), ({0: "AW"}, {}, [["ACD", "WCD"], []])],
)
def test_mlm_beam_sampler_handles_mixed_mutability_batches(
    first_allowed: dict[int, str],
    second_allowed: dict[int, str],
    expected: list[list[str]],
) -> None:
    """Keep immutable and mutable results in request order."""
    spaces = [
        _helpers._space("ACD", first_allowed),
        _helpers._space("ACD", second_allowed),
    ]
    generator = _helpers._FakeMLM({"A": 0.6, "W": 0.4})

    batches = beam.MLMBeamSampler(generator, max_samples=2).sample(spaces)

    assert [_helpers._decode(batches[0]), _helpers._decode(batches[1])] == expected


def test_mlm_beam_sampler_keeps_the_init_sequence_in_its_proposals() -> None:
    """The starting sequence retains its beam slot in the returned proposals."""
    space = _helpers._space("ACD", {0: "AW"})
    generator = _helpers._FakeMLM({"A": 0.6, "W": 0.4})

    batch = beam.MLMBeamSampler(generator, max_samples=2, beam_size=2).sample([space])[
        0
    ]

    assert _helpers._decode(batch) == ["ACD", "WCD"]


@pytest.mark.parametrize("sampling_order", ["lr", "random", "max_p"])
@pytest.mark.parametrize(
    ("allowed", "expected"),
    [
        ({0: "AW"}, {"ACD", "WCD"}),
        ({0: "AW", 2: "DW"}, {"ACD", "ACW", "WCD", "WCW"}),
    ],
)
def test_mlm_beam_sampler_keeps_oversized_beams_within_the_search_space(
    sampling_order: Literal["lr", "random", "max_p"],
    allowed: dict[int, str],
    expected: set[str],
) -> None:
    """Return only allowed mutants when the beam exceeds the search space."""
    space = _helpers._space("ACD", allowed)
    generator = _helpers._FakeMLM({"A": 0.1, "D": 0.1, "W": 0.8})

    batch = beam.MLMBeamSampler(
        generator,
        max_samples=25,
        beam_size=32,
        sampling_order=sampling_order,
        seed=7,
    ).sample([space])[0]

    assert set(_helpers._decode(batch)) == expected


def test_mlm_beam_sampler_decodes_spaces_with_different_mutable_counts() -> None:
    """Spaces that run out of masked positions early keep their own proposals."""
    spaces = [
        _helpers._space("ACD", {0: "AWY"}),
        _helpers._space("ACD", {0: "AWY", 2: "DWY"}),
    ]
    # Distinct site probabilities avoid a topk tie between YCW and WCY.
    generator = _helpers._FakeMLM(
        {"Y": 0.6, "W": 0.3, "A": 0.1},
        position_probs={2: {"Y": 0.5, "W": 0.4, "D": 0.1}},
    )

    batches = beam.MLMBeamSampler(generator, max_samples=2, beam_size=3).sample(spaces)

    assert [set(_helpers._decode(batch)) for batch in batches] == [
        {"YCD", "WCD"},
        {"YCY", "YCW"},
    ]


@pytest.mark.parametrize(
    ("sampling_order", "still_masked"),
    [("lr", [2]), ("max_p", [0])],
)
def test_mlm_beam_sampler_unmasks_in_the_order_it_was_configured_with(
    sampling_order: Literal["lr", "max_p"], still_masked: list[int]
) -> None:
    """`sampling_order` decides which masked position the beam commits to first."""
    space = _helpers._space("ACD", {0: "AW", 2: "DW"})
    # Position 2 carries the sharper distribution, so max_p commits to it first.
    generator = _helpers._FakeMLM(
        {"W": 0.5, "A": 0.5}, position_probs={2: {"W": 0.99, "D": 0.01}}
    )

    beam.MLMBeamSampler(
        generator,
        max_samples=2,
        beam_size=2,
        sampling_order=sampling_order,
    ).sample([space])

    assert _helpers._masked_positions(generator.requests[1].token_ids) == still_masked


@pytest.mark.parametrize(
    ("sampling_order", "expected"),
    [("lr", 0), ("max_p", 1)],
)
def test_get_unmask_position_picks_the_position_the_order_asks_for(
    sampling_order: Literal["lr", "max_p"], expected: int
) -> None:
    """Left-to-right takes the first mask; max_p takes the most confident one."""
    logp = torch.tensor([[[-3.0, -1.0], [-0.5, -2.0]]])

    assert beam._get_unmask_position(logp, sampling_order) == expected


def test_get_unmask_position_draws_a_random_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Random order draws uniformly over the remaining masked positions."""
    logp = torch.tensor([[[-3.0, -1.0], [-0.5, -2.0]]])
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([1]))

    assert beam._get_unmask_position(logp, "random") == 1
