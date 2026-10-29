from __future__ import annotations

import pytest
import torch

from protuna.optim import spaces
from protuna.optim.samplers import rlxf
from tests.unit.optim.samplers import _helpers


def test_rlxf_sampler_returns_empty_batch_for_immutable_space() -> None:
    """Return an empty token batch without requesting model inference."""
    space = _helpers._space("ACD", {})
    generator = _helpers._FakeMLM({"W": 1.0})

    batch = rlxf.RLXFSampler(generator, max_samples=2).sample([space])[0]

    assert batch.shape == (0, 3)
    assert batch.dtype == space.init_sequence.dtype
    assert batch.device == space.init_sequence.device
    assert generator.requests == []


@pytest.mark.parametrize(
    ("first_allowed", "second_allowed", "expected"),
    [({}, {0: "AW"}, [[], ["WCD"]]), ({0: "AW"}, {}, [["WCD"], []])],
)
def test_rlxf_sampler_handles_mixed_mutability_batches(
    first_allowed: dict[int, str],
    second_allowed: dict[int, str],
    expected: list[list[str]],
) -> None:
    """Keep immutable and mutable results in request order."""
    spaces = [
        _helpers._space("ACD", first_allowed),
        _helpers._space("ACD", second_allowed),
    ]
    generator = _helpers._FakeMLM({"W": 1.0})

    batches = rlxf.RLXFSampler(generator, max_samples=2).sample(spaces)

    assert [_helpers._decode(batches[0]), _helpers._decode(batches[1])] == expected


def _mutation_counts(batch: torch.Tensor, space: spaces.SearchSpace) -> list[int]:
    """Count how many positions each proposal changed from the initial sequence."""
    return (batch != space.init_sequence).sum(dim=1).tolist()


def test_prepare_per_position_masked_inputs_masks_one_site_per_row() -> None:
    """Each mutable site gets its own row carrying a single mask."""
    tokens = _helpers._TOKENIZER.encode("ACDEF")
    mutable = torch.tensor([True, False, True, False, False])

    batch = rlxf._prepare_per_position_masked_inputs(tokens, mutable)

    assert batch.shape == (2, 5)
    assert _helpers._masked_positions(batch[0]) == [0]
    assert _helpers._masked_positions(batch[1]) == [2]


def test_rlxf_scaffold_applies_residues_above_the_confidence_threshold() -> None:
    """A residue the model is confident about is written in outright."""
    space = _helpers._space("ACDEF", {0: "AW", 1: "CW"})
    rlxf_sampler = rlxf.RLXFSampler(
        _helpers._FakeMLM({"W": 0.95, "A": 0.05}), max_samples=4
    )

    scaffold = rlxf_sampler._get_scaffold(space)

    assert _helpers._TOKENIZER.decode(scaffold) == "WWDEF"


def test_rlxf_scaffold_keeps_residues_below_the_confidence_threshold() -> None:
    """A residue the model is unsure about is left at the wild type."""
    space = _helpers._space("ACDEF", {0: "AW", 1: "CW"})
    rlxf_sampler = rlxf.RLXFSampler(
        _helpers._FakeMLM({"W": 0.6, "A": 0.4}), max_samples=4
    )

    scaffold = rlxf_sampler._get_scaffold(space)

    assert _helpers._TOKENIZER.decode(scaffold) == "ACDEF"


def test_rlxf_scaffold_never_writes_a_residue_the_space_forbids() -> None:
    """A forbidden residue is excluded however confident the model is."""
    space = _helpers._space("ACDEF", {0: "AW"})
    rlxf_sampler = rlxf.RLXFSampler(
        _helpers._FakeMLM({"K": 0.95, "W": 0.05}), max_samples=4
    )

    scaffold = rlxf_sampler._get_scaffold(space)

    assert _helpers._TOKENIZER.decode(scaffold) == "ACDEF"


def test_rlxf_position_weights_are_the_non_wild_type_probability_mass() -> None:
    """A site is weighted by how much probability sits off the wild type."""
    space = _helpers._space("AACDE", {0: "AWY", 1: "AWY"})
    rlxf_sampler = rlxf.RLXFSampler(
        _helpers._FakeMLM({"A": 0.3, "W": 0.4, "Y": 0.3}), max_samples=4
    )

    weights = rlxf_sampler._get_position_weights(
        space, _helpers._TOKENIZER.encode("AACDE")
    )

    assert weights.tolist() == pytest.approx([0.7, 0.7, 0.0, 0.0, 0.0])


def test_rlxf_position_weights_exclude_residues_the_space_forbids() -> None:
    """Probability on a forbidden residue does not make a site eligible."""
    space = _helpers._space("AACDE", {0: "AW", 1: "AWY"})
    rlxf_sampler = rlxf.RLXFSampler(
        _helpers._FakeMLM({"A": 0.3, "W": 0.4, "Y": 0.3}), max_samples=4
    )

    weights = rlxf_sampler._get_position_weights(
        space, _helpers._TOKENIZER.encode("AACDE")
    )

    assert weights.tolist() == pytest.approx([0.4, 0.7, 0.0, 0.0, 0.0])


def test_rlxf_position_weights_drop_sites_below_the_position_threshold() -> None:
    """A site the model is happy to leave alone is never mutated."""
    space = _helpers._space("AACDE", {0: "AW", 1: "AW"})
    rlxf_sampler = rlxf.RLXFSampler(
        _helpers._FakeMLM({"A": 0.85, "W": 0.15}), max_samples=4
    )

    weights = rlxf_sampler._get_position_weights(
        space, _helpers._TOKENIZER.encode("AACDE")
    )

    assert weights.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0])


def test_rlxf_position_weights_zero_sites_already_mutated_in_the_scaffold() -> None:
    """Mutations the scaffold introduced are preserved, not sampled over."""
    space = _helpers._space("AACDE", {0: "AWY", 1: "AWY"})
    rlxf_sampler = rlxf.RLXFSampler(
        _helpers._FakeMLM({"A": 0.3, "W": 0.4, "Y": 0.3}), max_samples=4
    )

    weights = rlxf_sampler._get_position_weights(
        space, _helpers._TOKENIZER.encode("WACDE")
    )

    assert weights.tolist() == pytest.approx([0.0, 0.7, 0.0, 0.0, 0.0])


def test_rlxf_sampler_introduces_exactly_max_mutations_per_candidate() -> None:
    """Every proposal carries the requested number of mutations."""
    space = _helpers._space("ACDEF", {0: "AWY", 1: "CWY", 2: "DWY"})
    rlxf_sampler = rlxf.RLXFSampler(
        _helpers._FakeMLM({"W": 0.5, "Y": 0.5}), max_samples=6, max_mutations=2
    )

    batch = rlxf_sampler.sample([space])[0]

    assert set(_mutation_counts(batch, space)) == {2}


def test_rlxf_sampler_caps_mutations_at_the_number_of_mutable_sites() -> None:
    """Asking for more mutations than there are open sites mutates them all."""
    space = _helpers._space("ACDEF", {0: "AWY", 1: "CWY"})
    rlxf_sampler = rlxf.RLXFSampler(
        _helpers._FakeMLM({"W": 0.5, "Y": 0.5}), max_samples=6, max_mutations=5
    )

    batch = rlxf_sampler.sample([space])[0]

    assert set(_mutation_counts(batch, space)) == {2}


def test_rlxf_sampler_mutates_every_open_site_when_max_mutations_is_none() -> None:
    """`max_mutations=None` opens every mutable site to mutation."""
    space = _helpers._space("ACDEF", {0: "AWY", 1: "CWY", 2: "DWY"})
    rlxf_sampler = rlxf.RLXFSampler(
        _helpers._FakeMLM({"W": 0.5, "Y": 0.5}), max_samples=6
    )

    batch = rlxf_sampler.sample([space])[0]

    assert set(_mutation_counts(batch, space)) == {3}


def test_rlxf_sampler_only_proposes_residues_the_space_allows() -> None:
    """Residues the space forbids are masked out however high they score."""
    space = _helpers._space("ACDEF", {0: "AW"})
    rlxf_sampler = rlxf.RLXFSampler(
        _helpers._FakeMLM({"K": 0.6, "W": 0.4}), max_samples=10
    )

    batch = rlxf_sampler.sample([space])[0]

    assert _helpers._residues_at(batch, 0) <= {"A", "W"}


def test_rlxf_sampler_keeps_fixed_positions_at_the_wild_type_residue() -> None:
    """Positions outside the search space are never mutated."""
    space = _helpers._space("ACDEF", {0: "AWY", 1: "CWY"})
    rlxf_sampler = rlxf.RLXFSampler(
        _helpers._FakeMLM({"W": 0.5, "Y": 0.5}), max_samples=6, max_mutations=2
    )

    batch = rlxf_sampler.sample([space])[0]

    assert _helpers._residues_at(batch, 2) == {"D"}
    assert _helpers._residues_at(batch, 3) == {"E"}
    assert _helpers._residues_at(batch, 4) == {"F"}


def test_rlxf_sampler_returns_at_most_the_requested_number_of_samples() -> None:
    """The proposal batch never exceeds the requested sample count."""
    space = _helpers._space("ACDEF", {0: "AWY", 1: "CWY", 2: "DWY"})
    rlxf_sampler = rlxf.RLXFSampler(
        _helpers._FakeMLM({"W": 0.5, "Y": 0.5}), max_samples=2, max_mutations=1
    )

    batch = rlxf_sampler.sample([space])[0]

    assert batch.shape[0] <= 2
    assert batch.shape[1] == 5


def test_rlxf_sampler_collapses_duplicate_position_draws_into_one_candidate() -> None:
    """A space with a single open site draws that site for every design."""
    space = _helpers._space("ACDEF", {0: "AW"})
    rlxf_sampler = rlxf.RLXFSampler(
        _helpers._FakeMLM({"W": 0.5, "Y": 0.5}), max_samples=5, max_mutations=1
    )

    batch = rlxf_sampler.sample([space])[0]

    assert _helpers._decode(batch) == ["WCDEF"]


def test_rlxf_sampler_conditions_each_mutation_on_the_previous_ones() -> None:
    """Sites are filled one at a time, each seeing the mutations placed before it."""
    space = _helpers._space("ACDEF", {0: "AW", 1: "CW"})
    generator = _helpers._FakeMLM({"W": 0.5, "Y": 0.5})
    rlxf_sampler = rlxf.RLXFSampler(generator, max_samples=1, max_mutations=2)

    batch = rlxf_sampler.sample([space])[0]

    final_request = generator.requests[-1].token_ids.tolist()
    assert _helpers._MASK in final_request
    assert _helpers._TOKENIZER.token_to_id("W") in final_request
    assert _helpers._decode(batch) == ["WWDEF"]


def test_rlxf_sampler_returns_one_batch_per_space() -> None:
    """Each space is decoded independently against its own mutable positions."""
    spaces = [
        _helpers._space("ACDEF", {0: "AWY"}),
        _helpers._space("ACDEF", {2: "DWY"}),
    ]
    rlxf_sampler = rlxf.RLXFSampler(
        _helpers._FakeMLM({"W": 0.5, "Y": 0.5}), max_samples=4, max_mutations=1
    )

    batches = rlxf_sampler.sample(spaces)

    assert _helpers._residues_at(batches[0], 0) <= {"W", "Y"}
    assert _helpers._residues_at(batches[1], 2) <= {"W", "Y"}


def test_rlxf_sampler_returns_the_scaffold_alone_when_it_meets_max_mutations() -> None:
    """A scaffold that already spends the mutation budget is the only proposal."""
    space = _helpers._space("ACDEF", {0: "AW", 1: "CW"})
    rlxf_sampler = rlxf.RLXFSampler(
        _helpers._FakeMLM({"W": 0.95, "A": 0.05}), max_samples=4, max_mutations=1
    )

    batch = rlxf_sampler.sample([space])[0]

    assert _helpers._decode(batch) == ["WWDEF"]


def test_rlxf_sampler_warns_when_the_scaffold_meets_max_mutations(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exceeding the budget on the scaffold is reported rather than raised."""
    space = _helpers._space("ACDEF", {0: "AW", 1: "CW"})
    rlxf_sampler = rlxf.RLXFSampler(
        _helpers._FakeMLM({"W": 0.95, "A": 0.05}), max_samples=4, max_mutations=1
    )

    rlxf_sampler.sample([space])

    assert "High-confidence scaffold already meet/exceed" in caplog.messages[0]


def test_rlxf_sampler_returns_the_scaffold_when_no_site_is_eligible() -> None:
    """With every site below the position threshold nothing further is mutated."""
    space = _helpers._space("ACDEF", {0: "AW", 1: "CK"})
    rlxf_sampler = rlxf.RLXFSampler(
        _helpers._FakeMLM({"A": 0.05, "W": 0.95}), max_samples=4, max_mutations=2
    )

    batch = rlxf_sampler.sample([space])[0]

    assert _helpers._decode(batch) == ["WCDEF"]


def test_rlxf_sampler_mutates_fewer_sites_than_requested_when_few_are_eligible() -> (
    None
):
    """Only the eligible sites are mutated instead of raising."""
    space = _helpers._space("ACDEF", {0: "AWY", 1: "CW", 2: "DW"})
    generator = _helpers._FakeMLM(
        {"W": 0.5, "Y": 0.5},
        position_probs={1: {"C": 0.85, "W": 0.15}, 2: {"D": 0.85, "W": 0.15}},
    )
    rlxf_sampler = rlxf.RLXFSampler(generator, max_samples=6, max_mutations=3)

    batch = rlxf_sampler.sample([space])[0]

    assert set(_mutation_counts(batch, space)) == {1}


def test_rlxf_sampler_warns_when_fewer_sites_are_eligible_than_requested(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Running short of eligible sites is reported rather than raised."""
    space = _helpers._space("ACDEF", {0: "AWY", 1: "CW", 2: "DW"})
    generator = _helpers._FakeMLM(
        {"W": 0.5, "Y": 0.5},
        position_probs={1: {"C": 0.85, "W": 0.15}, 2: {"D": 0.85, "W": 0.15}},
    )
    rlxf_sampler = rlxf.RLXFSampler(generator, max_samples=6, max_mutations=3)

    rlxf_sampler.sample([space])

    assert "Only 1 are eligible for mutation" in caplog.messages[0]
