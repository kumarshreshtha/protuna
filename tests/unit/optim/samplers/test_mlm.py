from __future__ import annotations

from protuna.optim.samplers import mlm
from tests.unit.optim.samplers import _helpers


def test_mlm_sampler_masks_the_mutable_positions_of_each_space() -> None:
    """The generator is asked to fill in exactly the mutable positions."""
    generator = _helpers._FakeMLM({"W": 1.0})
    space = _helpers._space("ACD", {0: "AW", 2: "DW"})

    mlm.MLMSampler(generator, max_samples=4).sample([space])

    assert generator.requests[0].token_ids.tolist() == [
        _helpers._MASK,
        1,
        _helpers._MASK,
    ]


def test_mlm_sampler_leaves_the_search_space_init_sequence_untouched() -> None:
    """Masking happens on a copy, so the space is reusable afterwards."""
    space = _helpers._space("ACD", {0: "AW", 2: "DW"})

    mlm.MLMSampler(_helpers._FakeMLM({"W": 1.0}), max_samples=4).sample([space])

    assert _helpers._TOKENIZER.decode(space.init_sequence) == "ACD"


def test_mlm_sampler_fills_masked_positions_from_the_generator() -> None:
    """A generator peaked on one residue proposes that residue everywhere."""
    space = _helpers._space("ACD", {0: "AW", 2: "DW"})

    batch = mlm.MLMSampler(_helpers._FakeMLM({"W": 1.0}), max_samples=4).sample(
        [space]
    )[0]

    assert _helpers._decode(batch) == ["WCW"]


def test_mlm_sampler_returns_one_batch_per_space() -> None:
    """Each space is decoded independently against its own mutable positions."""
    spaces = [_helpers._space("ACD", {0: "AW"}), _helpers._space("ACD", {2: "DW"})]

    batches = mlm.MLMSampler(_helpers._FakeMLM({"W": 1.0}), max_samples=4).sample(
        spaces
    )

    assert [_helpers._decode(batch) for batch in batches] == [["WCD"], ["ACW"]]


def test_mlm_sampler_top_p_drops_residues_past_the_nucleus() -> None:
    """A residue outside the nucleus is never proposed."""
    space = _helpers._space("ACD", {0: "AWY"})
    generator = _helpers._FakeMLM({"W": 0.6, "Y": 0.3, "A": 0.1})

    batch = mlm.MLMSampler(generator, max_samples=100, top_p=0.9).sample([space])[0]

    assert _helpers._residues_at(batch, 0) <= {"W", "Y"}


def test_mlm_sampler_low_temperature_collapses_onto_the_top_residue() -> None:
    """Sharpening the distribution leaves only the most likely residue."""
    space = _helpers._space("ACD", {0: "AWY"})
    generator = _helpers._FakeMLM({"W": 0.6, "Y": 0.3, "A": 0.1})

    batch = mlm.MLMSampler(generator, max_samples=100, temperature=0.01).sample(
        [space]
    )[0]

    assert _helpers._decode(batch) == ["WCD"]
