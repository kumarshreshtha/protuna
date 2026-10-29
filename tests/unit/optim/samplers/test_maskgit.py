from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import math

import pytest
import torch

from protuna.optim.samplers import maskgit
from tests.unit.optim.samplers import _helpers


@pytest.mark.parametrize(
    ("schedule", "progress", "expected"),
    [
        (maskgit.cosine_mask_schedule, 0.0, 1.0),
        (maskgit.cosine_mask_schedule, 0.5, 0.7071067811865476),
        (maskgit.cosine_mask_schedule, 1.0, 0.0),
        (maskgit.linear_mask_schedule, 0.0, 1.0),
        (maskgit.linear_mask_schedule, 0.5, 0.5),
        (maskgit.linear_mask_schedule, 1.0, 0.0),
    ],
    ids=["cosine-0", "cosine-half", "cosine-1", "linear-0", "linear-half", "linear-1"],
)
def test_mask_schedule_falls_from_every_site_to_none(
    schedule: Callable[[float], float], progress: float, expected: float
) -> None:
    """Both schedules mask everything at the start and nothing at the end."""
    assert schedule(progress) == pytest.approx(expected)


@pytest.mark.parametrize(
    "scorer",
    [maskgit.sample_logp, maskgit.negative_entropy],
    ids=["probability", "negative_entropy"],
)
def test_site_scorers_return_one_score_per_masked_site(
    scorer: maskgit.SequenceScorer,
) -> None:
    """Every scorer reduces the vocabulary axis and keeps the batch and site axes."""
    logits = torch.zeros(3, 4, 20)
    samples = torch.zeros(3, 4, dtype=torch.long)

    assert scorer(logits, samples).shape == (3, 4)


def test_sample_logp_scores_a_site_by_the_draw_the_model_believed_in() -> None:
    """A site scores high when the residue drawn there was the likely one."""
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    samples = torch.tensor([[0, 0]])

    scores = maskgit.sample_logp(logits, samples)

    assert scores[0].tolist() == pytest.approx(
        [math.log(0.8807970779778823), math.log(0.11920292202211755)]
    )


def test_negative_entropy_scores_a_peaked_site_above_a_flat_one() -> None:
    """The site's own distribution is scored, not the residue drawn from it."""
    logits = torch.tensor([[[2.0, 0.0], [0.0, 0.0]]])
    samples = torch.tensor([[0, 0]])

    scores = maskgit.negative_entropy(logits, samples)

    assert scores[0].tolist() == pytest.approx(
        [-0.3653338938884451, -0.6931471805599453]
    )


def test_negative_entropy_ignores_the_residues_that_were_drawn() -> None:
    """Scoring the distribution alone makes the drawn residues irrelevant."""
    logits = torch.tensor([[[2.0, 0.0]]])

    drew_the_mode = maskgit.negative_entropy(logits, torch.tensor([[0]]))
    drew_the_tail = maskgit.negative_entropy(logits, torch.tensor([[1]]))

    assert torch.equal(drew_the_mode, drew_the_tail)


def test_negative_entropy_ignores_residues_the_space_masked_out() -> None:
    """A residue at ``-inf`` contributes nothing instead of turning the score NaN."""
    logits = torch.tensor([[[2.0, 0.0, -torch.inf]]])
    samples = torch.tensor([[0]])

    scores = maskgit.negative_entropy(logits, samples)

    assert scores[0].tolist() == pytest.approx([-0.3653338938884451])


def _masks(length: int, positions: Sequence[int], rows: int = 1) -> torch.Tensor:
    """Build a ``(rows, length)`` mask marking ``positions`` in every row."""
    masks = torch.zeros(rows, length, dtype=torch.bool)
    masks[:, positions] = True
    return masks


def _masked_per_row(masks: torch.Tensor) -> list[list[int]]:
    """List the positions each row of a ``(b, l)`` mask marks."""
    return [row.nonzero().flatten().tolist() for row in masks]


_FALLING_CONFIDENCE = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])


@pytest.mark.parametrize(
    "mask_generator",
    [
        maskgit.ConfidenceMaskGenerator(mask_schedule=maskgit.linear_mask_schedule),
        maskgit.RandomMaskGenerator(mask_schedule=maskgit.linear_mask_schedule),
    ],
    ids=["maskgit", "random"],
)
def test_mask_generators_settle_at_least_one_site_per_round(
    mask_generator: maskgit.MaskGenerator,
) -> None:
    """A schedule asking for every site back still leaves one site settled."""
    masks = _masks(5, [0, 1, 2, 3])
    n_mutable = 4

    # linear_mask_schedule(0.0) asks for all four mutable sites back.
    new_masks = mask_generator(
        0.0, n_mutable, torch.zeros(1, 5, dtype=torch.long), torch.zeros(1, 4, 3), masks
    )

    assert new_masks.sum(dim=1).tolist() == [3]


@pytest.mark.parametrize(
    "mask_generator",
    [
        maskgit.ConfidenceMaskGenerator(mask_schedule=maskgit.linear_mask_schedule),
        maskgit.RandomMaskGenerator(mask_schedule=maskgit.linear_mask_schedule),
    ],
    ids=["maskgit", "random"],
)
def test_mask_generators_hand_back_at_least_one_site_per_round(
    mask_generator: maskgit.MaskGenerator,
) -> None:
    """A schedule asking for no site back still redraws one."""
    masks = _masks(5, [0, 1, 2, 3])
    n_mutable = 4

    # linear_mask_schedule(1.0) asks for no site back.
    new_masks = mask_generator(
        1.0, n_mutable, torch.zeros(1, 5, dtype=torch.long), torch.zeros(1, 4, 3), masks
    )

    assert new_masks.sum(dim=1).tolist() == [1]


@pytest.mark.parametrize(
    "mask_generator",
    [
        maskgit.ConfidenceMaskGenerator(mask_schedule=maskgit.linear_mask_schedule),
        maskgit.RandomMaskGenerator(mask_schedule=maskgit.linear_mask_schedule),
    ],
    ids=["maskgit", "random"],
)
def test_mask_generators_never_hand_back_a_site_that_was_settled(
    mask_generator: maskgit.MaskGenerator,
) -> None:
    """Sites kept by an earlier round stay out of the mask, mutable or not."""
    masks = _masks(5, [1, 3])
    n_mutable = 4

    new_masks = mask_generator(
        0.5,
        n_mutable,
        torch.zeros(1, 5, dtype=torch.long),
        _FALLING_CONFIDENCE,
        masks,
    )

    assert _masked_per_row(new_masks & ~masks) == [[]]


def test_maskgit_mask_generator_hands_back_the_lowest_scoring_sites() -> None:
    """The sites the model was least confident about are the ones redrawn."""
    masks = _masks(5, [0, 1, 2, 3])
    n_mutable = 4
    # Confidence in the drawn residue falls from site 0 to site 3.
    logits = torch.tensor(
        [[[10.0, 0.0, 0.0], [5.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )

    # linear_mask_schedule(0.5) asks for two of the four mutable sites back.
    new_masks = maskgit.ConfidenceMaskGenerator(
        mask_schedule=maskgit.linear_mask_schedule, choice_temperature=0
    )(0.5, n_mutable, torch.zeros(1, 5, dtype=torch.long), logits, masks)

    assert _masked_per_row(new_masks) == [[2, 3]]


def test_maskgit_mask_generator_scores_each_candidate_on_its_own_draws() -> None:
    """Two candidates masked at the same sites can be handed back different ones."""
    masks = _masks(3, [0, 1], rows=2)
    n_mutable = 2
    # Both sites are peaked on residue 0, so the confident site is the one that
    # actually drew it.
    logits = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]).expand(2, 2, 3)
    samples = torch.tensor([[0, 1, 0], [1, 0, 0]])

    new_masks = maskgit.ConfidenceMaskGenerator(
        mask_schedule=maskgit.linear_mask_schedule, choice_temperature=0
    )(0.5, n_mutable, samples, logits, masks)

    assert _masked_per_row(new_masks) == [[1], [0]]


@pytest.mark.parametrize(
    ("scorer", "expected"),
    [
        (maskgit.sample_logp, [[1]]),
        (maskgit.negative_entropy, [[0]]),
    ],
    ids=["probability", "negative_entropy"],
)
def test_maskgit_mask_generator_picks_the_site_its_scorer_ranks_lowest(
    scorer: maskgit.SequenceScorer, expected: list[list[int]]
) -> None:
    """Site 0 is flat but drew its mode; site 1 is peaked but drew off it."""
    masks = _masks(3, [0, 1])
    n_mutable = 2
    logits = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 1.0, 0.0]]])
    samples = torch.tensor([[0, 1, 0]])

    new_masks = maskgit.ConfidenceMaskGenerator(
        mask_schedule=maskgit.linear_mask_schedule, scorer=scorer, choice_temperature=0
    )(0.5, n_mutable, samples, logits, masks)

    assert _masked_per_row(new_masks) == expected


_ROUND_1_LOGITS = torch.tensor(
    [[[10.0, 0.0, 0.0], [5.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
)


_ROUND_2_LOGITS = torch.tensor([[[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])


def _run_first_round(mask_generator: maskgit.MaskGenerator) -> torch.Tensor:
    """Play round 1 on a 5-site sequence with sites 0..3 open, handing back 2 and 3."""
    masks = _masks(5, [0, 1, 2, 3])
    n_mutable = 4
    # linear_mask_schedule(0.5) asks for two of the four mutable sites back.
    return mask_generator(
        0.5, n_mutable, torch.zeros(1, 5, dtype=torch.long), _ROUND_1_LOGITS, masks
    )


def test_maskgit_mask_generator_remasks_a_settled_site_when_allowed() -> None:
    """With remasking on, a settled site loses out to freshly confident redraws."""
    mask_generator = maskgit.ConfidenceMaskGenerator(
        mask_schedule=maskgit.linear_mask_schedule,
        choice_temperature=0,
        allow_remask=True,
    )
    masks = _run_first_round(mask_generator)
    n_mutable = 4

    # linear_mask_schedule(0.75) asks for one site back.
    new_masks = mask_generator(
        0.75, n_mutable, torch.zeros(1, 5, dtype=torch.long), _ROUND_2_LOGITS, masks
    )

    assert _masked_per_row(new_masks) == [[1]]


def test_maskgit_mask_generator_forgets_earlier_scores_without_remask() -> None:
    """With remasking off, only the sites redrawn this round are up for masking."""
    mask_generator = maskgit.ConfidenceMaskGenerator(
        mask_schedule=maskgit.linear_mask_schedule,
        choice_temperature=0,
        allow_remask=False,
    )
    masks = _run_first_round(mask_generator)
    n_mutable = 4

    new_masks = mask_generator(
        0.75, n_mutable, torch.zeros(1, 5, dtype=torch.long), _ROUND_2_LOGITS, masks
    )

    assert _masked_per_row(new_masks) == [[2]]


def test_maskgit_mask_generator_never_remasks_a_site_the_space_keeps_fixed() -> None:
    """Remasking only reaches sites that were open, never the fixed ones."""
    mask_generator = maskgit.ConfidenceMaskGenerator(
        mask_schedule=maskgit.linear_mask_schedule,
        choice_temperature=0,
        allow_remask=True,
    )
    masks = _masks(5, [0, 1, 2, 3])
    n_mutable = 4

    # linear_mask_schedule(0.0) asks for all four mutable sites back.
    new_masks = mask_generator(
        0.0, n_mutable, torch.zeros(1, 5, dtype=torch.long), _ROUND_1_LOGITS, masks
    )

    assert _masked_per_row(new_masks) == [[0, 1, 2, 3]]


def test_maskgit_mask_generator_may_hand_back_every_site_when_remasking() -> None:
    """Remasking drops the rule that each round must settle at least one site."""
    mask_generator = maskgit.ConfidenceMaskGenerator(
        mask_schedule=maskgit.linear_mask_schedule, allow_remask=True
    )
    masks = _masks(5, [0, 1, 2, 3])
    n_mutable = 4

    new_masks = mask_generator(
        0.0, n_mutable, torch.zeros(1, 5, dtype=torch.long), torch.zeros(1, 4, 3), masks
    )

    assert new_masks.sum(dim=1).tolist() == [4]


def test_maskgit_mask_generator_reset_forgets_the_scores_of_earlier_rounds() -> None:
    """After a reset, settled sites are no longer candidates for remasking."""
    mask_generator = maskgit.ConfidenceMaskGenerator(
        mask_schedule=maskgit.linear_mask_schedule,
        choice_temperature=0,
        allow_remask=True,
    )
    masks = _run_first_round(mask_generator)
    n_mutable = 4

    mask_generator.reset()
    new_masks = mask_generator(
        0.75, n_mutable, torch.zeros(1, 5, dtype=torch.long), _ROUND_2_LOGITS, masks
    )

    assert _masked_per_row(new_masks) == [[2]]


def test_maskgit_mask_generator_reset_drops_scores_of_a_different_shape() -> None:
    """A reset lets the generator move on to a batch of another size and length."""
    mask_generator = maskgit.ConfidenceMaskGenerator(
        mask_schedule=maskgit.linear_mask_schedule, allow_remask=True
    )
    _run_first_round(mask_generator)

    mask_generator.reset()
    new_masks = mask_generator(
        0.5,
        2,
        torch.zeros(3, 4, dtype=torch.long),
        torch.zeros(3, 2, 3),
        _masks(4, [0, 1], rows=3),
    )

    assert new_masks.shape == (3, 4)


def test_random_mask_generator_hands_back_the_scheduled_number_of_sites() -> None:
    """The schedule still sets how many sites each round redraws."""
    masks = _masks(5, [0, 1, 2, 3], rows=2)
    n_mutable = 4

    # linear_mask_schedule(0.5) asks for two of the four mutable sites back.
    new_masks = maskgit.RandomMaskGenerator(mask_schedule=maskgit.linear_mask_schedule)(
        0.5, n_mutable, torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, 4, 3), masks
    )

    assert new_masks.sum(dim=1).tolist() == [2, 2]


def test_random_mask_generator_maps_its_draws_onto_the_masked_sites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The residues drawn are indices into the masked sites, not into the sequence."""
    masks = _masks(5, [1, 3, 4])
    n_mutable = 5
    # Draw the first and third of the three masked sites, i.e. positions 1 and 4.
    monkeypatch.setattr(
        torch, "multinomial", lambda *args, **kwargs: torch.tensor([[0, 2]])
    )

    # linear_mask_schedule(0.6) asks for two of the five mutable sites back.
    new_masks = maskgit.RandomMaskGenerator(mask_schedule=maskgit.linear_mask_schedule)(
        0.6, n_mutable, torch.zeros(1, 5, dtype=torch.long), torch.zeros(1, 3, 3), masks
    )

    assert _masked_per_row(new_masks) == [[1, 4]]


def _masked_counts(generator: _helpers._FakeMLM) -> list[int]:
    """Count the positions each request the generator received left masked."""
    return [
        len(_helpers._masked_positions(request.token_ids))
        for request in generator.requests
    ]


def test_maskgit_sampler_masks_every_mutable_position_in_its_first_pass() -> None:
    """Decoding opens with all the open sites masked at once."""
    generator = _helpers._FakeMLM({"W": 1.0})
    space = _helpers._space("ACD", {0: "AW", 2: "DW"})

    maskgit.MaskGITSampler(generator, max_samples=4).sample([space])

    assert generator.requests[0].token_ids.tolist() == [
        _helpers._MASK,
        1,
        _helpers._MASK,
    ]


def test_maskgit_sampler_fills_masked_positions_from_the_generator() -> None:
    """A generator peaked on one residue proposes that residue everywhere."""
    space = _helpers._space("ACD", {0: "AW", 2: "DW"})

    batch = maskgit.MaskGITSampler(
        model=_helpers._FakeMLM({"W": 1.0}), max_samples=4
    ).sample([space])[0]

    assert _helpers._decode(batch) == ["WCW"]


def test_maskgit_sampler_returns_one_batch_per_space() -> None:
    """Each space is decoded independently against its own mutable positions."""
    spaces = [_helpers._space("ACD", {0: "AW"}), _helpers._space("ACD", {2: "DW"})]

    batches = maskgit.MaskGITSampler(
        _helpers._FakeMLM({"W": 1.0}), max_samples=4
    ).sample(spaces)

    assert [_helpers._decode(batch) for batch in batches] == [["WCD"], ["ACW"]]


def test_maskgit_sampler_leaves_the_search_space_untouched() -> None:
    """Masking happens on copies, so the space is reusable afterwards."""
    space = _helpers._space("ACD", {0: "AW", 2: "DW"})
    allowed = space.allowed.clone()

    maskgit.MaskGITSampler(_helpers._FakeMLM({"W": 1.0}), max_samples=4).sample([space])

    assert _helpers._TOKENIZER.decode(space.init_sequence) == "ACD"
    assert torch.equal(space.allowed, allowed)


def test_maskgit_sampler_hands_sites_back_on_the_cosine_schedule() -> None:
    """Each round masks fewer sites than the last, on the default schedule."""
    generator = _helpers._FakeMLM({"W": 1.0})
    space = _helpers._space("ACDEF", {0: "AW", 1: "CW", 2: "DW", 3: "EW"})

    maskgit.MaskGITSampler(generator, max_samples=4, n_iter=3).sample([space])

    assert _masked_counts(generator) == [4, 3, 2]


def test_maskgit_sampler_hands_sites_back_on_the_schedule_it_was_given() -> None:
    """A different schedule changes how many sites each round redraws."""
    generator = _helpers._FakeMLM({"W": 1.0})
    space = _helpers._space("ACDEF", {0: "AW", 1: "CW", 2: "DW", 3: "EW"})

    maskgit.MaskGITSampler(
        generator,
        max_samples=4,
        n_iter=3,
        mask_generator=maskgit.ConfidenceMaskGenerator(
            mask_schedule=maskgit.linear_mask_schedule
        ),
    ).sample([space])

    assert _masked_counts(generator) == [4, 2, 1]


def test_maskgit_sampler_redraws_only_the_sites_it_was_least_confident_about() -> None:
    """A site kept by one round is settled: later rounds never mask it again."""
    # Confidence falls from position 0 to position 2, whichever residue is drawn.
    generator = _helpers._FakeMLM(
        {"W": 1.0},
        position_probs={1: {"W": 0.5, "Y": 0.5}, 2: {"W": 0.4, "Y": 0.3, "D": 0.3}},
    )
    space = _helpers._space("ACDE", {0: "AW", 1: "CWY", 2: "DWY"})

    maskgit.MaskGITSampler(
        generator,
        max_samples=1,
        n_iter=2,
        # No Gumbel noise: the assertion is about the confidence ordering alone.
        mask_generator=maskgit.ConfidenceMaskGenerator(choice_temperature=0),
    ).sample([space])

    masked = [
        _helpers._masked_positions(request.token_ids) for request in generator.requests
    ]
    assert masked == [[0, 1, 2], [1, 2]]


def test_maskgit_sampler_caps_its_rounds_at_the_number_of_mutable_sites() -> None:
    """No round may give every site back, so two open sites allow two rounds."""
    generator = _helpers._FakeMLM({"W": 1.0})
    space = _helpers._space("ACD", {0: "AW", 2: "DW"})

    maskgit.MaskGITSampler(generator, max_samples=1, n_iter=10).sample([space])

    assert _masked_counts(generator) == [2, 1]


def test_maskgit_sampler_refines_a_batch_holding_more_than_one_candidate() -> None:
    """Rounds run over the whole batch, not just the first candidate."""
    torch.manual_seed(0)
    space = _helpers._space("ACD", {0: "AWY", 2: "DWY"})

    batch = maskgit.MaskGITSampler(
        _helpers._FakeMLM({"W": 0.5, "Y": 0.5}), max_samples=20, n_iter=3
    ).sample([space])[0]

    assert batch.shape[0] > 1
    assert _helpers._all_residues_allowed(batch, space)


def test_maskgit_sampler_returns_starting_sequence_when_nothing_can_mutate() -> None:
    """An immutable space yields repeated copies of its starting sequence."""
    batch = maskgit.MaskGITSampler(_helpers._FakeMLM({"W": 1.0}), max_samples=5).sample(
        [_helpers._space("ACD", {})]
    )[0]

    assert _helpers._decode(batch) == ["ACD"] * 5


def test_maskgit_sampler_top_p_drops_residues_past_the_nucleus() -> None:
    """A residue outside the nucleus is never proposed."""
    space = _helpers._space("ACD", {0: "AWY"})
    generator = _helpers._FakeMLM({"W": 0.6, "Y": 0.3, "A": 0.1})

    batch = maskgit.MaskGITSampler(generator, max_samples=100, top_p=0.9).sample(
        [space]
    )[0]

    assert _helpers._residues_at(batch, 0) <= {"W", "Y"}


def test_maskgit_sampler_low_temperature_collapses_onto_the_top_residue() -> None:
    """Sharpening the distribution leaves only the most likely residue."""
    space = _helpers._space("ACD", {0: "AWY"})
    generator = _helpers._FakeMLM({"W": 0.6, "Y": 0.3, "A": 0.1})

    batch = maskgit.MaskGITSampler(generator, max_samples=100, temperature=0.01).sample(
        [space]
    )[0]

    assert _helpers._decode(batch) == ["WCD"]


@dataclasses.dataclass
class _MaskGeneratorCall:
    """The arguments one refinement round handed the mask generator.

    Attributes:
        progress: How far through the rounds the sampler said it was.
        samples: The candidates drawn so far.
        logits: The model scores over the currently masked sites.
        masks: The sites that were masked to draw ``samples``.
        n_mutable: How many sites the search space leaves open.
    """

    progress: float
    samples: torch.Tensor
    logits: torch.Tensor
    masks: torch.Tensor
    n_mutable: int


class _RecordingMaskGenerator(maskgit.ConfidenceMaskGenerator):
    """Default mask generator that records what each round hands it.

    Attributes:
        calls: One entry per refinement round, in order.
    """

    def __init__(self):
        super().__init__()
        self.calls: list[_MaskGeneratorCall] = []

    def __call__(
        self,
        progress: float,
        n_mutable: int,
        samples: torch.Tensor,
        logits: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        self.calls.append(
            _MaskGeneratorCall(
                progress, samples.clone(), logits.clone(), masks.clone(), n_mutable
            )
        )
        return super().__call__(progress, n_mutable, samples, logits, masks)


class _AlternatingMaskGenerator(maskgit.MaskGenerator):
    """Hand site 0 back to every even candidate and site 1 to every odd one."""

    def __call__(
        self,
        progress: float,
        n_mutable: int,
        samples: torch.Tensor,
        logits: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        rows = torch.arange(masks.shape[0])
        new_masks = torch.zeros_like(masks)
        new_masks[rows, rows % 2] = True
        return new_masks


def test_maskgit_sampler_drives_the_mask_generator_it_was_given() -> None:
    """A generator picking sites at random still follows the round schedule."""
    generator = _helpers._FakeMLM({"W": 1.0})
    space = _helpers._space("ACDEF", {0: "AW", 1: "CW", 2: "DW", 3: "EW"})

    maskgit.MaskGITSampler(
        generator,
        max_samples=4,
        n_iter=3,
        mask_generator=maskgit.RandomMaskGenerator(),
    ).sample([space])

    assert _masked_counts(generator) == [4, 3, 2]


def test_maskgit_sampler_walks_the_schedule_from_the_first_round_to_the_last() -> None:
    """Progress runs from one round's worth up to 1.0 on the final round."""
    mask_generator = _RecordingMaskGenerator()
    space = _helpers._space("ACDEF", {0: "AW", 1: "CW", 2: "DW", 3: "EW"})

    maskgit.MaskGITSampler(
        _helpers._FakeMLM({"W": 1.0}),
        max_samples=4,
        n_iter=3,
        mask_generator=mask_generator,
    ).sample([space])

    assert [call.progress for call in mask_generator.calls] == pytest.approx(
        [0.3333333333333333, 0.6666666666666666]
    )


def test_maskgit_sampler_never_calls_mask_generator_for_one_shot_generation() -> None:
    """The mask the first round refines is the search space's own open sites."""
    mask_generator = _RecordingMaskGenerator()
    space = _helpers._space("ACDEF", {0: "AW", 2: "DW"})

    maskgit.MaskGITSampler(
        _helpers._FakeMLM({"W": 1.0}),
        max_samples=4,
        n_iter=1,
        mask_generator=mask_generator,
    ).sample([space])

    assert mask_generator.calls == []


def test_maskgit_sampler_tells_the_mask_generator_how_many_sites_are_open() -> None:
    """The number of mutable sites is handed over so schedules can scale."""
    mask_generator = _RecordingMaskGenerator()
    space = _helpers._space("ACDEF", {0: "AW", 2: "DW"})

    maskgit.MaskGITSampler(
        _helpers._FakeMLM({"W": 1.0}),
        max_samples=4,
        n_iter=2,
        mask_generator=mask_generator,
    ).sample([space])

    assert mask_generator.calls[0].n_mutable == 2


def test_maskgit_sampler_scores_every_candidate_against_its_own_row_of_logits() -> None:
    """The generator sees one row of logits per candidate, over the masked sites."""
    torch.manual_seed(0)
    mask_generator = _RecordingMaskGenerator()
    space = _helpers._space("ACDE", {0: "AW", 1: "CW"})
    generator = _helpers._FakeMLM(
        {"A": 0.5, "W": 0.5}, position_probs={1: {"C": 0.5, "W": 0.5}}
    )

    maskgit.MaskGITSampler(
        generator, max_samples=200, n_iter=2, mask_generator=mask_generator
    ).sample([space])

    assert mask_generator.calls[0].logits.shape == (4, 2, 20)


def test_maskgit_sampler_masks_each_candidate_at_its_own_sites() -> None:
    """Every candidate is re-masked where its own row of the mask says to."""
    torch.manual_seed(0)
    space = _helpers._space("ACDE", {0: "AW", 1: "CW"})
    generator = _helpers._FakeMLM(
        {"A": 0.5, "W": 0.5}, position_probs={1: {"C": 0.5, "W": 0.5}}
    )

    maskgit.MaskGITSampler(
        generator,
        max_samples=200,
        n_iter=2,
        mask_generator=_AlternatingMaskGenerator(),
    ).sample([space])

    refinement_requests = generator.requests[1:]
    assert [
        _helpers._masked_positions(request.token_ids) for request in refinement_requests
    ] == [[0], [1], [0], [1]]


def test_maskgit_sampler_carries_each_candidates_settled_residues_forward() -> None:
    """A site left out of the mask conditions the next round on its earlier draw."""
    torch.manual_seed(0)
    space = _helpers._space("ACDE", {0: "AW", 1: "CW"})
    generator = _helpers._FakeMLM(
        {"A": 0.5, "W": 0.5}, position_probs={1: {"C": 0.5, "W": 0.5}}
    )

    maskgit.MaskGITSampler(
        generator,
        max_samples=200,
        n_iter=2,
        mask_generator=_AlternatingMaskGenerator(),
    ).sample([space])

    settled = [
        _helpers._TOKENIZER.decode(
            request.token_ids[request.token_ids != _helpers._MASK]
        )
        for request in generator.requests[1:]
    ]
    assert settled == ["CDE", "ADE", "CDE", "WDE"]


class _ResetCountingMaskGenerator(maskgit.RandomMaskGenerator):
    """Random mask generator that counts how often it is told to reset.

    Attributes:
        resets: Number of times :meth:`reset` was called.
    """

    def __init__(self):
        super().__init__()
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


def test_maskgit_sampler_resets_the_mask_generator_once_per_space() -> None:
    """Each space starts the mask generator from a clean slate."""
    mask_generator = _ResetCountingMaskGenerator()
    spaces = [
        _helpers._space("ACDE", {0: "AW", 1: "CW"}),
        _helpers._space("FGH", {0: "FW", 1: "GW"}),
    ]

    maskgit.MaskGITSampler(
        _helpers._FakeMLM({"W": 1.0}),
        max_samples=4,
        n_iter=2,
        mask_generator=mask_generator,
    ).sample(spaces)

    assert mask_generator.resets == 2


def test_maskgit_sampler_remasks_spaces_of_different_lengths_in_one_call() -> None:
    """Scores carried across rounds never leak from one space into the next."""
    mask_generator = maskgit.ConfidenceMaskGenerator(allow_remask=True)
    # The second space is longer, so scores kept from the first would not fit it.
    spaces = [
        _helpers._space("FGH", {0: "FW", 1: "GW", 2: "HW"}),
        _helpers._space("ACDEF", {0: "AW", 1: "CW", 2: "DW", 3: "EW"}),
    ]

    batches = maskgit.MaskGITSampler(
        _helpers._FakeMLM({"W": 1.0}),
        max_samples=4,
        n_iter=3,
        mask_generator=mask_generator,
    ).sample(spaces)

    assert [batch.shape[1] for batch in batches] == [3, 5]
