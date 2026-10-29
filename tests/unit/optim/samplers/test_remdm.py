import math
from unittest import mock

import pytest
import torch

from protuna import data
from protuna import inference
from protuna.optim import spaces
from protuna.optim.samplers import remdm
from protuna.typing import SequenceLike, TokenIDs
from tests.unit.optim.samplers import _helpers


class _HalfPrecisionMLM(_helpers._FakeMLM):
    def _logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        return super()._logits(token_ids).to(torch.float16)


class _GuidanceModel(
    inference.ModelFreeLocalHandler[tuple[TokenIDs, float], torch.Tensor]
):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[tuple[TokenIDs, float]] = []

    def __call__(self, requests: SequenceLike[tuple[TokenIDs, float]]) -> torch.Tensor:
        self.requests.extend(requests)
        tokens = torch.stack([tokens for tokens, _ in requests])
        return -(tokens == data.tokenizer.mask_token_id).sum(dim=1) * math.log(2)


@pytest.mark.parametrize(
    "device_map",
    [
        {"model": -1, "guidance_model": 0},
        {"model": 2, "guidance_model": 0},
        {"model": 3, "guidance_model": 0},
        {"model": 0, "guidance_model": -1},
        {"model": 0, "guidance_model": 2},
        {"model": 0, "guidance_model": 3},
    ],
)
def test_device_map_rejects_invalid_indices_before_loading(
    device_map: remdm.ReMDMDeviceMap,
) -> None:
    """Reject invalid model or guidance indices before loading either model."""
    sampler = _sampler()
    guidance = _GuidanceModel()
    sampler.guidance_model = guidance
    sampler.device_map = device_map

    with (
        mock.patch.object(sampler.model, "load", autospec=True) as model_load,
        mock.patch.object(guidance, "load", autospec=True) as guidance_load,
        pytest.raises(ValueError, match="2 devices available.*requested device id"),
    ):
        sampler.load([torch.device("cuda:0"), torch.device("cuda:1")])

    model_load.assert_not_called()
    guidance_load.assert_not_called()


@pytest.mark.parametrize(
    ("sequence", "first_mask"),
    [("A__A", 1), ("AAAAAAAAAAAAAAAAAA__", 18), ("AAAAAAAAAAAAAAAAAAAA__", 20)],
)
def test_guidance_evaluates_single_site_unmask_candidates(
    sequence: str, first_mask: int
) -> None:
    """Change only the requested site at every supported sequence length."""
    tokens = data.tokenizer.encode(sequence)
    space = spaces.SearchSpace(
        data.tokenizer.encode(sequence.replace("_", "A")),
        torch.ones((len(sequence), 20), dtype=torch.bool),
    )
    model = _GuidanceModel()
    sampler = _sampler()
    sampler.guidance_model = model

    sampler.transition_scores(5, tokens, torch.zeros(2, 20), space)

    candidates = torch.stack([tokens for tokens, _ in model.requests[1:41]])
    torch.testing.assert_close(
        (candidates != tokens).sum(dim=1), torch.ones(40, dtype=torch.long)
    )
    torch.testing.assert_close(candidates[:20, first_mask], torch.arange(20))
    torch.testing.assert_close(candidates[20:, first_mask + 1], torch.arange(20))


@pytest.mark.parametrize("step, expected_time", [(10, 1.0), (5, 0.5), (1, 0.1)])
def test_guidance_receives_normalized_time(step: int, expected_time: float) -> None:
    """Pass the current diffusion time with every classifier request."""
    space = _helpers._space("A", {0: "AC"})
    model = _GuidanceModel()
    sampler = _sampler()
    sampler.guidance_model = model

    sampler.transition_scores(
        step, data.tokenizer.encode("_"), torch.zeros(1, 20), space
    )

    assert set(time for _, time in model.requests) == {expected_time}


def test_guidance_accepts_scalar_log_probabilities() -> None:
    """Weight both decoding and remasking with scalar classifier responses."""
    space = _helpers._space("AA", {0: "AC", 1: "AC"})
    sampler = _sampler()
    sampler.guidance_model = _GuidanceModel()
    logits = torch.full((1, 20), -torch.inf)
    logits[0, :2] = math.log(0.5)

    probs = sampler.transition_scores(
        5, data.tokenizer.encode("_A"), logits, space
    ).softmax(-1)

    torch.testing.assert_close(
        probs[:, [0, 1, 20]],
        torch.tensor(
            [
                [0.2307692308, 0.2307692308, 0.5384615385],
                [0.9473684211, 0.0, 0.0526315789],
            ],
            dtype=torch.float64,
        ),
    )


@pytest.mark.parametrize(
    ("mutable", "expected_mask_prob"), [(True, 0.0526315789), (False, 0.0)]
)
def test_guidance_handles_fully_decoded_sequences(
    mutable: bool, expected_mask_prob: float
) -> None:
    """Allow remasking or fixed-site preservation without any unmask candidates."""
    space = spaces.SearchSpace(
        data.tokenizer.encode("AA"), torch.full((2, 20), mutable)
    )
    sampler = _sampler()
    sampler.guidance_model = _GuidanceModel()

    scores = sampler.transition_scores(
        5, space.init_sequence, torch.empty(0, 20), space
    )

    assert scores.shape == (2, 21)
    assert torch.isfinite(scores.softmax(-1)).all()
    assert torch.equal(scores.argmax(-1), space.init_sequence)
    torch.testing.assert_close(
        scores.softmax(-1)[:, -1],
        torch.full((2,), expected_mask_prob, dtype=torch.float64),
    )


def _sampler(confidence: bool = False, eta: float = 0.1) -> remdm.ReMDMSampler:
    return remdm.ReMDMSampler(
        model=_helpers._FakeMLM({"W": 1.0}),
        scheduler=remdm.ReMDMSwitchScheduler(
            remdm.MaxCappedRemaskScheduler(eta), t_on=1.0
        ),
        n_steps=10,
        max_samples=4,
        top_p=1.0,
        confidence_remasking=confidence,
        seed=7,
    )


@pytest.mark.parametrize("confidence", [False, True])
@pytest.mark.parametrize("n_steps", [1, 10])
@pytest.mark.parametrize(
    "scheduler",
    [
        remdm.ReMDMSwitchScheduler(remdm.MaxCappedRemaskScheduler(0.1), 1.0),
        remdm.ReMDMLoopScheduler(remdm.MaxCappedRemaskScheduler(0.1), 0.65, 0.25, 0.8),
    ],
    ids=["switch", "loop"],
)
def test_sampling_returns_complete_constrained_candidates(
    confidence: bool, n_steps: int, scheduler: remdm.ReMDMScheduler
) -> None:
    """Generate complete batches while preserving fixed sites and the input."""
    space = _helpers._space("ACDE", {0: "AW", 1: "CW", 2: "DW"})
    sampler = _sampler(confidence)
    sampler.n_steps = n_steps
    sampler.scheduler = scheduler

    batch = sampler.sample([space])[0]

    assert batch.shape == (sampler.max_samples, 4)
    assert _helpers._decode(batch) == ["WWWE"] * sampler.max_samples
    assert data.tokenizer.decode(space.init_sequence) == "ACDE"


@pytest.mark.parametrize("confidence", [False, True])
def test_immutable_space_returns_the_starting_sequence(confidence: bool) -> None:
    """An immutable space yields repeated copies of its starting sequence."""
    space = _helpers._space("AC", {})

    sampler = _sampler(confidence)

    batch = sampler.sample([space])[0]

    assert _helpers._decode(batch) == ["AC"] * sampler.max_samples


@pytest.mark.parametrize("t_on", [0.5, 1.0])
def test_switch_handles_the_initial_all_masked_endpoint(t_on: float) -> None:
    """Avoid division by zero whether remasking is enabled or disabled."""
    scheduler = remdm.ReMDMSwitchScheduler(remdm.MaxCappedRemaskScheduler(0.1), t_on)

    alpha_t, alpha_s, sigma = scheduler(10, 10)

    assert alpha_t == 0.0
    assert alpha_s == pytest.approx(0.1)
    assert torch.isfinite(sigma).all()


@pytest.mark.parametrize(
    ("scheduler", "expected"),
    [
        (remdm.MaxCappedRemaskScheduler(0.2), 0.2),
        (remdm.RescaleRemaskScheduler(0.3), 0.3),
    ],
)
def test_remask_schedules_handle_zero_signal(
    scheduler: remdm.RemaskScheduler, expected: float
) -> None:
    """Use the finite limiting remask bound when alpha_t is zero."""
    sigma = scheduler(1.0, 0.0, 0.1)

    assert sigma.item() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("alpha_t", "alpha_s", "expected"),
    [(0.2, 0.3, 0.1), (0.8, 0.9, 0.0125)],
)
def test_rescale_tempers_the_bounded_maximum(
    alpha_t: float, alpha_s: float, expected: float
) -> None:
    """Apply rescaling after limiting the maximum remask probability to one."""
    sigma = remdm.RescaleRemaskScheduler(0.1)(0.5, alpha_t, alpha_s)

    assert sigma.item() == pytest.approx(expected)


@pytest.mark.parametrize("step", [4, 2])
def test_loop_crossing_steps_share_the_same_endpoint_alpha(step: int) -> None:
    """Keep consecutive transitions consistent across off-grid boundaries."""
    scheduler = remdm.ReMDMLoopScheduler(
        remdm.MaxCappedRemaskScheduler(0.1), 0.35, 0.15, 0.8
    )

    _, alpha_s, _ = scheduler(step, 10)
    next_alpha_t, _, _ = scheduler(step - 1, 10)

    assert alpha_s == next_alpha_t


def test_loop_bounds_remasking_when_crossing_into_final_phase() -> None:
    """Use the next endpoint's mask mass when bounding a crossing transition."""
    scheduler = remdm.ReMDMLoopScheduler(
        remdm.MaxCappedRemaskScheduler(1.0), 0.35, 0.15, 0.8
    )

    alpha_t, alpha_s, sigma = scheduler(2, 10)

    assert alpha_t == 0.8
    assert alpha_s == pytest.approx(0.8666666667)
    assert sigma.item() == pytest.approx(0.1666666667)


def test_transition_matches_masked_unmasked_and_fixed_probabilities() -> None:
    """Retain decoded residues and apply the ReMDM kernel only to mutable sites."""
    space = _helpers._space("AAA", {0: "AC", 1: "AC"})
    logits = torch.full((1, 20), -torch.inf)
    logits[0, :2] = torch.tensor([0.25, 0.75]).log()
    tokens = data.tokenizer.encode("A_A")

    probs = _sampler().transition_scores(5, tokens, logits, space).exp()

    torch.testing.assert_close(
        probs[:, [0, 1, 20]],
        torch.tensor(
            [[0.9, 0.0, 0.1], [0.075, 0.225, 0.7], [1.0, 0.0, 0.0]], dtype=torch.float64
        ),
    )
    assert torch.count_nonzero(probs[:, 2:20]) == 0
    torch.testing.assert_close(probs.sum(-1), torch.ones(3, dtype=torch.float64))


@pytest.mark.parametrize("length", [1, 3, 20])
def test_transition_applies_mass_to_each_site_not_vocabulary_columns(
    length: int,
) -> None:
    """Support arbitrary numbers of masked sites without incorrect broadcasting."""
    space = _helpers._space("A" * length, {})
    space = spaces.SearchSpace(
        space.init_sequence, torch.ones(length, 20, dtype=torch.bool)
    )
    tokens = torch.full((length,), data.tokenizer.mask_token_id)
    logits = torch.zeros(length, 20)

    probs = _sampler().transition_scores(5, tokens, logits, space).exp()

    torch.testing.assert_close(
        probs[:, :20], torch.full((length, 20), 0.015, dtype=torch.float64)
    )
    torch.testing.assert_close(
        probs[:, -1], torch.full((length,), 0.7, dtype=torch.float64)
    )


@pytest.mark.parametrize("length", [1, 3, 20])
def test_zero_remasking_keeps_every_decoded_token(length: int) -> None:
    """Recover the absorbing MDLM kernel when remasking is disabled."""
    space = _helpers._space("A" * length, {})
    space = spaces.SearchSpace(
        space.init_sequence, torch.ones(length, 20, dtype=torch.bool)
    )
    sampler = _sampler(eta=0.0)

    scores = sampler.transition_scores(
        5, space.init_sequence, torch.empty(0, 20), space
    )
    tokens = sampler.step(scores)

    assert torch.equal(tokens, space.init_sequence)
    assert tokens.shape == (length,)


def test_confidence_multiplies_remasking_and_leaves_masked_decoding_unchanged() -> None:
    """Share the base remasking budget across decoded sites only."""
    space = _helpers._space("AAA", {0: "AC", 1: "AC", 2: "AC"})
    logits = torch.full((1, 20), -torch.inf)
    logits[0, :2] = torch.tensor([0.5, 0.5]).log()

    probs = (
        _sampler(True)
        .transition_scores(
            5,
            data.tokenizer.encode("AA_"),
            logits,
            space,
            torch.tensor([0.5, 0.5, 0.1]),
        )
        .exp()
    )

    torch.testing.assert_close(
        probs[:, [0, 1, 20]],
        torch.tensor(
            [[0.95, 0.0, 0.05], [0.95, 0.0, 0.05], [0.1, 0.1, 0.8]], dtype=torch.float64
        ),
    )


def test_confidence_cannot_enable_remasking_in_final_step() -> None:
    """Decode all remaining masks even when finite confidence exists."""
    space = _helpers._space("AA", {0: "AC", 1: "AC"})
    logits = torch.full((1, 20), -torch.inf)
    logits[0, :2] = torch.tensor([0.5, 0.5]).log()

    probs = (
        _sampler(True)
        .transition_scores(
            1,
            data.tokenizer.encode("A_"),
            logits,
            space,
            torch.tensor([0.5, torch.inf]),
        )
        .exp()
    )

    torch.testing.assert_close(
        probs[:, [0, 1, 20]],
        torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=torch.float64),
    )


def test_lower_confidence_receives_more_of_the_remasking_budget() -> None:
    """Prefer remasking the token with lower decoding confidence."""
    space = _helpers._space("AA", {0: "AC", 1: "AC"})

    probs = (
        _sampler(True)
        .transition_scores(
            5,
            data.tokenizer.encode("AA"),
            torch.empty(0, 20),
            space,
            torch.tensor([0.0, 0.69314718056]),
        )
        .exp()
    )

    torch.testing.assert_close(
        probs[:, -1], torch.tensor([0.06666666667, 0.03333333333], dtype=torch.float64)
    )


@pytest.mark.parametrize("state", [None, torch.tensor([torch.inf, torch.inf])])
def test_confidence_handles_an_all_masked_sequence(state: torch.Tensor | None) -> None:
    """Avoid undefined softmax weights when no decoded token exists."""
    space = _helpers._space("AA", {0: "AC", 1: "AC"})
    logits = torch.full((2, 20), -torch.inf)
    logits[:, :2] = torch.tensor([0.5, 0.5]).log()

    probs = (
        _sampler(True)
        .transition_scores(5, data.tokenizer.encode("__"), logits, space, state)
        .exp()
    )

    torch.testing.assert_close(
        probs[:, -1], torch.tensor([0.8, 0.8], dtype=torch.float64)
    )


def test_confidence_uses_each_decoded_sites_own_probability() -> None:
    """Gather from separate model rows for newly decoded residues."""
    logits = torch.full((2, 20), -torch.inf)
    logits[:, :2] = torch.tensor([[0.1, 0.9], [0.8, 0.2]]).log()

    state = _sampler(True).update_state(
        data.tokenizer.encode("CA"), data.tokenizer.encode("__"), logits
    )

    torch.testing.assert_close(state, torch.tensor([0.9, 0.8]))


def test_confidence_skips_remaining_masks_and_handles_one_decoded_site() -> None:
    """Never gather a mask token from amino-acid logits."""
    logits = torch.full((2, 20), -torch.inf)
    logits[:, :2] = torch.tensor([[0.1, 0.9], [0.8, 0.2]]).log()

    state = _sampler(True).update_state(
        data.tokenizer.encode("C_"), data.tokenizer.encode("__"), logits
    )

    torch.testing.assert_close(state, torch.tensor([0.9, torch.inf]))


def test_confidence_retains_the_dimension_for_a_single_masked_site() -> None:
    """Keep confidence as a vector when exactly one site is decoded."""
    logits = torch.full((1, 20), -torch.inf)
    logits[0, :2] = torch.tensor([0.1, 0.9]).log()

    state = _sampler(True).update_state(
        data.tokenizer.encode("C"), data.tokenizer.encode("_"), logits
    )

    torch.testing.assert_close(state, torch.tensor([0.9]))


def test_confidence_resets_remasked_sites_and_preserves_unchanged_sites() -> None:
    """Clear stale confidence without modifying the previous state."""
    previous = torch.tensor([0.8, 0.6])

    state = _sampler(True).update_state(
        data.tokenizer.encode("_C"),
        data.tokenizer.encode("AC"),
        torch.empty(0, 20),
        previous,
    )

    torch.testing.assert_close(state, torch.tensor([torch.inf, 0.6]))
    torch.testing.assert_close(previous, torch.tensor([0.8, 0.6]))


def test_saturated_remasking_bound_has_no_nan_transition_scores() -> None:
    """Protect zero mask probability from floating-point cancellation."""
    space = _helpers._space("AA", {0: "AC", 1: "AC"})
    logits = torch.full((1, 20), -torch.inf)
    logits[0, :2] = torch.tensor([0.5, 0.5]).log()

    scores = _sampler(eta=1.0).transition_scores(
        3, data.tokenizer.encode("A_"), logits, space
    )

    assert not torch.isnan(scores).any()
    assert scores[1, -1].exp().item() == pytest.approx(0.0, abs=1e-7)
    torch.testing.assert_close(scores.exp().sum(-1), torch.ones(2, dtype=torch.float64))


def test_transition_log_probability_keeps_a_single_site_dimension() -> None:
    """Return a length-one vector for a length-one sequence."""
    scores = torch.full((1, 21), -torch.inf)
    scores[0, 0] = 0.0

    logp = _sampler().log_transition_probs(data.tokenizer.encode("A"), scores)

    assert logp.tolist() == [0.0]


def test_half_precision_temperature_scaling_preserves_finite_logits() -> None:
    """Avoid overflow when applying a small temperature to half-precision logits."""
    space = _helpers._space("A", {0: "AC"})
    sampler = _sampler()
    sampler.model = _HalfPrecisionMLM({"A": 0.5, "C": 0.5})
    sampler.temperature = 1e-5

    logits = sampler.denoising_logits([data.tokenizer.encode("_")], space)[0]

    assert torch.isfinite(logits[0, :2]).all()
    torch.testing.assert_close(logits.softmax(-1)[0, :2], torch.tensor([0.5, 0.5]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_logits_preserve_scheduled_unmask_probability(
    dtype: torch.dtype,
) -> None:
    """Preserve mask mass when normalizing low-precision residue logits."""
    space = _helpers._space("AA", {0: "AC", 1: "AC"})
    logits = torch.full((1, 20), -torch.inf, dtype=dtype)
    logits[0, :2] = 0

    probs = (
        _sampler(True)
        .transition_scores(
            5,
            data.tokenizer.encode("A_"),
            logits,
            space,
            torch.tensor([0.5, torch.inf]),
        )
        .softmax(-1)
    )

    assert probs[1, -1].item() == pytest.approx(0.8, abs=1e-10, rel=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_logits_preserve_decoding_confidence(dtype: torch.dtype) -> None:
    """Normalize decoding confidence before rounding it to the state dtype."""
    logits = torch.full((1, 20), -torch.inf, dtype=dtype)
    logits[0, :2] = torch.tensor([0.0, 1.0], dtype=dtype)

    state = _sampler(True).update_state(
        data.tokenizer.encode("C"), data.tokenizer.encode("_"), logits
    )

    torch.testing.assert_close(state, torch.tensor([0.7310586]), atol=1e-7, rtol=0)
