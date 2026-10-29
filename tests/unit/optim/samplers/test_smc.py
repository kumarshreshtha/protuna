from collections.abc import Callable, Sequence
import copy
import functools
import itertools
from typing import Literal
from unittest import mock

from jaxtyping import Float
import pytest
import torch

from protuna import inference
from protuna.data import tokenizer
from protuna.optim.samplers import remdm
from protuna.optim.samplers import smc
from protuna.typing import SequenceLike, TokenIDs
from protuna.utils import devices
from tests.unit.optim.samplers import _helpers


class _MixedDeviceMLM(_helpers._FakeMLM):
    def __init__(self, device: torch.device):
        super().__init__({"W": 1.0})
        self.output_devices = itertools.cycle((device, torch.device("cpu")))

    def __call__(
        self, requests: SequenceLike[inference.MLMLogitsRequest]
    ) -> list[Float[torch.Tensor, "m v"]]:
        return [
            logits.to(next(self.output_devices))
            for logits in super().__call__(requests)
        ]


class _RewardModel(inference.InferenceModel[TokenIDs, torch.Tensor]):
    """Assign reward two to cysteine and record model loading."""

    def __init__(self) -> None:
        super().__init__()
        self.devices: list[torch.device | Sequence[torch.device] | None] = []
        self.requests: list[TokenIDs] = []

    def forward(self, requests: SequenceLike[TokenIDs]) -> list[torch.Tensor]:
        """Return one reward for each sequence."""
        self.requests.extend(requests)
        return [torch.tensor(2.0 * float(tokens[0] == 1)) for tokens in requests]

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Record the requested device."""
        self.devices.append(device)

    def release(self) -> None:
        """Release no resources for this model."""


class _MixedDeviceRewardModel(_RewardModel):
    def __init__(self) -> None:
        super().__init__()
        self.output_devices = itertools.cycle(
            (torch.device("cuda:0"), torch.device("cpu"))
        )

    def forward(self, requests: SequenceLike[TokenIDs]) -> list[torch.Tensor]:
        return [
            reward.to(next(self.output_devices)) for reward in super().forward(requests)
        ]


def _proposal(
    n_steps: int = 3, probs: dict[str, float] | None = None
) -> remdm.ReMDMSampler:
    return remdm.ReMDMSampler(
        _helpers._FakeMLM(probs or {"A": 0.5, "C": 0.5}),
        scheduler=remdm.ReMDMSwitchScheduler(remdm.RescaleRemaskScheduler(0), 0),
        n_steps=n_steps,
        max_samples=1,
        seed=7,
    )


def test_default_sampler_completes_sampling() -> None:
    """Construct and run steering with the default ESS and difference potential."""
    sampler = smc.FKSteeringSampler(
        _proposal(), smc.LearnedSoftValue(_RewardModel()), 3, 10, seed=7
    )

    samples = sampler.sample_one(_helpers._space("A", {0: "AC"}))

    assert sampler.ess_threshold == 5
    assert samples.shape == (10, 1)
    assert torch.isin(samples, torch.tensor([0, 1])).all()


@pytest.mark.parametrize(
    "value_factory",
    [
        smc.LearnedSoftValue,
        functools.partial(smc.MonteCarloSoftValue, n_draws=2, seed=7),
    ],
    ids=["learned", "monte_carlo"],
)
def test_steering_loads_the_registered_reward_model(
    value_factory: Callable[[smc.PredictionModelLike], smc.SoftValueFunction],
) -> None:
    """Initialize and load reward registries through the steering sampler."""
    model = _RewardModel()
    soft_value = value_factory(model)
    sampler = smc.FKSteeringSampler(_proposal(), soft_value, 3, 10)

    sampler.load(torch.device("cpu"))

    assert model.devices == [torch.device("cpu")]


@pytest.mark.parametrize(
    "sampler_factory",
    [
        smc.FKSteeringSampler,
        functools.partial(smc.NestedSMCSampler, duplication_size=2),
    ],
    ids=["fk", "nested"],
)
@pytest.mark.parametrize("resolve", [False, True])
def test_device_map_places_concurrent_components(
    sampler_factory: Callable[..., smc.FKSteeringSampler], resolve: bool
) -> None:
    """Place each component on its selected explicit or resolved device."""
    proposal = _proposal()
    base_sampler = _proposal()
    model = _RewardModel()
    sampler = sampler_factory(
        proposal,
        smc.LearnedSoftValue(model),
        3,
        10,
        base_sampler=base_sampler,
        device_map={"proposal": 2, "base": 0, "soft_value": 1},
    )
    available = [torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2")]
    requested = None if resolve else available

    with (
        mock.patch.object(
            devices, "resolve_devices", autospec=True, return_value=available
        ),
        mock.patch.object(proposal, "load", autospec=True) as proposal_load,
        mock.patch.object(base_sampler, "load", autospec=True) as base_load,
    ):
        sampler.load(requested)

    proposal_load.assert_called_once_with(torch.device("cuda:2"))
    base_load.assert_called_once_with(torch.device("cuda:0"))
    assert model.devices == [torch.device("cuda:1")]


@pytest.mark.parametrize(
    "device_map", [None, {"proposal": 2, "base": 0, "soft_value": 1}]
)
def test_single_device_loading_preserves_default_placement(
    device_map: smc.SMCDeviceMap | None,
) -> None:
    """Load every component on a single device regardless of mapping."""
    proposal = _proposal()
    model = _RewardModel()
    sampler = smc.FKSteeringSampler(
        proposal,
        smc.LearnedSoftValue(model),
        3,
        10,
        device_map=device_map,
    )

    with mock.patch.object(proposal, "load", autospec=True) as proposal_load:
        sampler.load(torch.device("cpu"))

    proposal_load.assert_called_once_with(torch.device("cpu"))
    assert model.devices == [torch.device("cpu")]


@pytest.mark.parametrize("index", [-1, 2, 3])
def test_device_map_rejects_invalid_indices_before_loading(index: int) -> None:
    """Reject out-of-range placements before loading any component."""
    proposal = _proposal()
    model = _RewardModel()
    sampler = smc.FKSteeringSampler(
        proposal,
        smc.LearnedSoftValue(model),
        3,
        10,
        device_map={"proposal": 0, "base": 0, "soft_value": index},
    )

    with (
        mock.patch.object(proposal, "load", autospec=True) as proposal_load,
        pytest.raises(ValueError, match=f"2 devices available.*{index} for soft_value"),
    ):
        sampler.load([torch.device("cuda:0"), torch.device("cuda:1")])

    proposal_load.assert_not_called()
    assert model.devices == []


def test_device_map_skips_absent_components() -> None:
    """Ignore placement indices for an absent base and an unregistered reward."""
    proposal = _proposal()
    soft_value = mock.create_autospec(smc.SoftValueFunction, instance=True)
    sampler = smc.FKSteeringSampler(
        proposal,
        soft_value,
        3,
        10,
        base_sampler=proposal,
        device_map={"proposal": 0, "base": 5, "soft_value": 5},
    )

    with mock.patch.object(proposal, "load", autospec=True) as proposal_load:
        sampler.load([torch.device("cpu")])

    proposal_load.assert_called_once_with(torch.device("cpu"))


def test_monte_carlo_steering_scores_successor_states() -> None:
    """Provide logits for intermediate and fully decoded reward estimates."""
    sampler = smc.FKSteeringSampler(
        _proposal(),
        smc.MonteCarloSoftValue(_RewardModel(), n_draws=2, seed=7),
        3,
        10,
        seed=7,
    )

    samples = sampler.sample_one(_helpers._space("A", {0: "AC"}))

    assert samples.shape == (10, 1)
    assert torch.isin(samples, torch.tensor([0, 1])).all()


@pytest.mark.parametrize("potential", ["difference", "sum", "max"])
@pytest.mark.parametrize("ess_threshold", [0.0, 1001.0])
def test_terminal_samples_follow_the_reward_tilt(
    potential: Literal["difference", "sum", "max"], ess_threshold: float
) -> None:
    """Keep the target tilt with skipped or forced intermediate resampling."""
    sampler = smc.FKSteeringSampler(
        _proposal(),
        smc.LearnedSoftValue(_RewardModel()),
        3,
        1000,
        potential=potential,
        ess_threshold=ess_threshold,
        seed=7,
    )

    samples = sampler.sample_one(_helpers._space("A", {0: "AC"}))

    assert (samples[:, 0] == 1).float().mean().item() == pytest.approx(
        0.880797, abs=0.04
    )


@pytest.mark.parametrize("schedule", [[], [1], [2, 1]])
def test_sparse_schedules_retain_every_transition_importance_ratio(
    schedule: list[int],
) -> None:
    """Correct all proposal transitions even when weighting skips steps."""
    sampler = smc.FKSteeringSampler(
        _proposal(probs={"A": 0.2, "C": 0.8}),
        smc.LearnedSoftValue(_RewardModel()),
        3,
        1000,
        base_sampler=_proposal(probs={"A": 0.8, "C": 0.2}),
        resample_interval=schedule,
        ess_threshold=0,
        seed=7,
    )

    samples = sampler.sample_one(_helpers._space("A", {0: "AC"}))

    assert (samples[:, 0] == 1).float().mean().item() == pytest.approx(
        0.648786, abs=0.05
    )


def test_resampling_resets_weights_and_preserves_independent_histories() -> None:
    """Consume sampling weights while retaining copied potential histories."""
    sampler = smc.FKSteeringSampler(
        _proposal(), smc.LearnedSoftValue(_RewardModel()), 3, 2, ess_threshold=3
    )
    particle = smc._Particle(
        step=1,
        tokens=torch.tensor([1]),
        log_ratio=3,
        rewards=[0, 1],
        prev_log_weight=4,
    )

    with mock.patch.object(
        torch, "multinomial", autospec=True, return_value=torch.tensor([0, 0])
    ):
        selected = sampler._resample_particles(
            [particle, copy.deepcopy(particle)], _helpers._space("A", {0: "AC"})
        )

    assert selected[0].prev_log_weight == pytest.approx(-0.693147)
    assert selected[0].log_ratio == 0
    assert selected[0].rewards == [0, 1, 2]
    assert selected[0].rewards is not selected[1].rewards
    assert selected[0] is not selected[1]


def test_successors_have_independent_reward_histories() -> None:
    """Keep separately proposed descendants from changing each other's rewards."""
    sampler = smc.FKSteeringSampler(
        _proposal(), smc.LearnedSoftValue(_RewardModel()), 3, 2
    )
    space = _helpers._space("A", {0: "AC"})
    particle = smc._Particle(
        step=1,
        tokens=torch.tensor([1]),
        log_ratio=0,
        rewards=[0, 1],
        prev_log_weight=0,
    )
    sampler._set_logits([particle], space)

    first = sampler._update_particle(particle, space)
    second = sampler._update_particle(particle, space)
    first.rewards.append(2)

    assert second.rewards == [0, 1]
    assert particle.rewards == [0, 1]


def test_skipped_resampling_commits_each_potential_once() -> None:
    """Retain unconsumed weights without reapplying a previous potential."""
    sampler = smc.FKSteeringSampler(
        _proposal(), smc.LearnedSoftValue(_RewardModel()), 3, 1, ess_threshold=0
    )
    space = _helpers._space("A", {0: "AC"})
    particle = smc._Particle(
        step=1,
        tokens=torch.tensor([1]),
        log_ratio=3,
        rewards=[0],
        prev_log_weight=4,
    )

    selected = sampler._resample_particles([particle], space)
    sampler._set_logits(selected, space)
    successor = sampler._update_particle(selected[0], space)

    assert selected[0].prev_log_weight == 9
    assert selected[0].log_ratio == 0
    assert successor.prev_log_weight == 9
    assert successor.rewards == [0, 2]


@pytest.mark.parametrize(
    ("temperature", "expected"), [(1.0, [2.0, 0.0, 2.0]), (2.0, [1.0, 0.0, 1.0])]
)
def test_learned_value_scales_rewards_in_input_order(
    temperature: float, expected: list[float]
) -> None:
    """Return one temperature-scaled prediction per input without logits."""
    value = smc.LearnedSoftValue(_RewardModel())
    tokens = tokenizer.encode(["C_", "A_", "CA"])

    rewards = value(z_t=tokens, x0_logits=None, temperature=temperature)

    assert rewards.tolist() == expected
    assert value.requires_logits is False


@pytest.mark.parametrize(
    ("temperature", "expected"), [(1.0, 1.43378083), (2.0, 0.62011451)]
)
def test_monte_carlo_value_uses_log_mean_exponentiated_rewards(
    temperature: float, expected: float
) -> None:
    """Aggregate scaled completion rewards rather than averaging rewards."""
    value = smc.MonteCarloSoftValue(_RewardModel(), n_draws=2, seed=7)
    logits = torch.zeros(1, 20)

    with mock.patch.object(
        torch, "multinomial", autospec=True, return_value=torch.tensor([[0, 1]])
    ):
        rewards = value(
            z_t=tokenizer.encode(["_"]), x0_logits=[logits], temperature=temperature
        )

    torch.testing.assert_close(rewards, torch.tensor([expected]))


def test_monte_carlo_completions_preserve_fixed_sites_and_duplicate_draws() -> None:
    """Keep each particle's completions separate and leave inputs unchanged."""
    model = _RewardModel()
    value = smc.MonteCarloSoftValue(model, n_draws=3, seed=7)
    tokens = tokenizer.encode(["_D", "C_"])
    logits = torch.full((1, 20), -torch.inf)
    logits[:, 0] = 0.0

    rewards = value(tokens, [logits, logits], temperature=2.0)

    assert rewards.tolist() == [0.0, 1.0]
    assert tokenizer.decode(model.requests) == ["AD", "AD", "AD", "CA", "CA", "CA"]
    assert tokenizer.decode(tokens) == ["_D", "C_"]


def test_monte_carlo_value_scores_fully_decoded_particles() -> None:
    """Score terminal tokens without drawing replacements for fixed sites."""
    value = smc.MonteCarloSoftValue(_RewardModel(), n_draws=3, seed=7)

    rewards = value(tokenizer.encode(["CD", "AD"]), [torch.empty(0, 20)] * 2)

    assert rewards.tolist() == [2.0, 0.0]


def test_monte_carlo_value_requires_denoising_logits() -> None:
    """Reject completion scoring when no denoising distribution is supplied."""
    value = smc.MonteCarloSoftValue(_RewardModel(), n_draws=2, seed=7)

    with pytest.raises(ValueError, match="requires logits.*x0_logits=None"):
        value(tokenizer.encode(["_"]))


@pytest.mark.parametrize("logits", [[], [torch.zeros(1, 20)] * 2])
def test_monte_carlo_value_rejects_mismatched_particle_and_logit_counts(
    logits: list[torch.Tensor],
) -> None:
    """Reject missing or surplus distributions instead of truncating inputs."""
    value = smc.MonteCarloSoftValue(_RewardModel(), n_draws=2, seed=7)

    with pytest.raises(ValueError, match="zip\\(\\) argument 2 is (shorter|longer)"):
        value(tokenizer.encode(["_"]), logits)


def test_monte_carlo_seed_reproduces_completions() -> None:
    """Draw the same completions from independently seeded estimators."""
    first_model = _RewardModel()
    second_model = _RewardModel()
    first = smc.MonteCarloSoftValue(first_model, n_draws=20, seed=7)
    second = smc.MonteCarloSoftValue(second_model, n_draws=20, seed=7)

    first(tokenizer.encode(["__"]), [torch.zeros(2, 20)])
    second(tokenizer.encode(["__"]), [torch.zeros(2, 20)])

    assert tokenizer.decode(first_model.requests) == tokenizer.decode(
        second_model.requests
    )


def test_monte_carlo_rng_retains_its_state_between_calls() -> None:
    """Reuse the device generator without restarting its random stream."""
    value = smc.MonteCarloSoftValue(_RewardModel(), n_draws=2, seed=7)
    rng = value.rng(torch.device("cpu"))
    torch.rand(10, generator=rng)
    state = rng.get_state().clone()

    same_rng = value.rng(torch.device("cpu"))

    assert same_rng is rng
    assert torch.equal(same_rng.get_state(), state)


@pytest.mark.parametrize("global_resampling", [False, True])
@pytest.mark.parametrize(
    "value_factory",
    [
        smc.LearnedSoftValue,
        functools.partial(smc.MonteCarloSoftValue, n_draws=2, seed=7),
    ],
    ids=["learned", "monte_carlo"],
)
def test_nested_sampler_returns_complete_constrained_particles(
    global_resampling: bool,
    value_factory: Callable[[smc.PredictionModelLike], smc.SoftValueFunction],
) -> None:
    """Generate the requested particle count while preserving fixed residues."""
    space = _helpers._space("ACDE", {0: "AW", 1: "CW"})
    sampler = smc.NestedSMCSampler(
        _proposal(probs={"W": 1.0}),
        value_factory(_RewardModel()),
        3,
        4,
        duplication_size=2,
        global_resampling=global_resampling,
        seed=7,
    )

    samples = sampler.sample_one(space)

    assert _helpers._decode(samples) == ["WWDE", "WWDE", "WWDE", "WWDE"]
    assert tokenizer.decode(space.init_sequence) == "ACDE"


@pytest.mark.parametrize(
    ("global_resampling", "expected"), [(False, 0.8098), (True, 0.880797)]
)
def test_nested_sampler_selects_reward_weighted_inner_proposals(
    global_resampling: bool, expected: float
) -> None:
    """Apply local selection and optionally correct the outer distribution."""
    sampler = smc.NestedSMCSampler(
        _proposal(n_steps=1),
        smc.LearnedSoftValue(_RewardModel()),
        1,
        1000,
        duplication_size=4,
        global_resampling=global_resampling,
        seed=7,
    )

    samples = sampler.sample_one(_helpers._space("A", {0: "AC"}))

    assert (samples[:, 0] == 1).float().mean().item() == pytest.approx(
        expected, abs=0.05
    )


def test_nested_inner_weights_include_reward_and_transition_correction() -> None:
    """Return the inner normalizer with importance ratios included."""
    sampler = smc.NestedSMCSampler(
        _proposal(), smc.LearnedSoftValue(_RewardModel()), 3, 1, duplication_size=2
    )
    particles = [
        smc._Particle(1, tokenizer.encode("A"), 0.0, [0], 0.0),
        smc._Particle(1, tokenizer.encode("C"), -2.0, [0], 0.0),
    ]

    with mock.patch.object(
        torch, "multinomial", autospec=True, return_value=torch.tensor([1])
    ) as draw:
        selected, normalizer = sampler._sample_inner_particles(
            particles, _helpers._space("A", {0: "AC"})
        )

    torch.testing.assert_close(
        draw.call_args.args[0], torch.tensor([0.5, 0.5], dtype=torch.float64)
    )
    assert normalizer.item() == pytest.approx(0.69314718)
    assert selected.tokens.tolist() == [1]


@pytest.mark.parametrize(
    "sampler_factory",
    [
        smc.FKSteeringSampler,
        functools.partial(smc.NestedSMCSampler, duplication_size=2),
    ],
    ids=["fk", "nested"],
)
def test_sampling_keeps_duplicates_and_initial_sequence(
    sampler_factory: Callable[..., smc.FKSteeringSampler],
) -> None:
    """Keep all particles, including repeated proposals and starting sequences."""
    sampler = sampler_factory(
        _proposal(probs={"W": 1.0}),
        smc.LearnedSoftValue(_RewardModel()),
        3,
        4,
        seed=7,
    )

    samples = sampler.sample(
        [_helpers._space("AC", {0: "AW"}), _helpers._space("WC", {0: "AW"})]
    )

    assert _helpers._decode(samples[0]) == ["WC"] * 4
    assert _helpers._decode(samples[1]) == ["WC"] * 4


@pytest.mark.parametrize(
    "sampler_factory",
    [
        smc.FKSteeringSampler,
        functools.partial(smc.NestedSMCSampler, duplication_size=2),
    ],
    ids=["fk", "nested"],
)
def test_sampling_immutable_space_returns_starting_sequence(
    sampler_factory: Callable[..., smc.FKSteeringSampler],
) -> None:
    """An immutable space yields one starting sequence per particle."""
    sampler = sampler_factory(
        _proposal(), smc.LearnedSoftValue(_RewardModel()), 3, 4, seed=7
    )

    samples = sampler.sample([_helpers._space("AC", {})])

    assert _helpers._decode(samples[0]) == ["AC"] * 4


@pytest.mark.parametrize(
    "sampler_factory",
    [
        smc.FKSteeringSampler,
        functools.partial(smc.NestedSMCSampler, duplication_size=2),
    ],
    ids=["fk", "nested"],
)
def test_sampler_seed_reproduces_terminal_particles(
    sampler_factory: Callable[..., smc.FKSteeringSampler],
) -> None:
    """Reproduce diffusion and selection draws with independently built samplers."""
    first = sampler_factory(
        _proposal(), smc.LearnedSoftValue(_RewardModel()), 3, 20, seed=7
    )
    second = sampler_factory(
        _proposal(), smc.LearnedSoftValue(_RewardModel()), 3, 20, seed=7
    )
    space = _helpers._space("AD", {0: "AC", 1: "CD"})

    first_samples = first.sample_one(space)
    second_samples = second.sample_one(space)

    assert torch.equal(first_samples, second_samples)


@pytest.mark.parametrize(
    ("temperature", "expected"), [(0.5, 0.982014), (2.0, 0.731059)]
)
def test_tilt_temperature_controls_terminal_reward_distribution(
    temperature: float, expected: float
) -> None:
    """Temper rewards before selecting the terminal particle population."""
    sampler = smc.FKSteeringSampler(
        _proposal(n_steps=1),
        smc.LearnedSoftValue(_RewardModel()),
        1,
        1000,
        tilt_temperature=temperature,
        seed=7,
    )

    samples = sampler.sample_one(_helpers._space("A", {0: "AC"}))

    assert (samples[:, 0] == 1).float().mean().item() == pytest.approx(
        expected, abs=0.04
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    ("input_device", "output_device"), [("cpu", "cuda:0"), ("cuda:0", "cpu")]
)
@pytest.mark.parametrize(
    "sampler_factory",
    [
        smc.FKSteeringSampler,
        functools.partial(smc.NestedSMCSampler, duplication_size=2),
    ],
    ids=["fk", "nested"],
)
@pytest.mark.parametrize(
    "value_factory",
    [
        smc.LearnedSoftValue,
        functools.partial(smc.MonteCarloSoftValue, n_draws=2, seed=7),
    ],
    ids=["learned", "monte_carlo"],
)
def test_smc_keeps_particles_on_the_first_proposal_response_device(
    input_device: str,
    output_device: str,
    sampler_factory: Callable[..., smc.FKSteeringSampler],
    value_factory: Callable[[smc.PredictionModelLike], smc.SoftValueFunction],
) -> None:
    """Align particles despite mixed proposal, base, and reward response devices."""
    proposal = _proposal(probs={"W": 1.0})
    proposal.model = _MixedDeviceMLM(torch.device(output_device))
    base_sampler = _proposal(probs={"W": 1.0})
    base_sampler.model = _MixedDeviceMLM(torch.device(input_device))
    sampler = sampler_factory(
        proposal,
        value_factory(_MixedDeviceRewardModel()),
        3,
        4,
        base_sampler=base_sampler,
        seed=7,
    )
    space = _helpers._space("ACD", {0: "AW", 1: "CW"}).to(torch.device(input_device))

    samples = sampler.sample([space])[0]

    assert samples.device == torch.device(output_device)
    assert _helpers._decode(samples.cpu()) == ["WWD"] * 4
    assert space.init_sequence.device == torch.device(input_device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "value_factory",
    [
        smc.LearnedSoftValue,
        functools.partial(smc.MonteCarloSoftValue, n_draws=2, seed=7),
    ],
    ids=["learned", "monte_carlo"],
)
def test_soft_values_normalize_mixed_rewards_to_the_input_device(
    value_factory: Callable[[smc.PredictionModelLike], smc.SoftValueFunction],
) -> None:
    """Score CPU inputs with accelerator logits and mixed reward responses."""
    value = value_factory(_MixedDeviceRewardModel())
    tokens = tokenizer.encode(["_D", "C_"])
    logits = torch.full((1, 20), -torch.inf, device="cuda:0")
    logits[:, _helpers._TOKENIZER.token_to_id("W")] = 0.0

    rewards = value(tokens, [logits, logits], temperature=2.0)

    torch.testing.assert_close(rewards, torch.tensor([0.0, 1.0]))
