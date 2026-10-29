"""Steer diffusion particles with reward-based sequential Monte Carlo."""

from collections.abc import Callable, Iterable, Sequence
import copy
import dataclasses
import logging
import math
from typing import Literal, Protocol, TypedDict

from jaxtyping import Float
import torch

from protuna import inference
from protuna.data import tokenizer
from protuna.optim import spaces
from protuna.optim.samplers import base
from protuna.optim.samplers import common
from protuna.typing import BatchTokenIDs, SequenceLike, TokenIDs
from protuna.utils import devices

logger = logging.getLogger(__name__)


class SoftValueFunction(Protocol):
    """Structural interface for intermediate reward estimators.

    Attributes:
        requires_logits: Whether scoring requires denoising logits. The sampler
            supplies ``x0_logits`` when true and passes ``None`` otherwise.

    .. seealso::
        :class:`LearnedSoftValue` and :class:`MonteCarloSoftValue` for concrete
        implementations.
    """

    requires_logits: bool

    def __call__(
        self,
        z_t: Sequence[TokenIDs],
        x0_logits: Sequence[Float[torch.Tensor, "m v"]] | None = None,
        temperature: float = 1.0,
    ) -> Float[torch.Tensor, " b"]:
        """Score intermediate states optionally using denoising logits.

        Args:
            z_t: Particle token IDs at the current diffusion step.
            x0_logits: Denoising logits aligned with the particles, required when
                ``requires_logits`` is true.
            temperature: Divisor applied to rewards before aggregation.

        Returns:
            One soft value per particle, in input order.
        """
        ...


type PredictionModelLike = inference.Handler[TokenIDs, Float[torch.Tensor, ""]]


class LearnedSoftValue(inference.HandlerRegistryMixin, SoftValueFunction):
    r"""Estimate intermediate rewards with a trained soft value function.

    The model supplies the intermediate reward estimate given the partially masked
    sequences.
    """

    requires_logits = False

    def __init__(self, model: PredictionModelLike):
        """Register the model used to score masked and decoded sequences.

        Args:
            model: Prediction model used to estimate particle rewards from partially
                masked sequences.
        """
        super().__init__()
        self.model = model

    def __call__(
        self,
        z_t: Sequence[TokenIDs],
        x0_logits: Sequence[Float[torch.Tensor, "m v"]] | None = None,
        temperature: float = 1.0,
    ) -> Float[torch.Tensor, " b"]:
        """Return predicted rewards divided by temperature.

        Args:
            z_t: Particle token IDs to score.
            x0_logits: Unused; accepted for the soft value interface.
            temperature: Divisor applied to predicted rewards.

        Returns:
            One temperature-scaled reward per particle, in input order.
        """
        device = z_t[0].device
        return (
            torch.stack([reward.to(device) for reward in self.model(z_t)]) / temperature
        )


class MonteCarloSoftValue(inference.HandlerRegistryMixin, SoftValueFunction):
    r"""Estimate log mean exponentiated rewards from sampled completions.

    For a partially masked sequence :math:`z_t`, the method estimates the soft value

    .. math::

        v_t(z_t) = \log \mathbb{E}_{x_0 \sim q_t(\cdot \mid z_t)}
            \left[\exp\left(\frac{r(x_0)}{\tau}\right)\right]

    using the Monte Carlo approximation

    .. math::

        \hat{v}_t(z_t) = \log\left[\frac{1}{K}\sum_{k=1}^{K}
            \exp\left(\frac{r(x_0^{(k)})}{\tau}\right)\right],
        \qquad x_0^{(k)} \sim q_t(\cdot \mid z_t).

    Here :math:`r` is ``model``, :math:`\tau` is ``temperature``, and
    :math:`K` is ``n_draws``. The completion distribution :math:`q_t` samples
    masked positions independently from the categorical probabilities defined
    by ``x0_logits``, retaining unmasked tokens.
    """

    requires_logits = True

    def __init__(self, model: PredictionModelLike, n_draws: int, seed: int):
        """Set the reward model, number of completions, and sampling seed.

        Args:
            model: Prediction model used to score sampled completions.
            n_draws: Number of completions sampled per particle.
            seed: Seed for completion sampling on each device.
        """
        super().__init__()
        self.model = model
        self.n_draws = n_draws
        self.seed = seed

        self._rngs: dict[torch.device, torch.Generator] = {}

    def rng(self, device: torch.device) -> torch.Generator:
        """Return the persistent generator for the requested device.

        Args:
            device: Device on which to sample completions.

        Returns:
            The device's generator, initialized with the configured seed.
        """
        if device not in self._rngs:
            rng = torch.Generator(device=device)
            if self.seed is not None:
                rng.manual_seed(self.seed)
            self._rngs[device] = rng
        return self._rngs[device]

    def __call__(
        self,
        z_t: Sequence[TokenIDs],
        x0_logits: Sequence[Float[torch.Tensor, "m v"]] | None = None,
        temperature: float = 1.0,
    ) -> Float[torch.Tensor, " b"]:
        """Score sampled completions using temperature-scaled rewards.

        Args:
            z_t: Particle token IDs whose masked positions are completed.
            x0_logits: denoising logits from step t.
            temperature: Divisor applied to completion rewards before aggregation.

        Returns:
            The log mean exponentiated reward per particle, in input order.
        """
        if x0_logits is None:
            raise ValueError(
                f"{self.__class__.__name__} requires logits for value function "
                "approximation got x0_logits=None"
            )
        all_samples = []
        for masked_inputs, logits in zip(z_t, x0_logits, strict=True):
            masked_inputs = masked_inputs.to(logits.device)
            input_mask = masked_inputs == tokenizer.mask_token_id
            probs = torch.softmax(logits, dim=-1)
            samples = common.sample_from_weights(
                probs,
                masked_inputs,
                input_mask,
                self.n_draws,
                dedup=False,
                generator=self.rng(probs.device),
            )
            all_samples.extend([sample for sample in samples])
        device = z_t[0].device
        rewards = torch.stack(
            [reward.to(device) for reward in self.model(all_samples)]
        ).reshape(-1, self.n_draws)
        rewards = (rewards / temperature).logsumexp(dim=-1) - math.log(self.n_draws)
        return rewards


# rewards: [r_T, r_{T-1}, ..., r_t]
type Potential = Callable[[Sequence[float], int], float]


@dataclasses.dataclass
class _Particle[StateT1, StateT2]:
    step: int
    tokens: TokenIDs
    log_ratio: float
    rewards: list[float]
    prev_log_weight: float
    proposal_logits: Float[torch.Tensor, "m v"] | None = None
    pretrain_logits: Float[torch.Tensor, "m v"] | None = None
    proposal_state: StateT1 | None = None
    pretrain_state: StateT2 | None = None
    proposal_transition_score: Float[torch.Tensor, "l v+1"] | None = None
    pretrain_transition_score: Float[torch.Tensor, "l v+1"] | None = None

    def log_weight(self, potential_fn: Potential) -> float:
        potential = potential_fn(self.rewards, self.step)
        return self.log_ratio + potential + self.prev_log_weight


def sum_potential(rewards: Sequence[float], step: int) -> float:
    """Return the sum log potential with its terminal reward correction.

    Args:
        rewards: Reward history ordered from the initial to the current step.
        step: Current diffusion step; zero applies the terminal correction.

    Returns:
        The sum of rewards, or the terminal reward minus the sum of preceding
        cumulative reward sums at step zero.
    """
    if step != 0:
        return sum(rewards)
    return rewards[-1] - torch.tensor(rewards[:-1]).cumsum(dim=0).sum().item()


def max_potential(rewards: Sequence[float], step: int) -> float:
    """Return the maximum log potential with its terminal reward correction.

    Args:
        rewards: Reward history ordered from the initial to the current step.
        step: Current diffusion step; zero applies the terminal correction.

    Returns:
        The maximum reward, or the terminal reward minus the sum of preceding
        running reward maxima at step zero.
    """
    if step != 0:
        return max(rewards)
    return rewards[-1] - torch.tensor(rewards[:-1]).cummax(dim=0).values.sum().item()


class SMCDeviceMap(TypedDict):
    """Assign relative device indices to concurrent SMC components.

    Indices select positions in the device sequence passed to
    :meth:`FKSteeringSampler.load`, or the automatically resolved sequence when
    no devices are supplied. Components may share an index. The map is ignored
    when loading onto a single device.

    Args:
        proposal: Device index for the proposal sampler.
        base: Device index for the base sampler.
        soft_value: Device index for registered soft value models.

    Examples:
        Map components onto a reordered device sequence. The proposal and soft
        value models share ``cuda:2``; the base sampler uses ``cuda:0``::

            device_map = SMCDeviceMap(proposal=0, base=1, soft_value=0)
            # With sampler.load([torch.device("cuda:2"), torch.device("cuda:0")]):
            # proposal   -> devices[0] -> cuda:2
            # base       -> devices[1] -> cuda:0
            # soft_value -> devices[0] -> cuda:2
    """

    proposal: int
    base: int
    soft_value: int


class FKSteeringSampler(base.Sampler):
    r"""Reward-guided diffusion sampling with Feynman-Kac particle resampling.

    A population of partially masked sequences (particles) evolves through
    reverse diffusion steps generated by ``proposal_sampler``. ``soft_value``
    estimates future rewards and ``potential`` converts each particle's reward history
    into a log weight contribution.

    Resampling occurs at the terminal step and when the effective sample size falls
    below ``ess_threshold``.

    At reverse step :math:`t`, the particles approximate the path distribution
    :math:`\pi_t(z_{T:t}) \propto p(z_{T:t})\prod_{s=t}^{T-1}G_s(z_{T:s})`,
    where :math:`z_{T:t}` is the sequence of states from the initial masked
    state to step :math:`t`, :math:`p` is the base diffusion process, and
    :math:`G_s` are positive potentials.

    When the potentials satisfy :math:`\prod_{t=0}^{T-1}G_t = \exp(r(z_0)/\tau)`,
    the terminal particles approximate :math:`p(z_0)\exp(r(z_0)/\tau)`.

    See :cite:p:`FeynmanKac` for details.
    """

    def __init__(
        self,
        proposal_sampler: base.MaskedDiffusionSamplerBase,
        soft_value: SoftValueFunction,
        n_steps: int,
        n_particles: int,
        *,
        ess_threshold: float | None = None,
        potential: Potential | Literal["max", "sum", "difference"] = "difference",
        base_sampler: base.MaskedDiffusionSamplerBase | None = None,
        resample_interval: Iterable[int] | None = None,
        tilt_temperature: float = 1.0,
        seed: int | None = None,
        batch_size: int | None = None,
        device_map: SMCDeviceMap | None = None,
    ):
        """Configure diffusion proposals, rewards, and adaptive resampling.

        Args:
            proposal_sampler: Kernel used to generate particle transitions.
            soft_value: Intermediate reward estimator.
            n_steps: Number of reverse diffusion steps.
            n_particles: Number of particles per search space.
            ess_threshold: Resample below this effective sample size; defaults
                to half the particle count. Terminal resampling is unconditional.
            potential: Built-in log potential name or a callable accepting
                reward history and the current diffusion step.
            base_sampler: Target kernel for transition importance corrections;
                ``None`` uses the proposal kernel.
            resample_interval: Steps at which to evaluate potentials and check
                ESS; zero is always included. ``None`` evaluates every step.
            tilt_temperature: Divisor applied to log potentials.
            seed: Seed for particle selection.
            batch_size: Number of search spaces processed per request batch.
            device_map: Zero-based indices into the available device sequence for
                the proposal, base, and registered soft value models. Ignored
                when loading onto a single device; absent components are skipped.
        """
        super().__init__(batch_size=batch_size, seed=seed)
        self.proposal_sampler = proposal_sampler
        self.soft_value = soft_value
        self.n_steps = n_steps
        self.n_particles = n_particles
        self.ess_threshold = (
            self.n_particles / 2 if ess_threshold is None else ess_threshold
        )
        if isinstance(potential, str):
            match potential:
                case "max":
                    self.potential = max_potential
                case "sum":
                    self.potential = sum_potential
                case "difference":
                    self.potential = lambda x, _: x[-1] - x[-2]
        else:
            self.potential = potential
        if base_sampler is proposal_sampler:
            base_sampler = None
        self.base_sampler = base_sampler
        resample_interval = set(
            range(self.n_steps) if resample_interval is None else resample_interval
        )
        if 0 not in resample_interval:
            resample_interval.add(0)
        self.resample_interval = resample_interval
        self.tilt_temperature = tilt_temperature
        self.device_map = device_map

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Load components onto their mapped devices or use default placement.

        Args:
            device: Device or device sequence to load components onto. ``None``
                resolves available devices. A sequence uses ``device_map`` when
                configured; a single device loads all components together.
        """
        if self.device_map is None:
            return super().load(device)
        if device is None:
            device = devices.resolve_devices()
        if isinstance(device, torch.device):
            return super().load(device)
        components: dict[str, tuple[int, inference.HandlerRegistryMixin]] = {
            "proposal": (self.device_map["proposal"], self.proposal_sampler),
        }
        if self.base_sampler is not None:
            components["base"] = (self.device_map["base"], self.base_sampler)
        if isinstance(self.soft_value, inference.HandlerRegistryMixin):
            components["soft_value"] = (self.device_map["soft_value"], self.soft_value)
        for name, (index, _) in components.items():
            if not 0 <= index < len(device):
                raise ValueError(
                    f"{len(device)} devices available. Requested device id "
                    f"{index} for {name}."
                )
        for index, component in components.values():
            component.load(device[index])

    def _set_logits(
        self, particles: Sequence[_Particle], space: spaces.SearchSpace
    ) -> None:
        if particles[0].proposal_logits is not None:
            return
        logits = self.proposal_sampler.denoising_logits(
            [particle.tokens for particle in particles], space
        )
        device = (
            logits[0].device
            if particles[0].step == self.n_steps
            else particles[0].tokens.device
        )
        for particle, logits_ in zip(particles, logits, strict=True):
            particle.tokens = particle.tokens.to(device)
            particle.proposal_logits = logits_.to(device)

        if self.base_sampler is None:
            for particle in particles:
                particle.pretrain_logits = particle.proposal_logits
        else:
            space = space.to(device)
            logits = self.base_sampler.denoising_logits(
                [particle.tokens for particle in particles], space
            )
            for particle, logits_ in zip(particles, logits, strict=True):
                particle.pretrain_logits = logits_.to(device)

    def _set_transition_scores(
        self, particle: _Particle, space: spaces.SearchSpace
    ) -> None:
        if particle.proposal_transition_score is not None:
            return
        assert particle.proposal_logits is not None
        particle.proposal_transition_score = self.proposal_sampler.transition_scores(
            particle.step,
            particle.tokens,
            particle.proposal_logits,
            space,
            prev_state=particle.proposal_state,
        )
        if self.base_sampler is None:
            particle.pretrain_transition_score = particle.proposal_transition_score
        else:
            assert particle.pretrain_logits is not None
            particle.pretrain_transition_score = self.base_sampler.transition_scores(
                particle.step,
                particle.tokens,
                particle.pretrain_logits,
                space,
                prev_state=particle.pretrain_state,
            )

    def _get_log_ratio(self, particle: _Particle, z_s: TokenIDs) -> float:
        if self.base_sampler is None:
            return 0
        assert particle.pretrain_transition_score is not None
        assert particle.proposal_transition_score is not None
        pretrain_logp = self.base_sampler.log_transition_probs(
            z_s, particle.pretrain_transition_score
        )
        proposal_logp = self.proposal_sampler.log_transition_probs(
            z_s, particle.proposal_transition_score
        )
        return (pretrain_logp.sum() - proposal_logp.sum()).item()

    def _update_particle(
        self, particle: _Particle, space: spaces.SearchSpace
    ) -> _Particle:
        assert particle.proposal_logits is not None
        self._set_transition_scores(particle, space)
        assert particle.proposal_transition_score is not None
        z_s = self.proposal_sampler.step(particle.proposal_transition_score)
        proposal_state = self.proposal_sampler.update_state(
            z_s,
            particle.tokens,
            particle.proposal_logits,
            state=particle.proposal_state,
        )
        assert particle.pretrain_logits is not None
        if self.base_sampler is not None:
            pretrain_state = self.base_sampler.update_state(
                z_s,
                particle.tokens,
                particle.pretrain_logits,
                state=particle.pretrain_state,
            )
        else:
            pretrain_state = proposal_state
        next_particle = _Particle(
            step=particle.step - 1,
            tokens=z_s,
            log_ratio=self._get_log_ratio(particle, z_s),
            rewards=particle.rewards.copy(),
            prev_log_weight=particle.prev_log_weight + particle.log_ratio,
            proposal_state=proposal_state,
            pretrain_state=pretrain_state,
        )
        return next_particle

    def _update_rewards(
        self, particles: Sequence[_Particle], space: spaces.SearchSpace
    ) -> None:
        if self.soft_value.requires_logits:
            self._set_logits(particles, space)
            logits = []
            for particle in particles:
                assert particle.pretrain_logits is not None
                logits.append(particle.pretrain_logits)
        else:
            logits = None
        tokens = [particle.tokens for particle in particles]
        rewards = self.soft_value(tokens, logits, self.tilt_temperature)
        for particle, reward in zip(particles, rewards, strict=True):
            particle.rewards.append(reward.item())

    def _resample_particles(
        self, particles: Sequence[_Particle], space: spaces.SearchSpace
    ) -> Sequence[_Particle]:
        self._update_rewards(particles, space)
        weights = [particle.log_weight(self.potential) for particle in particles]
        log_weights = torch.tensor(weights, dtype=torch.float64)
        ess = log_weights.softmax(dim=-1).square().sum().reciprocal()
        logger.debug("At step = %s ESS = %s", particles[0].step, ess.item())
        if ess < self.ess_threshold or particles[0].step == 0:
            logger.debug(
                "resampling particles at step %s with ESS %s",
                particles[0].step,
                ess.item(),
            )
            indices = torch.multinomial(
                log_weights.softmax(dim=-1),
                num_samples=self.n_particles,
                replacement=True,
                generator=self.rng(log_weights.device),
            )
            particles = [copy.deepcopy(particles[idx]) for idx in indices]
            weights = [-math.log(self.n_particles)] * self.n_particles
        for particle, weight in zip(particles, weights, strict=True):
            particle.prev_log_weight = weight
            particle.log_ratio = 0.0
        return particles

    def sample_one(self, space: spaces.SearchSpace) -> BatchTokenIDs:
        """Draw terminal particles for one search space.

        Args:
            space: Search space defining the initial sequence and allowed mutations.

        Returns:
            Token IDs for the configured number of terminal particles.
        """
        tokens = space.init_sequence.clone()
        tokens[space.mutable_positions] = tokenizer.mask_token_id
        particle = _Particle(
            step=self.n_steps,
            tokens=tokens,
            prev_log_weight=-math.log(self.n_particles),
            rewards=[0],
            log_ratio=0,
        )
        self._set_logits([particle], space)
        space = space.to(particle.tokens.device)
        self._set_transition_scores(particle, space)
        particles = [copy.deepcopy(particle) for _ in range(self.n_particles)]
        for step in range(self.n_steps, 0, -1):
            self._set_logits(particles, space)
            particles = [
                self._update_particle(particle, space) for particle in particles
            ]
            if step - 1 in self.resample_interval:
                particles = self._resample_particles(particles, space)

        return torch.stack([particle.tokens for particle in particles])

    def sample(self, spaces: SequenceLike[spaces.SearchSpace]) -> list[BatchTokenIDs]:
        """Propose candidates for each space with :meth:`sample_one`.

        Args:
            spaces: Search spaces to sample candidates from.

        Returns:
            One batch of candidate token IDs per input search space.
        """
        return [self.sample_one(space) for space in spaces]


class NestedSMCSampler(FKSteeringSampler):
    r"""Steer diffusion sampling with nested particle proposals and resampling.

    A population of partially masked sequences (outer particles) evolves through
    reverse diffusion steps. At each step, ``proposal_sampler`` generates
    ``duplication_size`` candidate transitions per outer particle. ``soft_value``
    estimates their future rewards, and one candidate is selected in proportion
    to its importance weight.

    For a transition from :math:`z_t` to candidate :math:`z_{t-1}^{(j)}`, the
    inner weight is
    :math:`w_j = \exp(v_{t-1}(z_{t-1}^{(j)}))
    p(z_{t-1}^{(j)} \mid z_t) / q(z_{t-1}^{(j)} \mid z_t)`, where
    :math:`v_{t-1}` is the soft value evaluated with ``tilt_temperature``,
    :math:`p` is the base transition kernel, and :math:`q` is the proposal
    kernel. The ratio is one when no separate ``base_sampler`` is supplied.

    With ``global_resampling=True``, outer particles are resampled after every
    step using the sum of their inner weights. With ``global_resampling=False``,
    each outer particle retains its selected candidate independently, giving
    SVDD-style decoding :cite:p:`SVDD`.

    See :cite:p:`naesseth2024` (Algorithm 5) for nested SMC. The diffusion
    adaptation follows :cite:p:`uehara2025`.
    """

    def __init__(
        self,
        proposal_sampler: base.MaskedDiffusionSamplerBase,
        soft_value: SoftValueFunction,
        n_steps: int,
        n_particles: int,
        duplication_size: int,
        *,
        base_sampler: base.MaskedDiffusionSamplerBase | None = None,
        global_resampling: bool = True,
        tilt_temperature: float = 1.0,
        device_map: SMCDeviceMap | None = None,
        seed: int | None = None,
        batch_size: int | None = None,
    ):
        """Configure nested sampling; see :class:`FKSteeringSampler` for placement.

        Args:
            proposal_sampler: Kernel used to generate particle transitions.
            soft_value: Intermediate reward estimator.
            n_steps: Number of reverse diffusion steps.
            n_particles: Number of outer particles per search space.
            duplication_size: Number of proposed transitions per outer particle
                at each step, from which one is selected.
            base_sampler: Target kernel for transition importance corrections;
                ``None`` uses the proposal kernel.
            global_resampling: Whether to resample outer particles after each step.
            tilt_temperature: Divisor applied to rewards before weighting.
            device_map: Component device indices; see :class:`SMCDeviceMap`.
            seed: Seed for particle selection.
            batch_size: Number of search spaces processed per request batch.
        """
        super().__init__(
            proposal_sampler=proposal_sampler,
            soft_value=soft_value,
            n_steps=n_steps,
            n_particles=n_particles,
            ess_threshold=0,
            potential=lambda x, _: x[-1],
            base_sampler=base_sampler,
            resample_interval=None,
            tilt_temperature=tilt_temperature,
            seed=seed,
            batch_size=batch_size,
            device_map=device_map,
        )
        self.duplication_size = duplication_size
        self.global_resampling = global_resampling

    def _sample_inner_particles(
        self, particles: Sequence[_Particle], space: spaces.SearchSpace
    ) -> tuple[_Particle, Float[torch.Tensor, ""]]:
        self._update_rewards(particles, space)
        log_weights = [particle.log_weight(self.potential) for particle in particles]
        log_weights = torch.tensor(log_weights, dtype=torch.float64)
        idx = torch.multinomial(
            log_weights.softmax(dim=-1),
            num_samples=1,
            generator=self.rng(log_weights.device),
        )
        return particles[int(idx)], torch.logsumexp(log_weights, dim=-1)

    def sample_one(self, space: spaces.SearchSpace) -> BatchTokenIDs:
        """Draw terminal particles using nested importance sampling.

        Args:
            space: Search space defining the initial sequence and allowed mutations.

        Returns:
            Token IDs for the configured number of terminal particles.
        """
        tokens = space.init_sequence.clone()
        tokens[space.mutable_positions] = tokenizer.mask_token_id
        particle = _Particle(
            step=self.n_steps,
            tokens=tokens,
            prev_log_weight=0,
            rewards=[0],
            log_ratio=0,
        )
        self._set_logits([particle], space)
        space = space.to(particle.tokens.device)
        self._set_transition_scores(particle, space)
        particles = [copy.deepcopy(particle) for _ in range(self.n_particles)]
        for _ in range(self.n_steps, 0, -1):
            self._set_logits(particles, space)
            next_particles = []
            global_log_weights = []
            for particle in particles:
                inner_particles = [
                    self._update_particle(particle, space)
                    for _ in range(self.duplication_size)
                ]
                # no incremental weighting in SVDD
                for particle in inner_particles:
                    particle.prev_log_weight = 0
                selected_particle, log_weight = self._sample_inner_particles(
                    inner_particles, space
                )
                next_particles.append(selected_particle)
                global_log_weights.append(log_weight)

            if self.global_resampling:
                log_weights = torch.stack(global_log_weights)
                indices = torch.multinomial(
                    log_weights.softmax(dim=-1),
                    self.n_particles,
                    replacement=True,
                    generator=self.rng(log_weights.device),
                )
                particles = [copy.deepcopy(next_particles[idx]) for idx in indices]
            else:
                particles = next_particles
        return torch.stack([particle.tokens for particle in particles])
