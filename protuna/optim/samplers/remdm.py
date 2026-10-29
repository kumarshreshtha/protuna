"""Sample protein candidates with the remasking diffusion kernel."""

from collections.abc import Callable, Sequence
import dataclasses
import math
from typing import Protocol, TypedDict

import einops
from jaxtyping import Float
import torch

from protuna import inference
from protuna.data import tokenizer
from protuna.optim import spaces
from protuna.optim.samplers import base
from protuna.optim.samplers import common
from protuna.typing import BatchTokenIDs, SequenceLike, TokenIDs
from protuna.utils import devices


class RemaskScheduler(Protocol):
    """Structural interface for per-position remasking probabilities calculation."""

    def __call__(
        self,
        t: float,
        alpha_t: float,
        alpha_s: float,
    ) -> Float[torch.Tensor, " #l"]:
        """Return remasking probabilities for the transition from t to s.

        Args:
            t: Current normalized diffusion time in ``[0, 1]``.
            alpha_t: Signal probability at the current time.
            alpha_s: Signal probability at the next time.

        Returns:
            Remasking probabilities, broadcastable over sequence positions.
        """
        ...


def _sigma_max(alpha_t: float, alpha_s: float) -> float:
    return min(1.0, (1 - alpha_s) / alpha_t) if alpha_t > 0 else 1.0


class MaxCappedRemaskScheduler(RemaskScheduler):
    """Constant remasking probability capped to max allowed value for time t."""

    def __init__(self, eta_cap: float):
        """Set the remasking probability.

        Args:
            eta_cap: Finite remasking probability in ``[0, 1]``.
        """
        self.eta_cap = torch.tensor([eta_cap])

    def __call__(
        self,
        t: float,
        alpha_t: float,
        alpha_s: float,
    ) -> Float[torch.Tensor, " #l"]:
        """Return the smaller of the cap and the valid remasking bound.

        Args:
            t: Current normalized diffusion time; unused by this scheduler.
            alpha_t: Signal probability at the current time.
            alpha_s: Signal probability at the next time.

        Returns:
            Capped remasking probability, shared across sequence positions.
        """
        return torch.clamp_max(self.eta_cap, max=_sigma_max(alpha_t, alpha_s))


class RescaleRemaskScheduler(RemaskScheduler):
    """Rescale the maximum valid allowed remasking probability."""

    def __init__(self, eta_rescale: float):
        """Set the fraction of the remasking bound to use.

        Args:
            eta_rescale: Finite fraction of the remasking bound in ``[0, 1]``.
        """
        self.eta_rescale = torch.tensor([eta_rescale])

    def __call__(
        self,
        t: float,
        alpha_t: float,
        alpha_s: float,
    ) -> Float[torch.Tensor, " #l"]:
        """Return the valid remasking bound multiplied by ``eta_rescale``.

        Args:
            t: Current normalized diffusion time; unused by this scheduler.
            alpha_t: Signal probability at the current time.
            alpha_s: Signal probability at the next time.

        Returns:
            Rescaled remasking probability, shared across sequence positions.
        """
        return self.eta_rescale * _sigma_max(alpha_t, alpha_s)


class ReMDMScheduler(Protocol):
    """Inferface for calculating noise levels and remasking for each step."""

    def __call__(
        self,
        step: int,
        n_steps: int,
    ) -> tuple[float, float, Float[torch.Tensor, " #l"]]:
        """Return alpha at both endpoints and the remasking probability.

        Args:
            step: Current denoising step, counting down from ``n_steps`` to one.
            n_steps: Total number of denoising steps.

        Returns:
            Current signal probability, next signal probability, and remasking
            probabilities broadcastable over sequence positions.
        """
        ...


class ReMDMLoopScheduler(ReMDMScheduler):
    """Hold the noise level constant during a remasking interval."""

    def __init__(
        self,
        remask_scheduler: RemaskScheduler,
        t_on: float,
        t_off: float,
        alpha_on: float,
    ):
        """Set the loop interval and its constant signal probability.

        Args:
            remask_scheduler: Schedule providing valid remasking probabilities.
            t_on: Finite loop activation time satisfying
                ``0 < t_off < t_on < 1``. Need not align with a sampling step.
            t_off: Finite loop deactivation time satisfying
                ``0 < t_off < t_on < 1``. Need not align with a sampling step.
            alpha_on: Finite signal probability during the loop in ``(0, 1)``.
        """
        self.t_on = t_on
        self.t_off = t_off
        self.alpha_on = alpha_on
        self.remask_scheduler = remask_scheduler

    def _alpha(self, t: float) -> float:
        if t > self.t_on:
            return self.alpha_on * (1 - t) / (1 - self.t_on)
        if t > self.t_off:
            return self.alpha_on
        return 1 - (1 - self.alpha_on) * t / self.t_off

    def __call__(
        self,
        step: int,
        n_steps: int,
    ) -> tuple[float, float, Float[torch.Tensor, " #l"]]:
        """Return consistent noise levels across loop phase boundaries.

        Args:
            step: Current denoising step, counting down from ``n_steps`` to one.
            n_steps: Total number of denoising steps.

        Returns:
            Current signal probability, next signal probability, and remasking
            probabilities, which are zero outside the loop interval.
        """
        t = step / n_steps
        s = (step - 1) / n_steps
        alpha_t = self._alpha(t)
        alpha_s = self._alpha(s)
        if self.t_off < t <= self.t_on:
            sigma_t = self.remask_scheduler(t, alpha_t, alpha_s)
            sigma_t = torch.clamp_max(sigma_t, _sigma_max(alpha_t, alpha_s))
        else:
            sigma_t = torch.tensor([0.0])
        return alpha_t, alpha_s, sigma_t


def log_linear_noise_schedule(t: float) -> float:
    """Return the signal probability for a log-linear noise schedule.

    Args:
        t: Normalized diffusion time in ``[0, 1]``.

    Returns:
        Signal probability ``1 - t``.
    """
    return 1 - t


def cosine_noise_schedule(t: float) -> float:
    """Return the cosine signal probability.

    Args:
        t: Normalized diffusion time in ``[0, 1]``.

    Returns:
        Signal probability ``cos(pi * t / 2)``.
    """
    return math.cos(0.5 * math.pi * t)


def cosine_square_noise_schedule(t: float) -> float:
    """Return the squared cosine signal probability.

    Args:
        t: Normalized diffusion time in ``[0, 1]``.

    Returns:
        Signal probability ``cos(pi * t / 2) ** 2``.
    """
    return cosine_noise_schedule(t) ** 2


class ReMDMSwitchScheduler(ReMDMScheduler):
    """Enable remasking once time falls below ``t_on``."""

    def __init__(
        self,
        remask_scheduler: RemaskScheduler,
        t_on: float,
        noise_scheduler: Callable[[float], float] = log_linear_noise_schedule,
    ):
        """Set the activation time and noise and remasking schedules.

        Args:
            remask_scheduler: Schedule providing valid remasking probabilities.
            t_on: Finite activation time in ``[0, 1]``. Zero disables remasking.
            noise_scheduler: Nonincreasing signal probability as a function of
                time in ``[0, 1]``, with finite outputs in ``[0, 1]``, one at
                time zero, and zero at time one.
        """
        self.noise_scheduler = noise_scheduler
        self.remask_scheduler = remask_scheduler
        self.t_on = t_on

    def __call__(
        self,
        step: int,
        n_steps: int,
    ) -> tuple[float, float, Float[torch.Tensor, " #l"]]:
        """Return noise levels and a bounded, optionally enabled remask rate.

        Args:
            step: Current denoising step, counting down from ``n_steps`` to one.
            n_steps: Total number of denoising steps.

        Returns:
            Current signal probability, next signal probability, and remasking
            probabilities, which are zero above the activation time.
        """
        alpha_t = self.noise_scheduler(step / n_steps)
        alpha_s = self.noise_scheduler((step - 1) / n_steps)
        if step / n_steps <= self.t_on:
            sigma_t = self.remask_scheduler(step / n_steps, alpha_t, alpha_s)
        else:
            sigma_t = torch.tensor([0.0])
        sigma_t = torch.clamp_max(sigma_t, _sigma_max(alpha_t, alpha_s))
        return alpha_t, alpha_s, sigma_t


@dataclasses.dataclass
class _GuidanceScores:
    unmask_scores: Float[torch.Tensor, "m v"]
    remask_scores: Float[torch.Tensor, " u"]


type PredictionModelLike = inference.Handler[
    tuple[TokenIDs, float], Float[torch.Tensor, ""]
]


class ReMDMDeviceMap(TypedDict):
    """Relative device ids to place concurrent models for ReMDM."""

    model: int
    guidance_model: int


class ReMDMSampler(base.MaskedDiffusionSamplerBase):
    r"""Generate candidates with remasking diffusion and optional guidance.

    Remasking allows a decoded residue to be replaced on a later step using the
    remasking diffusion kernel from :cite:p:`ReMDM`. By default, remasking is disabled
    giving the masked diffusion language modeling (MDLM) kernel :cite:p:`MDLM`.

    Optional classifier guidance favors transitions that increase the target
    class probability, following :cite:p:`SimpleDiscreteGuidance,DiscreteGuidance`.
    It scores both single-site decoding and remasking proposals relative to the
    current sequence.
    """

    def __init__(
        self,
        model: inference.MLMForLogitsLike,
        n_steps: int,
        max_samples: int,
        *,
        scheduler: ReMDMScheduler | None = None,
        top_p: float = 1.0,
        temperature: float = 1.0,
        guidance_model: PredictionModelLike | None = None,
        guidance_strength: float = 1.0,
        confidence_remasking: bool = False,
        device_map: ReMDMDeviceMap | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
    ):
        """Configure denoising, remasking, and optional classifier guidance.

        Args:
            model: Masked language model returning amino-acid logits.
            n_steps: Positive integer number of denoising steps.
            max_samples: Positive integer number of candidate draws per space.
            scheduler: Schedule providing signal probabilities at both transition
                endpoints and per-position remasking probabilities. ``None``
                disables remasking and uses :func:`log_linear_noise_schedule`.
            top_p: Finite nucleus sampling threshold in ``(0, 1]``.
            temperature: Finite, strictly positive divisor of denoising logits.
            guidance_model: Optional classifier guidance handler accepting
                ``(token_ids, time)`` requests, where ``time = t / n_steps``
                is the current normalized diffusion time in ``[0, 1]``. Return
                one scalar target-class log probability per request. The model
                must support masked sequences. ``None`` disables guidance.
            guidance_strength: Multiplier for target-class log probability
                ratios. Zero disables their effect on transition probabilities.
            confidence_remasking: Weight scheduled remasking rates by a softmax
                of negative stored decoding probabilities over decoded mutable
                sites. Lower-confidence residues receive larger weights.
            device_map: Indices into the device sequence supplied to :meth:`load`
                for the denoising and guidance models. Ignored without a guidance
                model or when loading onto a single device.
            batch_size: Number of search spaces per forward pass; ``None``
                processes all. Does not set the model handler's batch size.
            seed: Seed for each device-specific sampling generator; ``None``
                uses its default state.
        """
        super().__init__(batch_size=batch_size, seed=seed)
        self.model = model
        if scheduler is None:
            scheduler = ReMDMSwitchScheduler(
                remask_scheduler=MaxCappedRemaskScheduler(0),
                t_on=0,
                noise_scheduler=log_linear_noise_schedule,
            )
        self.scheduler = scheduler
        self.n_steps = n_steps
        self.max_samples = max_samples
        self.top_p = top_p
        self.temperature = temperature
        self.confidence_remasking = confidence_remasking
        self.guidance_model = guidance_model
        self.guidance_strength = guidance_strength
        self.device_map = device_map

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Load models onto their mapped devices or use default placement.

        Args:
            device: Device or available devices for model placement. ``None``
                resolves available devices automatically.
        """
        if self.device_map is None or self.guidance_model is None:
            return super().load(device)
        if device is None:
            device = devices.resolve_devices()
        if isinstance(device, torch.device):
            return super().load(device)
        if not 0 <= self.device_map["model"] < len(device) or not (
            0 <= self.device_map["guidance_model"] < len(device)
        ):
            raise ValueError(
                f"{len(device)} devices available. requested device id "
                f"{self.device_map['model']} for model and "
                f"{self.device_map['guidance_model']} for guidance model."
            )
        self.model.load(device[self.device_map["model"]])
        self.guidance_model.load(device[self.device_map["guidance_model"]])

    def denoising_logits(
        self,
        z_t: SequenceLike[TokenIDs],
        space: spaces.SearchSpace,
        device: torch.device | None = None,
    ) -> SequenceLike[Float[torch.Tensor, "m v"]]:
        """Return constrained denoising logits for masked positions.

        Args:
            z_t: Partially masked sequences to denoise.
            space: Allowed residues for each sequence position.
            device: Output device; defaults to the first model response's device.

        Returns:
            One tensor of masked-position logits per input sequence, with
            temperature, nucleus sampling, and residue constraints applied.
        """
        responses = common.execute_mlm_batch(z_t, self.model, device=device)
        device = responses[0].device
        space = space.to(device)
        all_logits = []
        for tokens, response in zip(z_t, responses, strict=True):
            mask = (tokens == tokenizer.mask_token_id).to(device)
            allowed = space.allowed[mask]
            logits = common.modify_logits(
                response.to(dtype=torch.promote_types(response.dtype, torch.float32)),
                allowed,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            all_logits.append(logits)
        return all_logits

    def _guidance_scores(
        self, t: int, tokens: TokenIDs, space: spaces.SearchSpace
    ) -> _GuidanceScores | None:
        if self.guidance_model is None:
            return None

        aa_token_ids = torch.tensor(tokenizer.aa_token_ids, device=tokens.device)
        mask = tokens == tokenizer.mask_token_id
        masked_idx = (mask).nonzero(as_tuple=True)[0]
        unmask_tokens = tokens[None, None, :].repeat(
            masked_idx.shape[0], aa_token_ids.shape[0], 1
        )
        indices = torch.arange(masked_idx.shape[0], device=tokens.device)
        unmask_tokens[indices, :, masked_idx] = aa_token_ids
        unmask_tokens = einops.rearrange(unmask_tokens, "m v l -> (m v) l")

        unmasked_idx = (space.mutable_positions & ~mask).nonzero(as_tuple=True)[0]
        remask_tokens = tokens[None, :].repeat(unmasked_idx.shape[0], 1)
        indices = torch.arange(unmasked_idx.shape[0], device=tokens.device)
        remask_tokens[indices, unmasked_idx] = tokenizer.mask_token_id

        requests = [
            (candidate, t / self.n_steps)
            for candidate in [tokens, *list(unmask_tokens), *list(remask_tokens)]
        ]
        logp = torch.stack(
            [response.to(tokens.device) for response in self.guidance_model(requests)]
        )
        log_ratio = logp[1:] - logp[0]
        unmask_scores = log_ratio[: unmask_tokens.shape[0]].view(
            masked_idx.shape[0], aa_token_ids.shape[0]
        )
        remask_scores = log_ratio[unmask_tokens.shape[0] :]

        return _GuidanceScores(
            unmask_scores * self.guidance_strength,
            remask_scores * self.guidance_strength,
        )

    def transition_scores(
        self,
        t: int,
        z_t: TokenIDs,
        x0_logits: Float[torch.Tensor, "m v"],
        space: spaces.SearchSpace,
        prev_state: Float[torch.Tensor, " l"] | None = None,
    ) -> Float[torch.Tensor, "l v+1"]:
        """Return log transition scores over residues and the mask token.

        Args:
            t: Current denoising step, counting down from ``n_steps`` to one.
            z_t: Current partially masked sequence.
            x0_logits: Denoising logits for the masked positions in ``z_t``.
            space: Allowed residues and mutable positions.
            prev_state: Stored per-position decoding confidence, or ``None``
                when unavailable. Used when confidence remasking is enabled.

        Returns:
            Per-position log transition scores, with the mask token in the last
            column and disallowed transitions set to negative infinity.
        """
        score = self._guidance_scores(t, z_t, space)
        if score is not None:
            score.remask_scores = score.remask_scores.to(x0_logits.device)
            score.unmask_scores = score.unmask_scores.to(x0_logits.device)
        confidence = prev_state
        alpha_t, alpha_s, sigma_t = self.scheduler(t, self.n_steps)
        sigma_max = _sigma_max(alpha_t, alpha_s)
        alpha_t = torch.tensor(alpha_t, device=z_t.device, dtype=torch.float64)
        alpha_s = torch.tensor(alpha_s, device=z_t.device, dtype=torch.float64)
        sigma_t = sigma_t.to(device=z_t.device, dtype=torch.float64).expand(z_t.shape)
        mask = z_t == tokenizer.mask_token_id
        remask = ~mask & space.mutable_positions
        if self.confidence_remasking:
            if confidence is not None and remask.any():
                confidence = confidence.masked_fill(~remask, torch.inf)
                sigma_t = sigma_t * torch.softmax(-confidence, dim=0)
            else:
                sigma_t = torch.zeros_like(sigma_t)

        # Bound again in the calculation dtype to avoid rounded-up scheduler rates.
        sigma_t = torch.clamp_max(sigma_t, sigma_max)
        logp_unmask = ((alpha_s - alpha_t) + sigma_t * alpha_t).clamp_min(0).log() - (
            -alpha_t
        ).log1p()
        logp_stay_masked = (1 - alpha_s - sigma_t * alpha_t).clamp_min(0).log() - (
            -alpha_t
        ).log1p()
        logp_remask = sigma_t.log()
        logp_stay_unmasked = (-sigma_t).log1p()

        allowed = torch.zeros(
            space.allowed.shape[0],
            space.allowed.shape[1] + 1,
            dtype=torch.bool,
            device=space.allowed.device,
        )

        allowed[:, :-1] = space.allowed
        allowed[:, -1] = space.mutable_positions
        fixed = allowed.clone()
        fixed[space.mutable_positions] = False

        logp_z_s = x0_logits.log_softmax(dim=-1, dtype=torch.float64)

        new_logits = x0_logits.new_full(
            (z_t.shape[0], x0_logits.shape[1] + 1), -torch.inf, dtype=torch.float64
        )
        new_logits[mask, :-1] = logp_z_s + logp_unmask[mask, None]
        new_logits[mask, -1] = logp_stay_masked[mask]
        remask_idx = remask.nonzero(as_tuple=True)[0]
        new_logits[remask_idx, z_t[remask]] = logp_stay_unmasked[remask]
        new_logits[remask, -1] = logp_remask[remask]
        if score is not None:
            new_logits[mask, :-1] += score.unmask_scores
            new_logits[remask, -1] += score.remask_scores

        new_logits = new_logits.masked_fill(~allowed, -torch.inf)
        new_logits[fixed] = 0
        return new_logits

    def log_transition_probs(
        self, z_s: TokenIDs, transition_scores: Float[torch.Tensor, "l v+1"]
    ) -> Float[torch.Tensor, " l"]:
        """Return one log transition probability per sequence position.

        Args:
            z_s: Sequence after the transition.
            transition_scores: Per-position log transition scores, with the mask
                token in the last column.

        Returns:
            Log probabilities of the tokens in ``z_s``.
        """
        index = z_s.clone()
        index[z_s == tokenizer.mask_token_id] = len(tokenizer.aa_token_ids)
        logp = transition_scores.log_softmax(dim=-1)
        return logp.gather(dim=1, index=index[:, None]).squeeze(-1)

    def step(self, transition_scores: Float[torch.Tensor, "l v+1"]) -> TokenIDs:
        """Draw a sequence from the transition scores.

        Args:
            transition_scores: Per-position log transition scores, with the mask
                token in the last column.

        Returns:
            Sampled sequence of token IDs.
        """
        z_s = torch.multinomial(
            transition_scores.softmax(dim=-1),
            replacement=False,
            num_samples=1,
            generator=self.rng(transition_scores.device),
        ).squeeze(-1)
        z_s[z_s == len(tokenizer.aa_token_ids)] = tokenizer.mask_token_id
        return z_s

    def update_state(
        self,
        z_s: TokenIDs,
        z_t: TokenIDs,
        x0_logits: Float[torch.Tensor, "m v"],
        state: Float[torch.Tensor, " l"] | None = None,
    ) -> Float[torch.Tensor, " l"]:
        """Record decoding confidence and reset masked positions to infinity.

        Args:
            z_s: Sequence after the transition.
            z_t: Sequence before the transition.
            x0_logits: Denoising logits for the masked positions in ``z_t``.
            state: Stored per-position decoding confidence. ``None`` initializes
                all positions to infinity.

        Returns:
            Per-position confidence, updated when confidence remasking is enabled.
        """
        if state is None:
            state = torch.full_like(z_s, torch.inf, dtype=torch.float32)
        if self.confidence_remasking:
            confidence = state.clone()
            input_mask = z_t == tokenizer.mask_token_id
            unmasked = input_mask & (z_s != tokenizer.mask_token_id)
            probs = x0_logits[unmasked[input_mask]].softmax(
                dim=-1, dtype=torch.promote_types(x0_logits.dtype, torch.float32)
            )
            probs = probs.gather(-1, z_s[unmasked, None]).squeeze(-1)
            confidence[unmasked] = probs.to(confidence.dtype)
            confidence[z_s == tokenizer.mask_token_id] = torch.inf
            state = confidence
        return state

    def sample_one(self, space: spaces.SearchSpace) -> BatchTokenIDs:
        """Draw candidates for one search space.

        Args:
            space: Initial sequence and allowed mutations to sample from.

        Returns:
            A batch of ``max_samples`` candidate token sequences.
        """
        z_t = space.init_sequence.clone()
        z_t[space.mutable_positions] = tokenizer.mask_token_id
        x0_logits = self.denoising_logits([z_t], space)[0]
        device = x0_logits.device
        space = space.to(device)
        z_t = z_t.to(device)
        transition_score = self.transition_scores(
            self.n_steps, z_t, x0_logits, space, None
        )
        state = None
        next_tokens = []
        next_states = []
        for _ in range(self.max_samples):
            z_s = self.step(transition_score)
            next_state = self.update_state(z_s, z_t, x0_logits, state)
            next_tokens.append(z_s)
            next_states.append(next_state)

        z_t = next_tokens
        states = next_states
        for t in range(self.n_steps - 1, 0, -1):
            x0_logits = self.denoising_logits(z_t, space, device=device)
            next_states = []
            next_tokens = []
            for z, logits, state in zip(z_t, x0_logits, states, strict=True):
                transition_score = self.transition_scores(t, z, logits, space, state)
                z_s = self.step(transition_score)
                next_state = self.update_state(z_s, z, logits, state)
                next_states.append(next_state)
                next_tokens.append(z_s)
            z_t = next_tokens
            states = next_states
        return torch.stack(z_t)

    def sample(self, spaces: SequenceLike[spaces.SearchSpace]) -> list[BatchTokenIDs]:
        """Propose candidates for each space with :meth:`sample_one`.

        Args:
            spaces: Search spaces to sample candidates from.

        Returns:
            One batch of candidate token IDs per input search space.
        """
        return [self.sample_one(space) for space in spaces]
