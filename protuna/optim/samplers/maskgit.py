"""Refine masked language model candidates with scheduled iterative unmasking."""

from collections.abc import Callable
import dataclasses
import math
from typing import Protocol

from jaxtyping import Bool, Float, Integer
import torch

from protuna import inference
from protuna.data import tokenizer
from protuna.optim import spaces
from protuna.optim.samplers import base
from protuna.optim.samplers import common
from protuna.typing import BatchTokenIDs, SequenceLike


def cosine_mask_schedule(r: float) -> float:
    """Return the fraction of sites to keep masked, on a cosine schedule.

    Args:
        r: Decoding progress from 0 to 1.

    Returns:
        The fraction of mutable sites to keep masked.
    """
    return math.cos(r * math.pi * 0.5)


def linear_mask_schedule(r: float) -> float:
    """Return the fraction of sites to keep masked, on a linear schedule.

    Args:
        r: Decoding progress from 0 to 1.

    Returns:
        The fraction of mutable sites to keep masked.
    """
    return 1 - r


class MaskGenerator(Protocol):
    """Structural interface for choosing the sites a refinement round redraws.

    It decides both how many sites are masked, usually from a schedule such as
    :func:`cosine_mask_schedule`, and which ones.

    All members of a batch will always have the exact same number of sites masked as
    they are running on the same mask schedule.

    .. seealso::
        :class:`ConfidenceMaskGenerator` and :class:`RandomMaskGenerator`, the two
        concrete implementations.
    """

    def __call__(
        self,
        progress: float,
        n_mutable: int,
        samples: BatchTokenIDs,
        logits: Float[torch.Tensor, "b m v"],
        masks: Bool[torch.Tensor, "b l"],
    ) -> Bool[torch.Tensor, "b l"]:
        """Choose the sites to mask for the next round.

        Args:
            progress: How far decoding has progressed, in ``(0, 1)``.
            n_mutable: Number of unfixed sites.
            samples: The candidates drawn so far.
            logits: Model scores over the ``m`` sites ``masks`` mark as masked,
                aligned row-wise with ``samples``.
            masks: The sites that were masked to draw ``samples``.

        Returns:
            The ``(b, l)`` sites to mask next.
        """
        ...

    def reset(self) -> None:
        """Forget any state carried across rounds; called before each new space."""


def sample_logp(
    logits: Float[torch.Tensor, "b m v"], samples: BatchTokenIDs
) -> Float[torch.Tensor, "b m"]:
    """:py:type:`SequenceScorer` scoring sites by log-probability of the drawn residues.

    Args:
        logits: Model scores over the vocabulary at each site to score.
        samples: Token IDs drawn at those sites, aligned with ``logits``.

    Returns:
        The log-probability of each drawn token.
    """
    logp = torch.log_softmax(logits, dim=-1)
    logp = logp.gather(-1, samples[..., None]).squeeze(-1)
    return logp


def top_p_margin(
    logits: Float[torch.Tensor, "b m v"], samples: BatchTokenIDs
) -> Float[torch.Tensor, "b m"]:
    """:py:type:`SequenceScorer` scoring sites by difference in top two residue probs.

    Args:
        logits: Model scores over the vocabulary at each site to score.
        samples: Unused; accepted for compatibility with :py:type:`SequenceScorer`.

    Returns:
        The difference between the highest and second-highest probabilities
        at each site.
    """
    p = torch.softmax(logits, dim=-1)
    top_p, _ = p.topk(k=2, dim=-1)
    margin = top_p[:, :, 0] - top_p[:, :, 1]
    return margin


def negative_entropy(
    logits: Float[torch.Tensor, "b m v"], samples: BatchTokenIDs
) -> Float[torch.Tensor, "b m"]:
    """:py:type:`SequenceScorer` scoring sites by entropy of their distribution.

    Args:
        logits: Model scores over the vocabulary at each site to score.
        samples: Unused; accepted for compatibility with :py:type:`SequenceScorer`.

    Returns:
        The negative entropy at each site.
    """
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    return (p * logp).masked_fill(p == 0, 0).sum(dim=-1)


type SequenceScorer = Callable[
    [Float[torch.Tensor, "b m v"], BatchTokenIDs], Float[torch.Tensor, "b m"]
]


class ConfidenceMaskGenerator(MaskGenerator):
    """Unmask the sites the model was most confident about keeping others masked.

    ``mask_schedule`` sets how many sites remain masked and ``scorer``
    decides which ones, the lowest-scoring sites being masked first.

    By default, only currently masked sites are eligible, so once a site is unmasked
    it stays unmasked, following :cite:p:`MaskGIT`.

    With ``allow_remask=True``, previously unmasked sites remain eligible for remasking
    in later rounds, following the MaskPredict scheme :cite:p:`MaskPredict`. Each site
    retains its most recent score until it is redrawn and scored again, allowing
    low-confidence residues to be revisited as the surrounding sequence changes.
    """

    def __init__(
        self,
        *,
        mask_schedule: Callable[[float], float] = cosine_mask_schedule,
        scorer: SequenceScorer = sample_logp,
        temp_annealing: bool = True,
        choice_temperature: float = 1.0,
        allow_remask: bool = False,
        seed: int | None = None,
    ):
        """Configure the masking schedule and parameters.

        Args:
            mask_schedule: Fraction of the mutable sites to mask at each round,
                given how far through the rounds decoding is. Defaults to
                :func:`cosine_mask_schedule`; see also
                :func:`linear_mask_schedule`.
            scorer: How to score each masked site for unmasking. Defaults to
                :func:`sample_logp`; see also :func:`top_p_margin` and
                :func:`negative_entropy`.
            temp_annealing: Whether to anneal the noise added to the scores as
                decoding progresses.
            choice_temperature: Scale of the Gumbel noise added to the scores.
                Temperature of 0 picks the top k values deterministically.
            allow_remask: Whether to allow an unmasked position to be remasked again
                as per the MaskPredict :cite:p:`MaskPredict` scheme.
            seed: Seed for each device-specific mask sampling generator.
        """
        self.mask_schedule = mask_schedule
        self.scorer = scorer
        self.temp_annealing = temp_annealing
        self.choice_temperature = choice_temperature
        self.allow_remask = allow_remask
        self.seed = seed
        self._rngs: dict[torch.device, torch.Generator] = {}
        self._scores: Float[torch.Tensor, "b l"] | None = None

    def rng(self, device: torch.device) -> torch.Generator:
        """Return the persistent mask sampling generator for ``device``.

        Args:
            device: Device on which random tensors will be generated.

        Returns:
            The cached random number generator for the device.
        """
        if device not in self._rngs:
            rng = torch.Generator(device=device)
            if self.seed is not None:
                rng.manual_seed(self.seed)
            self._rngs[device] = rng
        return self._rngs[device]

    def _sample_positions(
        self, num_samples: int, progress: float
    ) -> Integer[torch.Tensor, "b k"]:
        assert self._scores is not None
        temperature = (
            (1 - progress) * self.choice_temperature
            if self.temp_annealing
            else self.choice_temperature
        )
        # pick the positions with the lowest scores
        scores = -self._scores.float()
        if temperature != 0:
            g = (
                -torch.empty_like(scores)
                .exponential_(generator=self.rng(scores.device))
                .log()
            )
            scores = scores / temperature + g

        _, positions = torch.topk(scores, k=num_samples)
        return positions

    def _update_score(
        self,
        sequence_length: int,
        scores: Float[torch.Tensor, "b m"],
        mask_idx: Integer[torch.Tensor, "b m"],
    ) -> None:
        if self._scores is None or not self.allow_remask:
            self._scores = torch.full(
                (scores.shape[0], sequence_length), torch.inf, device=scores.device
            )
        self._scores.scatter_(1, mask_idx, scores)

    def __call__(
        self,
        progress: float,
        n_mutable: int,
        samples: BatchTokenIDs,
        logits: Float[torch.Tensor, "b m v"],
        masks: Bool[torch.Tensor, "b l"],
    ) -> Bool[torch.Tensor, "b l"]:
        """Mask the lowest-scoring of the currently masked sites.

        Args:
            progress: How far decoding has progressed, in ``(0, 1)``.
            n_mutable: Number of unfixed sites used to scale the mask schedule.
            samples: Candidate token IDs drawn so far.
            logits: Model scores at the sites marked by ``masks``, aligned
                row-wise with ``samples``.
            masks: Sites that were masked to draw ``samples``.

        Returns:
            A boolean mask marking the sites to redraw next.
        """
        n_mask = int(self.mask_schedule(progress) * n_mutable)
        if not self.allow_remask:
            n_mask = max(1, min(n_mask, int(masks[0].sum()) - 1))

        device = logits.device

        mask_idx = masks.nonzero(as_tuple=True)[1].view(masks.shape[0], -1)
        masked_region = samples.gather(1, mask_idx)
        scores = self.scorer(logits, masked_region)
        self._update_score(masks.shape[1], scores, mask_idx)

        positions = self._sample_positions(n_mask, progress)
        new_masks = torch.zeros(masks.shape, dtype=torch.bool, device=device)
        indices = torch.arange(new_masks.shape[0], device=device)
        new_masks[indices[:, None], positions] = True
        return new_masks

    def reset(self) -> None:
        """Reset the scored scores for the search space to None."""
        self._scores = None


class RandomMaskGenerator(MaskGenerator):
    """Keep a uniformly random subset of the masked sites masked.

    ``mask_schedule`` decides how many sites are masked at each round, but which ones
    are decided at random from the currently masked sites.
    """

    def __init__(
        self,
        *,
        mask_schedule: Callable[[float], float] = cosine_mask_schedule,
        seed: int | None = None,
    ):
        """Setup the mask schedule and seed for random sampling of sites.

        Args:
            mask_schedule: Fraction of the mutable sites to mask at each round,
                given how far through the rounds decoding is. Defaults to
                :func:`cosine_mask_schedule`; see also
                :func:`linear_mask_schedule`.
            seed: Seed for each device-specific mask sampling generator.
        """
        self.mask_schedule = mask_schedule
        self.seed = seed
        self._rngs: dict[torch.device, torch.Generator] = {}

    def rng(self, device: torch.device) -> torch.Generator:
        """Return the persistent mask sampling generator for ``device``.

        Args:
            device: Device on which random tensors will be generated.

        Returns:
            The cached random number generator for the device.
        """
        if device not in self._rngs:
            rng = torch.Generator(device=device)
            if self.seed is not None:
                rng.manual_seed(self.seed)
            self._rngs[device] = rng
        return self._rngs[device]

    def __call__(
        self,
        progress: float,
        n_mutable: int,
        samples: BatchTokenIDs,
        logits: Float[torch.Tensor, "b m v"],
        masks: Bool[torch.Tensor, "b l"],
    ) -> Bool[torch.Tensor, "b l"]:
        """Mask a random subset of the currently masked sites.

        Args:
            progress: How far decoding has progressed, in ``(0, 1)``.
            n_mutable: Number of unfixed sites used to scale the mask schedule.
            samples: Candidate token IDs drawn so far.
            logits: Model scores whose device is used for mask sampling.
            masks: Sites that were masked to draw ``samples``.

        Returns:
            A boolean mask marking the sites to redraw next.
        """
        n_mask = int(self.mask_schedule(progress) * n_mutable)
        n_mask_prev = int(masks[0].sum())
        n_mask = max(1, min(n_mask, n_mask_prev - 1))

        device = logits.device

        mask_idx = masks.nonzero(as_tuple=True)[1].view(masks.shape[0], -1)
        selected_idx = torch.multinomial(
            torch.ones(samples.shape[0], n_mask_prev, device=device),
            n_mask,
            generator=self.rng(device),
        )
        positions = mask_idx.gather(dim=1, index=selected_idx)
        new_masks = torch.zeros_like(masks)
        indices = torch.arange(new_masks.shape[0], device=device)
        new_masks[indices[:, None], positions] = True

        return new_masks

    def reset(self) -> None:
        """No op as no state is stored between rounds."""
        return


@dataclasses.dataclass
class _MaskGITState:
    samples: BatchTokenIDs
    logits: Float[torch.Tensor, "b m v"]
    masks: Bool[torch.Tensor, "b l"]
    n_mutable: int

    def __post_init__(self) -> None:
        device = self.samples.device
        self.logits = self.logits.to(device)
        self.masks = self.masks.to(device)
        if self.logits.shape[0] == 1 and self.samples.shape[0] > 1:
            self.logits = self.logits.expand(self.samples.shape[0], -1, -1)
        if self.masks.ndim == 1:
            self.masks = self.masks.repeat(self.samples.shape[0], 1)


def _masked_logits_from_responses(
    responses: SequenceLike[Float[torch.Tensor, "m v"]],
    masks: Bool[torch.Tensor, "b l"],
    space: spaces.SearchSpace,
) -> torch.Tensor:
    logits = []
    for response, mask in zip(responses, masks, strict=True):
        space = space.to(response.device)
        logits.append(response.masked_fill(~space.allowed[mask], -torch.inf))
    return torch.stack(logits)


class MaskGITSampler(base.Sampler):
    """Draw candidates with iterative masked language model refinement.

    All mutable sites are first unmasked in a single pass, as in
    :class:`~protuna.optim.samplers.mlm.MLMSampler`. Subsequent rounds mask a
    subset of sites in each candidate and redraw their residues conditioned on
    the rest of that candidate. This lets later draws account for residues
    chosen in earlier rounds.

    ``mask_generator`` controls both the number of sites to redraw and their
    selection. The default :class:`ConfidenceMaskGenerator` progressively keeps
    confident residues and revisits less confident ones, following
    :cite:p:`MaskGIT`. By default, a site that is kept is never masked again.
    Pass a :class:`ConfidenceMaskGenerator` with ``allow_remask=True`` to allow
    previously kept sites to be revisited, or use :class:`RandomMaskGenerator`
    to select sites without consulting confidence scores.

    ``n_iter`` includes the initial draw and is capped at the number of mutable
    sites. Refinement stops early once only one site per candidate remains
    masked.
    """

    def __init__(
        self,
        model: inference.MLMForLogitsLike,
        max_samples: int,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n_iter: int = 10,
        mask_generator: MaskGenerator | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
    ):
        """Setup the sampling parameters and register the model.

        Args:
            model: The masked language model to sample from.
            max_samples: Number of candidates to draw per space.
            temperature: Temperature applied to the logits. Temperature of 0 picks the
                max logit deterministically.
            top_p: Keep only the most likely residues whose probability sums to
                ``top_p``.
            n_iter: Number of refinement rounds. Capped at the number of mutable
                sites.
            mask_generator: Which sites to mask at each round. Defaults
                to :class:`ConfidenceMaskGenerator`, which masks the sites the
                model was least confident about.
            batch_size: Number of search spaces per forward pass.
            seed: Seed for each device-specific sampling generator.
        """
        super().__init__(batch_size=batch_size, seed=seed)
        self.model = model
        self.max_samples = max_samples
        self.temperature = temperature
        self.top_p = top_p
        self.n_iter = n_iter
        self.mask_generator = (
            mask_generator
            if mask_generator is not None
            else ConfidenceMaskGenerator(seed=seed)
        )

    def sample_one(self, space: spaces.SearchSpace) -> BatchTokenIDs:
        """Propose candidates for a single space.

        Args:
            space: The space to propose candidates in.

        Returns:
            Candidates of shape ``(n, l)``, where ``n`` is at most
            ``max_samples``.
        """
        self.mask_generator.reset()
        n_mutable = int(space.mutable_positions.sum())
        n_iter = min(self.n_iter, n_mutable)
        input = space.init_sequence.clone()
        input[space.mutable_positions] = tokenizer.mask_token_id
        responses = common.execute_mlm_batch(input[None, :], self.model)
        response = responses[0]
        device = response.device
        space = space.to(device)
        samples = common.sample_from_logits(
            response,
            space.init_sequence,
            space.mutable_positions,
            space.allowed[space.mutable_positions],
            self.max_samples,
            temperature=self.temperature,
            top_p=self.top_p,
            generator=self.rng(device),
        )
        state = _MaskGITState(
            samples=samples,
            logits=_masked_logits_from_responses(
                responses, space.mutable_positions[None, :], space
            ),
            masks=space.mutable_positions,
            n_mutable=n_mutable,
        )
        for t in range(1, n_iter):
            if state.masks[0].sum() == 1:
                break
            masks = self.mask_generator(t / n_iter, **vars(state))
            masked_inputs = state.samples.clone()
            masked_inputs[masks] = tokenizer.mask_token_id
            responses = common.execute_mlm_batch(
                masked_inputs, self.model, device=device
            )
            new_samples: list[torch.Tensor] = []
            for input, mask, response in zip(
                state.samples, masks, responses, strict=True
            ):
                sample = common.sample_from_logits(
                    response,
                    input,
                    mask,
                    space.allowed[mask],
                    num_samples=1,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    generator=self.rng(device),
                )
                new_samples.append(sample)
            state = _MaskGITState(
                samples=torch.cat(new_samples),
                logits=_masked_logits_from_responses(responses, masks, space),
                masks=masks,
                n_mutable=n_mutable,
            )
        return state.samples

    def sample(self, spaces: SequenceLike[spaces.SearchSpace]) -> list[BatchTokenIDs]:
        """Propose candidates for each space.

        Args:
            spaces: Search spaces to sample candidates from.

        Returns:
            One batch of candidate token IDs per input search space.
        """
        return [self.sample_one(space) for space in spaces]
