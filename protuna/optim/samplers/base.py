"""Abstract base classes for samplers."""

import abc

from jaxtyping import Float
import torch

from protuna import inference
from protuna.optim import spaces
from protuna.optim.samplers import base
from protuna.typing import BatchTokenIDs, SequenceLike, TokenIDs

type SamplerLike = inference.Handler[spaces.SearchSpace, BatchTokenIDs]

# Sampler Pre-conditions:
# 1. All input search spaces are unique

# Sampler Post-conditions:
# 1. All generated samples respect the input search space mutation constraints

# Sampler Invariance:
# 1. Input search spaces: Input sequence and allowed residues shouldn't be edited
# inplace


class Sampler(
    inference.HandlerRegistryMixin,
    inference.InferenceModel[spaces.SearchSpace, BatchTokenIDs],
    abc.ABC,
):
    """Base class for candidate proposal strategies.

    Subclasses implement :meth:`sample`.

    Candidates must respect the supplied search space. Responses may contain
    repeated candidates and starting sequences; the pipeline removes these before
    storage and evaluation and raises a :class:`ValueError` for invalid candidates.
    """

    def __init__(self, *, seed: int | None = None, batch_size: int | None = None):
        """Configure sampling randomness and request batching.

        Args:
            seed: Seed for each device-specific generator; ``None`` uses its
                default state.
            batch_size: Number of search spaces per forward pass; ``None``
                processes all.
        """
        super().__init__(batch_size=batch_size)
        self.seed = seed
        self._rngs: dict[torch.device, torch.Generator] = {}

    def rng(self, device: torch.device) -> torch.Generator:
        """Return the persistent sampling generator for ``device``.

        Args:
            device: Device on which random tensors will be generated.

        Returns:
            Sampling generator associated with ``device``.
        """
        if device not in self._rngs:
            rng = torch.Generator(device=device)
            if self.seed is not None:
                rng.manual_seed(self.seed)
            self._rngs[device] = rng
        return self._rngs[device]

    @abc.abstractmethod
    def sample(self, spaces: SequenceLike[spaces.SearchSpace]) -> list[BatchTokenIDs]:
        """Propose a batch of candidates for each search space.

        Args:
            spaces: Search spaces to sample candidates from.

        Returns:
            One batch of candidate token IDs per input search space.
        """
        ...

    def forward(
        self, requests: SequenceLike[spaces.SearchSpace]
    ) -> list[BatchTokenIDs]:
        """Alias of :meth:`sample`."""
        return self.sample(requests)


class MaskedDiffusionSamplerBase[StateT](base.Sampler):
    """Interface for masked diffusion sampling and individual denoising transitions.

    In addition to the candidate generation API of :class:`Sampler`, this
    interface exposes the operations needed to advance and evaluate a diffusion
    trajectory.

    Notation:
        ``x_0``: Clean, fully unmasked amino acid sequence. The subscript zero
            denotes the endpoint of denoising.
        ``z_t``: Current sequence at diffusion timestep ``t``, potentially
            containing mask tokens.
        ``z_s``: Sequence sampled at the next denoising timestep ``s < t``.
            It may still contain masks; it becomes ``z_t`` for the next step.
            For consecutive integer timesteps, ``s = t - 1``.
        ``x0_logits``: unnormalized model scores predicting the clean residues of
            ``x_0`` at masked positions, conditioned on ``z_t``.
        ``transition_scores``: Unnormalized log scores for the outcomes of
            the transition from ``z_t`` to ``z_s``. Unlike ``x0_logits``, these
            cover every sequence position and include the mask outcome.
        ``StateT``: Sampler-specific information carried between transitions
            through ``prev_state`` or ``state``, such as residue confidence.

    The denoiser predicts clean residues from ``z_t``; the transition scores
    combine those predictions with the sampler's transition rules; :meth:`step`
    then samples ``z_s`` from those scores.
    """

    @abc.abstractmethod
    def denoising_logits(
        self, z_t: SequenceLike[TokenIDs], space: spaces.SearchSpace
    ) -> SequenceLike[Float[torch.Tensor, "m v"]]:
        r"""Return denoising logits :math:`\ell(x_0 \mid z_t)`.

        Args:
            z_t: Partially masked sequences to denoise.
            space: Search space constraining the denoised sequences.

        Returns:
            Logits for the masked positions in each input sequence.
        """

    @abc.abstractmethod
    def transition_scores(
        self,
        t: int,
        z_t: TokenIDs,
        x0_logits: Float[torch.Tensor, "m v"],
        space: spaces.SearchSpace,
        *,
        prev_state: StateT | None = None,
    ) -> Float[torch.Tensor, "l v+1"]:
        r"""Return unnormalized log transition scores.

        Scores represent :math:`\log \tilde{p}(z_s \mid z_t)`.

        Args:
            t: Current diffusion timestep.
            z_t: Sequence before the transition.
            x0_logits: Denoising logits for the masked positions in ``z_t``.
            space: Search space constraining the transition.
            prev_state: State carried into this transition, or ``None`` initially.

        Returns:
            Log scores for each position and outcome, including the mask outcome.
        """

    @abc.abstractmethod
    def step(self, transition_scores: Float[torch.Tensor, "l v+1"]) -> TokenIDs:
        """Return :math:`z_s` given unnormalized log transition probs.

        Args:
            transition_scores: Unnormalized log scores for each position and
                outcome, including the mask outcome.

        Returns:
            Token IDs sampled for the next diffusion step.
        """

    @abc.abstractmethod
    def update_state(
        self,
        z_s: TokenIDs,
        z_t: TokenIDs,
        x0_logits: Float[torch.Tensor, "m v"],
        *,
        state: StateT | None = None,
    ) -> StateT:
        """Return the state to carry forward after a denoising transition.

        Args:
            z_s: Sequence sampled for the next diffusion step.
            z_t: Sequence before the transition.
            x0_logits: Denoising logits computed from ``z_t``.
            state: State carried into this transition, or ``None`` initially.

        Returns:
            Updated sampler-specific state for the next transition.
        """
        ...

    @abc.abstractmethod
    def log_transition_probs(
        self, z_s: TokenIDs, transition_scores: Float[torch.Tensor, "l v+1"]
    ) -> Float[torch.Tensor, " l"]:
        """Return the log probability of each sampled token in a transition.

        Args:
            z_s: Sampled next-step tokens of shape ``(L,)``.
            transition_scores: Unnormalized log scores for each position and
                outcome, including the mask outcome.

        Returns:
            Normalized log probabilities of shape ``(L,)`` for the outcomes
            selected in ``z_s``.
        """
        ...
