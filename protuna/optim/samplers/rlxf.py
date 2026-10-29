"""Propose candidates by constructing and mutating model-guided scaffolds."""

import logging

from jaxtyping import Bool, Float
import torch

from protuna import inference
from protuna.data import tokenizer
from protuna.optim import spaces
from protuna.optim.samplers import base
from protuna.optim.samplers import common
from protuna.typing import BatchTokenIDs, SequenceLike, TokenIDs

logger = logging.getLogger(__name__)


def _prepare_per_position_masked_inputs(
    tokens: TokenIDs, mutable_mask: Bool[torch.Tensor, " l"]
) -> BatchTokenIDs:
    positions = mutable_mask.nonzero()[:, 0]  # M
    batch = tokens.repeat(positions.shape[0], 1)  # M, L
    indices = torch.arange(positions.shape[0], device=tokens.device)  # M
    batch[indices, positions] = tokenizer.mask_token_id
    return batch


def _responses_to_probs(
    responses: SequenceLike[Float[torch.Tensor, "m v"]],
) -> Float[torch.Tensor, "n l v"]:
    logits = torch.stack(list(responses))  # M, 1, V
    probs = torch.softmax(logits, dim=-1)
    return probs


class RLXFSampler(base.Sampler):
    """Refine a high-confidence scaffold by sampling residues one site at a time.

    Adapted from the sequence-generation strategy in :cite:p:`rlxf`.

    1. The model evaluates each mutable site of the initial sequence with that
        site masked. Allowed residues with probability above
        ``confidence_threshold`` form a shared scaffold.
    2. The model then reevaluates each mutable site in the scaffold context. Each
        site receives a weight equal to the total probability of its
        allowed non-initial residues. Sites with weight below
        ``position_threshold`` are excluded.
    3. For each candidate, the sampler selects distinct sites in proportion to
        these weights, using the mutation budget remaining after scaffold
        substitutions. The site weights stay fixed throughout generation.
    4. Each candidate starts from the scaffold. The sampler masks each selected
        site in turn and draws an allowed residue using ``temperature`` and
        ``top_p``, conditioned on the substitutions already made in that
        candidate.

    Differences from the original work's `official Hugging Face implementation
    <https://huggingface.co/RomeroLab-Duke/creilov-rlxf-esm2-650m>`_:

    * If scaffold substitutions meet or exceed ``max_mutations``, the sampler
      returns the scaffold alone with a warning. An over-budget scaffold is retained
      instead of raising ``ValueError``.
    * If too few sites meet ``position_threshold``, the sampler uses all
      eligible sites with a warning instead of raising ``ValueError``.
    * Residue sampling includes the initial residue when allowed, rather than
      forcing a substitution. Selected sites are visited only once, so the final
      mutation count can fall below ``max_mutations`` even with enough eligible
      sites. This lets a candidate retain the initial residue when preceding
      substitutions make it more probable.
    """

    def __init__(
        self,
        model: inference.MLMForLogitsLike,
        max_samples: int,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_mutations: int | None = None,
        confidence_threshold: float = 0.90,
        position_threshold: float = 0.25,
        seed: int | None = None,
        batch_size: int | None = None,
    ):
        """Configure the sampling parameters and register the model.

        Args:
            model: The masked language model to sample from.
            max_samples: Number of candidates to draw per space.
            temperature: Temperature applied to the logits; higher is more
                diverse.
            top_p: Keep only the most likely residues whose probability sums to
                ``top_p``.
            max_mutations: Number of mutations a candidate may carry, counting
                the ones already written into the scaffold. Defaults to the
                number of mutable sites.
            confidence_threshold: A residue the model assigns at least this
                probability is written into the scaffold outright.
            position_threshold: Sites whose non-wild-type probability falls
                below this are never mutated.
            seed: Seed for each device-specific sampling generator.
            batch_size: Number of search spaces per forward pass.

        Raises:
            ValueError: If ``confidence_threshold`` is below ``0.5``, which
                would leave more than one residue per site eligible for the
                scaffold.
        """
        if confidence_threshold < 0.5:
            raise ValueError(
                f"confidence_threshold must be >= 0.5 found {confidence_threshold=}"
            )
        super().__init__(seed=seed, batch_size=batch_size)
        self.model = model
        self.max_mutations = max_mutations
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.position_threshold = position_threshold
        self.max_samples = max_samples
        self.top_p = top_p

    def _get_scaffold(self, space: spaces.SearchSpace) -> TokenIDs:
        batch = _prepare_per_position_masked_inputs(
            space.init_sequence, space.mutable_positions
        )
        responses = common.execute_mlm_batch(batch, self.model)
        probs = _responses_to_probs(responses).squeeze(1)  # M, V=20
        space = space.to(probs.device)
        probs[~space.allowed[space.mutable_positions]] = 0  # exclude denied residues
        positions, aa = (probs > self.confidence_threshold).nonzero(as_tuple=True)
        mutable_positions = space.mutable_positions.nonzero(as_tuple=True)[0]
        confident_positions = mutable_positions[positions]
        scaffold = space.init_sequence.clone()
        scaffold[confident_positions] = aa
        return scaffold

    def _get_position_weights(
        self, space: spaces.SearchSpace, scaffold: TokenIDs
    ) -> Float[torch.Tensor, " l"]:
        space = space.to(scaffold.device)
        batch = _prepare_per_position_masked_inputs(scaffold, space.mutable_positions)
        responses = common.execute_mlm_batch(batch, self.model, device=scaffold.device)
        probs = _responses_to_probs(responses).squeeze(1)
        probs[~space.allowed[space.mutable_positions]] = 0  # exclude denied residues
        wt_probs = probs.gather(
            dim=1, index=space.init_sequence[space.mutable_positions, None]
        ).squeeze(1)
        weights = torch.zeros(scaffold.shape, dtype=torch.float32, device=probs.device)
        # due to masking above probs won't always sum to 1 so can't do 1 - wt_probs
        weights[space.mutable_positions] = probs.sum(dim=1) - wt_probs
        weights[weights < self.position_threshold] = 0
        weights[space.init_sequence != scaffold] = 0
        return weights

    def sample_one(self, space: spaces.SearchSpace) -> BatchTokenIDs:
        """Propose candidates for a single space.

        Residues the model is confident about are written into a scaffold in one
        shot; the remaining mutations are then drawn one site at a time, each
        conditioned on the mutations placed before it.

        Args:
            space: The space to propose candidates in.

        Returns:
            Candidates of shape ``(n, l)``, where ``n`` is at most
            ``max_samples``.
        """
        n_mutable = int(space.mutable_positions.sum())
        if n_mutable == 0:
            return space.init_sequence.new_empty((0, space.init_sequence.shape[0]))
        max_mutations = n_mutable if self.max_mutations is None else self.max_mutations
        max_mutations = min(max_mutations, n_mutable)
        scaffold = self._get_scaffold(space)
        device = scaffold.device
        space = space.to(device)
        curr_mutation_count = int((space.init_sequence != scaffold).sum())
        max_mutations -= curr_mutation_count
        if max_mutations <= 0:
            logger.warning(
                "High-confidence scaffold already meet/exceed self.max_mutations=%r "
                "mutations with %s mutations. No further mutations "
                "will be made. Consider increasing the confidence_threshold or "
                "max_mutations.",
                self.max_mutations,
                curr_mutation_count,
            )
            return scaffold[None, :]  # scaffold alone satisfies max_mutations
        pos_weights = self._get_position_weights(space, scaffold)
        available_positions_count = int((pos_weights != 0).sum())
        if available_positions_count < max_mutations:
            logger.warning(
                "Only %s are eligible for mutation, needed "
                "%s to satisfy %s requested mutations. "
                "You can lower position_threshold to make more sites eligible. "
                "This round will mutate a maximum of %s sites.",
                available_positions_count,
                max_mutations,
                self.max_mutations,
                available_positions_count + curr_mutation_count,
            )
        max_mutations = min(available_positions_count, max_mutations)
        if max_mutations == 0:
            return scaffold[None, :]  # no candidate positions
        sampled_positions = torch.stack(
            [
                torch.multinomial(
                    pos_weights,
                    num_samples=max_mutations,
                    generator=self.rng(device),
                )
                for _ in range(self.max_samples)
            ],
            dim=1,
        )
        # M, N
        sampled_positions = sampled_positions.unique(dim=1)
        inputs = scaffold.repeat(sampled_positions.shape[1], 1)
        for positions in sampled_positions:
            masked_inputs = inputs.clone()
            indices = torch.arange(positions.shape[0], device=device)
            masked_inputs[indices, positions] = tokenizer.mask_token_id
            responses = common.execute_mlm_batch(
                masked_inputs, self.model, device=device
            )
            next_inputs = []
            for input, pos, response in zip(inputs, positions, responses, strict=True):
                input_mask = torch.zeros_like(input, dtype=torch.bool)
                input_mask[pos] = True
                next_inputs.append(
                    common.sample_from_logits(
                        response,
                        input,
                        input_mask,
                        space.allowed[[int(pos)]],
                        temperature=self.temperature,
                        num_samples=1,
                        top_p=self.top_p,
                        generator=self.rng(device),
                    ).squeeze(0)
                )
            inputs = torch.stack(next_inputs)
        return inputs

    def sample(self, spaces: SequenceLike[spaces.SearchSpace]) -> list[BatchTokenIDs]:
        """Propose candidates for each space with :meth:`sample_one`.

        Args:
            spaces: Search spaces to sample candidates from.

        Returns:
            One batch of candidate token IDs per input search space.
        """
        return [self.sample_one(space) for space in spaces]
