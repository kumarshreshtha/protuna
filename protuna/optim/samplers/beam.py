"""Search masked language model proposals using a bounded beam."""

import dataclasses
from typing import Literal, Self

from jaxtyping import Float, Integer
import torch

from protuna import inference
from protuna.data import tokenizer
from protuna.optim import spaces
from protuna.optim.samplers import base
from protuna.optim.samplers import common
from protuna.typing import BatchTokenIDs, SequenceLike


# helper to organize all the context needed to decode
@dataclasses.dataclass
class _MLMBeamSamplerState:
    space: spaces.SearchSpace
    requests: list[inference.MLMLogitsRequest]
    mask_loc: Integer[torch.Tensor, " m"]
    topk_scores: Float[torch.Tensor, " k"] = dataclasses.field(
        default_factory=lambda: torch.zeros(1)
    )
    samples: list[Integer[torch.Tensor, " n"]] = dataclasses.field(default_factory=list)

    @classmethod
    def from_search_space(cls, space: spaces.SearchSpace) -> Self:
        mask_positions = space.mutable_positions.nonzero()[:, 0]
        if mask_positions.numel() == 0:
            return cls(space, [], mask_positions)
        token_ids = space.init_sequence.clone()
        token_ids[space.mutable_positions] = tokenizer.mask_token_id
        request = inference.MLMLogitsRequest(
            token_ids, masked_positions_only=True, aa_tokens_only=True
        )
        return cls(space, [request], mask_positions)


def _get_unmask_position(
    logp: Float[torch.Tensor, "b m v"],
    sampling_order: Literal["lr", "random", "max_p"] = "lr",
    generator: torch.Generator | None = None,
) -> int:
    if sampling_order == "lr":
        return 0
    if sampling_order == "random":
        return int(
            torch.randint(
                0, logp.shape[1], (1,), generator=generator, device=logp.device
            )
        )
    return int(torch.unravel_index(logp[0].argmax(), logp[0].shape)[0])


class MLMBeamSampler(base.Sampler):
    """Propose candidates with masked language model beam search.

    The sampler starts with all mutable sites masked and fills one site at a
    time with allowed amino acids in the configured ``sampling_order``.
    Partial sequences are ranked by the sum of their temperature-scaled
    conditional log probabilities, with up to ``beam_size`` retained at each
    step. Each prediction is conditioned on the residues already filled in
    that partial sequence.

    The sampler returns up to ``max_samples`` completed candidates per search
    space. The bounded beam does not guarantee globally optimal sequences,
    and results depend on the order in which sites are filled.
    """

    def __init__(
        self,
        model: inference.MLMForLogitsLike,
        max_samples: int,
        *,
        beam_size: int | None = None,
        temperature: float = 1.0,
        sampling_order: Literal["lr", "random", "max_p"] = "lr",
        seed: int | None = None,
        batch_size: int | None = None,
    ):
        """Initialize the beam search settings and register the generator model.

        Args:
            model: The masked language model to sample from.
            max_samples: Number of candidates to return per space.
            beam_size: Number of partial sequences to keep; defaults to
                ``max_samples``.
            temperature: Temperature applied to the logits.
            sampling_order: Which site to unmask next — left to right
                (``"lr"``), at random (``"random"``), or the most confident one
                (``"max_p"``).
            seed: Seed for each device-specific sampling generator.
            batch_size: Number of search spaces per forward pass.

        Raises:
            ValueError: If ``beam_size`` is smaller than
                ``max_samples``.
        """
        super().__init__(batch_size=batch_size, seed=seed)
        if beam_size is not None and beam_size < max_samples:
            raise ValueError(f"{beam_size=} must be >= {max_samples=}")
        self.model = model
        self.max_samples = max_samples
        self.beam_size = beam_size if beam_size is not None else max_samples
        self.temperature = temperature
        self.sampling_order = sampling_order

    def sample(self, spaces: SequenceLike[spaces.SearchSpace]) -> list[BatchTokenIDs]:
        """Propose the highest-likelihood candidates of each space.

        Args:
            spaces: Search spaces to sample candidates from.

        Returns:
            One batch of candidate token IDs per input search space.
        """
        states = [_MLMBeamSamplerState.from_search_space(space) for space in spaces]
        requests = [request for state in states for request in state.requests]
        device = None
        while requests:
            outputs = common.execute_mlm_batch(
                [request.token_ids for request in requests], self.model, device=device
            )
            device = outputs[0].device
            start_idx = 0
            for state in states:
                num_requests = len(state.requests)
                if num_requests == 0:
                    continue
                space_outputs = outputs[start_idx : start_idx + num_requests]
                start_idx = start_idx + num_requests
                # b, m, v
                logits = torch.stack(list(space_outputs)) / self.temperature
                state.space = state.space.to(device)
                state.mask_loc = state.mask_loc.to(device)
                state.topk_scores = state.topk_scores.to(device)
                logp = logits.log_softmax(dim=-1)
                logp[:, ~state.space.allowed[state.mask_loc]] = -torch.inf
                unmask_idx = _get_unmask_position(
                    logp, self.sampling_order, generator=self.rng(logp.device)
                )
                remaining_idx = torch.tensor(
                    [i for i in range(logp.shape[1]) if i != unmask_idx],
                    dtype=torch.long,
                    device=device,
                )
                allowed_tokens = state.space.allowed[
                    state.mask_loc[unmask_idx]
                ].nonzero(as_tuple=True)[0]
                new_scores = (
                    logp[:, unmask_idx, allowed_tokens] + state.topk_scores[:, None]
                )
                k = min(self.beam_size, new_scores.numel())
                topk_scores, topk_idxs = new_scores.flatten().topk(k=k)
                inputs, token_idxs = torch.unravel_index(topk_idxs, new_scores.shape)
                tokens = allowed_tokens[token_idxs]
                new_requests = []
                for input_idx, token_idx in zip(inputs, tokens, strict=True):
                    token_ids = state.requests[input_idx].token_ids.to(device).clone()
                    token_ids[state.mask_loc[unmask_idx]] = token_idx
                    if remaining_idx.numel() == 0:
                        if len(state.samples) < self.max_samples:
                            state.samples.append(token_ids)
                        continue
                    new_requests.append(
                        inference.MLMLogitsRequest(
                            token_ids, masked_positions_only=True, aa_tokens_only=True
                        )
                    )
                state.requests = new_requests
                state.topk_scores = topk_scores
                state.mask_loc = state.mask_loc[remaining_idx]
            requests = [request for state in states for request in state.requests]
        return [
            torch.stack(state.samples)
            if state.samples
            else state.space.init_sequence.new_empty(
                (0, state.space.init_sequence.shape[0])
            )
            for state in states
        ]
