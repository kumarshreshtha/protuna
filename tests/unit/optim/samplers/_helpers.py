"""Shared helpers for sampler tests."""

from __future__ import annotations

import math

from jaxtyping import Float
import torch

from protuna import inference
from protuna.data import tokenizers
from protuna.optim import spaces
from protuna.typing import SequenceLike

_TOKENIZER = tokenizers.ProteinTokenizer()


_MASK = _TOKENIZER.mask_token_id


_UNLIKELY = -50.0


def _space(init_sequence: str, allowed_at: dict[int, str]) -> spaces.SearchSpace:
    """Build a search space allowing exactly ``allowed_at[pos]`` at each position.

    Positions missing from ``allowed_at`` keep only their wild-type residue, which
    :class:`~protuna.optim.spaces.search.SearchSpace` restores on construction.
    """
    allowed = torch.zeros(
        len(init_sequence), len(_TOKENIZER.aa_token_ids), dtype=torch.bool
    )
    for position, residues in allowed_at.items():
        allowed[position, _TOKENIZER.encode(residues)] = True
    return spaces.SearchSpace(_TOKENIZER.encode(init_sequence), allowed)


def _decode(token_ids: torch.Tensor) -> list[str]:
    """Decode a ``(b, l)`` batch of token IDs into sequence strings."""
    return _TOKENIZER.decode(list(token_ids))


def _masked_positions(token_ids: torch.Tensor) -> list[int]:
    """List the sequence positions a request still leaves masked."""
    return (token_ids == _MASK).nonzero().flatten().tolist()


def _residues_at(token_ids: torch.Tensor, position: int) -> set[str]:
    """Collect the distinct residues a batch of proposals carries at a position."""
    return set(_TOKENIZER.decode(list(token_ids[:, position].unsqueeze(1))))


class _FakeMLM(
    inference.ModelFreeLocalHandler[
        inference.MLMLogitsRequest, Float[torch.Tensor, "m v"]
    ]
):
    """Masked LM stand-in returning the same distribution at every masked position.

    Attributes:
        residue_probs: Probability assigned to each named residue. Every other
            residue gets :data:`_UNLIKELY`.
        requests: Every request the sampler has sent, in order.
    """

    def __init__(
        self,
        residue_probs: dict[str, float],
        position_probs: dict[int, dict[str, float]] | None = None,
    ):
        super().__init__()
        self.residue_probs = residue_probs
        self.position_probs = position_probs or {}
        self.requests: list[inference.MLMLogitsRequest] = []

    def _logits(self, token_ids: torch.Tensor) -> Float[torch.Tensor, "m v"]:
        masked = _masked_positions(token_ids)
        logits = torch.full((len(masked), len(_TOKENIZER.aa_token_ids)), _UNLIKELY)
        for row, position in enumerate(masked):
            probs = self.position_probs.get(position, self.residue_probs)
            for residue, prob in probs.items():
                logits[row, _TOKENIZER.encode(residue)] = math.log(prob)
        return logits

    def __call__(
        self, requests: SequenceLike[inference.MLMLogitsRequest]
    ) -> list[Float[torch.Tensor, "m v"]]:
        for request in requests:
            assert isinstance(request, inference.MLMLogitsRequest)
            assert request.masked_positions_only is True
            assert request.aa_tokens_only is True
        self.requests.extend(requests)
        return [self._logits(request.token_ids) for request in requests]


def _all_residues_allowed(batch: torch.Tensor, space: spaces.SearchSpace) -> bool:
    """Whether every residue in a batch is one the space allows at its position."""
    positions = torch.arange(batch.shape[1]).expand_as(batch)
    return bool(space.allowed[positions, batch].all())
