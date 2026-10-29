"""Limit how far a candidate can go from the initial sequence in some space."""

from jaxtyping import Bool
import torch

from protuna import inference
from protuna.typing import SequenceLike, TokenIDs


# Build a trust region based on edit distance in the input space
class MaxMutationConstraint(
    inference.ModelFreeLocalHandler[TokenIDs, Bool[torch.Tensor, ""]]
):
    """Accept sequences with at most ``max_mutations`` changed residues."""

    def __init__(
        self,
        base_sequence: TokenIDs,
        max_mutations: float,
        *,
        batch_size: int | None = None,
    ):
        """Set the reference sequence and mutation limit.

        Args:
            base_sequence: Tokenized sequence from which mutations are counted.
            max_mutations: Maximum number of differing sites, inclusive.
            batch_size: Number of requests per batch while streaming.
        """
        super().__init__(batch_size=batch_size)
        self.base_sequence = base_sequence[None, :]
        self.max_mutations = max_mutations

    def __call__(self, requests: SequenceLike[TokenIDs]) -> Bool[torch.Tensor, " b"]:
        """Return whether each sequence is within the mutation limit.

        Args:
            requests: Tokenized candidate sequences of the same length as the
                reference sequence.

        Returns:
            Boolean mask indicating which candidates satisfy the mutation limit.
        """
        if len(requests) == 0:
            return torch.empty(0, dtype=torch.bool, device=self.base_sequence.device)
        batch_tokens = torch.stack(list(requests))
        num_mutations = (
            self.base_sequence.to(batch_tokens.device) != batch_tokens
        ).sum(dim=1)
        passed = num_mutations <= self.max_mutations
        return passed
