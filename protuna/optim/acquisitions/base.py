"""Acquisition base class, the identity acquisition and shared utilities."""

import abc

from jaxtyping import Float
import torch

from protuna import inference
from protuna.optim import candidates


class Acquisition(abc.ABC, inference.HandlerRegistryMixin):
    """Base class for acquisition strategies.

    Attributes:
        requires_std: boolean flag indicating whether the acquisition requires the
            objective's estimated standard deviation for it's computations
        requires_samples: boolean flag indicating whether the acquisition requires
            samples drawn from the objective posterior.

    """

    requires_std: bool = False
    requires_samples: bool = False

    @abc.abstractmethod
    def select(self, stream: candidates.CandidateStream) -> list[int]:
        """Return the ids of the candidates selected by the acquisition criteria.

        Args:
            stream: Candidate pool to select from.

        Returns:
            Sequence ids of the selected candidates.
        """
        ...


class IdentityAcquisition(Acquisition):
    """Identity acquisition function that selects everything it sees."""

    def select(self, stream: candidates.CandidateStream) -> list[int]:
        """Return the ids of all candidates in the stream.

        Args:
            stream: Candidate pool to select from.

        Returns:
            Sequence ids of all candidates in stream order.
        """
        return [seq_id for batch in stream for seq_id in batch.seq_id.tolist()]


def feasibility(batch: candidates.CandidateBatch) -> Float[torch.Tensor, " b"]:
    """Return the product of soft constraint probabilities for each candidate.

    Return ones when the batch has no soft constraints.

    Args:
        batch: Candidates with optional per-constraint feasibility values.

    Returns:
        Combined feasibility probability for each candidate.
    """
    if not batch.feasibility:
        return torch.ones_like(batch.seq_id, dtype=torch.float)
    constraints = torch.stack(
        [batch.feasibility[constraint] for constraint in batch.feasibility],
        dim=1,
    )
    return constraints.log().sum(dim=1).exp()
