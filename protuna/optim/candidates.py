"""Candidate streaming interface and the streamed batches for acquisition and recycling.

A :class:`CandidateBatch` is a collection of generated sequences, their campaign
lineage and optionally their evaluation results.

:class:`CandidateStream` defines the structural interface for streaming batches of
:class:`CandidateBatch` and to fetch them by given sequence identifiers.
"""

from collections.abc import Iterator, Mapping, Sequence
import dataclasses
from typing import Protocol

from jaxtyping import Float, Integer
import torch

from protuna.typing import BatchTokenIDs


@dataclasses.dataclass(frozen=True)
class CandidateBatch:
    """Group candidate sequences with their lineage and evaluation results.

    All tensors share a leading batch dimension ``b``, with each row referring
    to the same candidate across fields. Optional evaluation fields are ``None``
    when unavailable.

    Args:
        seq_id: Sequence identifiers.
        token_ids: Encoded sequences.
        parent_id: Identifiers of the immediate parent sequences.
        ancestor_id: Identifiers of the initial sequences from which candidates
            descend.
        cycle_id: Campaign cycles in which the sequences were generated.
        objective_mean: Predicted objective means, or ``None``.
        objective_std: Predicted objective standard deviations, or ``None``.
        objective_samples: Predicted objective samples, with shape ``[b, n]``
            for ``n`` samples per candidate, or ``None``.
        feasibility: Soft constraint names mapped to per-candidate probabilities
            of satisfying each constraint, or ``None`` when no soft constraint
            results are included.
    """

    seq_id: Integer[torch.Tensor, " b"]
    token_ids: BatchTokenIDs
    parent_id: Integer[torch.Tensor, " b"]
    ancestor_id: Integer[torch.Tensor, " b"]
    cycle_id: Integer[torch.Tensor, " b"]
    objective_mean: Float[torch.Tensor, " b"] | None
    objective_std: Float[torch.Tensor, " b"] | None
    objective_samples: Float[torch.Tensor, "b n"] | None
    feasibility: Mapping[str, torch.Tensor] | None


class CandidateStream(Protocol):
    """Provide candidate batches and lookup by sequence identifier.

    Iteration exposes an iterator over :class:`CandidateBatch` of the entire candidate
    pool; :meth:`candidates_from_ids` retrieves a selected subset in the requested
    order.
    """

    def __iter__(self) -> Iterator[CandidateBatch]:
        """Iterate over the candidate pool in batches.

        Yields:
            Candidate batches.
        """
        ...

    def candidates_from_ids(self, seq_ids: Sequence[int]) -> CandidateBatch:
        """Return candidates in the order of the requested identifiers.

        Args:
            seq_ids: Sequence identifiers to retrieve.

        Returns:
            A batch of candidates with the requested ``seq_ids``.
        """
        ...
