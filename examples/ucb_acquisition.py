"""Streaming upper confidence bound selection for the custom acquisitions guide."""

import math

from protuna.optim import acquisitions
from protuna.optim import candidates
from protuna.utils import data_structs


class UpperConfidenceBound(acquisitions.Acquisition):
    """Select candidates with the highest feasible upper confidence bounds."""

    requires_std = True
    requires_samples = False

    def __init__(
        self,
        budget: int,
        beta: float = 4.0,
        *,
        feasibility_threshold: float = 0.0,
    ) -> None:
        """Configure UCB ranking and feasibility filtering.

        Args:
            budget: Maximum number of candidates to select; must be nonnegative.
            beta: Finite, nonnegative confidence weight. Its square root multiplies
                the predicted standard deviation.
            feasibility_threshold: Minimum combined feasibility, excluded at
                equality. Must be between zero and one.
        """
        super().__init__()
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if not 0 <= feasibility_threshold <= 1:
            raise ValueError("feasibility_threshold must be between zero and one")
        self.budget = budget
        self.sqrt_beta = math.sqrt(beta)
        self.feasibility_threshold = feasibility_threshold

    # init-end
    def select(self, stream: candidates.CandidateStream) -> list[int]:
        """Return up to ``budget`` candidate IDs in descending UCB order.

        Args:
            stream: Candidates with objective means and standard deviations.

        Returns:
            Selected sequence IDs, with larger IDs breaking equal-score ties.

        Raises:
            ValueError: If a batch lacks objective means or standard deviations.
        """
        if self.budget == 0:
            return []
        heap = data_structs.TopKHeap[tuple[float, int]](self.budget)
        for batch in stream:
            if batch.objective_mean is None or batch.objective_std is None:
                raise ValueError("UCB requires objective means and standard deviations")
            feasible = acquisitions.feasibility(batch) > self.feasibility_threshold
            scores = (
                batch.objective_mean[feasible]
                + self.sqrt_beta * batch.objective_std[feasible]
            )
            seq_ids = batch.seq_id[feasible]
            for score, seq_id in zip(scores.tolist(), seq_ids.tolist(), strict=True):
                heap.push((score, seq_id))
        return [seq_id for _, seq_id in reversed(heap.items())]
