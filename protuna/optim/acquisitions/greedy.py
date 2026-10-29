"""Select candidates with the highest feasible objective predictions."""

from protuna.optim import candidates
from protuna.optim.acquisitions import base
from protuna.utils import data_structs


class Greedy(base.Acquisition):
    """Take the ``budget`` candidates with the highest predicted objective."""

    def __init__(self, budget: int, *, feasibility_threshold: float = 0.0):
        """Initialize greedy selection with the given budget and feasibility threshold.

        Args:
            budget: Number of sequences to select.
            feasibility_threshold: Minimum joint feasibility, excluded at equality.
        """
        super().__init__()
        self.budget = budget
        self.feasibility_threshold = feasibility_threshold

    def select(self, stream: candidates.CandidateStream) -> list[int]:
        """Select candidates with the highest feasible objective means.

        Args:
            stream: Candidate pool with predicted objective means.

        Returns:
            Sequence ids of up to ``budget`` candidates above the feasibility
            threshold, ordered by ascending objective mean.

        Raises:
            ValueError: If a batch has no predicted objective means.
        """
        heap = data_structs.TopKHeap(self.budget)
        for batch in stream:
            if batch.objective_mean is None:
                raise ValueError("Greedy acquisition requires objective mean values.")
            feasible = base.feasibility(batch) > self.feasibility_threshold
            objective_mean = batch.objective_mean[feasible]
            seq_ids = batch.seq_id[feasible]
            for seq_id, mean in zip(seq_ids, objective_mean, strict=True):
                heap.push((float(mean), int(seq_id)))

        return [sample_id for _, sample_id in heap.items()]
