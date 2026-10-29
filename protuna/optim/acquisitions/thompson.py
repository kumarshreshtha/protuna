"""Select candidates using sampled ensemble member rankings."""

import itertools

import torch

from protuna.optim import candidates
from protuna.optim.acquisitions import base
from protuna.utils import data_structs


class ThompsonSampling(base.Acquisition):
    """Pick candidates by sampling from an ensemble's posterior.

    Each ensemble member keeps its own top-``budget`` ranking; the batch is
    filled by drawing a member at random and taking its best unclaimed
    candidate :cite:`ParallelTS`.
    """

    requires_samples = True
    requires_std = False

    def __init__(
        self,
        budget: int,
        *,
        feasibility_threshold: float = 0.0,
        seed: int | None = None,
    ):
        """Initialize Thompson sampling.

        Args:
            budget: Number of sequences to select.
            feasibility_threshold: Minimum joint feasibility, excluded at equality.
            seed: Seed for ensemble member sampling.
        """
        super().__init__()
        self.budget = budget
        self.feasibility_threshold = feasibility_threshold
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def select(self, stream: candidates.CandidateStream) -> list[int]:
        """Select candidates using sampled ensemble member rankings.

        Args:
            stream: Candidate pool with per-model objective samples.

        Returns:
            Sequence ids of up to ``budget`` candidates above the feasibility
            threshold, with no guaranteed ordering.

        Raises:
            TypeError: If the prediction carries no per-model samples.
        """
        if self.budget == 0:
            return []
        iterator = iter(stream)
        first_batch = next(iterator, None)
        if first_batch is None:
            return []
        if first_batch.objective_samples is None:
            raise TypeError(
                "Thompson Sampling requires access to sampled values of objective. "
            )
        n_models = first_batch.objective_samples.shape[1]
        model_order: list[int] = torch.multinomial(
            torch.ones(n_models),
            self.budget,
            self.budget > n_models,
            generator=self.generator,
        ).tolist()
        heaps = [data_structs.TopKHeap(self.budget) for _ in range(n_models)]

        for batch in itertools.chain([first_batch], iterator):
            if batch.objective_samples is None:
                raise TypeError(
                    "Thompson Sampling requires access to sampled values of objective. "
                )
            feasible = base.feasibility(batch) > self.feasibility_threshold
            objective_samples = batch.objective_samples[feasible]
            seq_ids = batch.seq_id[feasible]
            for i, heap in enumerate(heaps):
                for seq_id, sample in zip(
                    seq_ids, objective_samples[:, i], strict=True
                ):
                    heap.push((float(sample), int(seq_id)))

        # This will happen if total qualified candidates were less than the budget
        if len(heaps[0].heap) < self.budget:
            return [seq_id for _, seq_id in heaps[0].heap]
        ordered_items = [heap.items() for heap in heaps]
        selections = set()
        for idx in model_order:
            _, sample_id = ordered_items[idx].pop()
            while sample_id in selections:
                _, sample_id = ordered_items[idx].pop()
            selections.add(sample_id)
        return list(selections)
