"""Propose candidates using sequence alignment-derived per-site residue scores."""

from typing import Self

from jaxtyping import Float
import torch

from protuna import data
from protuna.optim import spaces
from protuna.optim.samplers import base
from protuna.optim.samplers import common
from protuna.score import pssm
from protuna.typing import BatchTokenIDs, SequenceLike


class PSSMSampler(base.Sampler):
    """Draw candidates from an MSA's per-site residue frequencies."""

    def __init__(
        self,
        freq_matrix: Float[torch.Tensor, "l v"],
        max_samples: int,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
    ):
        """Configure candidate sampling from per-site residue frequencies.

        Args:
            freq_matrix: Per-site residue frequencies.
            max_samples: Number of candidates to draw per search space before
                removing duplicates.
            seed: Seed for each device-specific sampling generator; ``None``
                uses its default state.
            batch_size: Number of search spaces per forward pass; ``None``
                processes all.
        """
        super().__init__(seed=seed, batch_size=batch_size)
        self.max_samples = max_samples
        self.pssm_logits = freq_matrix
        self.device = self.pssm_logits.device

    @classmethod
    def from_msa(
        cls,
        msa: data.TokenizedMSA,
        max_samples: int,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
    ) -> Self:
        """Build the sampler from an alignment instead of a frequency matrix.

        Args:
            msa: Alignment used to compute per-site residue frequencies.
            max_samples: Number of candidates to draw per space.
            seed: Seed for each device-specific sampling generator; ``None``
                uses its default state.
            batch_size: Number of search spaces per forward pass; ``None``
                processes all.

        Returns:
            A sampler initialized with Laplace-smoothed alignment frequencies.
        """
        return cls(
            pssm.compute_frequency_matrix(msa, laplace_smoothing=True),
            max_samples=max_samples,
            seed=seed,
            batch_size=batch_size,
        )

    def sample(self, spaces: SequenceLike[spaces.SearchSpace]) -> list[BatchTokenIDs]:
        """Propose candidates drawn from the frequency matrix.

        Args:
            spaces: Search spaces to sample candidates from.

        Returns:
            One batch of candidate token IDs per input search space.
        """
        spaces = [space.to(self.device) for space in spaces]
        return [
            common.sample_from_logits(
                self.pssm_logits[space.mutable_positions],
                space.init_sequence,
                space.mutable_positions,
                space.allowed[space.mutable_positions],
                self.max_samples,
                generator=self.rng(self.pssm_logits.device),
            )
            for space in spaces
        ]
