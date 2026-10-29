"""Propose candidates by independently sampling all mutable sites from a masked LM."""

import torch

from protuna import inference
from protuna.data import tokenizer
from protuna.optim import spaces
from protuna.optim.samplers import base
from protuna.optim.samplers import common
from protuna.typing import BatchTokenIDs, SequenceLike


class MLMSampler(base.Sampler):
    """Draw candidates from a masked language model.

    The open sites are masked and filled in one pass, so the residues of a
    candidate are drawn independently of one another.
    """

    def __init__(
        self,
        model: inference.MLMForLogitsLike,
        max_samples: int,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
        batch_size: int | None = None,
    ):
        """Configure the sampling parameters and register the model.

        Args:
            model: The masked language model to sample from.
            max_samples: Number of candidates to draw per space.
            temperature: Temperature applied to the logits.
            top_p: Keep only the most likely residues whose probability sums to
                ``top_p``.
            seed: Seed for each device-specific sampling generator.
            batch_size: Number of search spaces per forward pass.
        """
        super().__init__(seed=seed, batch_size=batch_size)
        self.model = model
        self.max_samples = max_samples
        self.temperature = temperature
        self.top_p = top_p

    def sample(self, spaces: SequenceLike[spaces.SearchSpace]) -> list[BatchTokenIDs]:
        """Propose candidates drawn from the model's masked predictions.

        Args:
            spaces: Search spaces to sample candidates from.

        Returns:
            One batch of candidate token IDs per input search space.
        """
        mutable_mask = torch.stack([space.mutable_positions for space in spaces])
        batch = torch.stack([space.init_sequence for space in spaces])
        batch[mutable_mask] = tokenizer.mask_token_id
        responses = common.execute_mlm_batch(batch, self.model)
        samples = []
        spaces = [space.to(responses[0].device) for space in spaces]
        for logits, space in zip(responses, spaces, strict=True):
            samples.append(
                common.sample_from_logits(
                    logits,
                    space.init_sequence,
                    space.mutable_positions,
                    space.allowed[space.mutable_positions],
                    self.max_samples,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    generator=self.rng(logits.device),
                )
            )
        return samples
