"""Propose candidates by random draws or exhaustive mutation enumeration."""

import itertools

import torch

from protuna.optim import spaces
from protuna.optim.samplers import base
from protuna.typing import BatchTokenIDs, SequenceLike


class RandomSampler(base.Sampler):
    """Draw candidates uniformly from the allowed residues at every site."""

    def __init__(
        self,
        max_samples: int,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
    ):
        """Inits the sampler.

        Args:
            max_samples: Number of candidates to draw per space.
            seed: Seed for each device-specific sampling generator.
            batch_size: Number of search spaces per forward pass.
        """
        super().__init__(seed=seed, batch_size=batch_size)
        self.max_samples = max_samples

    def sample(self, spaces: SequenceLike[spaces.SearchSpace]) -> list[BatchTokenIDs]:
        """Propose random candidates.

        Args:
            spaces: Search spaces to sample candidates from.

        Returns:
            One batch of candidate token IDs per input search space.
        """
        samples = []
        for space in spaces:
            random_proteins = torch.multinomial(
                space.allowed.float(),
                num_samples=self.max_samples,
                replacement=True,
                generator=self.rng(space.allowed.device),
            )
            samples.append(random_proteins.T.contiguous())
        return samples


class SiteSaturationSampler(base.Sampler):
    """Enumerate every single mutant the space allows."""

    def sample(self, spaces: SequenceLike[spaces.SearchSpace]) -> list[BatchTokenIDs]:
        """Propose one candidate per allowed (site, residue) pair.

        Args:
            spaces: Search spaces to sample candidates from.

        Returns:
            One batch of candidate token IDs per input search space.
        """
        samples = []
        for space in spaces:
            token_ids = []
            for pos, row in enumerate(space.allowed):
                for aa in row.nonzero(as_tuple=True)[0]:
                    if aa == space.init_sequence[pos]:
                        continue
                    mutant = space.init_sequence.clone()
                    mutant[pos] = aa
                    token_ids.append(mutant)
            if token_ids:
                samples.append(torch.stack(token_ids))
            else:
                samples.append(
                    space.init_sequence.new_empty((0, space.init_sequence.shape[0]))
                )
        return samples


class BruteForceSampler(base.Sampler):
    """Enumerate every sequence the space contains.

    .. warning::
        The space grows exponentially with the number of open sites; check
        :meth:`~protuna.optim.spaces.search.SearchSpace.size` first.
    """

    def sample(self, spaces: SequenceLike[spaces.SearchSpace]) -> list[BatchTokenIDs]:
        """Propose every combination of allowed residues but the initial one.

        Args:
            spaces: Search spaces to sample candidates from.

        Returns:
            One batch of candidate token IDs per input search space.
        """
        samples = []
        for space in spaces:
            aa_per_pos = [row.nonzero(as_tuple=True)[0] for row in space.allowed]
            mutants = []
            for seq_tokens in itertools.product(*aa_per_pos):
                mutant = torch.tensor(seq_tokens, dtype=torch.long)
                if torch.equal(mutant, space.init_sequence):
                    continue
                mutants.append(mutant)
            if mutants:
                samples.append(torch.stack(mutants))
            else:
                samples.append(
                    space.init_sequence.new_empty((0, space.init_sequence.shape[0]))
                )
        return samples
