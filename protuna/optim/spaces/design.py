"""Encapsulates the initial search spaces and a strategy for sampling from them.

:class:`DesignSpace` brings together the starting sequences of a campaign round, the
mutations allowed on each, and a strategy for choosing which sites to explore
together.
"""

from collections.abc import Callable, Iterator
import hashlib
from typing import Self

from jaxtyping import Bool, Integer
import torch

from protuna import inference
from protuna.optim.spaces import search
from protuna.optim.spaces import subspaces

type SearchMask = Bool[torch.Tensor, "l v"]
type SearchRule = Callable[[SearchMask], SearchMask]


def _identity_sampler(space: search.SearchSpace) -> Iterator[search.SearchSpace]:
    yield space


class DesignSpace(inference.HandlerRegistryMixin):
    """Collection of search spaces and a subspace sampling strategy to divide them.

    Each :class:`~protuna.optim.spaces.search.SearchSpace` defines a starting
    sequence and its allowed mutations. :meth:`sample_search_spaces` applies the
    configured sampler to each starting space, or yields the starting spaces
    unchanged when no sampler is supplied.

    Examples:
        Split a starting sequence's mutable sites into two disjoint groups:

        >>> from protuna.optim import spaces
        >>> space = spaces.SearchSpace.from_rules("ACDE", [])
        >>> design_space = spaces.DesignSpace(
        ...     [space], space_sampler=spaces.PartitionSampler(n_partitions=2)
        ... )
        >>> [sub.mutable_positions.tolist()
        ...  for sub in design_space.sample_search_spaces()]
        [[True, True, False, False], [False, False, True, True]]
    """

    def __init__(
        self,
        init_spaces: list[search.SearchSpace],
        *,
        space_sampler: subspaces.SearchSpaceSampler | None = None,
    ):
        """Set the starting search spaces and optional subspace sampler.

        Args:
            init_spaces: Starting sequences and their allowed mutations.
            space_sampler: Sampler used to divide a search space into subspaces.
                If ``None``, yield each starting space unchanged.
        """
        super().__init__()
        self.init_spaces = init_spaces
        self.space_sampler = space_sampler

    def __repr__(self) -> str:
        """Return a compact summary of the design space.

        Returns:
            The starting space count and subspace sampler representation.
        """
        return (
            f"{type(self).__name__}(n_spaces={len(self.init_spaces)}, "
            f"space_sampler={self.space_sampler!r})"
        )

    @classmethod
    def from_rules(
        cls,
        rules: list[SearchRule],
        init_sequences: list[Integer[torch.Tensor, " l"]],
        space_sampler: subspaces.SearchSpaceSampler,
    ) -> Self:
        """Build starting search spaces by applying shared rules to each sequence.

        Args:
            rules: Mutation rules to apply to every starting sequence. An empty
                list allows all amino acids at every site.
            init_sequences: Tokenized starting sequences, in sampling order.
            space_sampler: Sampler to use to divide the initial search spaces into
                subspaces.

        Returns:
            A design space containing one starting space per input sequence.
        """
        init_spaces = [
            search.SearchSpace.from_rules(init_sequence, rules)
            for init_sequence in init_sequences
        ]
        return cls(init_spaces, space_sampler=space_sampler)

    def sample_search_spaces(self) -> Iterator[search.SearchSpace]:
        """Yield sampled subspaces with distinct mutable-site masks per starting space.

        If no sampler is configured, yield each starting space unchanged.

        Yields:
            Search spaces for candidate generation.
        """
        sampler = (
            _identity_sampler if self.space_sampler is None else self.space_sampler
        )
        for space in self.init_spaces:
            seen = set()
            for sampled_space in sampler(space):
                mask = sampled_space.mutable_positions.cpu().contiguous().numpy()
                key = hashlib.blake2b(memoryview(mask), digest_size=16).digest()
                if key in seen:
                    continue
                seen.add(key)
                yield sampled_space
