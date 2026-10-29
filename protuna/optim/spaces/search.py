"""Protein sequence search spaces with input space constraints.

:class:`SearchSpace` pairs an initial sequence with a mask of allowed residues.
Each :py:type:`SearchRule` transforms that mask to constrain the space.
"""

from collections.abc import Callable, Iterable
import dataclasses
import decimal
import functools
import math
from typing import Self

from jaxtyping import Bool, Integer
import torch

from protuna import data
from protuna.typing import BatchTokenIDs, TokenIDs

type SearchMask = Bool[torch.Tensor, "l v"]
type SearchRule = Callable[[SearchMask], SearchMask]


@dataclasses.dataclass(frozen=True)
class SearchSpace:
    """Starting sequence and the mutations allowed on top of it.

    Examples:
        Build an unconstrained space for a two-residue sequence:

        >>> from protuna.optim.spaces import search
        >>> # No rules: all 20 amino acids are allowed at each site.
        >>> space = search.SearchSpace.from_rules("AC", [])
        >>> # Both sites allow mutations away from the initial residue.
        >>> space.mutable_positions.tolist()
        [True, True]
        >>> # The space contains 20 * 20 possible sequences.
        >>> space.size()
        400

    Args:
        init_sequence: The tokenized sequence the mutations are made from.
        allowed: Mask where ``True`` at ``(site, aa)`` means amino acid ``aa``
            is allowed at that site. The residues of ``init_sequence`` are
            always allowed.
    """

    __hash__ = None
    init_sequence: Integer[torch.Tensor, " l"]
    allowed: Bool[torch.Tensor, "l v"]

    def __post_init__(self):
        """Re-open the residues of the initial sequence."""
        # initial sequence is always allowed
        init_sequence = self.init_sequence
        self.allowed[torch.arange(init_sequence.shape[0]), init_sequence] = True

    def __repr__(self) -> str:
        """Return a compact summary of the search space."""
        size = self.size()
        size_repr = (
            f"size≈{decimal.Decimal(size):.2e}" if size >= 1_000_000 else f"size={size}"
        )
        return (
            f"{type(self).__name__}(length={len(self.init_sequence)}, "
            f"mutable_sites={int(self.mutable_positions.sum())}, "
            f"{size_repr}, device={str(self.init_sequence.device)!r})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare the initial sequence and the allowed mask element-wise.

        Args:
            other: The object to compare with this space.

        Returns:
            Whether both sequences and masks match, or ``NotImplemented`` for
            objects that are not search spaces.
        """
        if not isinstance(other, SearchSpace):
            return NotImplemented
        seq_eq = torch.equal(self.init_sequence, other.init_sequence)
        if seq_eq:
            return torch.equal(self.allowed, other.allowed)
        return False

    @classmethod
    def from_rules(
        cls,
        init_sequence: str | Integer[torch.Tensor, " l"],
        rules: Iterable[SearchRule],
    ) -> Self:
        """Build a space by applying ``rules`` in order to a fully open mask.

        Args:
            init_sequence: The starting sequence, as a string or token ids.
            rules: The rules to apply, see :mod:`protuna.optim.rules`.

        Returns:
            A search space with the resulting allowed mask and the initial
            sequence's residues always allowed.
        """
        if isinstance(init_sequence, str):
            init_sequence = data.tokenizer.encode(init_sequence)
        full = torch.ones(
            init_sequence.shape[0],
            len(data.tokenizer.aa_token_ids),
            dtype=torch.bool,
            device=init_sequence.device,
        )
        allowed = functools.reduce(lambda mask, rule: rule(mask), rules, full)
        return cls(init_sequence, allowed)

    def to(self, device: torch.device) -> Self:
        """Return a copy of the space on ``device``, no-op if already on device.

        Args:
            device: The device to hold the initial sequence and allowed mask.

        Returns:
            The space on the requested device, or ``self`` if already there.
        """
        if self.init_sequence.device == torch.empty(0, device=device).device:
            return self
        return dataclasses.replace(
            self,
            init_sequence=self.init_sequence.to(device),
            allowed=self.allowed.to(device),
        )

    def size(self) -> int:
        """Return the number of sequences the space contains.

        Returns:
            The total number of allowed residue combinations.
        """
        return math.prod(self.allowed.sum(dim=1).tolist())

    @functools.cached_property
    def mutable_positions(self) -> Bool[torch.Tensor, " l"]:
        """Mask of the sites that can be mutated.

        A site is mutable if it allows for substitution of the wild-type residue.

        Returns:
            A mask marking sites with at least one allowed alternative residue.
        """
        return self.allowed.sum(dim=1) > 1

    def is_valid(self, sequence: str | TokenIDs) -> bool:
        """Whether ``sequence`` is within the input constraints of the space.

        Args:
            sequence: The sequence to check, as a string or token ids.

        Returns:
            Whether every supplied residue is allowed at its site.
        """
        if len(sequence) != self.init_sequence.shape[0]:
            return False
        seq_tok_ids = (
            data.tokenizer.encode(sequence) if isinstance(sequence, str) else sequence
        )
        positions = torch.arange(
            len(sequence), dtype=torch.long, device=seq_tok_ids.device
        )
        return bool(self.allowed[positions, seq_tok_ids].all())

    def is_valid_batch(self, batch: BatchTokenIDs) -> Bool[torch.Tensor, " b"]:
        """Whether sequences in the batch are within the input constraints of the space.

        Args:
            batch: The batch of sequence token ids to check.

        Returns:
            Whether each batch item is valid in the search space.
        """
        if batch.shape[1] != self.init_sequence.shape[0]:
            return batch.new_zeros(batch.shape[0], dtype=torch.bool)
        positions = torch.arange(batch.shape[1], device=batch.device)
        return self.allowed[positions, batch].all(dim=1)

    def available_aa_at_pos(self, pos: int) -> str:
        """Return the residues allowed at site ``pos``.

        Args:
            pos: The zero-based site index to inspect.

        Returns:
            The allowed amino acids in tokenizer vocabulary order.
        """
        return data.tokenizer.decode(torch.argwhere(self.allowed[pos]).flatten())
