"""Shared types and structural interfaces used throughout the library."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, overload, Protocol

from jaxtyping import Integer
import torch


class SequenceLike[T](Protocol):
    """Structural interface that supports array-like containers and lists/tuples."""

    def __len__(self) -> int:
        """Return the number of elements."""
        ...

    @overload
    def __getitem__(self, index: int, /) -> T: ...

    @overload
    def __getitem__(self, index: slice, /) -> SequenceLike[T]: ...

    def __getitem__(self, index: int | slice, /) -> T | SequenceLike[T]:
        """Return an element or a sliced sequence."""
        ...

    def __contains__(self, value: object, /) -> bool:
        """Return whether the sequence contains the value."""
        ...

    def __iter__(self) -> Iterator[T]:
        """Iterate over the elements in order."""
        ...


type TokenIDs = Integer[torch.Tensor, " l"]
type BatchTokenIDs = Integer[torch.Tensor, "b l"]

type AminoAcid = Literal[
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
