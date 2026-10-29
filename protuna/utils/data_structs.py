"""General purpose data structures."""

import heapq
from typing import Any, Protocol


class Comparable[T](Protocol):
    """Protocol for values that support ``<`` against a ``T``."""

    def __lt__(self, other: T, /) -> bool:
        """Return whether ``self`` sorts before ``other``."""
        ...


class TopKHeap[T: Comparable[Any]]:
    """Bounded min-heap retaining the ``k`` largest items pushed into it.

    Args:
        k: Maximum number of items to retain.
    """

    def __init__(self, k: int):
        """Initialize an empty heap retaining at most ``k`` items."""
        self.k = k
        self.heap: list[T] = []

    def push(self, value: T) -> None:
        """Add ``value``, evicting the smallest item once the heap is full.

        Args:
            value: Item to insert.
        """
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, value)
        else:
            heapq.heappushpop(self.heap, value)

    def items(self) -> list[T]:
        """Return the retained items in ascending order."""
        return sorted(self.heap, reverse=False)

    def reset(self) -> None:
        """Discard all retained items."""
        self.heap = []
