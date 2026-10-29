from __future__ import annotations

import pytest

from protuna.utils import data_structs


def test_items_empty_before_any_push() -> None:
    """A fresh heap holds nothing."""
    heap = data_structs.TopKHeap[int](3)

    assert heap.items() == []


def test_items_returns_ascending_order() -> None:
    """Items come back sorted smallest first, regardless of push order."""
    heap = data_structs.TopKHeap[int](3)
    heap.push(5)
    heap.push(1)
    heap.push(3)

    assert heap.items() == [1, 3, 5]


def test_keeps_only_k_largest_when_over_capacity() -> None:
    """Pushing past k drops the smallest values."""
    heap = data_structs.TopKHeap[int](3)
    heap.push(4)
    heap.push(1)
    heap.push(7)
    heap.push(2)
    heap.push(9)

    assert heap.items() == [4, 7, 9]


def test_push_below_capacity_keeps_every_value() -> None:
    """Fewer than k pushes are all retained."""
    heap = data_structs.TopKHeap[int](5)
    heap.push(2)
    heap.push(8)

    assert heap.items() == [2, 8]


def test_value_smaller_than_all_retained_is_discarded() -> None:
    """A new minimum is rejected once the heap is full."""
    heap = data_structs.TopKHeap[int](2)
    heap.push(10)
    heap.push(20)
    heap.push(1)

    assert heap.items() == [10, 20]


def test_duplicates_are_retained() -> None:
    """Equal values each occupy a slot."""
    heap = data_structs.TopKHeap[int](3)
    heap.push(6)
    heap.push(6)
    heap.push(6)
    heap.push(6)

    assert heap.items() == [6, 6, 6]


def test_k_zero_retains_nothing() -> None:
    """A zero-capacity heap discards everything pushed into it."""
    heap = data_structs.TopKHeap[int](0)
    heap.push(1)
    heap.push(2)

    assert heap.items() == []


def test_tuples_compare_lexicographically() -> None:
    """Tuple entries are ranked by their first element, then the second."""
    heap = data_structs.TopKHeap[tuple[float, int]](2)
    heap.push((0.5, 3))
    heap.push((0.5, 1))
    heap.push((0.1, 9))

    assert heap.items() == [(0.5, 1), (0.5, 3)]


def test_items_does_not_consume_the_heap() -> None:
    """Reading items leaves the contents in place."""
    heap = data_structs.TopKHeap[int](2)
    heap.push(1)
    heap.push(2)

    heap.items()

    assert heap.items() == [1, 2]


def test_reset_empties_the_heap() -> None:
    """After a reset the heap is empty again."""
    heap = data_structs.TopKHeap[int](2)
    heap.push(1)
    heap.push(2)

    heap.reset()

    assert heap.items() == []


def test_reset_restores_full_capacity() -> None:
    """A reset heap accepts k fresh values."""
    heap = data_structs.TopKHeap[int](2)
    heap.push(100)
    heap.push(200)

    heap.reset()
    heap.push(1)
    heap.push(2)

    assert heap.items() == [1, 2]


def test_push_of_uncomparable_value_raises() -> None:
    """Mixing incomparable types surfaces as a TypeError."""
    heap = data_structs.TopKHeap(2)
    heap.push(1)

    with pytest.raises(TypeError, match="not supported between instances"):
        heap.push("a")
