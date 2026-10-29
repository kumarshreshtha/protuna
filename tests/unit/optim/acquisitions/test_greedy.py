from collections.abc import Callable
import pathlib

import pytest

from protuna.optim import candidates
from protuna.optim.acquisitions import greedy


@pytest.mark.parametrize("scan_size", [1, 2, 10])
def test_greedy_selects_global_top_scores(
    tmp_path: pathlib.Path,
    scan_size: int,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Rank across batch boundaries, retaining the largest objectives."""
    stream = stream_factory(
        tmp_path, batch_factory(mean=[3.0, -1.0, 8.0, 2.0]), scan_size
    )

    assert greedy.Greedy(2).select(stream) == [1, 3]


def test_greedy_returns_python_ids(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Return serializable integer ids rather than scalar tensors."""
    selected = greedy.Greedy(1).select(
        stream_factory(tmp_path, batch_factory(1, mean=[2.0]))
    )

    assert type(selected[0]) is int


def test_greedy_filters_joint_feasibility_strictly(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Reject probabilities equal to the threshold as well as those below it."""
    stream = stream_factory(
        tmp_path,
        batch_factory(
            mean=[10.0, 9.0, 8.0, 7.0],
            feasibility={
                "a": [0.5, 0.5, 1.0, 0.0],
                "b": [0.5, 1.0, 1.0, 1.0],
            },
        ),
    )

    assert greedy.Greedy(4, feasibility_threshold=0.5).select(stream) == [3]


@pytest.mark.parametrize("n, budget, expected", [(0, 2, []), (1, 3, [1]), (4, 0, [])])
def test_greedy_handles_pool_and_budget_boundaries(
    tmp_path: pathlib.Path,
    n: int,
    budget: int,
    expected: list[int],
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Select at most the budget and the number of available candidates."""
    stream = stream_factory(tmp_path, batch_factory(n, mean=[1.0] * n))

    assert greedy.Greedy(budget).select(stream) == expected


def test_greedy_requires_objective_means(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Explain why candidates without means cannot be ranked."""
    with pytest.raises(ValueError, match="requires objective mean"):
        greedy.Greedy(1).select(stream_factory(tmp_path, batch_factory()))
