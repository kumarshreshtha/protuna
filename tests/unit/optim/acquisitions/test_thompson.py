from collections.abc import Callable
import pathlib

import pytest

from protuna.optim import candidates
from protuna.optim.acquisitions import thompson


@pytest.mark.parametrize("scan_size", [1, 2, 10])
def test_thompson_includes_the_first_batch(
    tmp_path: pathlib.Path,
    scan_size: int,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Keep the best sample even when it appears in the first batch."""
    stream = stream_factory(
        tmp_path, batch_factory(samples=[[10.0], [3.0], [2.0], [1.0]]), scan_size
    )

    assert thompson.ThompsonSampling(1, seed=2).select(stream) == [1]


def test_thompson_uses_each_drawn_ensemble_member(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Select the two distinct member optima when both members are drawn."""
    stream = stream_factory(
        tmp_path,
        batch_factory(samples=[[10.0, 0.0], [8.0, 1.0], [1.0, 8.0], [0.0, 10.0]]),
    )

    assert sorted(thompson.ThompsonSampling(2, seed=5).select(stream)) == [1, 4]


def test_thompson_does_not_select_a_shared_optimum_twice(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Fill the budget with distinct ids when ensemble rankings agree."""
    stream = stream_factory(
        tmp_path,
        batch_factory(samples=[[9.0, 9.0], [8.0, 8.0], [1.0, 1.0], [0.0, 0.0]]),
    )

    assert sorted(thompson.ThompsonSampling(2, seed=5).select(stream)) == [1, 2]


def test_thompson_filters_feasibility(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Exclude the best objective sample when its feasibility is too low."""
    stream = stream_factory(
        tmp_path,
        batch_factory(
            samples=[[99.0], [8.0], [7.0], [6.0]],
            feasibility={"a": [0.0, 0.5, 1.0, 0.2]},
        ),
    )

    assert thompson.ThompsonSampling(4, feasibility_threshold=0.5).select(stream) == [3]


def test_thompson_returns_python_ids(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Return plain integers from posterior sampling."""
    selected = thompson.ThompsonSampling(1, seed=1).select(
        stream_factory(tmp_path, batch_factory(samples=[[1.0], [2.0], [3.0], [4.0]]))
    )

    assert type(selected[0]) is int


@pytest.mark.parametrize("n, budget, expected", [(0, 2, []), (1, 3, [1]), (4, 0, [])])
def test_thompson_handles_pool_and_budget_boundaries(
    tmp_path: pathlib.Path,
    n: int,
    budget: int,
    expected: list[int],
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Select at most the budget and the number of available candidates."""
    stream = stream_factory(tmp_path, batch_factory(n, samples=[[1.0]] * n))

    assert thompson.ThompsonSampling(budget, seed=1).select(stream) == expected


def test_thompson_requires_posterior_samples(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Reject means alone with an explanation of the missing samples."""
    with pytest.raises(TypeError, match="requires access to sampled values"):
        thompson.ThompsonSampling(1).select(
            stream_factory(tmp_path, batch_factory(mean=[1.0] * 4))
        )


def test_thompson_seed_reproduces_selections(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Use the supplied seed consistently for ensemble draws."""
    stream = stream_factory(
        tmp_path,
        batch_factory(samples=[[9.0, 0.0], [8.0, 1.0], [1.0, 8.0], [0.0, 9.0]]),
    )

    assert thompson.ThompsonSampling(3, seed=17).select(
        stream
    ) == thompson.ThompsonSampling(3, seed=17).select(stream)
