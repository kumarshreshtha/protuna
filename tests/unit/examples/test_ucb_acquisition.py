from collections.abc import Callable
import pathlib

import pytest

from examples import ucb_acquisition
from protuna.optim import candidates


@pytest.mark.parametrize("scan_size", [1, 2, 10])
def test_ucb_ranks_means_and_uncertainty_across_batches(
    tmp_path: pathlib.Path,
    scan_size: int,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Include uncertainty when ranking the entire stream in descending order."""
    stream = stream_factory(
        tmp_path,
        batch_factory(mean=[3.0, 1.0, 4.0, 2.0], std=[0.0, 3.0, 1.0, 0.0]),
        scan_size,
    )

    assert ucb_acquisition.UpperConfidenceBound(2).select(stream) == [2, 3]


def test_ucb_zero_beta_ranks_only_means(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Disable the uncertainty bonus when beta is zero."""
    stream = stream_factory(
        tmp_path,
        batch_factory(mean=[3.0, 1.0, 4.0, 2.0], std=[0.0, 3.0, 1.0, 0.0]),
    )

    assert ucb_acquisition.UpperConfidenceBound(2, beta=0).select(stream) == [3, 1]


def test_ucb_filters_joint_feasibility_strictly(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Reject joint feasibility at or below the threshold before ranking."""
    stream = stream_factory(
        tmp_path,
        batch_factory(
            mean=[10.0, 9.0, 8.0, 7.0],
            std=[1.0, 1.0, 1.0, 1.0],
            feasibility={"a": [0.5, 0.5, 1.0, 0.0], "b": [0.5, 1.0, 1.0, 1.0]},
        ),
    )

    assert ucb_acquisition.UpperConfidenceBound(4, feasibility_threshold=0.5).select(
        stream
    ) == [3]


@pytest.mark.parametrize(
    ("n", "budget", "expected"),
    [(0, 2, []), (1, 3, [1]), (4, 0, []), (4, 2, [4, 3])],
)
def test_ucb_handles_budget_boundaries_and_breaks_ties_by_larger_id(
    tmp_path: pathlib.Path,
    n: int,
    budget: int,
    expected: list[int],
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Respect the available pool and budget, using larger IDs to break ties."""
    stream = stream_factory(tmp_path, batch_factory(n, mean=[1.0] * n, std=[0.0] * n))

    assert ucb_acquisition.UpperConfidenceBound(budget).select(stream) == expected


@pytest.mark.parametrize(("mean", "std"), [(None, [1.0]), ([1.0], None), (None, None)])
def test_ucb_requires_both_means_and_standard_deviations(
    tmp_path: pathlib.Path,
    mean: list[float] | None,
    std: list[float] | None,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Explain missing prediction fields instead of silently misranking."""
    stream = stream_factory(tmp_path, batch_factory(1, mean=mean, std=std))

    with pytest.raises(ValueError, match="requires objective means and standard"):
        ucb_acquisition.UpperConfidenceBound(1).select(stream)


@pytest.mark.parametrize(
    ("budget", "beta", "threshold", "message"),
    [
        (-1, 4.0, 0.0, "budget must be nonnegative"),
        (2, -1.0, 0.0, "beta must be finite and nonnegative"),
        (2, float("inf"), 0.0, "beta must be finite and nonnegative"),
        (2, float("nan"), 0.0, "beta must be finite and nonnegative"),
        (2, 4.0, -0.1, "feasibility_threshold must be between"),
        (2, 4.0, 1.1, "feasibility_threshold must be between"),
    ],
)
def test_ucb_rejects_invalid_configuration(
    budget: int, beta: float, threshold: float, message: str
) -> None:
    """Reject invalid budgets, confidence weights and feasibility thresholds."""
    with pytest.raises(ValueError, match=message):
        ucb_acquisition.UpperConfidenceBound(
            budget, beta, feasibility_threshold=threshold
        )


def test_ucb_requests_standard_deviations_without_posterior_samples() -> None:
    """Advertise the prediction fields the pipeline must supply."""
    acquisition = ucb_acquisition.UpperConfidenceBound(1)

    assert acquisition.requires_std is True
    assert acquisition.requires_samples is False
