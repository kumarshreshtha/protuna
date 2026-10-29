from collections.abc import Callable
import pathlib

import pytest
import torch

from protuna.optim import candidates
from protuna.optim.acquisitions import scores


def test_ucb_topk_returns_candidate_ids_and_values(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Return actual ids after ranking mean plus sqrt(beta) times deviation."""
    stream = stream_factory(
        tmp_path, batch_factory(mean=[1.0, 4.0, 2.0, -1.0], std=[0.0, 1.0, 3.0, 1.0])
    )

    values, ids = scores.UCBScore(2, beta=4.0, temperature=0)(stream)

    assert ids.tolist() == [3, 2]
    torch.testing.assert_close(values, torch.tensor([8.0, 6.0]))


def test_ucb_topk_maps_ids_after_feasibility_filtering(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Keep id alignment when the leading candidates are filtered out."""
    stream = stream_factory(
        tmp_path,
        batch_factory(
            mean=[10.0, 9.0, 3.0, 2.0],
            std=[0.0] * 4,
            feasibility={"a": [0.0, 0.5, 1.0, 1.0]},
        ),
    )

    _, ids = scores.UCBScore(2, beta=0.0, temperature=0, feasibility_threshold=0.5)(
        stream
    )

    assert ids.tolist() == [3, 4]


def test_ucb_sampling_keeps_scores_aligned_with_ids(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Preserve each candidate's score in a randomly ordered shortlist."""
    stream = stream_factory(
        tmp_path, batch_factory(mean=[1.0, 2.0, 3.0, 4.0], std=[0.5] * 4)
    )

    values, ids = scores.UCBScore(4, beta=4.0, sample_with_replacement=False, seed=3)(
        stream
    )

    assert sorted(zip(ids.tolist(), values.tolist(), strict=True)) == [
        (1, 2.0),
        (2, 3.0),
        (3, 4.0),
        (4, 5.0),
    ]


def test_ucb_can_sample_with_replacement(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Allow a shortlist larger than a one-candidate pool when replacing."""
    stream = stream_factory(tmp_path, batch_factory(1, mean=[2.0], std=[1.0]))

    values, ids = scores.UCBScore(3, beta=1.0, seed=3)(stream)

    assert ids.tolist() == [1, 1, 1]
    assert values.tolist() == [3.0, 3.0, 3.0]


def test_ucb_seed_reproduces_sampling(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Repeat the same shortlist with the same explicit seed."""
    stream = stream_factory(tmp_path, batch_factory(mean=[1.0] * 4, std=[1.0] * 4))

    _, first = scores.UCBScore(20, beta=1.0, seed=19)(stream)
    _, second = scores.UCBScore(20, beta=1.0, seed=19)(stream)

    torch.testing.assert_close(first, second)


@pytest.mark.parametrize(
    "mean, std, best_f, margin, probability, expected",
    [
        (1.0, 1.0, 1.0, 0.0, 1.0, 0.3989422804),
        (2.0, 1.0, 1.0, 0.0, 1.0, 1.0833154706),
        (0.0, 1.0, 1.0, 0.0, 1.0, 0.0833154706),
        (3.0, 0.0, 1.0, 0.5, 1.0, 1.5),
        (3.0, 0.0, 1.0, 0.5, 0.2, 0.3),
    ],
)
def test_ei_matches_analytic_values(
    tmp_path: pathlib.Path,
    mean: float,
    std: float,
    best_f: float,
    margin: float,
    probability: float,
    expected: float,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Compute Gaussian improvement, including margin and feasibility weighting."""
    stream = stream_factory(
        tmp_path,
        batch_factory(1, mean=[mean], std=[std], feasibility={"a": [probability]}),
    )

    values, ids = scores.EIScore(best_f, 1, margin=margin, seed=3)(stream)

    assert ids.tolist() == [1]
    assert values.tolist() == pytest.approx([expected])


def test_ei_excludes_zero_improvement_and_infeasible_candidates(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Sample only candidates with positive feasible expected improvement."""
    stream = stream_factory(
        tmp_path,
        batch_factory(
            mean=[-1.0, 0.0, 2.0, 20.0],
            std=[0.0] * 4,
            feasibility={"a": [1.0, 1.0, 1.0, 0.0]},
        ),
    )

    values, ids = scores.EIScore(0.0, 3, seed=7)(stream)

    assert ids.tolist() == [3, 3, 3]
    assert values.tolist() == [2.0, 2.0, 2.0]


def test_ei_without_replacement_preserves_score_id_alignment(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Retain all positive candidates once when sampling without replacement."""
    stream = stream_factory(
        tmp_path, batch_factory(mean=[1.0, 2.0, 3.0, 4.0], std=[0.0] * 4)
    )

    values, ids = scores.EIScore(0.0, 4, sample_with_replacement=False, seed=3)(stream)

    assert sorted(zip(ids.tolist(), values.tolist(), strict=True)) == [
        (1, 1.0),
        (2, 2.0),
        (3, 3.0),
        (4, 4.0),
    ]


def test_ei_seed_reproduces_sampling(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Use the provided generator for expected-improvement sampling."""
    stream = stream_factory(tmp_path, batch_factory(mean=[1.0] * 4, std=[1.0] * 4))

    _, first = scores.EIScore(0.0, 20, seed=19)(stream)
    _, second = scores.EIScore(0.0, 20, seed=19)(stream)

    torch.testing.assert_close(first, second)
