from collections.abc import Callable
import pathlib

import pytest
import torch

from protuna.optim import candidates
from protuna.optim.acquisitions import base


def test_identity_preserves_all_ids_across_batches(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Select every candidate in stream order without needing predictions."""
    stream = stream_factory(tmp_path, batch_factory())

    assert base.IdentityAcquisition().select(stream) == [1, 2, 3, 4]


def test_identity_accepts_an_empty_stream(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
) -> None:
    """Return no ids when no candidates were proposed."""
    assert (
        base.IdentityAcquisition().select(stream_factory(tmp_path, batch_factory(0)))
        == []
    )


@pytest.mark.parametrize(
    "constraints, expected",
    [
        (None, [1.0, 1.0, 1.0, 1.0]),
        ({}, [1.0, 1.0, 1.0, 1.0]),
        ({"a": [0.5, 1.0, 0.0, 0.8], "b": [0.4, 0.1, 1.0, 0.5]}, [0.2, 0.1, 0.0, 0.4]),
    ],
)
def test_feasibility_is_the_joint_probability(
    constraints: dict[str, list[float]] | None,
    expected: list[float],
    batch_factory: Callable[..., candidates.CandidateBatch],
) -> None:
    """Multiply constraint probabilities, treating no constraints as certainty."""
    torch.testing.assert_close(
        base.feasibility(batch_factory(feasibility=constraints)),
        torch.tensor(expected),
    )
