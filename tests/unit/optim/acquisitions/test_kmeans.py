from collections.abc import Callable
import pathlib
from typing import Literal

import pytest
import torch

from protuna.optim import candidates
from protuna.optim.acquisitions import kmeans

from .conftest import _Embedding
from .conftest import _Scores


@pytest.mark.parametrize("mode, expected", [("centroid", [1, 4]), ("max", [2, 3])])
def test_kmbbo_selects_one_representative_per_cluster(
    tmp_path: pathlib.Path,
    mode: Literal["centroid", "max"],
    expected: list[int],
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
    embedding_factory: type[_Embedding],
    scores_factory: type[_Scores],
) -> None:
    """Select centroid neighbors or maximum utility from two separated clusters."""
    stream = stream_factory(
        tmp_path, batch_factory(6, tokens=[[1], [0], [10], [11], [3], [13]])
    )
    strategy = kmeans.KMBBO(
        embedding_factory(),
        scores_factory([4.0, 9.0, 2.0, 8.0, 1.0, 3.0], [4, 2, 6, 3, 1, 5]),
        2,
        mode=mode,
        seed=7,
    )

    assert sorted(strategy.select(stream)) == expected


def test_kmbbo_propagates_score_requirements(
    embedding_factory: type[_Embedding], scores_factory: type[_Scores]
) -> None:
    """Request uncertainty and samples when the scoring function needs them."""
    strategy = kmeans.KMBBO(
        embedding_factory(),
        scores_factory([1.0], [1], requires_std=True, requires_samples=True),
        1,
    )

    assert (strategy.requires_std, strategy.requires_samples) == (True, True)


def test_kmbbo_loads_and_releases_the_embedding_model(
    embedding_factory: type[_Embedding], scores_factory: type[_Scores]
) -> None:
    """Make the embedding model available to pipeline lifecycle management."""
    model = embedding_factory()
    strategy = kmeans.KMBBO(model, scores_factory([1.0], [1]), 1)

    strategy.load(torch.device("cpu"))
    strategy.release()

    assert model.devices == [torch.device("cpu")]
    assert model.releases == 1
