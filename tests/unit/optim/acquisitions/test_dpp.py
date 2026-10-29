from collections.abc import Callable
import math
import pathlib

import pytest
import torch

from protuna.optim import candidates
from protuna.optim.acquisitions import dpp

from .conftest import _Embedding
from .conftest import _Kernel
from .conftest import _Scores


@pytest.mark.parametrize("chunk_size", [1, 2, 10])
def test_embedding_rbf_matches_pairwise_distances(
    chunk_size: int,
    batch_factory: Callable[..., candidates.CandidateBatch],
    embedding_factory: type[_Embedding],
) -> None:
    """Compute the same scalar-lengthscale kernel at every chunk size."""
    kernel = dpp.EmbeddingRBFKernel(
        embedding_factory(), torch.tensor([1.0]), chunk_size=chunk_size
    )

    actual = kernel(batch_factory(3, tokens=[[0, 0], [1, 0], [1, 2]]))

    torch.testing.assert_close(
        actual,
        torch.tensor(
            [
                [1.0, 0.60653066, 0.08208500],
                [0.60653066, 1.0, 0.13533528],
                [0.08208500, 0.13533528, 1.0],
            ]
        ),
    )


def test_embedding_rbf_supports_per_feature_lengthscales(
    batch_factory: Callable[..., candidates.CandidateBatch],
    embedding_factory: type[_Embedding],
) -> None:
    """Scale individual feature differences before combining distances."""
    kernel = dpp.EmbeddingRBFKernel(
        embedding_factory(), torch.tensor([1.0, 2.0]), chunk_size=2
    )

    actual = kernel(batch_factory(3, tokens=[[0, 0], [1, 0], [1, 2]]))

    torch.testing.assert_close(
        actual,
        torch.tensor(
            [
                [1.0, 0.60653066, 0.36787944],
                [0.60653066, 1.0, 0.60653066],
                [0.36787944, 0.60653066, 1.0],
            ]
        ),
    )


def test_ensemble_covariance_centers_each_candidate(
    batch_factory: Callable[..., candidates.CandidateBatch],
) -> None:
    """Compute unbiased covariance for a non-square candidate-by-member array."""
    batch = batch_factory(2, samples=[[1.0, 2.0, 3.0], [6.0, 4.0, 2.0]])

    torch.testing.assert_close(
        dpp.EnsembleCovarianceKernel()(batch),
        torch.tensor([[1.0, -2.0], [-2.0, 4.0]]),
    )


def test_ensemble_covariance_is_invariant_to_candidate_offsets(
    batch_factory: Callable[..., candidates.CandidateBatch],
) -> None:
    """Center rows even when candidate and ensemble dimensions happen to match."""
    batch = batch_factory(
        3, samples=[[10.0, 11.0, 12.0], [1.0, 3.0, 5.0], [7.0, 7.0, 7.0]]
    )

    torch.testing.assert_close(
        dpp.EnsembleCovarianceKernel()(batch),
        torch.tensor([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_composed_kernel_is_a_weighted_sum(
    batch_factory: Callable[..., candidates.CandidateBatch],
    kernel_factory: type[_Kernel],
) -> None:
    """Combine kernel values using their corresponding weights."""
    kernel = dpp.ComposeKernel(
        [
            kernel_factory([[1.0, 0.5], [0.5, 1.0]]),
            kernel_factory([[2.0, 0.0], [0.0, 3.0]]),
        ],
        weights=[2, 3],
    )

    torch.testing.assert_close(
        kernel(batch_factory(2)), torch.tensor([[8.0, 1.0], [1.0, 11.0]])
    )


@pytest.mark.parametrize(
    "needs_std, needs_samples",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_composed_kernel_propagates_prediction_requirements(
    needs_std: bool, needs_samples: bool, kernel_factory: type[_Kernel]
) -> None:
    """Request every prediction field required by a component kernel."""
    kernel = dpp.ComposeKernel(
        [
            kernel_factory([[1.0]], requires_std=needs_std),
            kernel_factory([[1.0]], requires_samples=needs_samples),
        ],
        weights=[1, 1],
    )

    assert (kernel.requires_std, kernel.requires_samples) == (needs_std, needs_samples)


def test_composed_kernel_loads_and_releases_nested_models(
    embedding_factory: type[_Embedding],
) -> None:
    """Forward model lifecycle calls through kernel composition."""
    model = embedding_factory()
    kernel = dpp.ComposeKernel(
        [dpp.EmbeddingRBFKernel(model, torch.tensor([1.0]))], weights=[1]
    )

    kernel.load(torch.device("cpu"))
    kernel.release()

    assert model.devices == [torch.device("cpu")]
    assert model.releases == 1


def test_greedy_dpp_trades_quality_for_diversity() -> None:
    """Prefer a lower-quality independent item to a near duplicate."""
    kernel = torch.tensor([[9.0, 8.4, 0.0], [8.4, 8.0, 0.0], [0.0, 0.0, 4.0]])

    assert dpp._greedy_kddp_map(kernel, 2) == [0, 2]


@pytest.mark.parametrize("k, expected", [(0, []), (1, [1]), (3, [1, 2, 0])])
def test_greedy_dpp_cardinality_boundaries(k: int, expected: list[int]) -> None:
    """Respect zero, unit and full-pool budgets."""
    assert (
        dpp._greedy_kddp_map(torch.diag(torch.tensor([1.0, 4.0, 2.0])), k) == expected
    )


def test_greedy_dpp_preserves_float64_precision() -> None:
    """Support double-precision kernels through repeated Cholesky updates."""
    assert dpp._greedy_kddp_map(
        torch.diag(torch.tensor([1.0, 4.0, 2.0], dtype=torch.float64)), 3
    ) == [1, 2, 0]


def test_sampled_dpp_returns_unique_indices() -> None:
    """Keep fixed cardinality throughout MCMC swaps."""
    selected = dpp._kdpp_sample(
        torch.eye(6), torch.zeros(6), 3, 30, torch.Generator().manual_seed(4)
    )

    assert len(set(selected)) == 3
    assert set(selected) <= {0, 1, 2, 3, 4, 5}


@pytest.mark.parametrize("n_iter", [0, 20])
def test_sampled_dpp_seed_controls_initialization_and_swaps(n_iter: int) -> None:
    """Reproduce draws without changing or depending on the global RNG."""
    with torch.random.fork_rng():
        torch.manual_seed(42)
        before = torch.random.get_rng_state()
        first = dpp._kdpp_sample(
            torch.eye(8), torch.zeros(8), 3, n_iter, torch.Generator().manual_seed(9)
        )
        after = torch.random.get_rng_state()
        second = dpp._kdpp_sample(
            torch.eye(8), torch.zeros(8), 3, n_iter, torch.Generator().manual_seed(9)
        )

    assert first == second
    torch.testing.assert_close(before, after)


@pytest.mark.parametrize("k, expected", [(0, []), (3, [0, 1, 2])])
def test_sampled_dpp_cardinality_boundaries(k: int, expected: list[int]) -> None:
    """Avoid invalid swaps when either side of the selection is empty."""
    assert (
        dpp._kdpp_sample(
            torch.eye(3), torch.zeros(3), k, 5, torch.Generator().manual_seed(2)
        )
        == expected
    )


def test_dpp_map_maps_indices_back_to_shortlist_ids(
    tmp_path: pathlib.Path,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
    scores_factory: type[_Scores],
    kernel_factory: type[_Kernel],
) -> None:
    """Use log quality and diversity while retaining shuffled candidate ids."""
    strategy = dpp.DPP(
        scores_factory([math.log(9.0), math.log(8.0), math.log(4.0)], [4, 1, 3]),
        kernel_factory([[1.0, 0.99, 0.0], [0.99, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        2,
        mode="map",
    )

    assert strategy.select(stream_factory(tmp_path, batch_factory())) == [4, 3]


@pytest.mark.parametrize("beta, expected", [(0.0, [4]), (2.0, [3])])
def test_dpp_beta_controls_quality_weight(
    tmp_path: pathlib.Path,
    beta: float,
    expected: list[int],
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
    scores_factory: type[_Scores],
    kernel_factory: type[_Kernel],
) -> None:
    """Ignore utility at beta zero and favor it at positive beta."""
    strategy = dpp.DPP(
        scores_factory([0.0, 1.0], [4, 3]),
        kernel_factory([[1.0, 0.0], [0.0, 1.0]]),
        1,
        mode="map",
        beta=beta,
    )

    assert strategy.select(stream_factory(tmp_path, batch_factory())) == expected


@pytest.mark.parametrize("n_iter", [None, 0, 10])
def test_dpp_sample_returns_candidate_ids(
    tmp_path: pathlib.Path,
    n_iter: int | None,
    batch_factory: Callable[..., candidates.CandidateBatch],
    stream_factory: Callable[..., candidates.CandidateStream],
    scores_factory: type[_Scores],
    kernel_factory: type[_Kernel],
) -> None:
    """Sample from shortlisted ids with either default or explicit MCMC length."""
    strategy = dpp.DPP(
        scores_factory([1.0, 1.0, 1.0], [4, 1, 3]),
        kernel_factory([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        2,
        mode="sample",
        n_iter=n_iter,
        seed=2,
    )

    selected = strategy.select(stream_factory(tmp_path, batch_factory()))

    assert len(set(selected)) == 2
    assert set(selected) <= {1, 3, 4}


def test_dpp_combines_quality_and_diversity_requirements(
    scores_factory: type[_Scores], kernel_factory: type[_Kernel]
) -> None:
    """Request the union of fields required by quality and diversity."""
    strategy = dpp.DPP(
        scores_factory([1.0], [1], requires_std=True),
        kernel_factory([[1.0]], requires_samples=True),
        1,
        mode="map",
    )

    assert (strategy.requires_std, strategy.requires_samples) == (True, True)


def test_dpp_loads_and_releases_diversity_models(
    embedding_factory: type[_Embedding], scores_factory: type[_Scores]
) -> None:
    """Forward model lifecycle calls into an embedding diversity kernel."""
    model = embedding_factory()
    strategy = dpp.DPP(
        scores_factory([1.0], [1]),
        dpp.EmbeddingRBFKernel(model, torch.tensor([1.0])),
        1,
        mode="map",
    )

    strategy.load(torch.device("cpu"))
    strategy.release()

    assert model.devices == [torch.device("cpu")]
    assert model.releases == 1
