"""Shared candidate and model factories for acquisition tests."""

from collections.abc import Callable, Iterator, Sequence
import itertools
import pathlib

from jaxtyping import Float
import pytest
import torch

from protuna import inference
from protuna.optim import candidates
from protuna.optim import storage
from protuna.typing import SequenceLike, TokenIDs


def _batch(
    n: int = 4,
    *,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    samples: list[list[float]] | None = None,
    feasibility: dict[str, list[float]] | None = None,
    tokens: list[list[int]] | None = None,
) -> candidates.CandidateBatch:
    return candidates.CandidateBatch(
        seq_id=torch.arange(1, n + 1),
        token_ids=torch.tensor(tokens)
        if tokens is not None
        else torch.arange(n)[:, None],
        parent_id=torch.zeros(n, dtype=torch.long),
        ancestor_id=torch.zeros(n, dtype=torch.long),
        cycle_id=torch.ones(n, dtype=torch.long),
        objective_mean=None if mean is None else torch.tensor(mean, dtype=torch.float),
        objective_std=None if std is None else torch.tensor(std, dtype=torch.float),
        objective_samples=None
        if samples is None
        else torch.tensor(samples, dtype=torch.float),
        feasibility=None
        if feasibility is None
        else {
            name: torch.tensor(values, dtype=torch.float)
            for name, values in feasibility.items()
        },
    )


class _ReplayStream(candidates.CandidateStream):
    # Freeze the batches so randomized storage join order cannot affect seed tests.
    def __init__(
        self,
        source: candidates.CandidateStream,
        batches: list[candidates.CandidateBatch],
    ):
        self.source = source
        self.batches = batches

    def __iter__(self) -> Iterator[candidates.CandidateBatch]:
        return iter(self.batches)

    def candidates_from_ids(self, seq_ids: Sequence[int]) -> candidates.CandidateBatch:
        return self.source.candidates_from_ids(seq_ids)


def _stream(
    tmp_path: pathlib.Path, batch: candidates.CandidateBatch, scan_size: int = 2
) -> candidates.CandidateStream:
    store = storage.CampaignStore(
        protein_length=batch.token_ids.shape[1],
        location=tmp_path,
        scan_size=scan_size,
        soft_constraint_names=tuple(batch.feasibility or {}),
    )
    store.store_samples(batch.token_ids, parent_id=0, ancestor_id=0, cycle_id=1)
    for i, seq_id in enumerate(batch.seq_id.tolist()):
        store.objective_store.store(
            {
                "seq_id": [seq_id],
                "objective_mean": [
                    None
                    if batch.objective_mean is None
                    else float(batch.objective_mean[i])
                ],
                "objective_std": [
                    None
                    if batch.objective_std is None
                    else float(batch.objective_std[i])
                ],
                "objective_samples": [
                    None
                    if batch.objective_samples is None
                    else batch.objective_samples[i].detach().cpu().numpy()
                ],
            }
        )
        for name, values in (batch.feasibility or {}).items():
            store.store_constraint(name, seq_id, values[i])
    source = store.get_candidate_stream()
    batches = [
        source.candidates_from_ids(ids)
        for ids in itertools.batched(batch.seq_id.tolist(), scan_size)
    ]
    return _ReplayStream(source, batches)


class _Embedding(inference.InferenceModel[TokenIDs, Float[torch.Tensor, " d"]]):
    def __init__(self):
        super().__init__()
        self.devices: list[torch.device | Sequence[torch.device] | None] = []
        self.releases = 0

    def forward(
        self, requests: SequenceLike[TokenIDs]
    ) -> list[Float[torch.Tensor, " d"]]:
        return [request.float() for request in requests]

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        self.devices.append(device)

    def release(self) -> None:
        self.releases += 1


class _Scores:
    def __init__(
        self,
        values: list[float],
        ids: list[int],
        *,
        requires_std: bool = False,
        requires_samples: bool = False,
    ):
        self.values = torch.tensor(values, dtype=torch.float)
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.requires_std = requires_std
        self.requires_samples = requires_samples

    def __call__(
        self, stream: candidates.CandidateStream
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.values, self.ids


class _Kernel:
    def __init__(
        self,
        values: list[list[float]],
        *,
        requires_std: bool = False,
        requires_samples: bool = False,
    ):
        self.values = torch.tensor(values)
        self.requires_std = requires_std
        self.requires_samples = requires_samples

    def __call__(self, batch: candidates.CandidateBatch) -> torch.Tensor:
        return self.values


@pytest.fixture
def batch_factory() -> Callable[..., candidates.CandidateBatch]:
    """Provide the shared batch factory."""
    return _batch


@pytest.fixture
def stream_factory() -> Callable[..., candidates.CandidateStream]:
    """Provide the shared stream factory."""
    return _stream


@pytest.fixture
def embedding_factory() -> type[_Embedding]:
    """Provide the shared embedding factory."""
    return _Embedding


@pytest.fixture
def scores_factory() -> type[_Scores]:
    """Provide the shared scores factory."""
    return _Scores


@pytest.fixture
def kernel_factory() -> type[_Kernel]:
    """Provide the shared kernel factory."""
    return _Kernel
