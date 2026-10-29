"""Tests for the model serving layer in protuna.inference.serve."""

from collections.abc import Sequence
import os

import torch

from protuna.inference import handlers
from protuna.inference.serve import pool as serving_pool
from protuna.typing import SequenceLike

# Actors are tiny, so give them a fraction of a CPU and let many share the
# cluster the session fixture starts.
_TINY_ACTOR: serving_pool.RayActorOptions = {"num_cpus": 0.1}


class _BatchSizeHandler(handlers.ModelFreeLocalHandler[int, tuple[int, int]]):
    """Echoes each request paired with the size of the batch it arrived in."""

    def __call__(self, requests: SequenceLike[int]) -> list[tuple[int, int]]:
        return [(request, len(requests)) for request in requests]


class _RecordingHandler(handlers.ModelFreeLocalHandler[int, int]):
    """Doubles each request and records every batch it was called with."""

    def __init__(self, batch_size: int | None = None):
        super().__init__(batch_size=batch_size)
        self.batches: list[list[int]] = []

    def __call__(self, requests: SequenceLike[int]) -> list[int]:
        self.batches.append(list(requests))
        return [request * 2 for request in requests]


class _LoadableHandler(handlers.LocalHandler[int, str]):
    """Reports the device it was loaded onto, so loading is observable."""

    def __init__(self):
        super().__init__()
        self.device = "unloaded"

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        self.device = "auto" if device is None else str(device)

    def release(self) -> None:
        self.device = "released"

    def __call__(self, requests: SequenceLike[int]) -> list[str]:
        return [self.device] * len(requests)


class _PidHandler(handlers.ModelFreeLocalHandler[int, int]):
    """Reports the process it runs in, so replica fan-out is observable."""

    def __call__(self, requests: SequenceLike[int]) -> list[int]:
        return [os.getpid()] * len(requests)


class _FailingHandler(handlers.ModelFreeLocalHandler[int, int]):
    """Always raises, to exercise the worker's error propagation."""

    def __call__(self, requests: SequenceLike[int]) -> list[int]:
        raise RuntimeError("handler exploded")


class _ShortHandler(handlers.ModelFreeLocalHandler[int, int]):
    """Drops the last response, violating one-response-per-request."""

    def __call__(self, requests: SequenceLike[int]) -> list[int]:
        return list(requests)[:-1]


class _TensorBatchHandler(handlers.ModelFreeLocalHandler[int, torch.Tensor]):
    def __call__(self, requests: SequenceLike[int]) -> torch.Tensor:
        return torch.tensor(list(requests)) * 2
