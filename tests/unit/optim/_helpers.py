"""Deterministic models for predictor and constraint tests."""

from collections.abc import Sequence

import torch

from protuna import data
from protuna import inference
from protuna.optim import predictors
from protuna.typing import SequenceLike


class _LookupPredictor(
    inference.InferenceModel[predictors.PredictionRequest, predictors.Prediction]
):
    def __init__(
        self,
        scores: dict[str, float | torch.Tensor],
        std: float | torch.Tensor = 0.5,
        batch_size: int | None = None,
    ):
        super().__init__(batch_size=batch_size)
        self.scores = scores
        self.std = std
        self.batches: list[list[predictors.PredictionRequest]] = []
        self.loaded_devices: list[torch.device | Sequence[torch.device] | None] = []
        self.released = False

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Record the requested device without allocating model weights."""
        self.loaded_devices.append(device)

    def release(self) -> None:
        self.released = True

    def forward(
        self, requests: SequenceLike[predictors.PredictionRequest]
    ) -> list[predictors.Prediction]:
        """Return fixed scores in request order and record the batch."""
        self.batches.append(list(requests))
        return [
            predictors.Prediction(
                mean=torch.as_tensor(
                    self.scores[data.tokenizer.decode(request.token_ids)]
                ),
                std=torch.as_tensor(self.std) if request.return_std else None,
            )
            for request in requests
        ]
