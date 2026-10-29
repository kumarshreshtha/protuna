"""Ensemble predictions and uncertainty from member disagreement."""

from collections.abc import Sequence
import dataclasses
import itertools

import torch

from protuna import inference
from protuna.optim.predictors import base
from protuna.typing import SequenceLike
from protuna.utils import devices as device_utils


class EnsemblePredictor(
    inference.InferenceModel[base.PredictionRequest, base.Prediction]
):
    """Average member predictions and optionally report their disagreement.

    Members are evaluated in parallel. Each candidate's mean is the average
    of the member means. Requested samples are the member means in ensemble
    order; requested standard deviations use the sample standard deviation
    across members, or zero for a single member. Uncertainty reported by
    individual members is ignored.
    """

    def __init__(
        self, predictors: Sequence[base.PredictorLike], *, batch_size: int | None = None
    ):
        """Initialize an ensemble from its member predictors.

        Args:
            predictors: The ensemble members, in sample order.
            batch_size: Candidates per forward pass. ``None`` processes all
                candidates passed to the predictor in one forward pass.

        Raises:
            ValueError: If the ensemble has no members.
        """
        if not predictors:
            raise ValueError("An ensemble requires at least one member")
        super().__init__(batch_size=batch_size)
        self.predictors = predictors

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Load ensemble members on devices in round-robin order.

        Args:
            device: A device to share amongst all members or multiple devices cycled
                across members if total devices is less than members.
                ``None`` selects visible CUDA devices, falling back to CPU.
                Served members manage their own devices and ignore this argument.
        """
        device = device_utils.resolve_devices() if device is None else device
        devices = [device] if isinstance(device, torch.device) else device
        for predictor, device in zip(
            self.predictors, itertools.cycle(devices), strict=False
        ):
            # if predictor is served, the device argument is ignored
            predictor.load(device)

    def release(self) -> None:
        """Release resources held by every ensemble member."""
        for predictor in self.predictors:
            predictor.release()

    def forward(
        self, requests: SequenceLike[base.PredictionRequest]
    ) -> list[base.Prediction]:
        """Score candidates with each member and combine their predictions.

        Args:
            requests: Candidate sequences and requested prediction outputs.

        Returns:
            Ensemble predictions, with standard deviations and
            member samples included when requested. Each prediction uses
            the device of the first member's mean.

        """
        model_requests = [
            dataclasses.replace(request, return_std=False, return_samples=False)
            for request in requests
        ]
        preds = inference.execute_parallel(self.predictors, model_requests)
        ensemble_preds = []
        for i, request in enumerate(requests):
            device = preds[0][i].mean.device
            samples = torch.stack(
                [model_pred[i].mean.to(device) for model_pred in preds]
            )
            mean = samples.mean()
            std = (
                samples.std(correction=1 if len(samples) > 1 else 0)
                if request.return_std
                else None
            )
            ensemble_preds.append(
                base.Prediction(
                    mean=mean,
                    std=std,
                    samples=samples if request.return_samples else None,
                )
            )
        return ensemble_preds
