"""Interfaces for predictor inputs and outputs."""

from collections.abc import Sequence
import dataclasses
from typing import Self

from jaxtyping import Float
import torch

from protuna import inference
from protuna.typing import TokenIDs
from protuna.utils import devices


@dataclasses.dataclass
class PredictionRequest:
    """A candidate sequence and the prediction outputs to return.

    Args:
        token_ids: The candidate sequence.
        return_std: Whether the prediction should carry a standard deviation.
        return_samples: Whether the prediction should return samples of the joint
            posterior; in case of point predictor ensembles this is simply the
            prediction from each ensemble member.
    """

    token_ids: TokenIDs
    return_std: bool = False
    return_samples: bool = False


@dataclasses.dataclass
class Prediction:
    """A candidate's predicted objective and optional uncertainty.

    Args:
        mean: The predicted value.
        std: Its standard deviation, when requested.
        samples: The individual samples of the objective, when requested.
    """

    mean: Float[torch.Tensor, ""]
    std: Float[torch.Tensor, ""] | None = None
    samples: Float[torch.Tensor, " n"] | None = None

    def to(self, device: torch.device) -> Self:
        """Return a prediction with all populated tensors on the requested device.

        Args:
            device: Destination device.
        """
        return dataclasses.replace(
            self,
            mean=self.mean.to(device),
            std=self.std.to(device) if self.std is not None else None,
            samples=self.samples.to(device) if self.samples is not None else None,
        )

    @classmethod
    def to_device_batch(
        cls, responses: Sequence[Self], device: torch.device
    ) -> list[Self]:
        """Return predictions with all populated tensors on the requested device.

        Args:
            responses: Predictions to transfer.
            device: Destination device.

        Returns:
            New predictions with all populated tensors on the destination device.
        """
        tensors = [
            tensor
            for response in responses
            for tensor in (response.mean, response.std, response.samples)
            if tensor is not None
        ]
        moved = iter(devices.to_device_batch(tensors, device))
        return [
            dataclasses.replace(
                response,
                mean=next(moved),
                std=next(moved) if response.std is not None else None,
                samples=next(moved) if response.samples is not None else None,
            )
            for response in responses
        ]


type PredictorLike = inference.Handler[PredictionRequest, Prediction]
