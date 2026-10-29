"""Interpolate predictor scores onto measured assay values."""

from collections.abc import Sequence
import functools
from typing import TypedDict

from jaxtyping import Float
import torch

from protuna import inference
from protuna.optim.predictors import base
from protuna.typing import BatchTokenIDs, SequenceLike


class AssayResult(TypedDict):
    """Sequences that have been measured in the lab, and their measurements."""

    token_ids: BatchTokenIDs
    measurement: Float[torch.Tensor, " b"]


def _interp(
    x: Float[torch.Tensor, " b"],
    xp: Float[torch.Tensor, " n"],
    fp: Float[torch.Tensor, " n"],
) -> Float[torch.Tensor, " b"]:
    if len(xp) == 1:
        return fp[0].expand_as(x)
    idx = torch.searchsorted(xp, x, right=True).clamp(1, len(xp) - 1)

    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]

    width = x1 - x0
    t = (x - x0) / torch.where(width == 0, torch.ones_like(width), width)
    interpolated = torch.where(width == 0, y1, y0 + t * (y1 - y0))

    return torch.where(x < xp[0], fp[0], torch.where(x >= xp[-1], fp[-1], interpolated))


class CalibratedPredictor(
    inference.InferenceModel[base.PredictionRequest, base.Prediction]
):
    """Map a predictor's scores onto the scale of measured assay values.

    The wrapped predictor is run once over the measured sequences, and every
    later score is linearly interpolated between those anchor points. This
    makes scores comparable to lab measurements; it does not change how
    candidates rank.

    The transform is useful for predictors that learn to rank but are used as
    constraints with threshold values in the original measurement units.
    """

    def __init__(
        self,
        predictor: base.PredictorLike,
        history: AssayResult,
        *,
        batch_size: int | None = None,
    ):
        """Initialize calibration with a predictor and measured assay results.

        Args:
            predictor: The predictor to calibrate.
            history: Sequences already measured in the lab and their
                measurements.
            batch_size: Number of candidates to score at a time.

        Raises:
            ValueError: If history is empty or sequence and measurement counts
                differ.
        """
        super().__init__(batch_size=batch_size)
        if len(history["token_ids"]) != len(history["measurement"]):
            raise ValueError("Sequences and measurements must have equal lengths")
        if len(history["token_ids"]) == 0:
            raise ValueError("Calibration requires at least one measured sequence")
        self.predictor = predictor
        self.history = history

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Load the wrapped predictor on the requested devices.

        Args:
            device: Device passed to the wrapped predictor.
        """
        self.predictor.load(device)

    def release(self) -> None:
        """Release resources held by the wrapped predictor."""
        self.predictor.release()

    @functools.cached_property
    def calibration_points(
        self,
    ) -> dict[str, Float[torch.Tensor, " b"]]:
        """Compute the calibration anchors in ascending predictor score order.

        Returns:
            Raw scores under ``xp`` and corresponding assay measurements under
            ``fp``, on the same device.
        """
        requests = [
            base.PredictionRequest(token_ids) for token_ids in self.history["token_ids"]
        ]
        predictions = self.predictor(requests)
        scores = torch.stack([prediction.mean for prediction in predictions])
        order = torch.argsort(scores, stable=True)
        calibration_points = {
            "xp": scores[order],
            "fp": self.history["measurement"].to(scores.device)[order],
        }
        return calibration_points

    def forward(
        self, requests: SequenceLike[base.PredictionRequest]
    ) -> list[base.Prediction]:
        """Score the candidates and interpolate onto the measured scale.

        Args:
            requests: Candidate sequences and requested prediction outputs.

        Returns:
            Predictions in request order, with calibrated means and no
            standard deviations or samples.

        Raises:
            ValueError: A request asks for standard deviation or samples.
        """
        if any(request.return_std or request.return_samples for request in requests):
            raise ValueError("CalibratedPredictor does not support std or samples.")
        if not requests:
            return []
        predictions = self.predictor(requests)
        raw_scores = torch.stack([prediction.mean for prediction in predictions])
        scores = _interp(
            raw_scores,
            self.calibration_points["xp"].to(raw_scores.device),
            self.calibration_points["fp"].to(raw_scores.device),
        )
        return [base.Prediction(mean=score) for score in scores]
