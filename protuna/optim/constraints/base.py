"""Soft and hard constraints on generated proposals."""

from jaxtyping import Bool, Float
import torch

from protuna import inference
from protuna.optim import predictors
from protuna.typing import TokenIDs

type ConstraintLike[ResponseT: (Bool[torch.Tensor, ""], Float[torch.Tensor, ""])] = (
    inference.Handler[TokenIDs, ResponseT]
)
type HardConstraintLike = ConstraintLike[Bool[torch.Tensor, ""]]
type SoftConstraintLike = ConstraintLike[Float[torch.Tensor, ""]]


class ModelBasedConstraint[
    ResponseT: (Bool[torch.Tensor, ""], Float[torch.Tensor, ""])
](inference.HandlerRegistryMixin, inference.InferenceModel[TokenIDs, ResponseT]):
    """Base class for constraints whose outputs come from a predictor."""

    def __init__(
        self, model: predictors.PredictorLike, *, batch_size: int | None = None
    ):
        """Initialize the constraint and register it's model.

        Args:
            model: The predictors that scores the candidates.
            batch_size: Number of candidates to score at a time.
        """
        super().__init__(batch_size=batch_size)
        self.model = model
