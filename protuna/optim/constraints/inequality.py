"""Filter or weight candidates using predicted property thresholds."""

import math

from jaxtyping import Bool, Float
import torch

from protuna.optim import predictors
from protuna.optim.constraints import base
from protuna.typing import SequenceLike, TokenIDs


class HardPredictionRangeConstraint(base.ModelBasedConstraint[Bool[torch.Tensor, ""]]):
    """Accept candidates whose predicted property falls in a range.

    At least one bound must be given; a candidate passes only when it satisfies
    every bound that was given.
    """

    def __init__(
        self,
        model: predictors.PredictorLike,
        *,
        le: float | None = None,
        ge: float | None = None,
        batch_size: int | None = None,
    ):
        """Set the lower and/or upper bounds that the candidate should satisfy.

        Args:
            model: The predictors that scores the candidates.
            le: Upper bound, inclusive.
            ge: Lower bound, inclusive.
            batch_size: Number of candidates to score at a time.

        Raises:
            ValueError: If no bound is given.
        """
        if all(comp is None for comp in (ge, le)):
            raise ValueError("Must provide at least one of `le`,`ge`.")
        super().__init__(model=model, batch_size=batch_size)
        self.ge = ge
        self.le = le

    def forward(self, requests: SequenceLike[TokenIDs]) -> Bool[torch.Tensor, " b"]:
        """Score the candidates and check each prediction against the bounds.

        Args:
            requests: Tokenized candidate sequences to evaluate.

        Returns:
            Boolean mask indicating which candidates satisfy all given bounds.
        """
        if len(requests) == 0:
            return torch.empty(0, dtype=torch.bool)
        preds = self.model(
            [predictors.PredictionRequest(token_ids) for token_ids in requests]
        )
        outputs = []
        for pred in preds:
            passed = torch.tensor(True, device=pred.mean.device)
            if self.ge is not None:
                passed = passed & (pred.mean >= self.ge)
            if self.le is not None:
                passed = passed & (pred.mean <= self.le)
            outputs.append(passed)
        return torch.stack(outputs)


class HardUCBInequalityConstraint(base.ModelBasedConstraint[Bool[torch.Tensor, ""]]):
    """Accept candidates whose upper confidence bound exceeds a threshold.

    Similar to CONFIG :cite:p:`CONFIG`, accept predictions when
    ``mean + sqrt(beta) * std > threshold``.

    """

    def __init__(
        self,
        model: predictors.PredictorLike,
        *,
        threshold: float,
        beta: int,
        batch_size: int | None = None,
    ):
        """Set the lower threshold and uncertainty margin.

        Args:
            model: Predictor providing means and standard deviations.
            threshold: Lower threshold, excluded at equality.
            beta: Nonnegative weight whose square root scales uncertainty.
            batch_size: Number of candidates to score per forward pass.
        """
        super().__init__(model=model, batch_size=batch_size)
        self.model = model
        self.sqrt_beta = math.sqrt(beta)
        self.threshold = threshold

    def forward(self, requests: SequenceLike[TokenIDs]) -> Bool[torch.Tensor, " b"]:
        """Check each candidate's upper confidence bound against the threshold.

        Args:
            requests: Tokenized candidate sequences to evaluate.

        Returns:
            Boolean mask indicating which candidates exceed the threshold.

        Raises:
            ValueError: If a prediction lacks a standard deviation.
        """
        if len(requests) == 0:
            return torch.empty(0, dtype=torch.bool)
        preds = self.model(
            [
                predictors.PredictionRequest(token_ids, return_std=True)
                for token_ids in requests
            ]
        )
        outputs = []
        for pred in preds:
            if pred.std is None:
                raise ValueError(
                    "Margin constraints require prediction standard deviations."
                )
            ucb = pred.mean + self.sqrt_beta * pred.std
            outputs.append(ucb > self.threshold)
        return torch.stack(outputs)


class SoftInequalityConstraint(base.ModelBasedConstraint[Float[torch.Tensor, ""]]):
    """Feasibility of candidates to have predicted property above given threshold."""

    def __init__(
        self,
        model: predictors.PredictorLike,
        *,
        threshold: float,
        temperature: float,
        batch_size: int | None = None,
    ):
        """Inits the constraint.

        Args:
            model: The predictors that scores the candidates.
            threshold: Lower bound, inclusive.
            temperature: Calibrated platt scaling temperature for the given threshold.
            batch_size: Number of candidates to score at a time.

        Raises:
            ValueError: If no bound is given.
        """
        super().__init__(model=model, batch_size=batch_size)
        self.threshold = threshold
        self.temperature = temperature

    def forward(self, requests: SequenceLike[TokenIDs]) -> Float[torch.Tensor, " b"]:
        """Compute sigmoid feasibility scores from predicted properties.

        Args:
            requests: Tokenized candidate sequences to evaluate.

        Returns:
            Feasibility scores from sigmoid-scaled prediction margins.
        """
        if len(requests) == 0:
            return torch.empty(0)
        preds = self.model(
            [predictors.PredictionRequest(token_ids) for token_ids in requests]
        )
        preds = torch.stack([pred.mean for pred in preds])
        return ((preds - self.threshold) / self.temperature).sigmoid()
