"""Interface for protein property predictors.

A predictor is a :class:`~protuna.inference.handlers.Handler` that estimates a
property for each candidate sequence and returns one
:class:`~protuna.optim.predictors.Prediction` per candidate. Each prediction
contains a mean and may include a standard deviation or samples to represent
uncertainty.

Inputs are a :py:type:`~protuna.typing.SequenceLike` of
:class:`~protuna.optim.predictors.PredictionRequest` objects, each containing a
tokenized candidate sequence and flags specifying which uncertainty outputs to
return. Outputs are a :py:type:`~protuna.typing.SequenceLike` of
:class:`~protuna.optim.predictors.Prediction` objects.
"""

from .base import Prediction as Prediction
from .base import PredictionRequest as PredictionRequest
from .base import PredictorLike as PredictorLike
from .calibrated import CalibratedPredictor as CalibratedPredictor
from .ensemble import EnsemblePredictor as EnsemblePredictor
from .zeroshot import MaskedMarginalPredictor as MaskedMarginalPredictor
from .zeroshot import MutantMarginalPredictor as MutantMarginalPredictor
from .zeroshot import PSSMZeroshotPredictor as PSSMZeroshotPredictor
from .zeroshot import WildTypeMarginalPredictor as WildTypeMarginalPredictor
