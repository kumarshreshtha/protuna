"""Adapters implementing the model contracts in :mod:`~protuna.inference.interfaces`."""

from .dplm import DplmInferenceModel as DplmInferenceModel
from .dplm import DplmLogitsModel as DplmLogitsModel
from .dplm import DplmModels as DplmModels
from .esm import EsmInferenceModel as EsmInferenceModel
from .esm import EsmLogitsModel as EsmLogitsModel
from .esm import EsmModels as EsmModels
from .esmc import EsmcInferenceModel as EsmcInferenceModel
from .esmc import EsmcLogitsModel as EsmcLogitsModel
from .esmc import EsmcModels as EsmcModels
from .shared import SharedLogitsModel as SharedLogitsModel
