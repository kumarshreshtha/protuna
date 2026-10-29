"""Shared inference contracts for different model categories.

Request and response types describe the inputs and outputs of each category,
while handler types express the corresponding callable interface. Model adapters
implement these contracts so algorithms can use compatible models interchangeably,
regardless of how or where inference runs or the interface exposed by the underlying
implementation.
"""

from .mlm import MaskedLMLike as MaskedLMLike
from .mlm import MLMForLogitsLike as MLMForLogitsLike
from .mlm import MLMLogitsRequest as MLMLogitsRequest
from .mlm import MLMOutput as MLMOutput
from .mlm import MLMOutputConfig as MLMOutputConfig
from .mlm import MLMRequest as MLMRequest
