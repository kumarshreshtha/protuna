"""Hard and soft constraints for evaluating generated candidates.

A constraint is a :class:`~protuna.inference.handlers.Handler` that evaluates
tokenized candidate sequences and returns one scalar tensor per candidate.
:py:type:`~protuna.optim.constraints.base.HardConstraintLike` returns boolean
pass/fail decisions;
:py:type:`~protuna.optim.constraints.base.SoftConstraintLike` returns
floating-point scores.

Inputs are a :py:type:`~protuna.typing.SequenceLike` of
:py:type:`~protuna.typing.TokenIDs`, such as a 2D tensor or a list of 1D tensors.
Outputs are a :py:type:`~protuna.typing.SequenceLike` of scalar tensors, such as
a 1D tensor.
"""

from .base import ConstraintLike as ConstraintLike
from .base import HardConstraintLike as HardConstraintLike
from .base import ModelBasedConstraint as ModelBasedConstraint
from .base import SoftConstraintLike as SoftConstraintLike
from .inequality import HardPredictionRangeConstraint as HardPredictionRangeConstraint
from .inequality import HardUCBInequalityConstraint as HardUCBInequalityConstraint
from .inequality import SoftInequalityConstraint as SoftInequalityConstraint
from .trust_region import MaxMutationConstraint as MaxMutationConstraint
