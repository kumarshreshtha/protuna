# Hard and soft constraints

## Source reference

- Read `protuna.optim.constraints.base` for the type alias definitions and the `ModelBasedConstraint` base class.
- Inspect `protuna.optim.constraints` for existing constraints and predictor wrappers and read what's
  relevant for your current implementation.
- Read `protuna.optim.predictors.base` for the base behavior of predictors when working with constraints based on predictors.

## Choose the contract

Constraints must implement the structural typing contract of either a hard or a
soft constraint. `HardConstraintLike` and `SoftConstraintLike` are the respective
handler type aliases.

Each request is a one-dimensional tensor of candidate token IDs. The response depends on the
constraint type:

- `constraints.HardConstraintLike` returns a scalar boolean tensor per sequence.
  `True` passes. Failed candidates are excluded from subsequent pipeline stages.
- `constraints.SoftConstraintLike` returns a scalar floating-point feasibility
  score interpreted as a probability in `[0, 1]`. Higher values indicate greater feasibility.

Handler `__call__` operates on `SequenceLike` batches of inputs and outputs. A
batch response can be a `[b]` tensor or a list of scalar tensors. Preserve
input order and cardinality. Each result must be a zero-dimensional tensor of
the appropriate dtype, not a Python `bool`/`float` or a `[1]` tensor.

## Reuse existing interfaces when they fit

| Existing constraint | Policy |
| --- | --- |
| `HardPredictionRangeConstraint(model, ge=..., le=...)` | Inclusive bounds on the predicted mean; supply at least one bound |
| `HardUCBInequalityConstraint(model, threshold=..., beta=...)` | Strict `mean + sqrt(beta) * std > threshold` |
| `SoftInequalityConstraint(model, threshold=..., temperature=...)` | Sigmoid of `(mean - threshold) / temperature`; score is `0.5` at equality |

`SoftInequalityConstraint` is meant for models with externally calibrated scaling temperatures. Only use it with calibrated
temperature values.

## Choose the base class

- Subclass `inference.ModelFreeLocalHandler` and implement `__call__` for
  a constraint that does not require a model, such as a simple constraint based
  on sequence properties.
- For a predictor-backed constraint, subclass `constraints.ModelBasedConstraint`.
- If the constraint owns a model or client that does not implement
  `PredictorLike`, use `inference.InferenceModel[TokenIDs, ResponseT]` with the
  appropriate resource lifecycle. `ModelBasedConstraint` assumes `PredictionRequest`
  inputs and `Prediction` outputs from its model.

Keep tensor computations on compatible devices: pipeline tokens arrive on CPU,
but a predictor's results may be on another device. If the results arrive on
GPU don't transfer them to CPU yourself. Return GPU tensors.

Read [adapters](adapters.md) if the constraint owns a custom model that does not implement `PredictorLike`.

Read [structure_constraint.py](../examples/structure_constraint.py) for a model based hard constraint
that filters candidates on pTM and pLDDT values from a structure prediction model.
