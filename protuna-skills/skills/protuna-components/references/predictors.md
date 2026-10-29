# Property predictors

## Source reference

- Read `protuna.optim.predictors.base` for the inputs and outputs of a predictor.
- Read `protuna.optim.predictors.zeroshot` for similar implementations if the predictor
  uses a zero-shot variant effect prediction (VEP) algorithm.
- Read `protuna.optim.predictors.ensemble` if working with multiple prediction models
  that will be used in an ensemble.
- Read `protuna.optim.predictors.calibrated` if working with a predictor whose predictions
  are in arbitrary units and need to be mapped to the units of an assay result. This is useful
  for ranking models used in constraints with thresholds in the units of the assay measurement.

## Contract

`predictors.PredictorLike` is a type alias for
`inference.Handler[predictors.PredictionRequest, predictors.Prediction]`.

There is no `Predictor` superclass to implement.

A predictor can inherit from `inference.InferenceModel` and implement `forward`, `load`, and `release`.
Inherited `__call__` handles batching.

See [adapters](adapters.md) for loading weights or clients, device placement,
token translation, and serving. A wrapper can inherit `HandlerRegistryMixin`
**before** `InferenceModel` to propagate loading and release to child handlers.
Assign those attributes after `super().__init__(batch_size=...)`. Register list
or dictionary members explicitly; those containers are not traversed.

Process input `PredictionRequest` flags separately for each request; a batch may
mix all four flag combinations.

Keep tensor computations on compatible devices: pipeline requests arrive on CPU,
but a predictor's results may be on another device. If the results arrive on
GPU don't transfer them to CPU yourself.


## Objective semantics and uncertainty

Raise a clear error for unsupported requested outputs rather
than silently returning `None`. A point estimate alone does not imply zero model
uncertainty.

For joint posterior samples, `samples[j]` must identify the same sampled function
across all candidates, batches, and replicas participating in selection. Use
a fixed ensemble member order or shared function-draw state. Independent marginal
draws for each sequence are insufficient for joint acquisition methods such as
`ThompsonSampling`.
