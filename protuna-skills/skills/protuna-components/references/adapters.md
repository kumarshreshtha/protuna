# Writing model adapters

## Source reference

Start by reading `protuna.inference.handlers` for the `Handler` protocol
that all adapters must follow. Other classes of interest in that file include
`InferenceModel` and `ModelFreeLocalHandler`. We discuss them [below](#choose-the-base-class).

Read `protuna.inference.interfaces.mlm` only if implementing a
masked language model or masked diffusion model adapter. In that case,
also read `protuna.inference.models.hf_mlm` and `protuna.inference.models.esm` for concrete examples.

## Establish the input and output contract

Start by determining the adapter's generic types `RequestT` and `ResponseT`.

Some examples are:

| Consumer expects | Individual request | Individual response |
| --- | --- | --- |
| `predictors.PredictorLike` | `predictors.PredictionRequest` | `predictors.Prediction` |
| `constraints.HardConstraintLike` | `TokenIDs` (`[l]`) | Scalar boolean tensor |
| `constraints.SoftConstraintLike` | `TokenIDs` (`[l]`) | Scalar float tensor, probability in `[0, 1]` |
| `inference.MLMForLogitsLike` | `inference.MLMLogitsRequest` | Float logits `[l, v]`, restricted by request flags |
| `inference.MaskedLMLike` | `inference.MLMRequest` | `inference.MLMOutput` |
| `EmbeddingModelLike` in acquisition `kmeans` | `TokenIDs` (`[l]`) | Float vector `[d]` |

If there is no existing consumer contract, define request and response types with
the new consumer.

Read [predictors](predictors.md) or [constraints](constraints.md) for their
output semantics when implementing those interfaces.

## Choose the base class

From `protuna.inference`, choose the appropriate base class.

Implement the full `inference.Handler` protocol only when the inherited behavior
cannot meet the task, such as completion-order streaming from a hosted API.

For an adapter that owns weights or an API client, use `inference.InferenceModel[RequestT, ResponseT]`:

| Method to implement | Work it owns |
| --- | --- |
| `__init__` | Call `super().__init__(batch_size=...)`; store configuration and initialize resource attributes to `None`. |
| `load(device=None)` | Create the model/tokenizer or client from configuration. Repeated loads should avoid duplicate allocation. |
| `forward(requests)` | Translate a batch, run inference, and return one `ResponseT` per request in input order. |
| `release()` | Close or drop loaded resources and clear their references so a later `load()` works. |

`forward` accepts `SequenceLike[RequestT]` and returns `SequenceLike[ResponseT]`.

`SequenceLike` can be a list, tuple, or tensor. The protocol is defined in `protuna.typing`.

The base class's final `__call__` splits requests into batches for the forward calls;
do not duplicate that batching loop. Inherited streaming yields each chunk after
its batch finishes; inherited `shutdown()` is a no-op.

For an inexpensive operation with no resources, use
`inference.ModelFreeLocalHandler[RequestT, ResponseT]` and implement `__call__` instead.
It supplies lifecycle no-ops and local streaming, but does not split direct batch calls.

## Implement `forward`

1. Decode incoming protein tokens with `data.tokenizer.decode` when the backend
   accepts strings, or build an explicit mapping to its token IDs. Translate
   special tokens such as masks to the backend's representation; passing Protuna
   IDs directly is valid only if the vocabularies have been verified identical.
2. Convert backend results into the consumer's response type. Reorder logit columns
   to `data.tokenizer.ordered_tokens`; returned protein token IDs also use
   Protuna's vocabulary. Pool embeddings over actual residues only.
3. A batch may mix output flags or sequence lengths. Reject unsupported requested
   outputs explicitly instead of silently returning missing fields or substitutes.
4. Don't manually transfer the results to CPU, the workflow will take care of it
  as needed.

See `HuggingFaceMLMBase`'s `translate` property for a concrete example of token
translation in `protuna.inference.models.hf_mlm` if your model requires it.

## Manage local resources

Keep weight loading and large allocations in `load`; keep training outside the
inference path. Define how the adapter handles multiple devices. Put local models
in evaluation mode and run inference in an appropriate inference context.
Ensure that calling `forward` before loading fails clearly.

## Make repeated `load()` calls a no-op

The pipeline calls `load()` on each cycle. If the model is already loaded with
the same inputs, return without reloading weights, recreating clients, or
allocating resources again.

## Preserve compatibility with serving

Protuna's serving layer wraps the adapter and supplies the distributed handler
interface automatically. The adapter's responsibility is serializable state and
payloads: exclude live clients, loaded weights, and reconstructible caches from
`__getstate__`, and recreate them in `load`. Keep configuration sufficient for a
worker to load independently, including reading credentials in that environment.
This also matters when a parent handler containing the adapter is served.

## Adapters for hosted services

Use `InferenceModel` with API calls inside `forward`; a hosted backend does not
require a Ray pool or a custom handler implementation. Inspect the provider's SDK
or API for its actual input format, supported outputs, and batching limits.

- Store the endpoint, model ID, timeout, and credential environment-variable name
  in the constructor. Create the client and read credentials in `load`. Close the
  client in `release` and exclude live sessions from serialized state.
- In `forward`, decode Protuna requests, submit the provider's batch call or
  bounded concurrent calls, and convert results to the required response types
  and devices. Restore input order if calls complete out of order. Surface API
  errors; never drop failed items and return a shorter batch.
- Enforce any concurrency, token, or rate limits through the client/executor,
  and use bounded timeouts and retries appropriate to the provider's failure semantics.

Use [biohub_adapter.py](../examples/biohub_adapter.py) as a concrete reference for client lifecycle,
token decoding, parallel requests, error handling, and CPU embedding responses.
