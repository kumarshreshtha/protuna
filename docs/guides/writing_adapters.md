# Custom Model Adapters

A model adapter connects a model to Protuna's request-handling interface. It
translates inputs into the model's format, runs inference, and returns outputs
that Protuna components can consume. The same interface supports models running
locally, models served through Ray, and models accessed through a hosted API.

This guide explains the contracts an adapter must satisfy and how to manage its
resources. For an overview of how handlers connect Protuna's components, support
local and distributed execution, and use `*Like` annotations, see the
[Handlers concept page](../concepts/handlers.md).

## Handler contract

An adapter implements the {py:class}`~protuna.inference.handlers.Handler`
protocol, without needing to inherit from a particular class. The protocol
exposes five operations:

| Operation | Contract |
| --- | --- |
| {py:meth}`__call__(requests) <protuna.inference.handlers.Handler.__call__>` | Returns one response per request, in input order. |
| {py:meth}`stream(requests) <protuna.inference.handlers.Handler.stream>` | Consumes `(request_id, request)` pairs and yields `(request_id, request, response)` triples as results become available. |
| {py:meth}`load(device=None) <protuna.inference.handlers.Handler.load>` | Initializes the resources needed for inference. |
| {py:meth}`release() <protuna.inference.handlers.Handler.release>` | Releases resources held by the adapter. |
| {py:meth}`shutdown() <protuna.inference.handlers.Handler.shutdown>` | Terminates the handler's service, when applicable. |

For most adapters, {py:class}`~protuna.inference.handlers.InferenceModel` provides
a convenient starting point. Subclasses implement `load`, `forward`, and
`release`, while the base class supplies the remaining handler operations,
including batching and local streaming. Batching is configured through
`super().__init__(batch_size=batch_size)` in the constructor.
The [local models](#local-models) section
explains resource management, and the [Biohub example](biohub-adapter-example)
shows a complete implementation for a hosted model.

A general-purpose handler can define its own request and response types. When an
adapter serves as a pipeline component or supplies a model to one, its request
and response types must match the interface that component expects. Some examples are:

| Interface | Request | Response |
| --- | --- | --- |
| {py:type}`~protuna.optim.predictors.base.PredictorLike` | {py:class}`~protuna.optim.predictors.base.PredictionRequest` | {py:class}`~protuna.optim.predictors.base.Prediction` |
| {py:type}`~protuna.optim.constraints.base.HardConstraintLike` | {py:type}`~protuna.typing.TokenIDs` | A scalar boolean tensor: `Bool[torch.Tensor, ""]` |
| {py:type}`~protuna.optim.constraints.base.SoftConstraintLike` | {py:type}`~protuna.typing.TokenIDs` | A scalar floating-point tensor: `Float[torch.Tensor, ""]` |
| {py:type}`~protuna.inference.interfaces.mlm.MLMForLogitsLike` | {py:class}`~protuna.inference.interfaces.mlm.MLMLogitsRequest` | A logits tensor: `Float[torch.Tensor, "l v"]` |
| {py:type}`~protuna.inference.interfaces.mlm.MaskedLMLike` | {py:class}`~protuna.inference.interfaces.mlm.MLMRequest` | {py:class}`~protuna.inference.interfaces.mlm.MLMOutput` |
| {py:type}`~protuna.optim.acquisitions.kmeans.EmbeddingModelLike` | {py:type}`~protuna.typing.TokenIDs` | An embedding vector: `Float[torch.Tensor, " d"]` |

These types describe individual requests and responses;
{py:meth}`~protuna.inference.handlers.Handler.__call__` receives a
batch of requests and returns a batch of responses. Both batches use
{py:class}`~protuna.typing.SequenceLike` containers. For example, a hard constraint
can return a one-dimensional boolean tensor whose elements are the scalar responses
for the batch.

Request options describe the outputs expected from the adapter. For instance, predictor
requests can ask for standard deviations or posterior samples. MLM requests select
which outputs to return and whether to restrict logits to masked positions or
amino acid columns. An adapter must support the options its consumers require.

## Translating protein tokens

Protein tokens passed to adapters use Protuna's vocabulary, exposed by
{py:class}`~protuna.data.tokenizers.ProteinTokenizer` through {py:class}`data.tokenizer <protuna.data.tokenizers.ProteinTokenizer>`.
The adapter is responsible for translating them into the model's vocabulary and back.

Returned protein tokens must use Protuna IDs, and logit columns must follow Protuna's
{py:attr}`vocabulary order <protuna.data.tokenizers.ProteinTokenizer.ordered_tokens>`.

For {py:type}`~protuna.inference.interfaces.mlm.MLMForLogitsLike`, when
`aa_tokens_only=True`, the response must contain only the first 20 columns,
corresponding to the standard amino acids in Protuna's token ID order.
When `masked_positions_only=True`, the logits must contain rows only for
masked input positions, in their original order.

Full per-position logits, per-position embeddings, and hidden states must contain one row per
input protein position, with model-added boundary tokens and batch padding
removed. Sequence-level embeddings instead contain one vector per sequence.

{py:class}`~protuna.inference.models.hf_mlm.HuggingFaceMLMBase` provides a working
implementation. Its
{py:meth}`~protuna.inference.models.hf_mlm.HuggingFaceMLMBase.prepare_inputs`
method handles input conversion, while its
{py:attr}`~protuna.inference.models.hf_mlm.HuggingFaceMLMBase.translation` mapping reorders model logits into Protuna's vocabulary.


## Serialization

For distributed execution, adapters must be picklable. {py:meth}`~object.__getstate__`
can be used to clear large cached tensors and other resources that
{py:meth}`~protuna.inference.handlers.Handler.load` can recreate. This keeps serialization lightweight.

Request and response payloads must also be serializable for distributed execution.
A custom response object containing tensors must provide a `to(device)` method
when used with `responses_to_cpu=True`, so serving workers can move its tensors to CPU.
{py:class}`~protuna.optim.predictors.base.Prediction` and
{py:class}`~protuna.inference.interfaces.mlm.MLMOutput` responses support
this operation.

## Local models

For models that own CPU or GPU resources,
{py:class}`~protuna.inference.handlers.InferenceModel` provides batch handling
and local streaming. It requires the following three abstract methods to be implemented:

- {py:meth}`load(device=None) <protuna.inference.handlers.LocalHandler.load>` loads weights and prepares inference resources.
- {py:meth}`forward(requests) <protuna.inference.handlers.InferenceModel.forward>` processes one batch and returns responses in request order.
- {py:meth}`release() <protuna.inference.handlers.LocalHandler.release>` drops the resources created by
  {py:meth}`~protuna.inference.handlers.LocalHandler.load`.

Weight loading, large tensor creation, and device allocation belong in
{py:meth}`~protuna.inference.handlers.LocalHandler.load`, keeping construction
lightweight and allowing workers to initialize their own resources.
{py:meth}`~protuna.inference.handlers.InferenceModel.forward` can receive inputs on any device and is responsible for moving them to the model's device.

{py:meth}`~protuna.inference.handlers.LocalHandler.load` accepts a
{py:class}`torch.device`, a sequence of devices, or `None`. The adapter determines
how these cases are handled, with `None` leaving device selection to its own
default.

Repeated calls to {py:meth}`~protuna.inference.handlers.LocalHandler.load` should
avoid loading duplicate weights. Calling it after
{py:meth}`~protuna.inference.handlers.LocalHandler.release` must recreate the resources
needed for inference.

The {py:class}`~protuna.inference.models.esm.EsmLogitsModel` implementation
illustrates how a concrete adapter supplies the {py:type}`~protuna.inference.interfaces.mlm.MLMForLogitsLike` contract
on top of the shared Hugging Face model operations.

### Batch size

Callers can submit batches of any size, so the adapter must control how much
work reaches the model at once.
{py:class}`~protuna.inference.handlers.InferenceModel` handles this in
{py:meth}`~protuna.inference.handlers.InferenceModel.__call__`
by splitting requests into forward passes of at most `batch_size` items.
With `batch_size=None`, it passes the entire input batch to
{py:meth}`~protuna.inference.handlers.InferenceModel.forward`.

When serving the handler (using {py:func}`~protuna.inference.serve.api.serve_routed_pool` or
{py:func}`~protuna.inference.serve.api.serve_queue_pool`), the pool's `max_batch_size` controls the batches sent
to each replica.

### Local streaming

The inherited {py:meth}`~protuna.inference.handlers.LocalHandler.stream` consumes
requests in chunks of `batch_size` and yields each chunk's responses in input
order, after the batch call completes. It does not yield individual requests as
soon as they finish within a batch.

With `batch_size=None`, it materializes the entire request iterator before running
inference. A finite batch size keeps consumption bounded for large or unbounded
streams.

## Hosted inference API

An adapter can also delegate inference to a hosted service, such as [Biohub](https://docs.biohub.ai/api/protein/models/logits),
[Tamarind Bio](https://app.tamarind.bio/api-docs),
or a deployed tool service like [proto-tools](https://proto.evodesign.org/docs/tools/introduction).
It follows the same handler and token contracts as a local model adapter.

The API client's lifecycle fits the same pattern:
{py:meth}`~protuna.inference.handlers.Handler.load` creates the client or
connection, and {py:meth}`~protuna.inference.handlers.Handler.release` closes it.
Live sessions and connections are excluded from pickled state and recreated in
{py:meth}`~protuna.inference.handlers.Handler.load`. Serialization can matter even
when the adapter itself is used locally: an enclosing handler may contain it
and later be distributed.

{py:class}`~protuna.inference.handlers.InferenceModel` is also useful here if you
want to implement API calls in {py:meth}`~protuna.inference.handlers.InferenceModel.forward` and reuse
its batching and streaming. Alternatively, the full {py:class}`~protuna.inference.handlers.Handler` protocol
can be implemented when the service needs different batching or streaming behavior.

Remote requests may complete out of order, while
{py:meth}`~protuna.inference.handlers.Handler.__call__` still returns responses in
input order. A custom
{py:meth}`~protuna.inference.handlers.Handler.stream` implementation can yield completed requests
immediately, preserving each request's ID and original payload.

(biohub-adapter-example)=
### Example: ESMC embeddings through Biohub

The following adapter wraps Biohub's hosted ESMC model. Each request is a
Protuna-tokenized sequence and each response is the mean embedding returned by
Biohub. This adapter satisfies the
{py:type}`~protuna.optim.acquisitions.kmeans.EmbeddingModelLike` contract, a
{py:class}`Handler[TokenIDs, Float[torch.Tensor, " d"]] <protuna.inference.handlers.Handler>`
interface accepted by some acquisition functions. You can also wrap the handler in
a predictor that converts embeddings into
{py:class}`~protuna.optim.predictors.base.Prediction` objects—for example, using a
trained linear head to predict a property of interest.

The complete source code is available in
[biohub_adapter.py](https://github.com/kumarshreshtha/protuna/blob/main/examples/biohub_adapter.py).
Running the example requires the [`esm` SDK](https://github.com/Biohub/esm#running-esmc-through-the-biohub-platform)
and a Biohub API key.

#### Setting up the adapter

The adapter subclasses {py:class}`~protuna.inference.handlers.InferenceModel`,
specifying a tokenized sequence as its request type and a embedding vector as
its response type. The constructor passes `batch_size` to the base class and
stores the configuration needed to create the API client on `load()`. Construction does not open a connection or read credentials.

```{literalinclude} ../../examples/biohub_adapter.py
:language: python
:start-after: "# init-start"
:end-before: "# init-end"
```


#### Managing the API client

The `load()` method creates the client if one does not already exist,
reading the API key from the environment. Since inference runs on
Biohub, the `device` argument is unused.

```{literalinclude} ../../examples/biohub_adapter.py
:language: python
:start-after: "# init-end"
:end-before: "# lifecycle-end"
:dedent: 4
```

The matching `release()` method closes the client and clears the reference, so
a later `load()` call can create a fresh connection. `__getstate__()` excludes
the live client from serialized state. A worker receiving the adapter can then
create its own client using the configuration and credentials in its environment.

#### Embedding one sequence

Before implementing batch inference, we define a helper for a single sequence.
It encodes a protein string with the Biohub client, requests its mean embedding,
and returns a floating-point vector on the CPU. Both API calls can return an
`ESMProteinError`, which the helper raises.

```{literalinclude} ../../examples/biohub_adapter.py
:language: python
:start-after: "# embed-start"
:end-before: "# embed-end"
```

#### Processing a batch

The adapter's `forward()` method connects Protuna requests to this helper:

1. It checks that the client was created.
2. Decodes the input token IDs into protein strings using `data.tokenizer`.
3. Submits the strings to the SDK's parallel executor, which calls
   `_embed_sequence` for each sequence.
4. Collects one embedding per request in input order, raising any
   `ESMProteinError` present in the executor's results.

```{literalinclude} ../../examples/biohub_adapter.py
:language: python
:start-after: "# lifecycle-end"
:end-before: "# forward-end"
:dedent: 4
```

The inherited batch handling limits each `forward()` call to `batch_size`
sequences. The executor manages concurrency and retries within that group.

#### Using the adapter

This adapter can then be used with any Protuna component that accepts an
`EmbeddingModelLike` handler. For example, the
{py:class}`~protuna.optim.acquisitions.kmeans.KMBBO` acquisition function performs
k-means clustering on the shortlisted candidates to pick high-fitness candidates
that are diverse in the embedding space.

The following snippet illustrates how this adapter fits in a
{py:class}`~protuna.optim.pipeline.Pipeline`. Here, `design_space` and `sampler`
are the design space and sampler from your campaign setup. The pipeline calls
`load()` on registered handlers, including the acquisition's embedding model,
when they are needed.

```py
from protuna import optim
from protuna.optim import acquisitions

model = BiohubEmbeddingModel(batch_size=128)
score_fn = acquisitions.UCBScore(subsample_size=1000, beta=4.0, temperature=0)
acquisition = acquisitions.KMBBO(model, score_fn, budget=10)
pipeline = optim.Pipeline(
    design_space=design_space,
    sampler=sampler,
    acquisition=acquisition,
    release_handlers_after_use=True,
)
```
