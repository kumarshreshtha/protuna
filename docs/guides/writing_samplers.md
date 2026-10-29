# Custom Samplers

A sampler proposes candidate sequences within a
{py:class}`~protuna.optim.spaces.search.SearchSpace`. It decides how to explore
the allowed mutations, optionally using a model to guide generation. The pipeline
then evaluates these proposals and selects candidates for lab testing.

This guide explains the sampler contract and walks through the implementation of a
genetic algorithm sampler that uses a fitness handler to evolve a population of sequences.

## Sampler contract

{py:class}`~protuna.optim.samplers.base.Sampler` inherits from
{py:class}`~protuna.inference.handlers.InferenceModel` and implements the
{py:class}`~protuna.inference.handlers.Handler` protocol. Subclasses implement the
{py:meth}`~protuna.optim.samplers.base.Sampler.sample` method, and the base class
forwards requests to it from {py:meth}`~protuna.inference.handlers.LocalHandler.stream`
and {py:meth}`~protuna.inference.handlers.InferenceModel.__call__`.

{py:meth}`~protuna.optim.samplers.base.Sampler.sample` has the following signature:

```py
def sample(self, spaces: SequenceLike[spaces.SearchSpace]) -> list[BatchTokenIDs]: ...
```

It takes a batch of search spaces as input and returns a list containing one candidate
tensor per input space, in the same order. Each response is a tensor of candidate
token IDs with shape `[n, l]`, where `n` is the number of candidates and `l` is the
sequence length. Different input spaces can lead to different numbers of generated
candidates.

See [Spaces](../concepts/spaces.md) for a conceptual overview of search spaces
and the {py:class}`~protuna.optim.spaces.search.SearchSpace` API reference for
details of its methods and attributes.

The sampler can assume that input search spaces are unique. Its outputs must
satisfy the following conditions:

| Condition | Contract |
| --- | --- |
| Allowed mutations | Every response satisfies the mutation constraints imposed by the space's {py:attr}`~protuna.optim.spaces.search.SearchSpace.allowed` mask. |
| Preserve inputs | The input sequence and allowed-residue mask carried by the input {py:class}`~protuna.optim.spaces.search.SearchSpace` are never modified in place. |

These conditions are the implementation's responsibility. Inheriting from
{py:class}`~protuna.optim.samplers.base.Sampler` does not enforce them automatically. The pipeline checks that proposals satisfy
the supplied search space and raises a `ValueError` if they do not. Responses may
include repeated candidates and starting sequences; the pipeline removes these
across all spaces and cycles before storage and evaluation. For an immutable
space, return either an empty integer tensor of shape `[0, l]` or copies of its
starting sequence.

## Models and resource management

Most model-guided samplers accept a handler at initialization rather than owning
model weights. {py:class}`~protuna.optim.samplers.base.Sampler` also inherits from
{py:class}`~protuna.inference.handlers.HandlerRegistryMixin`, which automatically
registers handlers assigned as attributes of the sampler. This requires `super().__init__()`
to be called before assigning a model in the sampler's constructor; otherwise,
an `AttributeError` is raised. The
{py:meth}`load(device) <protuna.inference.handlers.HandlerRegistryMixin.load>` and
{py:meth}`release() <protuna.inference.handlers.HandlerRegistryMixin.release>` methods
inherited from {py:class}`~protuna.inference.handlers.HandlerRegistryMixin` can
then propagate calls to registered components, including nested registries.

This lets the sampler work with a local adapter, a distributed handler, or a
hosted model through the same interface. A sampler that manages its own weights
should follow the loading, release, and serialization guidance in
[Custom Model Adapters](writing_adapters.md).

The default registry forwards the same device specification to every component.
When several models need different placements, override
{py:meth}`~protuna.inference.handlers.HandlerRegistryMixin.load` to route devices
to each one. The implementations of
{py:meth}`ReMDMSampler.load <protuna.optim.samplers.remdm.ReMDMSampler.load>` and
{py:meth}`FKSteeringSampler.load <protuna.optim.samplers.smc.FKSteeringSampler.load>`
illustrate this pattern. Sampling code must also account for model responses
arriving on a different device from its working tensors.

The sampler's `batch_size` counts search spaces per
{py:meth}`~protuna.optim.samplers.base.Sampler.sample` call. A model's own
batch size controls how many sequences it processes at once. These are separate
settings: one search space can generate many model requests.

## Example: Genetic algorithm sampler

The following sampler starts with a random population drawn from the allowed residues. In each generation, it keeps the highest-scoring quarter as parents, creates children by choosing between two parents at each site, and randomly redraws some residues. Parents survive into the next generation alongside their children.

The fitness model accepts one tokenized sequence per request and returns one scalar floating-point tensor per response. Higher scores are preferred.

The complete source code is available in
[genetic_algorithm_sampler.py](https://github.com/kumarshreshtha/protuna/blob/main/examples/genetic_algorithm_sampler.py).

### Setting up the sampler

The constructor calls `super().__init__()` before assigning `self.model`, so the model is registered as a handler and loading
and release calls propagate to it. The rest of the constructor stores the configuration
for the genetic algorithm.

```{literalinclude} ../../examples/genetic_algorithm_sampler.py
:language: python
:start-after: "# init-start"
:end-before: "# init-end"
```

### Proposing candidates

Next, the `sample()` method defines the logic for turning input search spaces into
proposed candidates.

We iterate over the search spaces one at a time, generating candidates and storing them
in `batches` in the input order. For an immutable search space, whose only valid state
is its initial sequence, we return an empty tensor.

```{literalinclude} ../../examples/genetic_algorithm_sampler.py
:language: python
:start-after: "# init-end"
:end-before: "# base-end"
:dedent: 4
```
This is followed by the genetic algorithm implementation, which proceeds in three steps:

1. **Population initialization.** The initial population is drawn from
   `space.allowed` using {py:func}`torch.multinomial`.
2. **Parent selection.** In each generation, the sampler sends the population to the
   fitness handler and retains the highest-scoring quarter as parents.
3. **Creating the next generation.** Crossover combines parents at corresponding
   positions, and mutation draws replacement residues from `space.allowed`.
   Both operations preserve the mutation constraints. The parents and children
   form the next population.

```{literalinclude} ../../examples/genetic_algorithm_sampler.py
:language: python
:start-after: "# base-end"
:end-before: "# loop-end"
:dedent: 4
```

{py:meth}`self.rng(device) <protuna.optim.samplers.base.Sampler.rng>` supplies
a persistent generator for each device. Passing it to every random operation makes
the sampler's draws reproducible for a fixed seed
and call order. Repeated calls advance the generator rather than restarting it;
the fitness handler may have its own sources of randomness.

For mutable spaces, the returned batch contains `population_size` candidates,
including any repeated sequences and starting sequences in the final population.
The pipeline removes these before storage and evaluation.

### Using the sampler

To complete the example, we define a toy fitness model that rewards higher
alanine content in the proposals:

```{literalinclude} ../../examples/genetic_algorithm_sampler.py
:language: python
:start-after: "# fitness-toy-start"
:end-before: "# fitness-toy-end"
```

With our `fitness_model` defined, the sampler can be called directly as follows:

```{literalinclude} ../../examples/genetic_algorithm_sampler.py
:language: python
:start-after: "# main-start"
:end-before: "# main-end"
:dedent: 4
```

Here, {py:meth}`~protuna.optim.spaces.search.SearchSpace.from_rules` receives no
search rules, so all amino acids are allowed at every site.
{py:meth}`data.tokenizer.decode <protuna.data.tokenizers.ProteinTokenizer.decode>`
converts the candidate token IDs into sequence strings.
In a campaign, the sampler is passed as the `sampler` argument to
{py:class}`~protuna.optim.pipeline.Pipeline`. The pipeline manages loading the
sampler before generation and can release its registered handlers afterward when
`release_handlers_after_use=True`.
