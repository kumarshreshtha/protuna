# Candidate samplers

This reference covers candidate sequence generation. To choose which
search spaces to explore, read [search rules and subspaces](spaces.md) instead.

## Source reference

- Read `protuna.optim.samplers.base` for the `Sampler` base class.
  Read `MaskedDiffusionSamplerBase` only when implementing denoising transitions.
- Read `protuna.optim.spaces.search` for the search-space dataclass. Samplers take a `SequenceLike` of `SearchSpace` objects as input.
- Inspect `protuna.optim.samplers` to locate and read existing sampler implementations
  closest to the sampler you're implementing.
- Read [genetic_algorithm_sampler.py](../examples/genetic_algorithm_sampler.py) for a simple example illustrating the sampler contract.
- `protuna.optim.samplers.smc` has examples of samplers that perform alignment during inference
  through particle filtering.
- `protuna.optim.samplers.remdm` implements a masked diffusion sampler with an optional guidance
  model for guidance based on a classifier during sampling.
- `protuna.optim.samplers.maskgit`, `protuna.optim.samplers.rlxf`, and `protuna.optim.samplers.beam` are examples of iterative refinement samplers without guidance
  or reward tilting.

## Sampler contract

Subclass `samplers.Sampler` and implement the `sample` method.
Call `super().__init__(seed=seed, batch_size=batch_size)` before assigning handler
attributes.
Each request is one `SearchSpace`; each response is one integer tensor `[n, l]`
of Protuna token IDs. Return a list with exactly one response per input space, in
input order. Candidate counts (`n`) can differ between spaces.

The sampler's `batch_size` counts **search spaces per `sample` call**, not proposed
sequences or model requests.

Callers are responsible for the precondition; sampler implementations are responsible for
the postconditions and invariant. Subclassing `Sampler` does not automatically
validate or enforce them.

- Precondition: all input search spaces in a call are unique.
- Invariant: preserve the input search spaces: never modify `init_sequence`,
  `allowed`, or cached masks in place. Clone tensors before editing them; `space.to(device)`
  may return the same object.
- Postcondition: Respect mutation constraints. Every residue in every candidate must be
   allowed by that space's `allowed` mask. Final outputs contain no mask,
   padding, or boundary tokens.

The pipeline validates mutation constraints and raises a `ValueError` for invalid
proposals. Responses may contain repeated candidates and starting sequences.
The pipeline deduplicates across all spaces and cycles and excludes starting
sequences before storage and evaluation.

For immutable spaces, return either copies of the starting sequence or an empty
integer tensor `[0, l]` on the working device, keeping its place in the response
list. `init_sequence.new_empty((0, init_sequence.numel()))` preserves dtype and
device.

## Randomness, devices, and models

Pass `self.rng(device)` as `generator` to random tensor operations on that device.
Generators persist and advance across calls; do not recreate or reseed them in
`sample`. Compare fresh instances with an explicit identical seed, device, and
call order for reproducibility. This does not seed Python/NumPy randomness or
an external model, and does not promise identical draws across devices.

Choose a working device explicitly, for example the first response's device.
Model responses may be on different devices. If you're working on a GPU
don't transfer the final results to CPU yourself. The workflow will take care
of it.

Prefer a model handler argument over loading weights in the sampling algorithm.
Assigning `self.model = model` after `super().__init__` registers a handler for
`load`/`release` propagation. Containers of handlers are not traversed automatically.

If the sampler hosts multiple models, you can manage their device placement explicitly
when the sampler receives multiple GPU inputs. For examples of explicit
device placement, see `protuna.optim.samplers.smc` or `protuna.optim.samplers.remdm`.

For owned resources, read [model adapters](adapters.md).
