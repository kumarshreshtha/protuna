# Search rules and subspace sampling

A **subspace sampler** is used to sample a narrower search space
for a given input `SearchSpace`. For code that produces candidate
sequences, read [samplers](samplers.md) instead.

## Source reference

- Read `protuna.optim.spaces.rules` for built-in rules if writing a new search rule.
- Read `protuna.optim.spaces.subspaces` for built-in search space samplers if writing a new search space sampler.
- Read `protuna.optim.spaces.design` and `protuna.optim.spaces.search` for how the rules and search space samplers interact
  with `DesignSpace` and `SearchSpace`.

## Writing search rules

A rule takes a boolean mask of shape `[L, 20]`, where `L` is the protein length,
and returns a boolean mask of the same shape on the same device after applying
its criteria. A callable needing a target sequence, alignment, or other
context must capture it in configuration.

`spaces.SearchSpace.from_rules(init_sequence, rule_list)` accepts a string or
token tensor. It starts with every amino acid allowed, passes each rule's return
value to the next rule, and restores initial residues **after all rules run**.

## Writing a subspace sampler

`spaces.SearchSpaceSampler` is a structural callable protocol:
`sampler(space: spaces.SearchSpace) -> Iterator[spaces.SearchSpace]`.
A function or callable class is sufficient.

- For every yielded subspace, keep the initial sequence unchanged, preserve shape
  and dtype, and ensure `child.allowed` is a subset of the parent mask.
- Do not modify the input or previously yielded spaces.
- For model-backed sampling, inherit `inference.HandlerRegistryMixin`, call
  `super().__init__()` before assigning handlers, and assign the model handler to
  an attribute such as `self.model`. Direct handler/registry attributes register
  automatically; containers require explicit registration of their members.
- Read [adapters](adapters.md) for model contracts and resource
  composition if building a search space sampler that requires model inputs.

For an example of a search space sampler that uses a model, see `LogitsEntropyMaskSampler`.
