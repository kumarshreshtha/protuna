---
name: protuna-components
description: Implement new Protuna components including samplers, acquisitions and recyclers, model adapters, predictors, constraints, search space samplers, and search rules. Use when extending a campaign with new behavior or writing compatibility adapters for existing libraries or hosted services.
---

# Extend Protuna with new components

Read referenced `protuna` modules from the local checkout or the package installed in the project's Python environment.
Resolve Markdown links relative to the file containing them.

Read the reference for the component being implemented. Additionally, read the adapter
reference when the new component owns model weights, a hosted client, or nested handlers.

- [Samplers](references/samplers.md) for implementing or extending a candidate sampler
- [Acquisitions and recyclers](references/acquisitions.md) for implementing or extending an acquisition or recycler
- [Predictors](references/predictors.md) for implementing or adapting a property predictor
- [Constraints](references/constraints.md) for implementing or extending a hard or soft constraint
- [Model adapters and handlers](references/adapters.md) for writing a model adapter or managing model weights, hosted clients, or nested handlers in any component
- [Search rules and subspaces](references/spaces.md) for writing search rules or implementing a subspace sampler

Read `protuna.data.tokenizers` for `ProteinTokenizer` if your component receives or returns
protein tokens or logits. All protein token inputs come from this tokenizer, and
all outputs must follow its vocabulary and token order. Translate at the input/output
boundary of your component if it requires a different vocabulary.

For samplers, predictors, constraints, and model adapters, start by reading
`protuna.inference.handlers` for the `Handler` protocol that each of them implements.
