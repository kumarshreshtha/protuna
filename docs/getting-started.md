# Get started

## Installation

```{include} ../README.md
:start-after: <!-- installation-start -->
:end-before: <!-- installation-end -->
```

## Quickstart

This example uses the GB1 protein {cite:p}`GB1Landscape` as the starting sequence
and its precomputed multiple sequence alignment (MSA). The MSA downloads on first use (262 kB);
see [Storage](#storage) below to choose where downloads and pipeline artifacts
are saved before running the code.

```{include} ../README.md
:start-after: <!-- quickstart-code-start -->
:end-before: <!-- quickstart-code-end -->
```

You can pass your own protein sequence string to {py:meth}`tokenizer.encode <protuna.data.tokenizers.ProteinTokenizer.encode>` and it's MSA file to {py:meth}`TokenizedMSA.from_file() <protuna.data.msa.core.TokenizedMSA.from_file()>`. Setting `rules` to `[]` will create a search space where all mutations are permitted.

## Storage

```{include} ../README.md
:start-after: <!-- storage-start -->
:end-before: <!-- storage-end -->
```

See the [storage reference](api/optim/storage.rst) for configuring pipeline storage.

## Next steps

- Read the [campaign concepts](concepts/overview.md) for the broader workflow.
- Explore [mutation spaces and rules](concepts/spaces.md) to tailor allowed mutations.
- Browse the [examples](examples/index.md) for more elaborate pipelines.
