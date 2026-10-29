# In-silico evolution of CreiLOV

This example runs five rounds of simulated protein engineering to improve the
fluorescence of CreiLOV {cite:p}`creiLOV`. In each round, a Protuna pipeline
proposes variants and selects 96 candidates for evaluation. A trained oracle supplies
their fitness values in place of wet-lab measurements. These observations guide the next round.

The complete source code, including model training and plotting, is available in
[insilico_simulation.py](https://github.com/kumarshreshtha/protuna/blob/main/examples/insilico_simulation.py).

## Setup

In addition to protuna's dependencies this example also requires `peft` and
`matplotlib`. If using uv you can run it with the additional dependencies using:

```bash
uv run --with peft --with matplotlib python insilico_simulation.py
```

The script has an expected runtime of about 11 minutes on an NVIDIA A10G,
plus the model download time on the first run. It uses ESMC-300M for sequence
generation, fitness ranking, and the embeddings used by the oracle.

The configuration sets the number of rounds, parents, search spaces, and candidates
to acquire. `max_samples` is the number of proposals per search space;
`model_batch_size` controls the number of sequences processed per model call.

```{literalinclude} ../../examples/insilico_simulation.py
:language: python
:start-after: "# docs-config-start"
:end-before: "# docs-config-end"
```

## Oracle and objective predictor

The simulation uses two different models for evaluation and selection:

- **Oracle.** An MLP is trained to predict fluorescence from mean ESMC-300M
  embeddings of variants in the CreiLOV dataset. Replicate measurements are
  averaged, and training uses variants with up to four substitutions. The oracle
  is trained once, or loaded from its saved checkpoint, and remains fixed during
  the campaign.
- **Objective predictor.** The first round uses zero-shot wild-type marginal
  scores from ESMC-300M. Subsequent rounds train a new LoRA adapter on all fitness
  observations collected so far. A pairwise ranking loss aligns the marginal
  scores with observed fitness, while a KL penalty discourages deviation from
  the base model's logits at the wild-type sequence.

The oracle evaluates the acquired batch. The objective predictor ranks the
pool of generated candidates. Training happens before constructing each pipeline;
the pipeline itself performs inference-time search. See `Oracle` and `train_predictor()`
in the full script for the training implementations.

## Building the campaign

### Generating candidates

`Campaign` builds a {py:class}`~protuna.optim.samplers.maskgit.MaskGITSampler` using the base
ESMC-300M model throughout the campaign. The sampler fills the twelve masked sites over six iterations,
committing the most confident sites first, with nucleus sampling at `top_p=0.9`.

```{literalinclude} ../../examples/insilico_simulation.py
:language: python
:start-after: "# docs-campaign-init-start"
:end-before: "# docs-campaign-init-end"
```

### Choosing parents and mutation sites

The first round starts from the wild type. Later rounds start from the four
highest-fitness candidates across all observations collected so far.

The empty rule list allows all standard amino acids at every site.
{py:class}`~protuna.optim.spaces.subspaces.RandomMaskSampler` draws 100 search
spaces per parent, each opening up to 12 uniformly selected sites for mutation.

```{literalinclude} ../../examples/insilico_simulation.py
:language: python
:start-after: "# docs-design-space-start"
:end-before: "# docs-design-space-end"
:dedent: 4
```

With ten proposals per space, the first round generates up to 1,000 candidates
and subsequent rounds up to 4,000. The actual number can be smaller because the
pipeline removes duplicate proposals before storage and evaluation.

### Ranking and acquiring candidates

`get_pipeline()` prepares the objective predictor. With no observations, it shares the sampler's
base model for zero-shot scoring otherwise it trains the predictor on the newly acquired sequences.

{py:class}`~protuna.optim.acquisitions.greedy.Greedy` selects the 96 candidates
with the highest objective scores for oracle evaluation. Parent selection for
the next round happens in `get_design_space()` as described above.

```{literalinclude} ../../examples/insilico_simulation.py
:language: python
:start-after: "# docs-pipeline-start"
:end-before: "# docs-pipeline-end"
:dedent: 4
```

## Running the simulation

`Evaluator` loads the dataset, initializes the campaign from its target sequence,
and prepares the oracle. Each iteration builds a pipeline, generates the acquired
batch with `pipeline.run()`, and asks the oracle for its fitness values. Those
sequences and fitness values are passed into the next iteration.

```{literalinclude} ../../examples/insilico_simulation.py
:language: python
:start-after: "# docs-evaluator-start"
:end-before: "# docs-evaluator-end"
```

The `_record()` helper collects each round's fitness values, mean, and maximum.
`summarize()` prints these statistics and saves the two plots below.
Outputs are written under `$PROTUNA_HOME/insilico_simulation`, which defaults to
`~/.cache/protuna/insilico_simulation`. This directory contains the oracle
checkpoint, campaign stores, predictor adapter, and plots.

## Results

The plots use zero-based round numbers, so round 0 is the initial zero-shot round.
The dashed line marks the oracle's predicted fitness for the wild type.

````{container} simulation-plots
```{figure} ../_static/fitness_evolution.png
:alt: Mean and maximum oracle fitness increase over five rounds; mean fitness exceeds the wild-type baseline at round 2.

Mean and maximum oracle fitness of the acquired batch in each round.
```

```{figure} ../_static/fitness_densities.png
:alt: Fitness density curves shift toward higher oracle fitness in later rounds, with rounds 3 and 4 mostly above the wild-type baseline.

Kernel density estimates of oracle fitness for each round's acquired candidates.
```
````

In this run, mean fitness increases from about 4.10 to 4.48, exceeding the
wild-type baseline of about 4.24 by round 2. The maximum increases from about
4.36 to 4.74. The density curves show the acquired population shifting toward
higher oracle fitness as observations accumulate.
