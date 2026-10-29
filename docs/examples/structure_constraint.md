# Filtering by pTM and pLDDT

This example generates CreiLOV {cite:p}`creiLOV` variants and filters them using a
custom hard constraint. DPLM-150M proposes candidate sequences, and ESMFold2 predicts their
structures. A candidate passes only if its mean residue pLDDT is at least 0.7
and its pTM is at least 0.5.

The complete source code is available in
[structure_constraint.py](https://github.com/kumarshreshtha/protuna/blob/main/examples/structure_constraint.py).

## Setup

The example loads [Synthyra/DPLM-150M](https://huggingface.co/Synthyra/DPLM-150M) and [Synthyra/ESMFold2](https://huggingface.co/Synthyra/ESMFold2) through Hugging Face to avoid additional dependencies. The reported
run below takes about five on an NVIDIA A10G, excluding model downloads.

## Writing the hard constraint

### Defining the interface and thresholds

A hard constraint returns one scalar Boolean tensor per candidate: `True` keeps
the sequence and `False` excludes it from subsequent pipeline stages.
`StructureConfidenceConstraint` subclasses `inference.InferenceModel`, taking
Protuna token IDs as input and returning Boolean scalar tensors.

Model loading is deferred to `load()`.

```{literalinclude} ../../examples/structure_constraint.py
:language: python
:start-after: "# docs-constraint-init-start"
:end-before: "# docs-constraint-init-end"
```

### Loading the structure model

`load()` selects the first supplied or automatically resolved device and
loads ESMFold2 in evaluation mode. `__getstate__()` leaves loaded model and device
resources out of the serialized state, so a worker can load its own copy.

```{literalinclude} ../../examples/structure_constraint.py
:language: python
:start-after: "# docs-constraint-load-start"
:end-before: "# docs-constraint-load-end"
:dedent: 4
```

### Evaluating candidates

`forward()` decodes each candidate to an amino-acid sequence and calls
`infer_protein()` with three loops, 50 sampling steps, and one diffusion sample.
For clarity, requests are processed one at a time.

The implementation averages the returned pLDDT values and compares both scores
against inclusive thresholds.

```{literalinclude} ../../examples/structure_constraint.py
:language: python
:start-after: "# docs-constraint-forward-start"
:end-before: "# docs-constraint-forward-end"
:dedent: 4
```

`release()` drops the model and device references.

```{literalinclude} ../../examples/structure_constraint.py
:language: python
:start-after: "# docs-constraint-release-start"
:end-before: "# docs-constraint-release-end"
:dedent: 4
```

## Building the pipeline

### Choosing mutation sites

We use the CreiLOV wild-type sequence as the starting sequence. An empty rule list
allows all standard amino acids at every site. `RandomMaskSampler` draws a single
search space, opening 80% of the sequence positions for mutation: 95 sites
for the 119-residue wild type. The remaining sites stay fixed.

```{literalinclude} ../../examples/structure_constraint.py
:language: python
:start-after: "# docs-design-space-start"
:end-before: "# docs-design-space-end"
:dedent: 4
```

### Generating candidates

We build a `ReMDMSampler` with the DPLM-150M model to generate up to 128 proposals from
this search space over 16 sampling steps. The default `ReMDMSampler` doesn't introduce
a remasking schedule so is equivalent to a MDLM sampler with a log linear noise schedule.

```{literalinclude} ../../examples/structure_constraint.py
:language: python
:start-after: "# docs-sampler-start"
:end-before: "# docs-sampler-end"
:dedent: 4
```

### Filtering candidates

With `objective=None` and `acquisition=None`, the pipeline performs a single
generation and filtering cycle and returns all survivors.

```{literalinclude} ../../examples/structure_constraint.py
:language: python
:start-after: "# docs-pipeline-start"
:end-before: "# docs-pipeline-end"
:dedent: 4
```
The temporary campaign directory is deleted when the
context exits. Use a persistent location for `CampaignStore` if you want to
inspect the stored sequences and constraint verdicts afterward.

## Results

Running the script on an NVIDIA A10G produced:

```text
10 / 128 generated candidates passed mean pLDDT >= 0.7 and pTM >= 0.5.
total runtime: 328.65 seconds
```

About 7.8% of the generated candidates passed both confidence thresholds. The
runtime excludes model download time.
