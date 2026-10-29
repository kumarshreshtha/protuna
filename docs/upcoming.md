# Future Releases

For individual feature requests, bug reports, and implementation discussions, see
the [GitHub issue tracker](https://github.com/kumarshreshtha/protuna/issues).
This page provides a high-level roadmap for upcoming releases.

## Broader sequence exploration

The current pipeline expects starting sequences to be variants of the same
wild type, with a shared sequence length. The next minor release is planned to
allow homologs of different lengths as starting points in one design space.
This will let campaigns explore several related proteins together.

Insertion and deletion support is a longer-term direction. Today's search spaces
represent allowed substitutions using a fixed-length residue mask, and samplers
generate candidates within that mask. Supporting indels would extend generation
beyond substitutions so that candidate lengths can change during exploration.
Whether this added flexibility leads to meaningful gains in protein optimization
remains to be established through practical evaluation.
See [Spaces](concepts/spaces.md) for the current representation and its limits.

## Reliable, scalable distributed serving

Protuna already distributes handlers across Ray actors using routed and queue
pools. The next improvements focus on keeping large searches running reliably
and reducing the overhead of coordinating many workers:

- **Actor failure recovery:** Recover work interrupted by worker or broker
  failures. Currently, if a queue worker fails after taking requests, it leaves
  its responses pending without requeueing the work.
- **Larger queue pools:** Improve throughput as worker counts grow, including
  support for multiple brokers to address the current single-broker bottleneck.
- **Ray Direct Transport experiments:** Explore RDT for direct GPU transfers
  between handlers and between handlers and the driver. This is an experimental
  direction; the current serving API does not enable RDT.

Actor failure recovery and larger queue pools are near-term priorities. RDT
experiments are a longer-term direction rather than an immediate release goal.

The [distributed serving architecture](concepts/distributed.md) describes the
existing execution model and failure handling.

## Campaign management and resumable runs

A `Pipeline` currently performs the computational search that proposes one
experimental batch. Incorporating lab measurements and preparing the next
experimental round, however, requires coordinating separate pipeline runs.

A higher-level `Campaign` abstraction is planned on a longer horizon to manage
this broader lead optimization workflow. It will coordinate pipelines across
experimental rounds, combine trainers for updating predictors and generators
with new measurements, and use schedulers to adjust exploration and exploitation
within a budget of experimental rounds. See [Pipeline](concepts/pipeline.md) for
the current boundary between generation cycles and experimental rounds.

Resuming interrupted pipeline runs from checkpoints is a near-term priority.
`CampaignStore` already records candidates, lineage, and evaluations, but those
records do not currently provide an execution checkpoint from which a pipeline
can resume. Checkpoint support will allow long searches to continue after an
interruption without restarting the entire run.

## Training predictors and adapting generators

The planned `protuna.train` module will expose trainers and recipes for different
modeling choices, covering both property prediction and generator fine-tuning
and preference alignment. These capabilities are expected to arrive around the
same time, making it easier to learn from campaign measurements and update the
models used to score and propose new candidates.

Predictors in `protuna.optim` remain inference components; fitting and updating
models belongs in the training layer. Generator training will complement the
inference-time steering methods already available in Protuna.

The initial trainers and recipes will provide starting points for these workflows,
but users may still need substantial hyperparameter optimization to find settings
that work well for their data. Automating model selection and hyperparameter
search is a further, longer-term direction.

## Structure- and physics-informed design

Native samplers and constraints informed by protein structure and physical
properties are planned to complement the existing sequence-based methods.
The aim is to make structural and physical criteria easier to incorporate both
when proposing candidates and when evaluating their acceptability.

Custom handlers already provide an extension point for these methods. Native
components will reduce the amount of campaign-specific integration needed;
the [custom sampler guide](guides/writing_samplers.md) and
[model adapter guide](guides/writing_adapters.md) describe today's interfaces.

## More diversity-aware acquisitions

Future releases will expand the acquisition strategies available for selecting
an experimental batch that combines promising predictions with diversity and
information for subsequent rounds. This builds on existing methods such as
`DPP` and `KMBBO`, which already support diversity-aware selection.

See the [acquisition reference](api/optim/acquisitions.rst) for current strategies
and the [custom acquisition guide](guides/writing_acquisitions.md) for integrating
other selection methods.

## Agent-guided workflows

LLM-based scientific agents are another direction under active exploration,
with the aim of supporting end-to-end automated workflows that use Protuna
alongside other scientific tools. This will most likely take the form of skills
that help existing agent platforms configure and operate Protuna, rather than
a standalone agent built into the library.

We are exploring how Protuna can fit into the broader ecosystem of tools used
by scientific agents, including platforms such as Rosalind Workbench and
Claude Science, so they can make effective use of its optimization capabilities.
