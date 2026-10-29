# Pipeline examples

Run with the project's interpreter, for example `uv run python example.py`.
These examples use existing components and temporary local storage. For real runs, replace
the temporary directory with a persistent output location and retain the store.

## Score, filter, recycle, and inspect a small run

This standalone example uses a synthetic alignment to avoid checkpoint downloads.
Its score and feasibility threshold demonstrate wiring, not biological evidence
or calibrated probabilities. Replace them with validated campaign inputs. PSSM
scores provide means only, which is sufficient for `Greedy`.

Here, we're using the same predictor for both the soft constraint and the objective.
This is just to show the wiring; in real campaigns, they'd normally be different.

```python
import tempfile

import torch

from protuna import data
from protuna import optim
from protuna.optim import acquisitions
from protuna.optim import constraints
from protuna.optim import predictors
from protuna.optim import rules
from protuna.optim import samplers
from protuna.optim import spaces

wt = "MKTA"
wt_tokens = data.tokenizer(wt)
msa = data.TokenizedMSA(
    msa_tokens=torch.stack([data.tokenizer(seq) for seq in [wt, "MATA", "MKAA"]]),
    valid_cols=torch.ones(len(wt), dtype=torch.bool),
)
# Step 1: define the design space
design_space = spaces.DesignSpace.from_rules(
    [rules.AllowOnly(positions=[1, 2]), rules.DenyResidues(amino_acids=["C"])],
    [wt_tokens],
    space_sampler=spaces.PartitionSampler(n_partitions=2),
)
# Step 2: define the sampler
sampler = samplers.SiteSaturationSampler(batch_size=1)
# Step 3: define hard constraints
hard_constraints = {
    "distance_from_wt": constraints.MaxMutationConstraint(
        base_sequence=wt_tokens, max_mutations=2, batch_size=32
    ),
}
# Step 4: define soft constraints
predictor = predictors.PSSMZeroshotPredictor(
    wt_tokens=wt_tokens,
    msa_tokens=msa,
    laplace_smoothing=True,
    batch_size=16,
)
soft_constraints = {
    "score_feasibility": constraints.SoftInequalityConstraint(
        model=predictor, threshold=-10.0, temperature=2.0, batch_size=16
    ),
}
# Step 5: define the objective
objective = predictor
# Step 6: define the acquisition
acquisition = acquisitions.Greedy(budget=4, feasibility_threshold=0.01)
# Step 7: define the recycler
recycler = acquisitions.Greedy(budget=2, feasibility_threshold=0.01)

with tempfile.TemporaryDirectory() as output_dir:
    store = optim.CampaignStore(
        protein_length=len(wt),
        location=output_dir,
        max_size_mb=8,
        scan_size=16,
        soft_constraint_names=tuple(soft_constraints),
    )
    pipeline = optim.Pipeline(
        design_space=design_space,
        sampler=sampler,
        objective=objective,
        hard_constraints=hard_constraints,
        soft_constraints=soft_constraints,
        acquisition=acquisition,
        recycler=recycler,
        num_cycles=2,
        lock_mutations_on_recycle=True,
        evaluated_mutants=[wt_tokens],
        release_handlers_after_use=True,
    )
    output = pipeline.run(store=store)
    sequences = data.tokenizer.decode(list(output.selection.token_ids))
    assert 0 < len(sequences) <= 4
    assert all(design_space.init_spaces[0].is_valid(seq) for seq in sequences)
    for batch in store.get_candidate_stream():
        assert batch.objective_mean is not None
        assert torch.isfinite(batch.objective_mean).all()
        assert batch.feasibility is not None
        assert "score_feasibility" in batch.feasibility
    print(sequences)
    print("Unique selected sequences:", len(set(sequences)))
```

The pipeline deduplicates proposals across parents, spaces, and cycles, so the
selected candidates have distinct sequences. The selection can contain fewer
than four candidates if fewer eligible unique proposals are available.

Sharing the predictor between the objective and soft constraint does not cache
predictions across stages. With release-after-use enabled, it is also released
and loaded between stages.

## Enumerate single substitutions without model inference

This standalone smoke check should return exactly 76 distinct variants.

```python
import tempfile

from protuna import data
from protuna import optim
from protuna.optim import samplers
from protuna.optim import spaces

with tempfile.TemporaryDirectory() as output_dir:
    store = optim.CampaignStore(protein_length=4, location=output_dir)
    pipeline = optim.Pipeline(
        design_space=spaces.DesignSpace([spaces.SearchSpace.from_rules("MKTA", [])]),
        sampler=samplers.SiteSaturationSampler(batch_size=1),
    )
    output = pipeline.run(store=store)
    sequences = data.tokenizer.decode(list(output.selection.token_ids))
    assert len(sequences) == len(set(sequences)) == 76
    assert all(
        sum(a != b for a, b in zip(seq, "MKTA", strict=True)) == 1 for seq in sequences
    )
```
