"""The following code builds and runs a simple `Pipeline`.

It starts by defining which mutations are allowed and how to sample narrower search
spaces. The pipeline then samples random protein sequences in those subspaces,
uses a multiple sequence alignment (MSA) to score them based on amino acid
frequencies at the mutated sites, and selects up to 20 of the highest-scoring
sequences.
"""

from protuna import data
from protuna import optim
from protuna.optim import acquisitions
from protuna.optim import predictors
from protuna.optim import rules
from protuna.optim import samplers
from protuna.optim import spaces

dataset = data.get_dataset("gb1", source=data.DatasetSource.FLIP)
tokens = data.tokenizer.encode(dataset.target_sequence)
msa = data.TokenizedMSA.from_file(
    dataset.get_msa_path(data.MSAProtocol.ColabFold), format="a3m"
)
design_space = spaces.DesignSpace.from_rules(
    rules=[
        # Positions are illustrative and use zero-based indexing.
        rules.AllowOnly(positions=list(range(100, 125))),
        rules.DenyResidues(amino_acids=["C"]),
        rules.FixSites(positions=[105, 120]),
        # Protect sites where the query residue is >= 95% conserved.
        rules.MSAConserved(tokenized_msa=msa, threshold=0.95),
    ],
    init_sequences=[tokens],
    # Explore up to 4 subspaces, each with at most 3 mutable sites.
    space_sampler=spaces.RandomMaskSampler(max_mutations=3, n_subspaces=4),
)
pipeline = optim.Pipeline(
    design_space=design_space,
    sampler=samplers.RandomSampler(max_samples=100),  # Propose up to 4x100 variants
    objective=predictors.PSSMZeroshotPredictor(tokens, msa, laplace_smoothing=True),
    acquisition=acquisitions.Greedy(budget=20),
)
output = pipeline.run()
# Up to 20 sequences
candidates = data.tokenizer.decode(list(output.selection.token_ids))
