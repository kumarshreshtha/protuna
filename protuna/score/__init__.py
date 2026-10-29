"""Functional library of protein-related scoring functions.

The module provides functions for scoring protein sequences including zero-shot fitness
scores for mutants and pairwise residue couplings:

- :mod:`protuna.score.mlm` — masked protein language models, via marginals
  and pseudo-log-likelihoods.
- :mod:`protuna.score.pssm` — position-specific frequency matrices built from
  a multiple sequence alignment.
- :mod:`protuna.score.coupling` — pairwise residue coupling scores
"""

from .coupling import categorical_jacobian as categorical_jacobian
from .mlm import batch_pseudo_log_likelihood as batch_pseudo_log_likelihood
from .mlm import masked_marginal as masked_marginal
from .mlm import mutant_marginal as mutant_marginal
from .mlm import pseudo_log_likelihood as pseudo_log_likelihood
from .mlm import ScoreOutput as ScoreOutput
from .mlm import wildtype_marginal as wildtype_marginal
from .mlm import wildtype_marginal_from_logits as wildtype_marginal_from_logits
from .pssm import compute_frequency_matrix as compute_frequency_matrix
from .pssm import compute_msa_frequency_score as compute_msa_frequency_score
