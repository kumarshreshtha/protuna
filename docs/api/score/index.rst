Scoring
=======

.. automodule:: protuna.score

Masked language model likelihoods
-----------------------------------------

.. automodule:: protuna.score.mlm
.. currentmodule:: protuna.score.mlm

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   wildtype_marginal
   mutant_marginal
   masked_marginal
   pseudo_log_likelihood
   wildtype_marginal_from_logits
   masked_marginal_inference
   batch_pseudo_log_likelihood
   ScoreOutput

Type interfaces
~~~~~~~~~~~~~~~

.. autoclass:: protuna.score.mlm.HasLogits
   :members:
   :undoc-members:

.. autoclass:: protuna.score.mlm.LogitsModel
   :members:
   :undoc-members:

Position-specific scoring matrices (PSSM)
------------------------------------------

.. automodule:: protuna.score.pssm
.. currentmodule:: protuna.score.pssm

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   compute_frequency_matrix
   compute_msa_frequency_score

Coupling
-----------------

.. automodule:: protuna.score.coupling
.. currentmodule:: protuna.score.coupling

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   categorical_jacobian
