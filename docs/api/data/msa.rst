Multiple sequence alignment
============================

.. automodule:: protuna.data.msa

Data structures and operations
------------------------------

.. automodule:: protuna.data.msa.core
.. currentmodule:: protuna.data.msa.core

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   MSAFilter
   TokenizedMSA
   get_seq_identity_to_query
   compute_pairwise_identity
   compute_sequence_weights

Sampling MSA
------------------

.. automodule:: protuna.data.msa.sample
.. currentmodule:: protuna.data.msa.sample

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   greedy_sample_msa
   random_sample_msa
   weighted_sample_msa
   stratified_sample_msa
   MSASamplingMethod
   sample_msa

IO
--------------------------

.. automodule:: protuna.data.msa.io
.. currentmodule:: protuna.data.msa.io

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   parse_alignment
   sequences_from_alignment
   iterate_fasta
