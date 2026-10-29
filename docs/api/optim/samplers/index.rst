Samplers
========

.. automodule:: protuna.optim.samplers

.. toctree::
   :hidden:
   :maxdepth: 1

   maskgit
   remdm
   smc

Interfaces
------------------

.. automodule:: protuna.optim.samplers.base
.. currentmodule:: protuna.optim.samplers.base

.. autosummary::
   :toctree: ../../generated
   :nosignatures:

   Sampler
   MaskedDiffusionSamplerBase

Type aliases
~~~~~~~~~~~~

.. autotype:: protuna.optim.samplers.base.SamplerLike

Baselines
-----------------

.. automodule:: protuna.optim.samplers.baselines
.. currentmodule:: protuna.optim.samplers.baselines

.. autosummary::
   :toctree: ../../generated
   :nosignatures:

   RandomSampler
   SiteSaturationSampler
   BruteForceSampler

PSSM
-------------

.. automodule:: protuna.optim.samplers.pssm
.. currentmodule:: protuna.optim.samplers.pssm

.. autosummary::
   :toctree: ../../generated
   :nosignatures:

   PSSMSampler

Masked language model
------------------------------

.. automodule:: protuna.optim.samplers.mlm
.. currentmodule:: protuna.optim.samplers.mlm

.. autosummary::
   :toctree: ../../generated
   :nosignatures:

   MLMSampler

Iterative Refinement
---------------------

Fill all mutable sites one or few at a time, conditioning each step on the sites filled before.

.. currentmodule:: protuna.optim.samplers

.. container:: joined-autosummaries

   .. autosummary::
      :toctree: ../../generated
      :nosignatures:

      MLMBeamSampler
      RLXFSampler

   .. autosummary::
      :nosignatures:

      ~maskgit.MaskGITSampler
      ~remdm.ReMDMSampler

See :doc:`maskgit` for decoding strategies. See :doc:`remdm` for noise and remasking schedules


Reward Guided
---------------

Sample from a reward tilted distribution balancing "naturalness" with high fitness of candidates.

.. currentmodule:: protuna.optim.samplers

.. autosummary::
   :nosignatures:

   ~remdm.ReMDMSampler
   ~smc.FKSteeringSampler
   ~smc.NestedSMCSampler

See :doc:`remdm` for guidance models and :doc:`smc` for soft value functions and steering utilities.

Sampling utilities
------------------

.. automodule:: protuna.optim.samplers.common
.. currentmodule:: protuna.optim.samplers.common

.. autosummary::
   :toctree: ../../generated
   :nosignatures:

   sample_from_weights
   top_p_mask
   modify_logits
   sample_from_logits
   sample_from_mlm_response
   execute_mlm_batch
