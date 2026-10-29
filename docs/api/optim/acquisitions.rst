Acquisition
============

.. automodule:: protuna.optim.acquisitions


Interface
-------------

.. currentmodule:: protuna.optim.acquisitions.base

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Acquisition

Identity
--------

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   IdentityAcquisition


Greedy selection
----------------

.. automodule:: protuna.optim.acquisitions.greedy
.. currentmodule:: protuna.optim.acquisitions.greedy

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Greedy

Thompson sampling
-----------------

.. automodule:: protuna.optim.acquisitions.thompson
.. currentmodule:: protuna.optim.acquisitions.thompson

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ThompsonSampling

Score functions
---------------

.. automodule:: protuna.optim.acquisitions.scores
.. currentmodule:: protuna.optim.acquisitions.scores

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ScoreFunction
   UCBScore
   EIScore

K-means batch selection
-----------------------

.. automodule:: protuna.optim.acquisitions.kmeans
.. currentmodule:: protuna.optim.acquisitions.kmeans

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   KMBBO

Type aliases
~~~~~~~~~~~~

.. autotype:: protuna.optim.acquisitions.kmeans.EmbeddingModelLike

Determinantal point processes
-----------------------------

.. automodule:: protuna.optim.acquisitions.dpp
.. currentmodule:: protuna.optim.acquisitions.dpp

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   DPP
   DiversityKernel
   EmbeddingRBFKernel
   EnsembleCovarianceKernel
   ComposeKernel

Type aliases
~~~~~~~~~~~~

.. autotype:: protuna.optim.acquisitions.dpp.EmbeddingModelLike


Common
-------

.. currentmodule:: protuna.optim.acquisitions.base

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   feasibility
