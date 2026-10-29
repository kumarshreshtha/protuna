Spaces
========================

.. automodule:: protuna.optim.spaces

Search space
-------------

.. automodule:: protuna.optim.spaces.search
.. currentmodule:: protuna.optim.spaces.search

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   SearchSpace

Type aliases
~~~~~~~~~~~~

.. autotype:: protuna.optim.spaces.search.SearchRule

.. autotype:: protuna.optim.spaces.search.SearchMask

Design space
-------------

.. automodule:: protuna.optim.spaces.design
.. currentmodule:: protuna.optim.spaces.design

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   DesignSpace

Subspace sampling
-----------------

.. automodule:: protuna.optim.spaces.subspaces
.. currentmodule:: protuna.optim.spaces.subspaces

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   SearchSpaceSampler
   PartitionSampler
   RandomMaskSampler
   MSAEntropyMaskSampler
   LogitsEntropyMaskSampler
   CouplingMaskSampler
   SiteSelector
   random_site_selector


Type aliases
~~~~~~~~~~~~

.. autotype:: protuna.optim.spaces.subspaces.LogitsModelLike

Mutation rules
--------------

.. automodule:: protuna.optim.spaces.rules
.. currentmodule:: protuna.optim.spaces.rules

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   AllowOnly
   Allow
   Deny
   FixSites
   DenyResidues
   Preserve
   MSAConserved
   MSACovered

Type aliases
~~~~~~~~~~~~

.. autotype:: protuna.optim.spaces.rules.SearchMask
