Distributed serving
===================

.. automodule:: protuna.inference.serve

API
--------------------

.. automodule:: protuna.inference.serve.api
.. currentmodule:: protuna.inference.serve.api

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   serve_queue_pool
   serve_routed_pool
   execute_parallel


Model pools
-----------

.. automodule:: protuna.inference.serve.pool
.. currentmodule:: protuna.inference.serve.pool

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   RayActorOptions
   ReplicaGroup
   ModelPool
   RoutedPool
   QueuePool


Type aliases
~~~~~~~~~~~~~~

.. autotype:: Router

Request routing
---------------

.. automodule:: protuna.inference.serve.routing
.. currentmodule:: protuna.inference.serve.routing

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   RoundRobinRouter
   MaskAffinityRouter

Ray actors
----------

.. automodule:: protuna.inference.serve.actors
.. currentmodule:: protuna.inference.serve.actors

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   PushWorker
   Broker
   PullWorker
