Predictors
==========

.. automodule:: protuna.optim.predictors

Interfaces
------------------------

.. automodule:: protuna.optim.predictors.base
.. currentmodule:: protuna.optim.predictors

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   PredictionRequest
   Prediction

Type aliases
~~~~~~~~~~~~

.. autotype:: protuna.optim.predictors.base.PredictorLike

Ensemble
--------

.. automodule:: protuna.optim.predictors.ensemble
.. currentmodule:: protuna.optim.predictors

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   EnsemblePredictor

Calibration
-----------

.. automodule:: protuna.optim.predictors.calibrated
.. currentmodule:: protuna.optim.predictors

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   CalibratedPredictor

Type interfaces
~~~~~~~~~~~~~~~

.. autoclass:: protuna.optim.predictors.calibrated.AssayResult
   :members:
   :undoc-members:


Zero-shot
--------------------

.. automodule:: protuna.optim.predictors.zeroshot
.. currentmodule:: protuna.optim.predictors.zeroshot

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   MaskedMarginalPredictor
   WildTypeMarginalPredictor
   MutantMarginalPredictor
   PSSMZeroshotPredictor
