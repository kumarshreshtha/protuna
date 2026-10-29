"""Acquisition functions to select from proposed candidates.

.. note::

    The current API supports pointwise acquisitions and acquisitions that use samples
    from the joint posterior, such as Monte Carlo acquisitions. For acquisitions that
    need direct access to the joint posterior, set ``objective=None`` in the pipeline
    and pass the trained Gaussian process directly to the acquisition.
"""

from .base import Acquisition as Acquisition
from .base import feasibility as feasibility
from .base import IdentityAcquisition as IdentityAcquisition
from .dpp import DiversityKernel as DiversityKernel
from .dpp import DPP as DPP
from .greedy import Greedy as Greedy
from .kmeans import KMBBO as KMBBO
from .scores import EIScore as EIScore
from .scores import ScoreFunction as ScoreFunction
from .scores import UCBScore as UCBScore
from .thompson import ThompsonSampling as ThompsonSampling
