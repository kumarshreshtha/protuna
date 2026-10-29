"""Definitions of search spaces and strategies to divide them into subspaces.

A :class:`~protuna.optim.spaces.search.SearchSpace` pairs a starting sequence with
the mutations allowed on it, built from the composable rules in
:mod:`~protuna.optim.spaces.rules`.
:class:`~protuna.optim.spaces.subspaces.SearchSpaceSampler` divides a single search
space into smaller subspaces that can be independently processed by a
:class:`~protuna.optim.samplers.base.Sampler` to propose candidates.

The :class:`~protuna.optim.spaces.design.DesignSpace` is a collection of one or more
initial search spaces along with an optional subspace sampling strategy.
"""

from .design import DesignSpace as DesignSpace
from .search import SearchSpace as SearchSpace
from .subspaces import CouplingMaskSampler as CouplingMaskSampler
from .subspaces import LogitsEntropyMaskSampler as LogitsEntropyMaskSampler
from .subspaces import MSAEntropyMaskSampler as MSAEntropyMaskSampler
from .subspaces import PartitionSampler as PartitionSampler
from .subspaces import RandomMaskSampler as RandomMaskSampler
from .subspaces import SearchSpaceSampler as SearchSpaceSampler
