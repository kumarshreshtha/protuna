"""Shared fixtures for sampler tests."""

from __future__ import annotations

import pytest
import torch

from protuna.optim import spaces
from protuna.optim.samplers import base
from protuna.optim.samplers import baselines
from protuna.optim.samplers import beam
from protuna.optim.samplers import maskgit
from protuna.optim.samplers import mlm
from protuna.optim.samplers import pssm
from protuna.optim.samplers import rlxf
from tests.unit.optim.samplers import _helpers

_EVEN_PROBS = {"A": 0.2, "C": 0.2, "D": 0.2, "W": 0.2, "Y": 0.2}


_NUM_SAMPLES = 25


@pytest.fixture
def shared_space() -> spaces.SearchSpace:
    """Space with three open sites and two positions pinned to the wild type."""
    return _helpers._space("ACDEF", {0: "AWY", 1: "CWY", 2: "DWY"})


@pytest.fixture(
    params=[
        lambda: baselines.RandomSampler(max_samples=_NUM_SAMPLES),
        lambda: baselines.SiteSaturationSampler(),
        lambda: baselines.BruteForceSampler(),
        lambda: pssm.PSSMSampler(torch.zeros(5, 20), _NUM_SAMPLES),
        lambda: mlm.MLMSampler(_helpers._FakeMLM(_EVEN_PROBS), _NUM_SAMPLES),
        lambda: beam.MLMBeamSampler(_helpers._FakeMLM(_EVEN_PROBS), _NUM_SAMPLES),
        lambda: rlxf.RLXFSampler(_helpers._FakeMLM(_EVEN_PROBS), _NUM_SAMPLES),
        lambda: maskgit.MaskGITSampler(_helpers._FakeMLM(_EVEN_PROBS), _NUM_SAMPLES),
    ],
    ids=[
        "random",
        "site_saturation",
        "brute_force",
        "pssm",
        "mlm",
        "mlm_beam",
        "rlxf",
        "maskgit",
    ],
)
def any_sampler(request: pytest.FixtureRequest) -> base.Sampler:
    """A fresh instance of each concrete sampler in the module."""
    return request.param()
