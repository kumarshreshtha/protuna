from __future__ import annotations

from collections.abc import Sequence
import re

import pytest
import torch

from protuna import inference
from protuna.optim import spaces
from protuna.optim.samplers import base
from protuna.typing import SequenceLike
from tests.unit.optim.samplers import _helpers


class _FakeInferenceModel(inference.InferenceModel):
    """Inference model that records the devices it was asked to load onto."""

    def __init__(self, batch_size: int | None = None):
        super().__init__(batch_size=batch_size)
        self.devices: list[torch.device | Sequence[torch.device] | None] = []

    def forward(self, requests: SequenceLike[object]) -> list[object]:
        return list(requests)

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        self.devices.append(device)

    def release(self) -> None:
        return


class _ModelSampler(base.Sampler):
    """Minimal concrete sampler that echoes each space's init sequence."""

    model: inference.InferenceModel

    def sample(self, spaces: SequenceLike[spaces.SearchSpace]) -> list[torch.Tensor]:
        return [space.init_sequence.unsqueeze(0) for space in spaces]


def test_assigned_inference_models_are_loaded_onto_the_requested_device() -> None:
    """Models assigned as attributes are registered and loaded by `load`."""
    model = _FakeInferenceModel(batch_size=2)
    candidate_sampler = _ModelSampler()
    candidate_sampler.model = model

    candidate_sampler.load(torch.device("cpu"))

    assert model.devices == [torch.device("cpu")]


def test_deleting_a_model_attribute_stops_it_from_being_loaded() -> None:
    """A deleted model attribute is unregistered and never loaded."""
    model = _FakeInferenceModel(batch_size=2)
    candidate_sampler = _ModelSampler()
    candidate_sampler.model = model
    del candidate_sampler.model

    candidate_sampler.load(torch.device("cpu"))

    assert model.devices == []


def test_assigning_a_model_before_init_raises() -> None:
    """Registering a model requires `Sampler.__init__` to have run."""

    class _UninitializedSampler(base.Sampler):
        def __init__(self):
            self.model = _FakeInferenceModel()

        def sample(
            self, spaces: SequenceLike[spaces.SearchSpace]
        ) -> list[torch.Tensor]:
            return []

    with pytest.raises(
        AttributeError,
        match=re.escape("cannot register component before super().__init__() call"),
    ):
        _UninitializedSampler()


def test_calling_a_sampler_returns_one_batch_per_requested_space() -> None:
    """`__call__` fans the requests out to `sample` and keeps their order."""
    spaces = [_helpers._space("ACD", {}), _helpers._space("WYK", {})]

    batches = _ModelSampler()(spaces)

    assert [_helpers._decode(batch) for batch in batches] == [["ACD"], ["WYK"]]
