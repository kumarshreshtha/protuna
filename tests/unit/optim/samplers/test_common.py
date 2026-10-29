from __future__ import annotations

import pytest
import torch

from protuna import inference
from protuna.optim import spaces
from protuna.optim.samplers import base
from protuna.optim.samplers import common
from protuna.typing import SequenceLike
from tests.unit.optim.samplers import _helpers


def test_every_sampler_only_proposes_residues_the_space_allows(
    any_sampler: base.Sampler, shared_space: spaces.SearchSpace
) -> None:
    """Proposals respect the search space mutation constraints."""
    batch = any_sampler.sample([shared_space])[0]

    assert _helpers._all_residues_allowed(batch, shared_space)


@pytest.mark.parametrize(
    ("top_p", "expected"),
    [
        (1.0, [0.5, 0.25, 0.25]),
        (0.7, [0.5, 0.25, 0.0]),
        (0.5, [0.5, 0.0, 0.0]),
    ],
)
def test_mask_above_top_p_keeps_the_nucleus_and_zeroes_the_tail(
    top_p: float, expected: list[float]
) -> None:
    """Residues past the cumulative probability threshold are zeroed out."""
    probs = torch.tensor([[0.5, 0.25, 0.25]])

    mask = common.top_p_mask(probs, top_p)

    assert (probs * mask).tolist() == [expected]


def test_modify_logits_applies_top_p_without_mutating_the_input() -> None:
    """Remove tail residues from the returned logits only."""
    logits = torch.tensor([[0.6, 0.3, 0.1]]).log()
    original = logits.clone()

    result = common.modify_logits(
        logits, torch.ones(1, 3, dtype=torch.bool), temperature=1.0, top_p=0.5
    )

    torch.testing.assert_close(result.softmax(-1), torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.equal(logits, original)


class _TensorBatchMLM(
    inference.ModelFreeLocalHandler[inference.MLMLogitsRequest, torch.Tensor]
):
    def __call__(
        self, requests: SequenceLike[inference.MLMLogitsRequest]
    ) -> torch.Tensor:
        return torch.stack(_helpers._FakeMLM({"W": 1.0})(requests))


def test_mlm_batch_accepts_tensor_responses() -> None:
    """Accept an entire logits tensor as the handler response batch."""
    model = _TensorBatchMLM()
    batch = torch.tensor([[_helpers._MASK], [_helpers._MASK]])

    responses = common.execute_mlm_batch(batch, model)

    assert len(responses) == 2
    assert responses[0].shape == torch.Size([1, 20])
    assert responses[0].argmax(-1).tolist() == [18]
    torch.testing.assert_close(responses[0], responses[1])
