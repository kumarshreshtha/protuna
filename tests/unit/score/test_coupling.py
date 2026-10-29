from collections.abc import Iterator

import pytest
import torch

from protuna import inference
from protuna.data import tokenizer
from protuna.score import coupling
from protuna.typing import SequenceLike


class _InteractionHandler(
    inference.ModelFreeLocalHandler[inference.MLMLogitsRequest, torch.Tensor]
):
    """Return logits from directed interactions with a known categorical contrast."""

    def __init__(
        self,
        weights: torch.Tensor,
        *,
        reverse: bool = False,
        offset: float = 0.0,
        gauge: float = 0.0,
    ) -> None:
        super().__init__(batch_size=7)
        self.weights = weights
        self.reverse = reverse
        self.offset = offset
        self.gauge = gauge
        self.requests: list[inference.MLMLogitsRequest] = []

    def _logits(self, request: inference.MLMLogitsRequest) -> torch.Tensor:
        contrast = self.weights.new_zeros(20)
        contrast[0] = 1
        contrast[1] = -1
        values = contrast[request.token_ids.long()]
        # A->C has contrast -2; all other amino acids have contrast zero.
        signal = (self.weights.T @ values)[:, None] * contrast[None, :]
        baseline = (
            self.offset
            * torch.arange(20, dtype=self.weights.dtype, device=self.weights.device)[
                None, :
            ]
        )
        # Sequence-dependent shifts shared by output amino acids are pure gauge.
        shifts = self.gauge * values[:, None]
        return signal + baseline + shifts

    def __call__(
        self, requests: SequenceLike[inference.MLMLogitsRequest]
    ) -> list[torch.Tensor]:
        self.requests.extend(requests)
        return [self._logits(request) for request in requests]

    def stream(
        self, requests: Iterator[tuple[int, inference.MLMLogitsRequest]]
    ) -> Iterator[tuple[int, inference.MLMLogitsRequest, torch.Tensor]]:
        results = list(super().stream(requests))
        yield from reversed(results) if self.reverse else results


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("token_dtype", [torch.long, torch.uint8])
@pytest.mark.parametrize("logit_dtype", [torch.float32, torch.float64])
def test_categorical_jacobian_matches_analytical_couplings(
    reverse: bool, token_dtype: torch.dtype, logit_dtype: torch.dtype
) -> None:
    """Recover centered, symmetric interactions regardless of completion order."""
    # After symmetrizing and spatial centering, the site weights are
    # [[3, -1, -2], [-1, 4, -3], [-2, -3, 5]]. The categorical outer
    # product has norm 2, giving off-diagonal strengths 2, 4, 6.
    # Row sums 6, 8, 10 and total 24 give APC scores 0, 3/2, 8/3.
    model = _InteractionHandler(
        torch.tensor(
            [[4.0, 4.0, -5.0], [-3.0, 6.0, 3.0], [5.0, -4.0, 8.0]],
            dtype=logit_dtype,
        ),
        reverse=reverse,
    )
    wt_tokens = tokenizer.encode("ACY", dtype=token_dtype)

    scores = coupling.categorical_jacobian(wt_tokens, model)

    expected = torch.tensor(
        [
            [0.0, 0.0, 1.5],
            [0.0, 0.0, 2.6666666666666665],
            [1.5, 2.6666666666666665, 0.0],
        ],
        dtype=logit_dtype,
    )
    torch.testing.assert_close(scores, expected, atol=1e-6, rtol=1e-6)


def test_categorical_jacobian_enumerates_each_single_mutant_once() -> None:
    """Send the wild type once and every non-wild-type amino acid substitution."""
    model = _InteractionHandler(torch.eye(3))
    wt_tokens = tokenizer.encode("ACY")

    coupling.categorical_jacobian(wt_tokens, model)

    tokens = torch.stack([request.token_ids for request in model.requests])
    torch.testing.assert_close(tokens[0], wt_tokens)
    assert tokens.shape == (58, 3)
    assert torch.unique(tokens, dim=0).shape == (58, 3)
    assert torch.all((tokens[1:] != wt_tokens).sum(dim=1) == 1)
    counts = torch.nn.functional.one_hot(tokens[1:], num_classes=20).sum(dim=0)
    expected_counts = torch.ones(3, 20, dtype=torch.long)
    expected_counts[0, 0] = 38
    expected_counts[1, 1] = 38
    expected_counts[2, 19] = 38
    torch.testing.assert_close(counts, expected_counts)
    assert all(request.aa_tokens_only for request in model.requests)
    assert all(not request.masked_positions_only for request in model.requests)


def test_categorical_jacobian_preserves_wild_type_tokens() -> None:
    """Leave the caller's sequence unchanged while generating mutants."""
    model = _InteractionHandler(torch.eye(3))
    wt_tokens = tokenizer.encode("ACY")

    coupling.categorical_jacobian(wt_tokens, model)

    torch.testing.assert_close(wt_tokens, torch.tensor([0, 1, 19]))


@pytest.mark.parametrize(("offset", "gauge"), [(7.0, 0.0), (0.0, 5.0), (7.0, 5.0)])
def test_categorical_jacobian_removes_baselines_and_logit_gauge(
    offset: float, gauge: float
) -> None:
    """Ignore unchanged logits and mutation-dependent amino-acid-uniform shifts."""
    model = _InteractionHandler(
        torch.tensor([[4.0, 4.0, -5.0], [-3.0, 6.0, 3.0], [5.0, -4.0, 8.0]]),
        offset=offset,
        gauge=gauge,
    )

    scores = coupling.categorical_jacobian(tokenizer.encode("ACY"), model)

    torch.testing.assert_close(
        scores,
        torch.tensor([[0.0, 0.0, 1.5], [0.0, 0.0, 2.6666667], [1.5, 2.6666667, 0.0]]),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("sequence", ["", "A", "AC", "ACY"])
def test_categorical_jacobian_returns_zero_for_constant_logits(sequence: str) -> None:
    """Return finite zeros when mutations have no effect, including empty input."""
    model = _InteractionHandler(torch.zeros(len(sequence), len(sequence)), offset=3.0)

    scores = coupling.categorical_jacobian(tokenizer.encode(sequence), model)

    torch.testing.assert_close(scores, torch.zeros(len(sequence), len(sequence)))


def test_categorical_jacobian_retains_negative_apc_scores() -> None:
    """Keep below-background couplings instead of clipping them to zero."""
    # Centered weights have off-diagonal norms 2, 4, 0. APC gives
    # 2 - 6*2/12 = 1; 4 - 6*4/12 = 2; 0 - 2*4/12 = -2/3.
    model = _InteractionHandler(
        torch.tensor([[3.0, -1.0, -2.0], [-1.0, 1.0, 0.0], [-2.0, 0.0, 2.0]])
    )

    scores = coupling.categorical_jacobian(tokenizer.encode("ACY"), model)

    torch.testing.assert_close(
        scores,
        torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, -0.6666667], [2.0, -0.6666667, 0.0]]),
    )
