"""Tests for the masked-LM zero-shot scorers in protuna.score.mlm."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import functools
import math

from jaxtyping import Float
import pytest
import torch
from torch import nn

from protuna.score import mlm

_VOCAB_SIZE = 6
_MASK_TOKEN_ID = 5

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FakeOutput:
    """Minimal :class:`~protuna.typing.HasLogits` value returned by the fake model."""

    logits: Float[torch.Tensor, "b l v"]


class _FakeMLM(nn.Module):
    """Fake masked LM returning configured logits and recording every input.

    The inputs the model was called with are exposed via :attr:`calls` so tests
    can assert exactly which tokens a scoring function fed to ``forward``.
    """

    def __init__(
        self,
        logits: Float[torch.Tensor, "1 l v"] | None = None,
        logits_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        """Configures how logits are produced on each forward call.

        Args:
            logits: Per-position logits of shape ``(1, L, V)``, broadcast to
                every row of the input batch. The model output is then
                independent of the input tokens.
            logits_fn: Callable mapping input tokens to a logits tensor; takes
                precedence over ``logits`` when provided.
        """
        super().__init__()
        self._logits = logits
        self._logits_fn = logits_fn
        self.calls: list[torch.Tensor] = []

    def forward(self, tokens: torch.Tensor) -> _FakeOutput:
        """Records the input and returns the configured logits."""
        self.calls.append(tokens.clone())
        if self._logits_fn is not None:
            return _FakeOutput(logits=self._logits_fn(tokens))
        assert self._logits is not None, "one of logits or logits_fn is required"
        return _FakeOutput(logits=self._logits.expand(tokens.shape[0], -1, -1))


def _token_dependent_logits(tokens: torch.Tensor) -> torch.Tensor:
    """Logits derived deterministically from the input tokens.

    The value at ``(b, l, v)`` depends only on the token at ``(b, l)``. This
    mimics the input-dependence of a real masked LM, so implementations that
    share logits across masking patterns are exercised against a model whose
    output actually reacts to the masking pattern.
    """
    one_hot = torch.nn.functional.one_hot(tokens, num_classes=_VOCAB_SIZE).float()
    positions = torch.arange(tokens.shape[1], dtype=torch.float)[None, :, None]
    return one_hot * 2.0 + positions * 0.05


def _flat_logits(tokens: torch.Tensor) -> torch.Tensor:
    """All-zero logits, giving a uniform distribution over the vocabulary."""
    return torch.zeros(tokens.shape[0], tokens.shape[1], _VOCAB_SIZE)


def _mask_favouring_logits(tokens: torch.Tensor) -> torch.Tensor:
    """Logits placing almost all probability mass on the mask token."""
    logits = torch.full((tokens.shape[0], tokens.shape[1], _VOCAB_SIZE), -1e3)
    logits[..., _MASK_TOKEN_ID] = 0.0
    return logits


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def position_logits() -> torch.Tensor:
    """Logits of shape ``(1, 4, 6)`` that differ from position to position.

    Because ``log_softmax(x)[i] - log_softmax(x)[j] == x[i] - x[j]``, a marginal
    score reduces to the plain logit difference at each mutated position, which
    keeps the expected scores exact and readable.
    """
    return torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 0.0],
                [1.0, 0.0, 1.5, 2.0, 2.0, 0.0],
                [0.0, 0.5, 1.0, 1.5, 2.0, 0.0],
                [2.0, 1.0, 0.0, 1.0, 2.0, 0.0],
            ]
        ]
    )


@pytest.fixture
def prob_logits() -> torch.Tensor:
    """Logits of shape ``(1, 4, 6)`` whose softmax is a known distribution.

    Every position carries the same probabilities, so a token is worth the same
    wherever it appears: token 4 has probability 1/2, token 1 has 1/4, and the
    four remaining tokens have 1/16 each.
    """
    probabilities = torch.tensor([1 / 16, 1 / 4, 1 / 16, 1 / 16, 1 / 2, 1 / 16])
    return probabilities.log().expand(1, 4, _VOCAB_SIZE).contiguous()


@pytest.fixture
def wt_tokens() -> torch.Tensor:
    """Wild-type sequence of shape ``(1, 4)``."""
    return torch.tensor([[0, 1, 2, 3]])


@pytest.fixture
def single_mutant() -> torch.Tensor:
    """One mutant of shape ``(1, 4)`` carrying token 4 in place of token 1."""
    return torch.tensor([[0, 4, 2, 3]])


@pytest.fixture
def double_mutant() -> torch.Tensor:
    """One mutant of shape ``(1, 4)`` mutated at positions 0 and 3."""
    return torch.tensor([[4, 1, 2, 0]])


# Every marginal scorer takes ``(model, wt_tokens, mt_tokens)`` and shares the
# temperature, weight, log-odds and return_logits options.
_MARGINALS = [
    pytest.param(mlm.wildtype_marginal, id="wildtype_marginal"),
    pytest.param(mlm.mutant_marginal, id="mutant_marginal"),
    pytest.param(
        functools.partial(mlm.masked_marginal, mask_token_id=_MASK_TOKEN_ID),
        id="masked_marginal",
    ),
    pytest.param(
        functools.partial(mlm.masked_marginal_inference, mask_token_id=_MASK_TOKEN_ID),
        id="masked_marginal_inference",
    ),
]

_Marginal = Callable[..., mlm.ScoreOutput]


# ---------------------------------------------------------------------------
# Marginal scorers — shared contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scorer", _MARGINALS)
def test_marginal_returns_one_score_per_mutant(
    scorer: _Marginal, position_logits: torch.Tensor, wt_tokens: torch.Tensor
) -> None:
    """A ScoreOutput is returned whose score has shape ``(B,)``."""
    mutants = torch.tensor([[0, 4, 2, 3], [4, 1, 2, 0], [0, 1, 2, 3]])
    model = _FakeMLM(logits=position_logits)

    out = scorer(model, wt_tokens, mutants)

    assert isinstance(out, mlm.ScoreOutput)
    assert out.score.shape == (3,)


@pytest.mark.parametrize("scorer", _MARGINALS)
def test_marginal_withholds_logits_by_default(
    scorer: _Marginal,
    position_logits: torch.Tensor,
    wt_tokens: torch.Tensor,
    single_mutant: torch.Tensor,
) -> None:
    """``ScoreOutput.logits`` is None unless the caller asks for it."""
    model = _FakeMLM(logits=position_logits)

    out = scorer(model, wt_tokens, single_mutant)

    assert out.logits is None


@pytest.mark.parametrize("scorer", _MARGINALS)
def test_marginal_returns_logits_on_request(
    scorer: _Marginal,
    position_logits: torch.Tensor,
    wt_tokens: torch.Tensor,
    single_mutant: torch.Tensor,
) -> None:
    """``return_logits=True`` attaches the forward pass logits to the output."""
    model = _FakeMLM(logits=position_logits)

    out = scorer(model, wt_tokens, single_mutant, return_logits=True)

    assert out.logits is not None
    assert out.logits.shape[-2:] == (4, _VOCAB_SIZE)


@pytest.mark.parametrize("scorer", _MARGINALS)
def test_marginal_scores_the_wild_type_as_zero(
    scorer: _Marginal, position_logits: torch.Tensor, wt_tokens: torch.Tensor
) -> None:
    """A mutant identical to the wild-type has no mutated position to score."""
    mutants = wt_tokens.expand(2, -1).contiguous()
    model = _FakeMLM(logits=position_logits)

    out = scorer(model, wt_tokens, mutants)

    torch.testing.assert_close(out.score, torch.zeros(2))


@pytest.mark.parametrize("scorer", _MARGINALS)
def test_marginal_scores_a_single_mutation_by_its_log_prob_ratio(
    scorer: _Marginal,
    position_logits: torch.Tensor,
    wt_tokens: torch.Tensor,
    single_mutant: torch.Tensor,
) -> None:
    """Token 1 to token 4 at position 1 is worth ``2.0 - 0.0``."""
    model = _FakeMLM(logits=position_logits)

    out = scorer(model, wt_tokens, single_mutant)

    torch.testing.assert_close(out.score, torch.tensor([2.0]))


@pytest.mark.parametrize("scorer", _MARGINALS)
def test_marginal_sums_over_mutated_positions(
    scorer: _Marginal,
    position_logits: torch.Tensor,
    wt_tokens: torch.Tensor,
    double_mutant: torch.Tensor,
) -> None:
    """A double mutant scores ``4.0`` at position 0 plus ``1.0`` at position 3."""
    model = _FakeMLM(logits=position_logits)

    out = scorer(model, wt_tokens, double_mutant)

    torch.testing.assert_close(out.score, torch.tensor([5.0]))


@pytest.mark.parametrize("scorer", _MARGINALS)
def test_marginal_weight_scales_each_position(
    scorer: _Marginal,
    position_logits: torch.Tensor,
    wt_tokens: torch.Tensor,
    single_mutant: torch.Tensor,
) -> None:
    """Doubling the weight of the only mutated position doubles the score."""
    weight = torch.tensor([1.0, 2.0, 1.0, 1.0])
    model = _FakeMLM(logits=position_logits)

    out = scorer(model, wt_tokens, single_mutant, weight=weight)

    torch.testing.assert_close(out.score, torch.tensor([4.0]))


@pytest.mark.parametrize("scorer", _MARGINALS)
def test_marginal_temperatures_scale_each_side_independently(
    scorer: _Marginal,
    prob_logits: torch.Tensor,
    wt_tokens: torch.Tensor,
    single_mutant: torch.Tensor,
) -> None:
    """The mutant and wild-type terms are read off separately tempered softmaxes."""
    # The mutant residue keeps p = 1/2 at mt_temp = 1. Halving wt_temp squares
    # and renormalises the distribution, taking the wild-type residue from
    # p = 1/4 to (1/4)^2 / sum(p^2) = (1/16) / (21/64) = 4/21.
    model = _FakeMLM(logits=prob_logits)

    out = scorer(model, wt_tokens, single_mutant, wt_temp=0.5, mt_temp=1.0)

    torch.testing.assert_close(out.score, torch.tensor([math.log(21 / 8)]))


@pytest.mark.parametrize("scorer", _MARGINALS)
def test_marginal_log_odds_replaces_log_probabilities(
    scorer: _Marginal,
    prob_logits: torch.Tensor,
    wt_tokens: torch.Tensor,
    single_mutant: torch.Tensor,
) -> None:
    """``use_log_odds`` scores ``log(p / (1 - p))`` at the mutated position."""
    # Token 4 has p = 1/2 and token 1 has p = 1/4, so the log-odds difference is
    # log(1) - log(1/3) = log(3), where log-probabilities would give log(2).
    model = _FakeMLM(logits=prob_logits)

    out = scorer(model, wt_tokens, single_mutant, use_log_odds=True)

    torch.testing.assert_close(out.score, torch.tensor([math.log(3)]))


# ---------------------------------------------------------------------------
# wildtype_marginal
# ---------------------------------------------------------------------------


def test_wildtype_marginal_runs_one_forward_pass_on_the_wild_type(
    position_logits: torch.Tensor, wt_tokens: torch.Tensor
) -> None:
    """The mutants never reach the model; only the wild-type is encoded."""
    mutants = torch.tensor([[0, 4, 2, 3], [4, 1, 2, 0]])
    model = _FakeMLM(logits=position_logits)

    mlm.wildtype_marginal(model, wt_tokens, mutants)

    assert len(model.calls) == 1
    torch.testing.assert_close(model.calls[0], wt_tokens)


# ---------------------------------------------------------------------------
# wildtype_marginal_from_logits
# ---------------------------------------------------------------------------


def test_wildtype_marginal_from_logits_matches_the_model_based_scorer(
    position_logits: torch.Tensor, wt_tokens: torch.Tensor
) -> None:
    """Pre-computed wild-type logits give the same scores as a forward pass."""
    mutants = torch.tensor([[0, 4, 2, 3], [4, 1, 2, 0]])
    model = _FakeMLM(logits=position_logits)

    expected = mlm.wildtype_marginal(model, wt_tokens, mutants)
    out = mlm.wildtype_marginal_from_logits(position_logits, wt_tokens, mutants)

    torch.testing.assert_close(out.score, expected.score)


def test_wildtype_marginal_from_logits_returns_the_given_logits(
    position_logits: torch.Tensor,
    wt_tokens: torch.Tensor,
    single_mutant: torch.Tensor,
) -> None:
    """``return_logits=True`` echoes back the logits it was handed."""
    out = mlm.wildtype_marginal_from_logits(
        position_logits, wt_tokens, single_mutant, return_logits=True
    )

    torch.testing.assert_close(out.logits, position_logits)


# ---------------------------------------------------------------------------
# mutant_marginal
# ---------------------------------------------------------------------------


def test_mutant_marginal_runs_one_forward_pass_on_the_mutants(
    position_logits: torch.Tensor, wt_tokens: torch.Tensor
) -> None:
    """The unmasked mutant sequences are what the model sees."""
    mutants = torch.tensor([[0, 4, 2, 3], [4, 1, 2, 0]])
    model = _FakeMLM(logits=position_logits)

    mlm.mutant_marginal(model, wt_tokens, mutants)

    assert len(model.calls) == 1
    torch.testing.assert_close(model.calls[0], mutants)


def test_mutant_marginal_scores_each_mutant_under_its_own_context(
    wt_tokens: torch.Tensor,
) -> None:
    """Row ``i`` of the logits scores mutant ``i``, not a shared wild-type row."""
    # _token_dependent_logits gives a token a logit of 2.0 (plus a position
    # offset) where it appears, so each mutant sees its own residue favoured.
    mutants = torch.tensor([[0, 4, 2, 3], [0, 3, 2, 3]])
    model = _FakeMLM(logits_fn=_token_dependent_logits)

    out = mlm.mutant_marginal(model, wt_tokens, mutants)

    # Mutant 0 carries token 4 at position 1, mutant 1 carries token 3 there;
    # in both cases the mutant residue leads the wild-type residue by 2.0.
    torch.testing.assert_close(out.score, torch.tensor([2.0, 2.0]))


# ---------------------------------------------------------------------------
# masked_marginal
# ---------------------------------------------------------------------------


def test_masked_marginal_masks_only_the_mutated_positions(
    position_logits: torch.Tensor, wt_tokens: torch.Tensor
) -> None:
    """Mutated positions are blanked and the rest of the sequence is untouched."""
    mutants = torch.tensor([[0, 4, 2, 3], [0, 1, 4, 4]])
    model = _FakeMLM(logits=position_logits)

    mlm.masked_marginal(model, wt_tokens, mutants, mask_token_id=_MASK_TOKEN_ID)

    assert len(model.calls) == 1
    expected = torch.tensor(
        [[0, _MASK_TOKEN_ID, 2, 3], [0, 1, _MASK_TOKEN_ID, _MASK_TOKEN_ID]]
    )
    torch.testing.assert_close(model.calls[0], expected)


# ---------------------------------------------------------------------------
# masked_marginal_inference
# ---------------------------------------------------------------------------


def test_masked_marginal_inference_matches_masked_marginal(
    wt_tokens: torch.Tensor,
) -> None:
    """Deduplicating masking patterns does not change the scores."""
    mutants = torch.tensor([[0, 4, 2, 3], [0, 3, 2, 3], [4, 1, 2, 0]])
    reference_model = _FakeMLM(logits_fn=_token_dependent_logits)
    dedup_model = _FakeMLM(logits_fn=_token_dependent_logits)

    expected = mlm.masked_marginal(
        reference_model, wt_tokens, mutants, mask_token_id=_MASK_TOKEN_ID
    )
    out = mlm.masked_marginal_inference(
        dedup_model, wt_tokens, mutants, mask_token_id=_MASK_TOKEN_ID
    )

    torch.testing.assert_close(out.score, expected.score)


def test_masked_marginal_inference_encodes_each_masking_pattern_once(
    position_logits: torch.Tensor, wt_tokens: torch.Tensor
) -> None:
    """Mutants sharing a set of mutated positions are forwarded as one row."""
    # The first two mutants differ only in the residue at position 1, so both
    # collapse onto the same masked input; the third masks position 0 instead.
    mutants = torch.tensor([[0, 4, 2, 3], [0, 3, 2, 3], [4, 1, 2, 3]])
    model = _FakeMLM(logits=position_logits)

    mlm.masked_marginal_inference(
        model, wt_tokens, mutants, mask_token_id=_MASK_TOKEN_ID
    )

    expected = torch.tensor([[0, _MASK_TOKEN_ID, 2, 3], [_MASK_TOKEN_ID, 1, 2, 3]])
    torch.testing.assert_close(torch.cat(model.calls), expected)


def test_masked_marginal_inference_batches_patterns_per_forward_call(
    position_logits: torch.Tensor, wt_tokens: torch.Tensor
) -> None:
    """``batch_size`` bounds how many masking patterns share a forward call."""
    mutants = torch.tensor([[0, 4, 2, 3], [4, 1, 2, 3], [0, 1, 4, 3], [0, 1, 2, 4]])
    model = _FakeMLM(logits=position_logits)

    mlm.masked_marginal_inference(
        model, wt_tokens, mutants, mask_token_id=_MASK_TOKEN_ID, batch_size=2
    )

    assert len(model.calls) == 2


def test_masked_marginal_inference_returns_logits_per_mutant(
    position_logits: torch.Tensor, wt_tokens: torch.Tensor
) -> None:
    """Returned logits are expanded back from patterns to one row per mutant."""
    mutants = torch.tensor([[0, 4, 2, 3], [0, 3, 2, 3], [4, 1, 2, 3]])
    model = _FakeMLM(logits=position_logits)

    out = mlm.masked_marginal_inference(
        model,
        wt_tokens,
        mutants,
        mask_token_id=_MASK_TOKEN_ID,
        return_logits=True,
    )

    assert out.logits is not None
    assert out.logits.shape == (3, 4, _VOCAB_SIZE)


# ---------------------------------------------------------------------------
# pseudo_log_likelihood
# ---------------------------------------------------------------------------


def test_pseudo_log_likelihood_returns_one_score_per_sequence() -> None:
    """A ScoreOutput is returned whose score has shape ``(B,)``."""
    sequences = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 1, 2, 2, 3, 3]])
    model = _FakeMLM(logits_fn=_token_dependent_logits)

    out = mlm.pseudo_log_likelihood(
        model, sequences, mask_token_id=_MASK_TOKEN_ID, mask_rate=3
    )

    assert isinstance(out, mlm.ScoreOutput)
    assert out.score.shape == (2,)


def test_pseudo_log_likelihood_masks_every_mask_rate_th_position() -> None:
    """Pass ``i`` masks positions ``i``, ``i + mask_rate``, and so on."""
    sequences = torch.tensor([[0, 1, 2, 3, 4, 0]])
    model = _FakeMLM(logits_fn=_token_dependent_logits)

    mlm.pseudo_log_likelihood(
        model, sequences, mask_token_id=_MASK_TOKEN_ID, mask_rate=3
    )

    expected = torch.tensor(
        [
            [[_MASK_TOKEN_ID, 1, 2, _MASK_TOKEN_ID, 4, 0]],
            [[0, _MASK_TOKEN_ID, 2, 3, _MASK_TOKEN_ID, 0]],
            [[0, 1, _MASK_TOKEN_ID, 3, 4, _MASK_TOKEN_ID]],
        ]
    )
    torch.testing.assert_close(torch.stack(model.calls), expected)


def test_pseudo_log_likelihood_masks_each_position_alone_without_a_mask_rate() -> None:
    """``mask_rate=None`` degrades to one forward pass per position."""
    sequences = torch.tensor([[0, 1, 2, 3, 4, 0]])
    model = _FakeMLM(logits_fn=_token_dependent_logits)

    mlm.pseudo_log_likelihood(
        model, sequences, mask_token_id=_MASK_TOKEN_ID, mask_rate=None
    )

    assert len(model.calls) == 6


def test_pseudo_log_likelihood_sums_the_per_position_log_probabilities() -> None:
    """Uniform logits give every position ``log(1 / V)``."""
    sequences = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 1, 2, 2, 3, 3]])
    model = _FakeMLM(logits_fn=_flat_logits)

    out = mlm.pseudo_log_likelihood(
        model, sequences, mask_token_id=_MASK_TOKEN_ID, mask_rate=3
    )

    torch.testing.assert_close(out.score, torch.full((2,), -6 * math.log(_VOCAB_SIZE)))


# ---------------------------------------------------------------------------
# batch_pseudo_log_likelihood
# ---------------------------------------------------------------------------


def test_batch_pseudo_log_likelihood_matches_the_sequential_implementation() -> None:
    """Batching the masking patterns leaves the scores unchanged."""
    sequences = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 1, 2, 2, 3, 3]])
    sequential_model = _FakeMLM(logits_fn=_token_dependent_logits)
    batched_model = _FakeMLM(logits_fn=_token_dependent_logits)

    expected = mlm.pseudo_log_likelihood(
        sequential_model, sequences, mask_token_id=_MASK_TOKEN_ID, mask_rate=3
    )
    out = mlm.batch_pseudo_log_likelihood(
        batched_model, sequences, mask_token_id=_MASK_TOKEN_ID, mask_rate=3
    )

    torch.testing.assert_close(out.score, expected.score)


def test_batch_pseudo_log_likelihood_forwards_all_patterns_at_once() -> None:
    """Every masking pattern of every sequence goes out in a single call."""
    sequences = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 1, 2, 2, 3, 3]])
    model = _FakeMLM(logits_fn=_token_dependent_logits)

    mlm.batch_pseudo_log_likelihood(
        model, sequences, mask_token_id=_MASK_TOKEN_ID, mask_rate=3
    )

    assert len(model.calls) == 1
    assert model.calls[0].shape == (6, 6)


def test_batch_pseudo_log_likelihood_expands_to_one_pattern_per_position() -> None:
    """``mask_rate=None`` gives a ``B * L`` row batch."""
    sequences = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 1, 2, 2, 3, 3]])
    model = _FakeMLM(logits_fn=_token_dependent_logits)

    mlm.batch_pseudo_log_likelihood(
        model, sequences, mask_token_id=_MASK_TOKEN_ID, mask_rate=None
    )

    assert model.calls[0].shape == (12, 6)


def test_batch_pseudo_log_likelihood_masks_each_position_exactly_once() -> None:
    """The strided patterns together cover every position of the sequence."""
    sequences = torch.tensor([[0, 1, 2, 3, 4, 0]])
    model = _FakeMLM(logits_fn=_token_dependent_logits)

    mlm.batch_pseudo_log_likelihood(
        model, sequences, mask_token_id=_MASK_TOKEN_ID, mask_rate=3
    )

    masked_per_position = (model.calls[0] == _MASK_TOKEN_ID).sum(dim=0)
    torch.testing.assert_close(masked_per_position, torch.ones(6, dtype=torch.long))


def test_batch_pseudo_log_likelihood_scores_the_original_residues() -> None:
    """The score reads ``log p(original token)``, never ``log p(mask token)``."""
    sequences = torch.tensor([[0, 1, 2, 3]])
    model = _FakeMLM(logits_fn=_mask_favouring_logits)

    out = mlm.batch_pseudo_log_likelihood(
        model, sequences, mask_token_id=_MASK_TOKEN_ID, mask_rate=2
    )

    # Every original residue is given a logit 1e3 below the mask token's.
    assert out.score.item() < -1000


def test_batch_pseudo_log_likelihood_sums_the_per_position_log_probabilities() -> None:
    """Uniform logits give every position ``log(1 / V)``."""
    sequences = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 1, 2, 2, 3, 3]])
    model = _FakeMLM(logits_fn=_flat_logits)

    out = mlm.batch_pseudo_log_likelihood(
        model, sequences, mask_token_id=_MASK_TOKEN_ID, mask_rate=3
    )

    torch.testing.assert_close(out.score, torch.full((2,), -6 * math.log(_VOCAB_SIZE)))


def test_batch_pseudo_log_likelihood_withholds_logits_by_default() -> None:
    """``ScoreOutput.logits`` is None unless the caller asks for it."""
    sequences = torch.tensor([[0, 1, 2, 3, 4, 0]])
    model = _FakeMLM(logits_fn=_token_dependent_logits)

    out = mlm.batch_pseudo_log_likelihood(
        model, sequences, mask_token_id=_MASK_TOKEN_ID, mask_rate=3
    )

    assert out.logits is None


def test_batch_pseudo_log_likelihood_returns_logits_on_request() -> None:
    """Requested logits are collapsed back to one row per input sequence."""
    sequences = torch.tensor([[0, 1, 2, 3, 4, 0]])
    model = _FakeMLM(logits_fn=_token_dependent_logits)

    out = mlm.batch_pseudo_log_likelihood(
        model,
        sequences,
        mask_token_id=_MASK_TOKEN_ID,
        mask_rate=3,
        return_logits=True,
    )

    assert out.logits is not None
    assert out.logits.shape == (1, 6, _VOCAB_SIZE)


@pytest.mark.parametrize(("batch_size", "expected_calls"), [(1, 6), (4, 2), (5, 2)])
def test_batch_pseudo_log_likelihood_chunks_the_expanded_batch(
    batch_size: int, expected_calls: int
) -> None:
    """``batch_size`` splits the ``B * mask_rate`` rows across forward calls."""
    sequences = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 1, 2, 2, 3, 3]])
    model = _FakeMLM(logits_fn=_token_dependent_logits)

    mlm.batch_pseudo_log_likelihood(
        model,
        sequences,
        mask_token_id=_MASK_TOKEN_ID,
        mask_rate=3,
        batch_size=batch_size,
    )

    assert len(model.calls) == expected_calls


@pytest.mark.parametrize("batch_size", [1, 4, 5])
def test_batch_pseudo_log_likelihood_chunking_preserves_scores(
    batch_size: int,
) -> None:
    """Chunked forward passes score the same as a single expanded batch."""
    sequences = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 1, 2, 2, 3, 3]])
    whole_model = _FakeMLM(logits_fn=_token_dependent_logits)
    chunked_model = _FakeMLM(logits_fn=_token_dependent_logits)

    expected = mlm.batch_pseudo_log_likelihood(
        whole_model, sequences, mask_token_id=_MASK_TOKEN_ID, mask_rate=3
    )
    out = mlm.batch_pseudo_log_likelihood(
        chunked_model,
        sequences,
        mask_token_id=_MASK_TOKEN_ID,
        mask_rate=3,
        batch_size=batch_size,
    )

    torch.testing.assert_close(out.score, expected.score)
