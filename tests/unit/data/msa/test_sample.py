"""Tests for MSA sampling strategies in protuna.data.msa.sample."""

from __future__ import annotations

from collections.abc import Callable
import pathlib

import pytest
import torch

from protuna.data import tokenizers
from protuna.data.msa import core
from protuna.data.msa import sample

_TOKENIZER = tokenizers.ProteinTokenizer()
_GAP = _TOKENIZER.gap_token_id


def _make_msa(*sequences: str) -> core.TokenizedMSA:
    tokens = torch.stack(_TOKENIZER.encode(list(sequences)), dim=0)
    return core.TokenizedMSA(
        msa_tokens=tokens, valid_cols=torch.ones(tokens.shape[1], dtype=torch.bool)
    )


def _rows(tokens: torch.Tensor) -> set[tuple[int, ...]]:
    """The rows of a token tensor as a set of hashable tuples."""
    return {tuple(row.tolist()) for row in tokens}


@pytest.fixture
def small_msa() -> core.TokenizedMSA:
    """4-sequence MSA with distinct, descending identities to the query.

    query  ACDE  identity 1.00
    seq1   ACDF  identity 0.75  (A, C, D match; E→F)
    seq2   AFGE  identity 0.50  (A, E match; C→F, D→G)
    seq3   MMME  identity 0.25  (only E matches)
    """
    return _make_msa("ACDE", "ACDF", "AFGE", "MMME")


@pytest.fixture
def a3m_msa(a3m_file: pathlib.Path) -> core.TokenizedMSA:
    """TokenizedMSA loaded from the test alignment.a3m fixture file."""
    return core.TokenizedMSA.from_file(a3m_file, format="a3m")


# ---------------------------------------------------------------------------
# MSASamplingMethod
# ---------------------------------------------------------------------------


def test_msa_sampler_values() -> None:
    """All enum values match their expected string literals."""
    assert sample.MSASamplingMethod.Greedy == "greedy"
    assert sample.MSASamplingMethod.Random == "random"
    assert sample.MSASamplingMethod.Weighted == "weighted"
    assert sample.MSASamplingMethod.Stratified == "stratified"


# ---------------------------------------------------------------------------
# greedy_sample_msa
# ---------------------------------------------------------------------------


def test_greedy_sample_msa_output_shape(small_msa: core.TokenizedMSA) -> None:
    """Output tensor has shape (max_sequences, L)."""
    result = sample.greedy_sample_msa(small_msa, max_sequences=4)
    assert result.shape == (4, 4)


def test_greedy_sample_msa_query_is_first(small_msa: core.TokenizedMSA) -> None:
    """Query sequence is always the first row of the result."""
    result = sample.greedy_sample_msa(small_msa, max_sequences=2)
    torch.testing.assert_close(result[0], small_msa.msa_tokens[0])


def test_greedy_sample_msa_max_sequences_limits_output(
    small_msa: core.TokenizedMSA,
) -> None:
    """Output contains at most max_sequences rows."""
    result = sample.greedy_sample_msa(small_msa, max_sequences=2)
    assert result.shape[0] == 2


def test_greedy_sample_msa_budget_larger_than_msa_returns_all(
    small_msa: core.TokenizedMSA,
) -> None:
    """When budget exceeds MSA depth, all sequences are returned."""
    result = sample.greedy_sample_msa(small_msa, max_sequences=100)
    assert result.shape[0] == 4


def test_greedy_sample_msa_descending_identity_order(
    small_msa: core.TokenizedMSA,
) -> None:
    """Greedy returns sequences ordered by descending identity to query."""
    # small_msa rows are ordered by descending identity
    # (1.0, 0.75, 0.5, 0.25), so greedy preserves the original order.
    result = sample.greedy_sample_msa(small_msa, max_sequences=4)
    torch.testing.assert_close(result, small_msa.msa_tokens)


def test_greedy_sample_msa_least_similar_excluded_when_budget_tight(
    small_msa: core.TokenizedMSA,
) -> None:
    """Least-similar sequence is dropped when budget is tight."""
    # MMME has identity 0.25 — dropped when max_sequences=3.
    result = sample.greedy_sample_msa(small_msa, max_sequences=3)
    mmme = tuple(small_msa.msa_tokens[3].tolist())
    assert mmme not in _rows(result)


def test_greedy_sample_msa_max_tokens_limits_output(
    small_msa: core.TokenizedMSA,
) -> None:
    """Output respects a token budget when max_tokens is provided."""
    # Each sequence is 4 non-gap tokens; cumsum: 4, 8, 12, 16.
    # max_tokens=8 → first two sequences fit (4 <= 8, 8 <= 8; 12 > 8).
    result = sample.greedy_sample_msa(small_msa, max_tokens=8, count_gap_tokens=False)
    assert result.shape[0] == 2


@pytest.mark.parametrize(
    ("count_gap_tokens", "expected_depth"), [(False, 2), (True, 1)]
)
def test_greedy_sample_msa_count_gap_tokens_changes_budget_accounting(
    count_gap_tokens: bool, expected_depth: int
) -> None:
    """Gap tokens are counted or skipped based on count_gap_tokens."""
    # Query "ACDE" (4 non-gap); homolog "A--E" (2 non-gap, 4 total).
    # max_tokens=6:
    #   count_gap_tokens=False → costs [4, 2]; cumsum=[4, 6] → 2 seqs.
    #   count_gap_tokens=True  → costs [4, 4]; cumsum=[4, 8] → 1 seq.
    msa = core.TokenizedMSA(
        msa_tokens=torch.tensor([[0, 1, 2, 3], [0, _GAP, _GAP, 3]]),
        valid_cols=torch.ones(4, dtype=torch.bool),
    )
    result = sample.greedy_sample_msa(
        msa, max_tokens=6, count_gap_tokens=count_gap_tokens
    )
    assert result.shape[0] == expected_depth


# ---------------------------------------------------------------------------
# random_sample_msa
# ---------------------------------------------------------------------------


def test_random_sample_msa_query_is_first(small_msa: core.TokenizedMSA) -> None:
    """Query sequence is always the first row of the result."""
    torch.manual_seed(0)
    result = sample.random_sample_msa(small_msa, max_sequences=3)
    torch.testing.assert_close(result[0], small_msa.msa_tokens[0])


def test_random_sample_msa_max_sequences_limits_output(
    small_msa: core.TokenizedMSA,
) -> None:
    """Output contains at most max_sequences rows."""
    torch.manual_seed(0)
    result = sample.random_sample_msa(small_msa, max_sequences=2)
    assert result.shape[0] == 2


def test_random_sample_msa_budget_larger_than_msa_returns_all(
    small_msa: core.TokenizedMSA,
) -> None:
    """When budget exceeds MSA depth, all sequences are returned."""
    torch.manual_seed(0)
    result = sample.random_sample_msa(small_msa, max_sequences=100)
    assert result.shape[0] == 4


def test_random_sample_msa_output_rows_are_from_original_msa(
    small_msa: core.TokenizedMSA,
) -> None:
    """Every row in the output is present in the original MSA."""
    torch.manual_seed(0)
    result = sample.random_sample_msa(small_msa, max_sequences=4)
    assert _rows(result) <= _rows(small_msa.msa_tokens)


def test_random_sample_msa_no_duplicate_rows(small_msa: core.TokenizedMSA) -> None:
    """Sampling without replacement produces no duplicate rows."""
    torch.manual_seed(0)
    result = sample.random_sample_msa(small_msa, max_sequences=4)
    assert len(_rows(result)) == 4


def test_random_sample_msa_max_tokens_limits_output(
    small_msa: core.TokenizedMSA,
) -> None:
    """Output respects a token budget when max_tokens is provided."""
    torch.manual_seed(0)
    # 4 non-gap tokens per sequence; max_tokens=4 → only query fits.
    result = sample.random_sample_msa(small_msa, max_tokens=4, count_gap_tokens=False)
    assert result.shape[0] == 1
    torch.testing.assert_close(result[0], small_msa.msa_tokens[0])


# ---------------------------------------------------------------------------
# weighted_sample_msa
# ---------------------------------------------------------------------------


def test_weighted_sample_msa_query_is_first(small_msa: core.TokenizedMSA) -> None:
    """Query sequence is always the first row of the result."""
    torch.manual_seed(0)
    result = sample.weighted_sample_msa(small_msa, max_sequences=3)
    torch.testing.assert_close(result[0], small_msa.msa_tokens[0])


def test_weighted_sample_msa_output_shape(small_msa: core.TokenizedMSA) -> None:
    """Output tensor has shape (max_sequences, L)."""
    torch.manual_seed(0)
    result = sample.weighted_sample_msa(small_msa, max_sequences=4)
    assert result.shape == (4, 4)


def test_weighted_sample_msa_max_sequences_limits_output(
    small_msa: core.TokenizedMSA,
) -> None:
    """Output contains at most max_sequences rows."""
    torch.manual_seed(0)
    result = sample.weighted_sample_msa(small_msa, max_sequences=2)
    assert result.shape[0] == 2


def test_weighted_sample_msa_no_duplicate_rows(small_msa: core.TokenizedMSA) -> None:
    """Multinomial without replacement produces no duplicate rows."""
    torch.manual_seed(0)
    result = sample.weighted_sample_msa(small_msa, max_sequences=4)
    assert len(_rows(result)) == 4


def test_weighted_sample_msa_output_rows_are_from_original_msa(
    small_msa: core.TokenizedMSA,
) -> None:
    """Every row in the output is present in the original MSA."""
    torch.manual_seed(0)
    result = sample.weighted_sample_msa(small_msa, max_sequences=4)
    assert _rows(result) <= _rows(small_msa.msa_tokens)


def test_weighted_sample_msa_max_tokens_limits_output(
    small_msa: core.TokenizedMSA,
) -> None:
    """Output respects a token budget when max_tokens is provided."""
    torch.manual_seed(0)
    result = sample.weighted_sample_msa(small_msa, max_tokens=4, count_gap_tokens=False)
    assert result.shape[0] == 1
    torch.testing.assert_close(result[0], small_msa.msa_tokens[0])


def test_weighted_sample_msa_threshold_parameter_accepted(
    small_msa: core.TokenizedMSA,
) -> None:
    """Threshold parameter is accepted without error."""
    torch.manual_seed(0)
    result = sample.weighted_sample_msa(small_msa, max_sequences=4, threshold=0.5)
    assert result.shape[0] == 4


# ---------------------------------------------------------------------------
# stratified_sample_msa
# ---------------------------------------------------------------------------


def test_stratified_sample_msa_query_is_first(small_msa: core.TokenizedMSA) -> None:
    """Query sequence is always the first row of the result."""
    torch.manual_seed(0)
    result = sample.stratified_sample_msa(
        small_msa, max_sequences=4, seq_identity_buckets=(0.0, 1.0)
    )
    torch.testing.assert_close(result[0], small_msa.msa_tokens[0])


def test_stratified_sample_msa_excludes_sequences_outside_bucket_range(
    a3m_msa: core.TokenizedMSA,
) -> None:
    """Sequences with identity outside all bucket ranges are excluded."""
    # All a3m sequences except homolog7 have identity 1.0 to the query.
    # Default upper bound is 0.98; identity-1.0 sequences are not < 0.98
    # → excluded. Only homolog7 (identity 19/20 = 0.95) falls in (0.8, 0.98).
    result = sample.stratified_sample_msa(a3m_msa, max_sequences=10)
    assert result.shape[0] == 2
    torch.testing.assert_close(result[0], a3m_msa.msa_tokens[0])


def test_stratified_sample_msa_single_wide_bucket_includes_all_non_query_sequences(
    small_msa: core.TokenizedMSA,
) -> None:
    """A single wide bucket captures all non-query sequences."""
    # With one bucket (0.0, 1.0), all three non-query sequences
    # (ids 0.75, 0.5, 0.25) satisfy id >= 0.0 and id < 1.0.
    torch.manual_seed(0)
    result = sample.stratified_sample_msa(
        small_msa, max_sequences=10, seq_identity_buckets=(0.0, 1.0)
    )
    assert result.shape[0] == 4


def test_stratified_sample_msa_max_sequences_limits_output(
    small_msa: core.TokenizedMSA,
) -> None:
    """Output contains at most max_sequences rows."""
    torch.manual_seed(0)
    result = sample.stratified_sample_msa(
        small_msa, max_sequences=2, seq_identity_buckets=(0.0, 1.0)
    )
    assert result.shape[0] <= 2


def test_stratified_sample_msa_max_tokens_limits_output(
    small_msa: core.TokenizedMSA,
) -> None:
    """Output respects a token budget when max_tokens is provided."""
    torch.manual_seed(0)
    result = sample.stratified_sample_msa(
        small_msa,
        max_tokens=4,
        seq_identity_buckets=(0.0, 1.0),
        count_gap_tokens=False,
    )
    assert result.shape[0] == 1
    torch.testing.assert_close(result[0], small_msa.msa_tokens[0])


def test_stratified_sample_msa_output_rows_are_from_original_msa(
    small_msa: core.TokenizedMSA,
) -> None:
    """Every row in the output is present in the original MSA."""
    torch.manual_seed(0)
    result = sample.stratified_sample_msa(
        small_msa, max_sequences=4, seq_identity_buckets=(0.0, 1.0)
    )
    assert _rows(result) <= _rows(small_msa.msa_tokens)


def test_stratified_sample_msa_shuffle_samples_keeps_query_first(
    small_msa: core.TokenizedMSA,
) -> None:
    """Query is still first even when shuffle_samples=True."""
    torch.manual_seed(0)
    result = sample.stratified_sample_msa(
        small_msa,
        max_sequences=4,
        seq_identity_buckets=(0.0, 1.0),
        shuffle_samples=True,
    )
    torch.testing.assert_close(result[0], small_msa.msa_tokens[0])


def test_stratified_sample_msa_round_robin_interleaves_buckets(
    small_msa: core.TokenizedMSA,
) -> None:
    """Round-robin draw produces sequences from every non-empty bucket."""
    # Two buckets: (0.0, 0.6) contains seq2 (0.5) and seq3 (0.25);
    # (0.6, 1.0) contains seq1 (0.75).  Round-robin picks one from each
    # bucket per iteration, so the result contains sequences from both.
    torch.manual_seed(0)
    result = sample.stratified_sample_msa(
        small_msa, max_sequences=10, seq_identity_buckets=(0.0, 0.6, 1.0)
    )
    sampled = _rows(result)
    original = small_msa.msa_tokens
    bucket1 = {tuple(original[1].tolist())}  # ACDF — identity 0.75, in (0.6, 1.0)
    bucket0 = {tuple(original[2].tolist()), tuple(original[3].tolist())}
    assert sampled & bucket1
    assert sampled & bucket0


# ---------------------------------------------------------------------------
# sample_msa (dispatch)
# ---------------------------------------------------------------------------


def test_sample_msa_raises_if_both_limits_given(small_msa: core.TokenizedMSA) -> None:
    """Raises ValueError when both max_sequences and max_tokens are given."""
    with pytest.raises(ValueError, match="exactly one"):
        sample.sample_msa(
            small_msa, sample.MSASamplingMethod.Greedy, max_sequences=2, max_tokens=8
        )


def test_sample_msa_raises_if_no_limit_given(small_msa: core.TokenizedMSA) -> None:
    """Raises ValueError when neither max_sequences nor max_tokens is given."""
    with pytest.raises(ValueError, match="exactly one"):
        sample.sample_msa(small_msa, sample.MSASamplingMethod.Greedy)


def test_sample_msa_raises_for_unknown_strategy(small_msa: core.TokenizedMSA) -> None:
    """Raises ValueError for an unrecognised sampling strategy string."""
    with pytest.raises(ValueError, match="Unknown Sampling Strategy"):
        sample.sample_msa(
            small_msa,
            "unknown_strategy",  # ty: ignore[invalid-argument-type]
            max_sequences=2,
        )


@pytest.mark.parametrize(
    ("strategy", "sampler"),
    [
        (sample.MSASamplingMethod.Greedy, sample.greedy_sample_msa),
        (sample.MSASamplingMethod.Random, sample.random_sample_msa),
        (sample.MSASamplingMethod.Weighted, sample.weighted_sample_msa),
        (sample.MSASamplingMethod.Stratified, sample.stratified_sample_msa),
    ],
)
def test_sample_msa_dispatches_to_strategy(
    strategy: sample.MSASamplingMethod,
    sampler: Callable[..., torch.Tensor],
    small_msa: core.TokenizedMSA,
) -> None:
    """Each MSASamplingMethod value dispatches to its matching sampling function."""
    torch.manual_seed(7)
    expected = sampler(small_msa, max_sequences=3)
    torch.manual_seed(7)
    result = sample.sample_msa(small_msa, strategy, max_sequences=3)
    torch.testing.assert_close(result, expected)


def test_sample_msa_passes_sequence_wt_threshold_to_weighted(
    small_msa: core.TokenizedMSA,
) -> None:
    """sequence_wt_threshold kwarg is forwarded to weighted_sample_msa."""
    torch.manual_seed(0)
    result = sample.sample_msa(
        small_msa,
        sample.MSASamplingMethod.Weighted,
        max_sequences=4,
        sequence_wt_threshold=0.5,
    )
    assert result.shape[0] == 4


def test_sample_msa_passes_seq_identity_buckets_to_stratified(
    small_msa: core.TokenizedMSA,
) -> None:
    """seq_identity_buckets kwarg is forwarded to stratified_sample_msa."""
    torch.manual_seed(0)
    result = sample.sample_msa(
        small_msa,
        sample.MSASamplingMethod.Stratified,
        max_sequences=10,
        seq_identity_buckets=(0.0, 1.0),
    )
    assert result.shape[0] == 4
