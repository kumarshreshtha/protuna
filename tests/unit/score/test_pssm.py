"""Tests for the MSA frequency scorer in protuna.score.pssm."""

from __future__ import annotations

import math
import pathlib

import pytest
import torch

from protuna.data import tokenizers
from protuna.data.msa import core
from protuna.score import pssm

_TOKENIZER = tokenizers.tokenizer

# Token ids follow the alphabet "ACDEFGHIKLMNPQRSTVWY-_^*?":
# A=0  C=1  D=2  E=3  F=4 ... W=18  Y=19  gap=20
_GAP_COLUMN = 20

# Wild-type "ACDE" and the two mutants used across the scoring tests.
_WT = torch.tensor([[0, 1, 2, 3]])
_MT_E_TO_F = torch.tensor([[0, 1, 2, 4]])
_MT_A_TO_W = torch.tensor([[18, 1, 2, 3]])


def _make_msa(*sequences: str) -> core.TokenizedMSA:
    """Build an unfiltered TokenizedMSA from aligned sequence strings."""
    tokens = torch.stack(_TOKENIZER.encode(list(sequences)), dim=0)
    return core.TokenizedMSA(
        msa_tokens=tokens, valid_cols=torch.ones(tokens.shape[1], dtype=torch.bool)
    )


@pytest.fixture
def two_thirds_msa() -> core.TokenizedMSA:
    """An MSA whose last column holds E twice and F once."""
    return _make_msa("ACDE", "ACDE", "ACDF")


@pytest.fixture
def freq_matrix(two_thirds_msa: core.TokenizedMSA) -> torch.Tensor:
    """Frequency matrix in which E scores 2/3 and F scores 1/3 at position 3."""
    return pssm.compute_frequency_matrix(two_thirds_msa)


@pytest.fixture
def smoothed_freq_matrix(two_thirds_msa: core.TokenizedMSA) -> torch.Tensor:
    """Frequency matrix with every cell strictly inside ``(0, 1)``.

    The logit transform is undefined at 0 and 1, so the log-odds tests need a
    matrix that a unit pseudocount has pulled away from both bounds. Position 3
    then holds E at 3/23, F at 2/23, and 1/23 for every other residue.
    """
    return pssm.compute_frequency_matrix(
        two_thirds_msa, laplace_smoothing=True, pseudocount=1.0
    )


# ---------------------------------------------------------------------------
# compute_frequency_matrix
# ---------------------------------------------------------------------------


def test_compute_frequency_matrix_covers_the_amino_acid_alphabet() -> None:
    """The default matrix has one column per standard amino acid."""
    freq = pssm.compute_frequency_matrix(_make_msa("ACDE", "ACDF", "WCDE"))
    assert freq.shape == (4, 20)


def test_compute_frequency_matrix_appends_a_gap_column_on_request() -> None:
    """``aa_cols_only=False`` adds a trailing column for the gap token."""
    msa = _make_msa("ACDE", "AC-E", "WCDE")
    freq = pssm.compute_frequency_matrix(msa, aa_cols_only=False)
    assert freq.shape == (4, 21)
    assert freq[2, _GAP_COLUMN].item() == pytest.approx(1 / 3)


def test_compute_frequency_matrix_normalises_each_position() -> None:
    """Every position is a probability distribution over the vocabulary."""
    freq = pssm.compute_frequency_matrix(_make_msa("ACDE", "ACDF", "WCDE"))
    torch.testing.assert_close(freq.sum(dim=-1), torch.ones(4))


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_compute_frequency_matrix_is_one_hot_for_a_conserved_column(
    position: int,
) -> None:
    """A fully conserved position puts all its mass on the observed residue."""
    freq = pssm.compute_frequency_matrix(_make_msa("ACDE", "ACDE", "ACDE"))
    assert freq[position, position].item() == pytest.approx(1.0)


def test_compute_frequency_matrix_counts_residue_occurrences(
    freq_matrix: torch.Tensor,
) -> None:
    """A column holding two E and one F splits 2/3 to 1/3."""
    assert freq_matrix[3, 3].item() == pytest.approx(2 / 3)
    assert freq_matrix[3, 4].item() == pytest.approx(1 / 3)


def test_compute_frequency_matrix_reweighting_downweights_duplicates(
    two_thirds_msa: core.TokenizedMSA,
) -> None:
    """Two identical sequences share one sequence's worth of counts."""
    # The duplicated "ACDE" rows are neighbours and take weight 1/2 each, so E and
    # F both accumulate a count of 1 at position 3.
    freq = pssm.compute_frequency_matrix(two_thirds_msa, use_sample_reweighting=True)
    assert freq[3, 3].item() == pytest.approx(0.5)


def test_compute_frequency_matrix_laplace_smoothing_lifts_unobserved_residues(
    two_thirds_msa: core.TokenizedMSA,
) -> None:
    """A pseudocount moves probability mass onto residues the MSA never shows."""
    freq = pssm.compute_frequency_matrix(
        two_thirds_msa, laplace_smoothing=True, pseudocount=1.0
    )
    # Counts at position 3 become E=3, F=2 and 1 for each of the other 18 residues.
    assert freq[3, 3].item() == pytest.approx(3 / 23)
    assert freq[3, 0].item() == pytest.approx(1 / 23)


def test_compute_frequency_matrix_laplace_smoothing_keeps_positions_normalised(
    two_thirds_msa: core.TokenizedMSA,
) -> None:
    """Smoothed positions remain probability distributions."""
    freq = pssm.compute_frequency_matrix(two_thirds_msa, laplace_smoothing=True)
    torch.testing.assert_close(freq.sum(dim=-1), torch.ones(4))


def test_compute_frequency_matrix_regularization_mixes_toward_uniform() -> None:
    """Frequency regularization pulls each cell toward ``1 / V`` by the amplitude."""
    freq = pssm.compute_frequency_matrix(
        _make_msa("ACDE", "ACDE"),
        use_frequency_regularization=True,
        freq_regularization_amplitude=0.1,
    )
    assert freq[0, 0].item() == pytest.approx(0.9 + 0.1 / 20)
    assert freq[0, 1].item() == pytest.approx(0.1 / 20)


def test_compute_frequency_matrix_regularization_keeps_positions_normalised() -> None:
    """Regularized positions remain probability distributions."""
    freq = pssm.compute_frequency_matrix(
        _make_msa("ACDE", "ACDF"), use_frequency_regularization=True
    )
    torch.testing.assert_close(freq.sum(dim=-1), torch.ones(4))


def test_compute_frequency_matrix_blanks_masked_columns() -> None:
    """Columns excluded by ``valid_cols`` come back as nan."""
    msa = _make_msa("ACDE", "ACDF")
    msa.valid_cols = torch.tensor([True, True, False, True])
    freq = pssm.compute_frequency_matrix(msa)
    assert torch.isnan(freq[2]).all()
    assert not torch.isnan(freq[[0, 1, 3]]).any()


def test_compute_frequency_matrix_from_a3m_file(a3m_file: pathlib.Path) -> None:
    """A matrix built from a real alignment spans the query columns."""
    msa = core.TokenizedMSA.from_file(a3m_file, format="a3m")
    freq = pssm.compute_frequency_matrix(msa)
    assert freq.shape == (20, 20)
    torch.testing.assert_close(freq.sum(dim=-1), torch.ones(20))


# ---------------------------------------------------------------------------
# compute_msa_frequency_score
# ---------------------------------------------------------------------------


def test_compute_msa_frequency_score_returns_one_score_per_mutant(
    freq_matrix: torch.Tensor,
) -> None:
    """Output shape is ``(B,)`` for ``(B, L)`` mutants."""
    mutants = torch.cat([_MT_E_TO_F, _MT_A_TO_W, _WT])
    assert pssm.compute_msa_frequency_score(_WT, mutants, freq_matrix).shape == (3,)


def test_compute_msa_frequency_score_is_zero_for_the_wild_type(
    freq_matrix: torch.Tensor,
) -> None:
    """A mutant identical to the wild-type has no mutated positions to score."""
    mutants = _WT.expand(3, -1).contiguous()
    score = pssm.compute_msa_frequency_score(_WT, mutants, freq_matrix)
    torch.testing.assert_close(score, torch.zeros(3))


def test_compute_msa_frequency_score_is_the_log_ratio_at_the_mutated_site(
    freq_matrix: torch.Tensor,
) -> None:
    """E (2/3) to F (1/3) scores ``log(1/3) - log(2/3)``."""
    score = pssm.compute_msa_frequency_score(_WT, _MT_E_TO_F, freq_matrix)
    torch.testing.assert_close(score, torch.tensor([-math.log(2)]))


def test_compute_msa_frequency_score_sums_over_mutated_sites() -> None:
    """A double mutant scores the sum of its per-position log ratios."""
    # Both position 0 (A 2/3 -> W 1/3) and position 3 (E 2/3 -> F 1/3) halve.
    freq = pssm.compute_frequency_matrix(_make_msa("ACDE", "ACDE", "WCDF"))
    score = pssm.compute_msa_frequency_score(_WT, torch.tensor([[18, 1, 2, 4]]), freq)
    torch.testing.assert_close(score, torch.tensor([-2 * math.log(2)]))


def test_compute_msa_frequency_score_scores_each_mutant_independently(
    freq_matrix: torch.Tensor,
) -> None:
    """Batched scoring matches scoring the mutants one at a time."""
    batched = pssm.compute_msa_frequency_score(
        _WT, torch.cat([_MT_E_TO_F, _MT_A_TO_W]), freq_matrix
    )
    expected = torch.cat(
        [
            pssm.compute_msa_frequency_score(_WT, _MT_E_TO_F, freq_matrix),
            pssm.compute_msa_frequency_score(_WT, _MT_A_TO_W, freq_matrix),
        ]
    )
    torch.testing.assert_close(batched, expected)


def test_compute_msa_frequency_score_weight_scales_the_mutated_site(
    freq_matrix: torch.Tensor,
) -> None:
    """Doubling the weight of the only mutated position doubles the score."""
    weight = torch.tensor([1.0, 1.0, 1.0, 2.0])
    score = pssm.compute_msa_frequency_score(
        _WT, _MT_E_TO_F, freq_matrix, weight=weight
    )
    torch.testing.assert_close(score, torch.tensor([-2 * math.log(2)]))


def test_compute_msa_frequency_score_softmax_transform_renormalises_frequencies(
    freq_matrix: torch.Tensor,
) -> None:
    """``softmax_transform`` scores the softmax of the frequencies, not their log."""
    score = pssm.compute_msa_frequency_score(
        _WT, _MT_E_TO_F, freq_matrix, softmax_transform=True
    )
    # log_softmax differences reduce to the raw frequency difference: 1/3 - 2/3.
    torch.testing.assert_close(score, torch.tensor([-1 / 3]))


def test_compute_msa_frequency_score_log_odds_uses_the_logit_transform(
    smoothed_freq_matrix: torch.Tensor,
) -> None:
    """``use_log_odds`` scores ``logit(f)`` instead of ``log(f)``."""
    score = pssm.compute_msa_frequency_score(
        _WT, _MT_E_TO_F, smoothed_freq_matrix, use_log_odds=True
    )
    # logit(2/23) - logit(3/23) = log(2/21) - log(3/20) = log(40/63).
    torch.testing.assert_close(score, torch.tensor([math.log(40 / 63)]))


def test_compute_msa_frequency_score_log_odds_overrides_softmax_transform(
    smoothed_freq_matrix: torch.Tensor,
) -> None:
    """Setting both transforms applies the log-odds one."""
    log_odds = pssm.compute_msa_frequency_score(
        _WT, _MT_E_TO_F, smoothed_freq_matrix, use_log_odds=True
    )
    both = pssm.compute_msa_frequency_score(
        _WT, _MT_E_TO_F, smoothed_freq_matrix, use_log_odds=True, softmax_transform=True
    )
    torch.testing.assert_close(both, log_odds)
