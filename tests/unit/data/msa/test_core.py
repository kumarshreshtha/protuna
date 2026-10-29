"""Tests for the TokenizedMSA container and identity utilities in msa.core."""

from __future__ import annotations

import inspect
import pathlib
import types
from unittest import mock

import psutil
import pytest
import torch

from protuna.data import tokenizers
from protuna.data.msa import core

_TOKENIZER = tokenizers.tokenizer

# Token ids follow the alphabet "ACDEFGHIKLMNPQRSTVWY-_^*?":
# A=0  C=1  D=2  E=3  F=4 ... M=10 ... W=18  Y=19  gap=20  unk=24
_GAP = _TOKENIZER.gap_token_id


def _tokens(*sequences: str) -> torch.Tensor:
    """Encode ``sequences`` into an ``(N, L)`` token tensor."""
    return torch.stack(_TOKENIZER.encode(list(sequences)), dim=0)


@pytest.mark.parametrize("shape", [(0, 4), (3, 0), (1, 4), (3, 4)])
def test_batch_size_is_bounded_for_small_and_empty_alignments(
    shape: tuple[int, int],
) -> None:
    """Keep batch sizes usable for empty inputs and bounded by alignment depth."""
    msa = torch.empty(shape, dtype=torch.int64)

    batch_size = core._get_batch_size(msa)

    assert 1 <= batch_size <= max(1, shape[0])


def test_batch_size_accounts_for_comparisons_and_reduction_copies() -> None:
    """Budget for boolean comparisons and their integer reduction copies."""
    msa = torch.zeros((), dtype=torch.int64).expand(100_000, 1000)

    with mock.patch.object(
        psutil,
        "virtual_memory",
        autospec=True,
        return_value=types.SimpleNamespace(total=16 * 1024**3, available=8 * 1024**3),
    ):
        batch_size = core._get_batch_size(msa)

    assert 1 < batch_size < 5


@pytest.mark.parametrize(
    ("total_memory", "available_memory", "expected"),
    [
        (8 * 1024**3, 8 * 1024**3, 2),
        (32 * 1024**3, 32 * 1024**3, 9),
        (32 * 1024**3, 2 * 1024**3, 1),
        (32 * 1024**3, 0, 1),
    ],
)
def test_cpu_batch_size_limits_total_and_available_ram(
    total_memory: int,
    available_memory: int,
    expected: int,
) -> None:
    """Scale with system RAM while leaving headroom when available RAM is low."""
    msa = torch.zeros((), dtype=torch.int64).expand(100_000, 1000)
    with mock.patch.object(
        psutil,
        "virtual_memory",
        autospec=True,
        return_value=types.SimpleNamespace(
            total=total_memory, available=available_memory
        ),
    ):
        batch_size = core._get_batch_size(msa)

    assert batch_size == expected


@pytest.mark.parametrize(
    ("free_memory", "expected"), [(0, 1), (1024, 1), (1024**3, 100)]
)
def test_cuda_batch_size_uses_free_memory_on_the_alignment_device(
    free_memory: int,
    expected: int,
) -> None:
    """Honor free CUDA memory, including an exhausted device."""
    msa = mock.create_autospec(torch.Tensor, instance=True)
    msa.shape = (100, 1000)
    msa.device = torch.device("cuda:1")
    msa.element_size.return_value = 8
    with mock.patch.object(
        torch.cuda,
        "mem_get_info",
        autospec=True,
        return_value=(free_memory, 8 * 1024**3),
    ) as memory_info:
        batch_size = core._get_batch_size(msa)

    assert batch_size == expected
    memory_info.assert_called_once_with(torch.device("cuda:1"))


# ---------------------------------------------------------------------------
# get_seq_identity_to_query
# ---------------------------------------------------------------------------


def test_get_seq_identity_to_query_returns_one_value_per_sequence() -> None:
    """Output shape is ``(N,)`` for an ``(N, L)`` MSA."""
    msa = _tokens("ACDE", "ACDF", "WCDE")
    assert core.get_seq_identity_to_query(msa).shape == (3,)


def test_get_seq_identity_to_query_scores_the_query_against_itself_as_one() -> None:
    """The first row is the query and always has identity 1.0."""
    msa = _tokens("ACDE", "ACDF")
    assert core.get_seq_identity_to_query(msa)[0].item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("homolog", "expected"),
    [
        ("ACDE", 1.0),  # identical to the query
        ("ACDF", 0.75),  # A, C, D match; E -> F
        ("MMMM", 0.0),  # shares no residue with ACDE
    ],
)
def test_get_seq_identity_to_query_counts_matching_positions(
    homolog: str, expected: float
) -> None:
    """Identity is the fraction of positions matching the query."""
    msa = _tokens("ACDE", homolog)
    assert core.get_seq_identity_to_query(msa)[1].item() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("homolog", "expected"),
    [
        # "A-DE": 3 non-gap positions, all matching the query -> 3/3.
        ([0, _GAP, 2, 3], 1.0),
        # "W-DE": 3 non-gap positions, D and E match, W does not -> 2/3.
        ([18, _GAP, 2, 3], 2 / 3),
    ],
)
def test_get_seq_identity_to_query_normalises_by_non_gap_length(
    homolog: list[int], expected: float
) -> None:
    """Gapped positions are excluded from the denominator."""
    msa = torch.tensor([[0, 1, 2, 3], homolog])
    assert core.get_seq_identity_to_query(msa)[1].item() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# compute_pairwise_identity
# ---------------------------------------------------------------------------


def test_compute_pairwise_identity_returns_a_square_matrix() -> None:
    """Output shape is ``(N, N)`` for an ``(N, L)`` MSA."""
    msa = _tokens("ACDE", "ACDF", "WCDE")
    assert core.compute_pairwise_identity(msa).shape == (3, 3)


def test_compute_pairwise_identity_scores_self_identity_as_one() -> None:
    """The diagonal holds the self-identity of each sequence."""
    msa = _tokens("ACDE", "ACDF", "WCDE")
    matrix = core.compute_pairwise_identity(msa)
    torch.testing.assert_close(matrix.diagonal(), torch.ones(3))


def test_compute_pairwise_identity_is_symmetric() -> None:
    """Identity does not depend on the order of the two sequences."""
    msa = _tokens("ACDE", "ACDF", "WCDE")
    matrix = core.compute_pairwise_identity(msa)
    torch.testing.assert_close(matrix, matrix.T)


def test_compute_pairwise_identity_counts_shared_residues() -> None:
    """ACDE against ACDF shares 3 of 4 positions."""
    matrix = core.compute_pairwise_identity(_tokens("ACDE", "ACDF"))
    assert matrix[0, 1].item() == pytest.approx(0.75)


def test_compute_pairwise_identity_does_not_count_gap_gap_matches() -> None:
    """A shared gap column raises neither the numerator nor the denominator."""
    # "A-DE" against "A-DF": non-gap length 3 each; A and D match -> 2/3.
    msa = torch.tensor([[0, _GAP, 2, 3], [0, _GAP, 2, 4]])
    matrix = core.compute_pairwise_identity(msa)
    assert matrix[0, 1].item() == pytest.approx(2 / 3)


def test_compute_pairwise_identity_normalises_by_the_shorter_sequence() -> None:
    """The denominator is the shorter of the two non-gap lengths."""
    # "ACDE" (length 4) against "AC--" (length 2): A and C match -> 2/2.
    msa = torch.tensor([[0, 1, 2, 3], [0, 1, _GAP, _GAP]])
    matrix = core.compute_pairwise_identity(msa)
    assert matrix[0, 1].item() == pytest.approx(1.0)


def test_compute_pairwise_identity_is_unchanged_by_batching() -> None:
    """Chunking the computation does not change the result."""
    msa = _tokens("ACDE", "ACDF", "WCDE", "MMMM")
    torch.testing.assert_close(
        core.compute_pairwise_identity(msa, batch_size=1),
        core.compute_pairwise_identity(msa),
    )


# ---------------------------------------------------------------------------
# compute_sequence_weights
# ---------------------------------------------------------------------------


def test_compute_sequence_weights_returns_one_weight_per_sequence() -> None:
    """Output shape is ``(N,)`` for an ``(N, L)`` MSA."""
    msa = _tokens("ACDE", "ACDF", "WCDE")
    assert core.compute_sequence_weights(msa).shape == (3,)


def test_compute_sequence_weights_splits_weight_across_near_duplicates() -> None:
    """Each of three identical sequences carries a third of the weight."""
    msa = _tokens("ACDE", "ACDE", "ACDE")
    weights = core.compute_sequence_weights(msa, threshold=0.2)
    torch.testing.assert_close(weights, torch.full((3,), 1 / 3))


def test_compute_sequence_weights_gives_full_weight_to_distinct_sequences() -> None:
    """A sequence with no neighbour above the threshold keeps weight 1.0."""
    # threshold=0.01 admits only neighbours >99% identical; these share nothing.
    msa = _tokens("ACDE", "MMMM", "WWWW")
    weights = core.compute_sequence_weights(msa, threshold=0.01)
    torch.testing.assert_close(weights, torch.ones(3))


def test_compute_sequence_weights_widening_the_threshold_merges_neighbours() -> None:
    """A looser threshold turns two half-identical sequences into neighbours."""
    # "ACDE" and "ACMM" are 50% identical: neighbours only once 1 - threshold < 0.5.
    msa = _tokens("ACDE", "ACMM")
    weights = core.compute_sequence_weights(msa, threshold=0.6)
    torch.testing.assert_close(weights, torch.full((2,), 0.5))


def test_compute_sequence_weights_restricts_comparison_to_valid_columns() -> None:
    """Masked-out columns take no part in the neighbour decision."""
    # "ACDE" and "ACMM" agree on columns 0 and 1 only: identical once masked.
    msa = _tokens("ACDE", "ACMM")
    valid_cols = torch.tensor([True, True, False, False])
    weights = core.compute_sequence_weights(msa, valid_cols=valid_cols, threshold=0.2)
    torch.testing.assert_close(weights, torch.full((2,), 0.5))


def test_compute_sequence_weights_is_unchanged_by_batching() -> None:
    """Chunking the computation does not change the result."""
    msa = _tokens("ACDE", "ACDF", "WCDE", "MMMM")
    torch.testing.assert_close(
        core.compute_sequence_weights(msa, batch_size=1),
        core.compute_sequence_weights(msa),
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "min_sequence_coverage",
        "min_site_coverage",
        "min_sequence_identity",
        "max_sequence_identity",
        "drop_sequences_with_unk_aa",
    ],
)
def test_msa_filter_requires_keyword_arguments(field_name: str) -> None:
    """Expose filter thresholds as keyword-only constructor arguments."""
    signature = inspect.signature(core.MSAFilter)

    assert signature.parameters[field_name].kind == inspect.Parameter.KEYWORD_ONLY


def test_msa_filter_max_identity_does_not_override_min_identity() -> None:
    """The maximum-identity exemption applies only to its own threshold."""
    msa_filter = core.MSAFilter(min_sequence_identity=1.1, max_sequence_identity=0.9)

    filtered, _ = msa_filter.apply(_tokens("ACDE", "ACDF"))

    assert filtered.shape == (0, 4)


def test_msa_filter_identity_handles_rows_removed_by_coverage() -> None:
    """Identity filtering accepts an alignment emptied by coverage filtering."""
    msa_filter = core.MSAFilter(min_sequence_coverage=1.0, max_sequence_identity=0.9)

    filtered, valid_cols = msa_filter.apply(_tokens("AC--", "A---"))

    assert filtered.shape == (0, 4)
    torch.testing.assert_close(valid_cols, torch.tensor([True, True, True, True]))


def test_msa_filter_computes_site_coverage_after_identity_filtering() -> None:
    """Removed near-duplicates do not contribute to site coverage."""
    msa_filter = core.MSAFilter(min_site_coverage=0.5, max_sequence_identity=0.9)

    filtered, valid_cols = msa_filter.apply(_tokens("ACDE", "ACD-", "ACD-"))

    torch.testing.assert_close(filtered, _tokens("ACDE"))
    torch.testing.assert_close(valid_cols, torch.tensor([True, True, True, True]))


def test_msa_filter_checks_unknown_residues_only_in_retained_columns() -> None:
    """Mask low-coverage columns before checking for unknown residues."""
    msa_filter = core.MSAFilter(min_site_coverage=0.75, drop_sequences_with_unk_aa=True)

    filtered, valid_cols = msa_filter.apply(_tokens("ACDX", "ACD-"))

    torch.testing.assert_close(filtered, _tokens("ACDX", "ACD-"))
    torch.testing.assert_close(valid_cols, torch.tensor([True, True, True, False]))


# ---------------------------------------------------------------------------
# TokenizedMSA.from_file
# ---------------------------------------------------------------------------


def test_tokenized_msa_from_file_keeps_every_sequence_and_query_column(
    a3m_file: pathlib.Path,
) -> None:
    """The unfiltered fixture yields 8 sequences over the 20 query columns."""
    msa = core.TokenizedMSA.from_file(a3m_file, format="a3m")
    assert msa.msa_tokens.shape == (8, 20)


def test_tokenized_msa_from_file_marks_all_columns_valid_without_a_filter(
    a3m_file: pathlib.Path,
) -> None:
    """``valid_cols`` is all-True when no site-coverage filter is requested."""
    msa = core.TokenizedMSA.from_file(a3m_file, format="a3m")
    torch.testing.assert_close(msa.valid_cols, torch.ones(20, dtype=torch.bool))


def test_tokenized_msa_query_sequence_is_the_first_row(a3m_file: pathlib.Path) -> None:
    """``query_sequence`` exposes row 0 of the token matrix."""
    msa = core.TokenizedMSA.from_file(a3m_file, format="a3m")
    torch.testing.assert_close(msa.query_sequence, msa.msa_tokens[0])


def test_tokenized_msa_from_file_drops_low_coverage_sequences(
    a3m_file: pathlib.Path,
) -> None:
    """min_sequence_coverage removes homolog5, which is gapped at 2 of 20 sites."""
    opts = core.MSAFilter(min_sequence_coverage=0.95)
    msa = core.TokenizedMSA.from_file(a3m_file, format="a3m", msa_filter=opts)
    assert msa.msa_tokens.shape == (7, 20)


def test_tokenized_msa_from_file_drops_near_identical_sequences(
    a3m_file: pathlib.Path,
) -> None:
    """Retain the query and homolog7 while removing near-identical homologs."""
    opts = core.MSAFilter(max_sequence_identity=0.99)
    msa = core.TokenizedMSA.from_file(a3m_file, format="a3m", msa_filter=opts)
    torch.testing.assert_close(
        msa.msa_tokens, _tokens("ACDEFGHIKLMNPQRSTVWY", "WCDEFGHIKLMNPQRSTVWY")
    )


def test_tokenized_msa_from_file_drops_remote_sequences(
    a3m_file: pathlib.Path,
) -> None:
    """min_sequence_identity removes homolog7, the only sequence below 1.0."""
    opts = core.MSAFilter(min_sequence_identity=0.99)
    msa = core.TokenizedMSA.from_file(a3m_file, format="a3m", msa_filter=opts)
    assert msa.msa_tokens.shape == (7, 20)


def test_tokenized_msa_from_file_masks_low_coverage_columns(
    a3m_file: pathlib.Path,
) -> None:
    """min_site_coverage masks the three columns gapped in homolog2 or homolog5."""
    # homolog5 is gapped at columns 0 and 17; homolog2 at column 9.
    opts = core.MSAFilter(min_site_coverage=1.0)
    msa = core.TokenizedMSA.from_file(a3m_file, format="a3m", msa_filter=opts)
    assert torch.nonzero(~msa.valid_cols).flatten().tolist() == [0, 9, 17]


def test_tokenized_msa_from_file_drops_sequences_with_unknown_residues(
    tmp_path: pathlib.Path,
) -> None:
    """drop_sequences_with_unk_aa removes rows holding non-alphabet residues."""
    alignment = tmp_path / "unknown.fasta"
    alignment.write_text(">query\nACDE\n>homolog\nACDX\n")
    opts = core.MSAFilter(drop_sequences_with_unk_aa=True)
    msa = core.TokenizedMSA.from_file(alignment, format="fasta", msa_filter=opts)
    assert msa.msa_tokens.shape == (1, 4)


def test_tokenized_msa_from_file_applies_no_filter_by_default(
    tmp_path: pathlib.Path,
) -> None:
    """Sequences that every filter would reject survive when none is configured."""
    alignment = tmp_path / "unknown.fasta"
    alignment.write_text(">query\nACDE\n>homolog\nACDX\n")
    msa = core.TokenizedMSA.from_file(alignment, format="fasta")
    assert msa.msa_tokens.shape == (2, 4)


# ---------------------------------------------------------------------------
# TokenizedMSA.get_sequence_weights
# ---------------------------------------------------------------------------


def test_get_sequence_weights_returns_one_weight_per_sequence(
    a3m_file: pathlib.Path,
) -> None:
    """Output shape is ``(N,)`` for an ``(N, L)`` MSA."""
    msa = core.TokenizedMSA.from_file(a3m_file, format="a3m")
    assert msa.get_sequence_weights().shape == (8,)


def test_get_sequence_weights_uses_the_column_mask(a3m_file: pathlib.Path) -> None:
    """The instance's ``valid_cols`` is forwarded to compute_sequence_weights."""
    opts = core.MSAFilter(min_site_coverage=1.0)
    msa = core.TokenizedMSA.from_file(a3m_file, format="a3m", msa_filter=opts)
    expected = core.compute_sequence_weights(
        msa.msa_tokens, valid_cols=msa.valid_cols, threshold=0.2
    )
    torch.testing.assert_close(msa.get_sequence_weights(threshold=0.2), expected)


def test_get_sequence_weights_downweights_the_near_identical_fixture_rows(
    a3m_file: pathlib.Path,
) -> None:
    """Seven of the eight fixture rows are mutual neighbours by default."""
    # Only homolog7 (95% identity) falls outside the 80% neighbourhood of the rest.
    msa = core.TokenizedMSA.from_file(a3m_file, format="a3m")
    weights = msa.get_sequence_weights()
    torch.testing.assert_close(weights, torch.full((8,), 1 / 8))


# ---------------------------------------------------------------------------
# TokenizedMSA.to
# ---------------------------------------------------------------------------


def test_to_moves_both_tensors_and_preserves_content(a3m_file: pathlib.Path) -> None:
    """``to`` returns a new MSA holding the same tokens and column mask."""
    msa = core.TokenizedMSA.from_file(a3m_file, format="a3m")
    moved = msa.to(torch.device("cpu"))
    torch.testing.assert_close(moved.msa_tokens, msa.msa_tokens)
    torch.testing.assert_close(moved.valid_cols, msa.valid_cols)
