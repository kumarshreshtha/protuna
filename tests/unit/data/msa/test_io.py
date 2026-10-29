"""Tests for FASTA/A2M/A3M readers in protuna.data.msa.io."""

from __future__ import annotations

import io
import pathlib

import pytest

from protuna.data.msa import io as align_io


def _handle(text: str) -> io.StringIO:
    """An in-memory file handle over ``text``."""
    return io.StringIO(text)


# ---------------------------------------------------------------------------
# iterate_fasta
# ---------------------------------------------------------------------------


def test_iterate_fasta_yields_header_sequence_pairs() -> None:
    """Each record is yielded as a ``(header, sequence)`` pair, in file order."""
    source = _handle(">s1\nACDE\n>s2\nKLMN\n")
    assert list(align_io.iterate_fasta(source)) == [("s1", "ACDE"), ("s2", "KLMN")]


def test_iterate_fasta_concatenates_multiline_sequences() -> None:
    """A sequence wrapped over several lines is joined into one string."""
    source = _handle(">s1\nACDE\nFGHI\n>s2\nKLMN\n")
    assert list(align_io.iterate_fasta(source)) == [("s1", "ACDEFGHI"), ("s2", "KLMN")]


def test_iterate_fasta_keeps_only_the_id_before_the_first_tab() -> None:
    """A tab-delimited description is dropped from the header."""
    source = _handle(">id1\tdescription text\nACDE\n")
    assert list(align_io.iterate_fasta(source)) == [("id1", "ACDE")]


def test_iterate_fasta_yields_final_record_without_trailing_newline() -> None:
    """The last record is emitted even when the file does not end in a newline."""
    source = _handle(">s1\nACDE\n>s2\nKLMN")
    assert list(align_io.iterate_fasta(source)) == [("s1", "ACDE"), ("s2", "KLMN")]


def test_iterate_fasta_yields_nothing_for_an_empty_source() -> None:
    """An empty file yields no records."""
    assert list(align_io.iterate_fasta(_handle(""))) == []


def test_iterate_fasta_skips_a_header_with_no_sequence() -> None:
    """A header with no residue lines before the next header is not yielded."""
    source = _handle(">empty\n>s2\nKLMN\n")
    assert list(align_io.iterate_fasta(source)) == [("s2", "KLMN")]


def test_iterate_fasta_accepts_a_path(tmp_path: pathlib.Path) -> None:
    """A filesystem path is accepted in place of a file handle."""
    path = tmp_path / "seqs.fasta"
    path.write_text(">s1\nACDE\n>s2\nKLMN\n")
    assert list(align_io.iterate_fasta(path)) == [("s1", "ACDE"), ("s2", "KLMN")]


def test_iterate_fasta_is_lazy(tmp_path: pathlib.Path) -> None:
    """Records are produced on demand rather than read up front."""
    path = tmp_path / "seqs.fasta"
    path.write_text(">s1\nACDE\n>s2\nKLMN\n")
    records = align_io.iterate_fasta(path)
    assert next(records) == ("s1", "ACDE")


def test_iterate_fasta_reads_every_record_of_the_a3m_fixture(
    a3m_file: pathlib.Path,
) -> None:
    """The alignment fixture holds a query plus seven homologs."""
    headers = [header for header, _ in align_io.iterate_fasta(a3m_file)]
    assert headers == [
        "query",
        "homolog1 insert after F",
        "homolog2 gap at L",
        "homolog3 inserts at start",
        "homolog4 identical to query",
        "homolog5 gaps at A and V",
        "homolog6 insert between P and Q",
        "homolog7 A to W substitution",
    ]


# ---------------------------------------------------------------------------
# parse_alignment normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("AC.EF", ["ACDEF", "AC-EF"]),
        ("ACdEF", ["ACDEF", "ACEF"]),
        ("Ac.dEF", ["ACDEF", "A-EF"]),
        ("ACDEF", ["ACDEF", "ACDEF"]),
        ("", ["ACDEF", ""]),
    ],
)
def test_parse_alignment_normalizes_query_columns(
    sequence: str, expected: list[str]
) -> None:
    """Replace dots with dashes and delete lowercase inserts in homologs."""
    source = _handle(f">q\nACDEF\n>h1\n{sequence}\n")
    assert (
        align_io.parse_alignment(source, format="a3m", query_cols_only=True) == expected
    )


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("AC.EF", "AC-EF"),
        ("ACdEF", "ACdEF"),
        ("Ac.dEF", "Ac-dEF"),
        ("ACDEF", "ACDEF"),
    ],
)
def test_parse_alignment_normalizes_all_columns(sequence: str, expected: str) -> None:
    """Replace dots with dashes and retain lowercase inserts."""
    source = _handle(f">q\n{sequence}\n")
    assert align_io.parse_alignment(source, format="a3m", query_cols_only=False) == [
        expected
    ]


# ---------------------------------------------------------------------------
# parse_alignment
# ---------------------------------------------------------------------------


def test_parse_alignment_rejects_an_unsupported_format() -> None:
    """An unrecognised format raises ValueError."""
    with pytest.raises(ValueError, match="Only fasta, a3m and a2m"):
        align_io.parse_alignment(
            _handle(">s1\nACDE\n"),
            format="stockholm",  # ty: ignore[invalid-argument-type]
        )


def test_parse_alignment_fasta_returns_one_sequence_per_record() -> None:
    """Every record of a FASTA alignment is returned, in file order."""
    source = _handle(">q\nACDE\n>h1\nACDF\n>h2\nA-DE\n")
    assert align_io.parse_alignment(source, format="fasta") == ["ACDE", "ACDF", "A-DE"]


def test_parse_alignment_strips_gaps_from_the_query() -> None:
    """Gap columns introduced by the query are removed from the query itself."""
    source = _handle(">q\nAC-DE\n>h1\nACFDE\n")
    sequences = align_io.parse_alignment(source, format="fasta", query_cols_only=True)
    assert sequences[0] == "ACDE"


def test_parse_alignment_a3m_removes_lowercase_inserts() -> None:
    """A3M inserts are deleted so homologs align to the query columns."""
    source = _handle(">q\nACDE\n>h1\nACdDE\n")
    sequences = align_io.parse_alignment(source, format="a3m", query_cols_only=True)
    assert sequences == ["ACDE", "ACDE"]


def test_parse_alignment_a3m_keeps_inserts_when_query_cols_only_is_false() -> None:
    """Insert columns survive when column trimming is switched off."""
    source = _handle(">q\nACDE\n>h1\nACdDE\n")
    sequences = align_io.parse_alignment(source, format="a3m", query_cols_only=False)
    assert sequences == ["ACDE", "ACdDE"]


def test_parse_alignment_a2m_replaces_dots_with_dashes() -> None:
    """A2M dot inserts become the canonical gap character."""
    source = _handle(">q\nACDE\n>h1\nAC.E\n")
    sequences = align_io.parse_alignment(source, format="a2m", query_cols_only=False)
    assert sequences == ["ACDE", "AC-E"]


def test_parse_alignment_a3m_fixture_aligns_every_sequence_to_query_length(
    a3m_file: pathlib.Path,
) -> None:
    """All parsed A3M sequences share the ungapped query length of 20."""
    sequences = align_io.parse_alignment(a3m_file, format="a3m", query_cols_only=True)
    assert {len(sequence) for sequence in sequences} == {20}


def test_parse_alignment_a3m_fixture_query_has_no_gaps(a3m_file: pathlib.Path) -> None:
    """The query row of a parsed A3M is the plain wild-type sequence."""
    sequences = align_io.parse_alignment(a3m_file, format="a3m", query_cols_only=True)
    assert sequences[0] == "ACDEFGHIKLMNPQRSTVWY"


def test_parse_alignment_a3m_fixture_keeps_homolog_gaps(
    a3m_file: pathlib.Path,
) -> None:
    """Gaps in a homolog are retained at the query-aligned columns."""
    sequences = align_io.parse_alignment(a3m_file, format="a3m", query_cols_only=True)
    assert sequences[2] == "ACDEFGHIK-MNPQRSTVWY"


# ---------------------------------------------------------------------------
# sequences_from_alignment
# ---------------------------------------------------------------------------


def test_sequences_from_alignment_rejects_an_unsupported_format() -> None:
    """An unrecognised format raises ValueError."""
    with pytest.raises(ValueError, match="Only fasta, a3m and a2m"):
        # ty: ignore[invalid-argument-type]
        align_io.sequences_from_alignment(_handle(">s1\nACDE\n"), format="clustal")


def test_sequences_from_alignment_strips_gaps_and_dots() -> None:
    """Both gap characters are removed from every record."""
    source = _handle(">q\nAC-DE\n>h1\nA.CDE\n")
    assert align_io.sequences_from_alignment(source, format="fasta") == [
        "ACDE",
        "ACDE",
    ]


def test_sequences_from_alignment_uppercases_inserts() -> None:
    """Lowercase insert residues are kept but uppercased."""
    source = _handle(">q\nACDE\n>h1\nacde\n")
    assert align_io.sequences_from_alignment(source, format="fasta") == [
        "ACDE",
        "ACDE",
    ]


def test_sequences_from_alignment_a3m_fixture_returns_ungapped_sequences(
    a3m_file: pathlib.Path,
) -> None:
    """Insert columns lengthen a homolog once alignment columns are discarded."""
    sequences = align_io.sequences_from_alignment(a3m_file, format="a3m")
    assert sequences[1] == "ACDEFAGHIKLMNPQRSTVWY"
    assert sequences[2] == "ACDEFGHIKMNPQRSTVWY"
