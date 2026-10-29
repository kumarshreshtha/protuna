"""Low-level I/O utilities for reading FASTA, A2M, and A3M sequence files."""

from collections.abc import Generator, Iterator
import contextlib
import logging
import os
from typing import IO, Literal

from protuna import constants

logger = logging.getLogger(__name__)

AA_LOWERCASE_BYTES = constants.AA_UPPERCASE.lower().encode()

File = str | os.PathLike[str] | IO[str]


@contextlib.contextmanager
def _open_file(source: File) -> Generator[IO[str], None, None]:
    if isinstance(source, (str, os.PathLike)):
        with open(source) as f:
            yield f
    else:
        yield source


def iterate_fasta(source: File) -> Iterator[tuple[str, str]]:
    """Iterate over records in a FASTA-format file.

    Args:
        source: Path, path-like object, or open file handle for the FASTA file.

    Yields:
        A ``(header, sequence)`` tuple for each record.
    """
    curr_uuid = None
    curr_sequence = None
    with _open_file(source) as f:
        for line in f:
            if line.startswith(">"):
                if curr_sequence is not None and curr_uuid is not None:
                    yield curr_uuid, curr_sequence
                curr_uuid: str = line.strip().split("\t", 1)[0][1:]
                curr_sequence = None
            elif curr_uuid is not None:
                fragment = line.strip()
                curr_sequence: str = (
                    fragment if curr_sequence is None else curr_sequence + fragment
                )
        if curr_sequence is not None and curr_uuid is not None:
            yield curr_uuid, curr_sequence


def _format_a3m_sequence(seq: str, *, query_cols_only: bool = True) -> str:
    """Normalize a single A3M/A2M sequence string.

    Replaces dot insert characters with ``-`` and, when ``query_cols_only=True``,
    deletes lowercase insert residues so the sequence spans only query columns.

    Args:
        seq: Raw sequence string from an A3M or A2M record.
        query_cols_only: If ``True`` (default), remove lowercase (insert) residues.

    Returns:
        Normalised sequence string.
    """
    seq = seq.replace(".", "-")
    if query_cols_only:
        seq = seq.encode().translate(None, delete=AA_LOWERCASE_BYTES).decode()
    return seq


def parse_alignment(
    source: File,
    *,
    format: Literal["fasta", "a2m", "a3m"] = "fasta",
    query_cols_only: bool = True,
) -> list[str]:
    """Read an alignment file and return sequences aligned to the query.

    The first sequence is assumed to be the query. With ``query_cols_only=True``
    its gap characters are removed so it contains only residues, giving the canonical
    sequence length *L*. All subsequent sequences are trimmed to the same
    column set and may retain ``-`` at gapped alignment positions.

    Using ``query_cols_only=False`` with ``format="a3m"`` produces
    sequences of variable length because insert columns differ between
    records.

    Args:
        source: Path, path-like object, or open file handle for the alignment
            file.
        format: Alignment file format. One of ``"fasta"``, ``"a2m"``, or
            ``"a3m"``. Defaults to ``"fasta"``.
        query_cols_only: If ``True`` (default), remove insert columns so every
            sequence has the same length as the query.

    Returns:
        List of sequence strings.

    Raises:
        ValueError: If ``format`` is not one of the supported values.
    """
    if format not in ("a3m", "a2m", "fasta"):
        raise ValueError("Only fasta, a3m and a2m formats are supported for alignment.")
    if not query_cols_only and format == "a3m":
        logger.warning(
            "Parsing a3m file while keeping all columns. The resulting "
            "sequences will have variable length."
        )
    sequences = []
    query = None
    for _, seq in iterate_fasta(source):
        seq = _format_a3m_sequence(seq, query_cols_only=query_cols_only)
        if query_cols_only and query is None:
            seq = "".join([aa for aa in seq if aa != "-"])
            query = seq
        sequences.append(seq)
    return sequences


def sequences_from_alignment(
    source: File,
    *,
    format: Literal["fasta", "a2m", "a3m"] = "fasta",
) -> list[str]:
    """Read sequences (without gaps) from an alignment file.

    Strips all gap (``-``) and insert (``.``) characters and uppercases the
    result, returning the raw amino-acid sequences without alignment information.

    Args:
        source: Path, path-like object, or open file handle for the alignment
            file.
        format: Alignment file format — one of ``"fasta"``, ``"a2m"``, or
            ``"a3m"``. Defaults to ``"fasta"``.

    Returns:
        List of ungapped, uppercased sequence strings.

    Raises:
        ValueError: If ``format`` is not one of the supported values.
    """
    if format not in ("a3m", "a2m", "fasta"):
        raise ValueError("Only fasta, a3m and a2m formats are supported for alignment.")
    sequences = []
    for _, seq in iterate_fasta(source):
        seq = "".join([aa for aa in seq if aa not in [".", "-"]])
        seq = seq.upper()
        sequences.append(seq)
    return sequences
