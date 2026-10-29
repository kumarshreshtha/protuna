"""Core data structures and algorithms for handling sequence alignments.

This module provides the :class:`TokenizedMSA` dataclass as the central
representation for a tokenized alignment, along with functions for computing
per-sequence identity, pairwise identity matrices, and sequence weights used
for MSA reweighting.

Example:
    Load an alignment file and compute reweighting coefficients:

    >>> import pathlib
    >>> from protuna.data.msa import core
    >>> _tests = pathlib.Path(__file__).resolve().parents[3] / "tests"
    >>> _a3m = _tests / "fixtures" / "alignment.a3m"
    >>> msa = core.TokenizedMSA.from_file(_a3m, format="a3m")
    >>> msa.msa_tokens.shape
    torch.Size([8, 20])
    >>> weights = msa.get_sequence_weights(threshold=0.2)
    >>> weights.shape
    torch.Size([8])

    Compute the full pairwise identity matrix:

    >>> identity_matrix = core.compute_pairwise_identity(msa.msa_tokens)
    >>> identity_matrix.shape
    torch.Size([8, 8])
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import os
from typing import IO, Literal, Self

from jaxtyping import Bool, Float, Integer
import psutil
import torch

from protuna.data.msa import io
from protuna.data.tokenizers import tokenizer

logger = logging.getLogger(__name__)


def get_seq_identity_to_query(
    msa_tokens: Integer[torch.Tensor, "n l"],
) -> Float[torch.Tensor, " n"]:
    r"""Compute the sequence identity of each MSA row to the query sequence.

    The query is assumed to be the first row of ``msa_tokens``. Identity is
    defined as the fraction of positions that match the query, normalised by
    the non-gap length of each sequence:

    .. math::

        \text{identity}_i =
        \frac{\displaystyle\sum_k \mathbf{1}[x_{i,k} = x_{0,k}]}
             {\displaystyle\sum_k \mathbf{1}[x_{i,k} \neq \text{gap}]}

    Args:
        msa_tokens: Tokenised MSA of shape ``(N, L)``.  Row ``0`` is treated
            as the query sequence.

    Returns:
        Float tensor of shape ``(N,)`` with per-sequence identity to the query.
        The first element is always ``1.0``.
    """
    seq_lengths = (msa_tokens != tokenizer.gap_token_id).sum(dim=1)
    seq_identity = (msa_tokens == msa_tokens[0]).sum(dim=1)
    seq_identity = seq_identity / seq_lengths
    return seq_identity


def _get_batch_size(msa_tokens: Integer[torch.Tensor, "n l"]) -> int:
    num_sequences, sequence_length = msa_tokens.shape
    if num_sequences == 0:
        return 1
    # Leave 20% of available memory for allocator overhead and workspaces.
    if msa_tokens.device.type == "cuda":
        available_memory, _ = torch.cuda.mem_get_info(msa_tokens.device)
        memory_budget = available_memory * 4 // 5
    else:
        system_memory = psutil.virtual_memory()
        # reserve at least 75% of system RAM for the rest of the process
        memory_budget = min(system_memory.total // 4, system_memory.available * 4 // 5)

    comparison_memory = (
        num_sequences * sequence_length * (torch.bool.itemsize + torch.int64.itemsize)
    )
    token_memory = sequence_length * msa_tokens.element_size()
    reduction_memory = (
        num_sequences
        * (2 * torch.int64.itemsize + 2 * torch.float32.itemsize + torch.bool.itemsize)
        + torch.int64.itemsize
    )
    memory_per_query = comparison_memory + token_memory + reduction_memory
    return max(1, min(num_sequences, memory_budget // memory_per_query))


def compute_pairwise_identity(
    msa_tokens: Integer[torch.Tensor, "n l"],
    *,
    batch_size: int | None = None,
) -> Float[torch.Tensor, "n n"]:
    r"""Compute the :math:`N \times N` pairwise sequence identity matrix.

    Identity between two sequences is the number of positions where both
    sequences carry the same amino-acid token (gap–gap and gap–residue
    comparisons are excluded), divided by the shorter non-gap length:

    .. math::

        \text{identity}(i, j) =
        \frac{\displaystyle\sum_k
              \mathbf{1}[x_{i,k} = x_{j,k},\; x_{i,k} \neq \text{gap}]}
             {\min(|i|,\, |j|)}

    where :math:`|i|` denotes the number of non-gap positions in sequence
    :math:`i`.

    Args:
        msa_tokens: Tokenised MSA of shape ``(N, L)``.
        batch_size: Number of query sequences processed per forward pass.
            Inferred from available memory when ``None``.

    Returns:
        Float tensor of shape ``(N, N)`` where entry ``[i, j]`` is the
        pairwise identity between sequences ``i`` and ``j``.
    """
    gaps = msa_tokens == tokenizer.gap_token_id
    lengths = (~gaps).sum(dim=1)
    # this makes sure that (-,-) is not considered a match for sim comp.
    new_token = max(tokenizer.all_token_ids) + 1
    msa_tokens_gap_masked = msa_tokens.clone()
    msa_tokens_gap_masked[gaps] = new_token
    batch_size = _get_batch_size(msa_tokens) if batch_size is None else batch_size
    logger.debug("Computing pairwise identities with batch_size=%r", batch_size)
    pairwise_id = torch.empty(msa_tokens.shape[0], msa_tokens.shape[0])
    for idx in itertools.batched(range(msa_tokens.shape[0]), batch_size):
        idx = list(idx)
        # B x L
        batch_tokens = msa_tokens[idx]
        batch_lengths = lengths[idx]
        min_lengths = torch.minimum(batch_lengths[:, None], lengths[None, :])
        identity = (batch_tokens[:, None] == msa_tokens_gap_masked[None, :]).sum(dim=2)
        identity = identity / min_lengths
        pairwise_id[idx, :] = identity
    return pairwise_id


def compute_sequence_weights(
    msa_tokens: Integer[torch.Tensor, "n l"],
    *,
    valid_cols: Bool[torch.Tensor, " l"] | None = None,
    threshold: float = 0.2,
    batch_size: int | None = None,
) -> Float[torch.Tensor, " n"]:
    r"""Compute per-sequence weights for MSA reweighting.

    Implements the reweighting scheme from EVMutation :cite:p:`EVMutation`.
    The weight of sequence :math:`i` is the inverse of the number of sequences
    within normalised Hamming distance :math:`\theta`:

    .. math::

        w_i = \left( \sum_j I\!\left[ D_H(x_i, x_j) < \theta \right] \right)^{-1}

    where :math:`D_H(x_i, x_j)` is the normalised Hamming distance between
    sequences :math:`i` and :math:`j` (fraction of non-gap positions that
    differ, divided by the length of sequence :math:`i`), and :math:`\theta`
    is ``threshold``.

    Args:
        msa_tokens: Tokenised MSA of shape ``(N, L)``.
        valid_cols: Boolean mask of shape ``(L,)`` selecting the columns to
            include in the distance computation.  All columns are used when
            ``None``.
        threshold: Sequences within Hamming distance :math:`\theta` are
            treated as near-duplicates. Defaults to ``0.2`` (the threshold
            used in EVMutation).
        batch_size: Number of query sequences processed per forward pass.
            Inferred from available memory when ``None``.

    Returns:
        Float tensor of shape ``(N,)`` with a weight for each sequence.
    """
    if valid_cols is not None:
        msa_tokens = msa_tokens[:, valid_cols]
    gaps = msa_tokens == tokenizer.gap_token_id
    lengths = (~gaps).sum(dim=1, keepdim=True)
    # this makes sure that (-,-) is not considered a match for identity comp.
    new_token = max(tokenizer.all_token_ids) + 1
    msa_tokens_gap_masked = msa_tokens.clone()
    msa_tokens_gap_masked[gaps] = new_token
    batch_size = _get_batch_size(msa_tokens) if batch_size is None else batch_size
    logger.debug("Computing sequence weights with batch_size=%r", batch_size)
    wts = []
    for idx in itertools.batched(range(msa_tokens.shape[0]), batch_size):
        idx = list(idx)
        batch_tokens = msa_tokens[idx]
        identity = (batch_tokens[:, None] == msa_tokens_gap_masked[None, :]).sum(dim=2)
        identity = identity / lengths[idx]
        wts.append(torch.reciprocal((identity > (1 - threshold)).sum(dim=1)))
    wts = torch.cat(wts)
    return wts


@dataclasses.dataclass(kw_only=True)
class MSAFilter:
    """Filter sequences and low coverage columns in an MSA.

    All thresholds are inclusive. The query sequence is always retained during
    filtering.

    Filters apply in the order:

    1. Minimum sequence coverage
    2. Minimum sequence identity to query
    3. Maximum sequence identity to query
    4. Minimum coverage per site
    5. Removal of sequences with unknown residues in the retained columns


    Example:
        Keep the query and homologs with identity at most 90%:

        >>> msa_filter = MSAFilter(max_sequence_identity=0.9)
        >>> tokens = torch.stack(tokenizer.encode(["ACDE", "ACDE", "ACDF"]))
        >>> filtered, valid_cols = msa_filter.apply(tokens)
        >>> filtered.shape
        torch.Size([2, 4])

    Args:
        min_sequence_coverage: Minimum fraction of non-gap positions required
            for a sequence to be retained.  Sequences below this threshold are
            dropped.
        min_site_coverage: Minimum fraction of non-gap positions required for a
            column to be retained.  Columns below this threshold are masked.
            Sequences that are entirely composed of gaps in the retained columns
            are also removed.
        min_sequence_identity: Minimum identity to the query (first) sequence.
            Sequences below this threshold are dropped.
        max_sequence_identity: Maximum identity to the query sequence.
            Sequences above this threshold are dropped (useful for removing
            near-identical sequences). The query sequence is always retained.
        drop_sequences_with_unk_aa: If ``True``, sequences containing unknown
            amino-acid tokens in the retained columns are removed.
    """

    min_sequence_coverage: float | None = None
    min_site_coverage: float | None = None
    min_sequence_identity: float | None = None
    max_sequence_identity: float | None = None
    drop_sequences_with_unk_aa: bool = False

    def apply(
        self,
        msa_tokens: Integer[torch.Tensor, "n l"],
    ) -> tuple[Integer[torch.Tensor, "m l"], Bool[torch.Tensor, " l"]]:
        """Apply the filtering stages to the given MSA.

        Args:
            msa_tokens: Aligned sequences, with the query in the first row.

        Returns:
            Retained rows with their original column width, and a boolean mask
            selecting valid columns. Row order is preserved; the input is unchanged.
        """
        init_seq_count = msa_tokens.shape[0]
        if self.min_sequence_coverage is not None:
            non_gap_positions = msa_tokens != tokenizer.gap_token_id
            seq_coverage = non_gap_positions.float().mean(dim=1)
            valid_seqs = seq_coverage >= self.min_sequence_coverage
            invalid_count = msa_tokens.shape[0] - valid_seqs.sum().item()
            logger.debug(
                "%s/%s sequences have coverage < %s and will be removed.",
                invalid_count,
                msa_tokens.shape[0],
                self.min_sequence_coverage,
            )
            msa_tokens = msa_tokens[valid_seqs].contiguous()

        if msa_tokens.shape[0] > 0 and (
            self.min_sequence_identity is not None
            or self.max_sequence_identity is not None
        ):
            seq_identity = get_seq_identity_to_query(msa_tokens)
            valid_seqs = msa_tokens.new_ones(msa_tokens.shape[0], dtype=torch.bool)
            if self.min_sequence_identity is not None:
                valid_seqs = seq_identity >= self.min_sequence_identity
                invalid_count = msa_tokens.shape[0] - valid_seqs.sum().item()
                logger.debug(
                    "%s/%s sequences have sequence identity < %s and will be removed.",
                    invalid_count,
                    msa_tokens.shape[0],
                    self.min_sequence_identity,
                )
            if self.max_sequence_identity is not None:
                below_max_identity = seq_identity <= self.max_sequence_identity
                below_max_identity[0] = True
                valid_seqs = valid_seqs & below_max_identity
                invalid_count = msa_tokens.shape[0] - valid_seqs.sum().item()
                logger.debug(
                    "%s/%s sequences have sequence identity > %s and will be removed.",
                    invalid_count,
                    msa_tokens.shape[0],
                    self.max_sequence_identity,
                )
            msa_tokens = msa_tokens[valid_seqs].contiguous()

        valid_cols = msa_tokens.new_ones(msa_tokens.shape[1], dtype=torch.bool)
        if self.min_site_coverage is not None:
            non_gap_positions = msa_tokens != tokenizer.gap_token_id
            site_coverage = non_gap_positions.float().mean(dim=0)
            valid_cols = site_coverage >= self.min_site_coverage
            invalid_count = msa_tokens.shape[1] - valid_cols.sum().item()
            logger.debug(
                "%s/%s sites have coverage < %s and will be masked.",
                invalid_count,
                msa_tokens.shape[1],
                self.min_site_coverage,
            )
            invalid_seqs = (msa_tokens[:, valid_cols] == tokenizer.gap_token_id).all(
                dim=1
            )
            msa_tokens = msa_tokens[~invalid_seqs].contiguous()

        if self.drop_sequences_with_unk_aa:
            valid_seqs = (msa_tokens[:, valid_cols] != tokenizer.unk_token_id).all(
                dim=1
            )
            msa_tokens = msa_tokens[valid_seqs].contiguous()

        logger.debug(
            "%s/%s sequences remain after filtering.",
            msa_tokens.shape[0],
            init_seq_count,
        )

        return msa_tokens, valid_cols


@dataclasses.dataclass
class TokenizedMSA:
    """Tokenized MSA as a 2D tensor and a mask representing its valid alignment columns.

    Stores the MSA as a 2D integer tensor with one row per sequence and one
    column per alignment position. It is assumed that the first sequence is the query
    sequence and is gap free (any insertions relative to the query are removed before
    construction).

    Args:
        msa_tokens: Integer tensor of shape ``(N, L)`` containing the tokenized
            sequences.  Row ``0`` is always the query sequence.
        valid_cols: Boolean mask of shape ``(L,)`` marking columns that passed the
            site-coverage filter.
    """

    msa_tokens: Integer[torch.Tensor, "n l"]
    valid_cols: Bool[torch.Tensor, " l"]

    def __repr__(self) -> str:
        """Return a compact summary of the tokenized alignment."""
        n_sequences, length = self.msa_tokens.shape
        return (
            f"{type(self).__name__}(n_sequences={n_sequences}, "
            f"length={length}, valid_cols={int(self.valid_cols.sum())}, "
            f"device={str(self.msa_tokens.device)!r})"
        )

    @property
    def query_sequence(self) -> Integer[torch.Tensor, " l"]:
        """Token IDs of the query (first) sequence, shape ``(L,)``."""
        return self.msa_tokens[0]

    def get_sequence_weights(self, threshold: float = 0.2) -> Float[torch.Tensor, " n"]:
        r"""Compute per-sequence reweighting coefficients.

        See :func:`compute_sequence_weights` for details.

        Args:
            threshold: the normalized Hamming distance below which sequences are
                treated as near-duplicates. Defaults to ``0.2``.

        Returns:
            Float tensor of shape ``(N,)`` with a weight for each sequence.
        """
        return compute_sequence_weights(
            self.msa_tokens, valid_cols=self.valid_cols, threshold=threshold
        )

    def to(self, device: torch.device) -> Self:
        """Move the MSA to the given device."""
        return dataclasses.replace(
            self,
            msa_tokens=self.msa_tokens.to(device),
            valid_cols=self.valid_cols.to(device),
        )

    @classmethod
    def from_file(
        cls,
        alignment_file: str | os.PathLike | IO,
        *,
        format: Literal["fasta", "a2m", "a3m"],
        msa_filter: MSAFilter | None = None,
    ) -> TokenizedMSA:
        """Load and tokenize an alignment from a file.

        Args:
            alignment_file: Path to or open file handle for the alignment.
            format: Alignment file format. One of ``"fasta"``, ``"a2m"``,
                or ``"a3m"``.
            msa_filter: Filters to apply to the constructed MSA. No filtering is
                performed when ``None``.

        Returns:
            The filtered, tokenized alignment.
        """
        sequences = io.parse_alignment(alignment_file, format=format)
        msa_tokens = torch.stack(tokenizer.encode(sequences), dim=0)
        msa_filter = MSAFilter() if msa_filter is None else msa_filter
        msa_tokens, valid_cols = msa_filter.apply(msa_tokens)
        return cls(msa_tokens=msa_tokens, valid_cols=valid_cols)
