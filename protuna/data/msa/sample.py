"""Strategies for sampling a subset of aligned rows from a multiple sequence alignment.

Several protein language models incorporate evolutionary context at inference time by
conditioning generation on a prefix of sequences from the same protein family. This
module provides strategies for sampling sequences from an MSA for such inference time
conditioning.
"""

from collections.abc import Sequence
import enum
import itertools

from jaxtyping import Integer
import torch

from protuna.data import tokenizers
from protuna.data.msa import core


class MSASamplingMethod(enum.StrEnum):
    """Available strategies for MSA sub-selection."""

    Greedy = "greedy"
    Weighted = "weighted"
    Random = "random"
    Stratified = "stratified"


def _ordered_sample(
    tokenized_msa: core.TokenizedMSA,
    priority_order: Integer[torch.Tensor, " n"],
    *,
    max_sequences: int | None = None,
    max_tokens: int | None = None,
    count_gap_tokens: bool = False,
) -> Integer[torch.Tensor, "s l"]:
    msa_tokens = tokenized_msa.msa_tokens
    num_seq, seq_len = msa_tokens.shape
    if max_tokens is not None:
        if count_gap_tokens:
            token_count = torch.full((num_seq,), fill_value=seq_len)
        else:
            gap_id = tokenizers.tokenizer.gap_token_id
            token_count = (msa_tokens != gap_id).sum(dim=1)
        token_count = token_count[priority_order]
        selected_seqs_mask = torch.cumsum(token_count, dim=0) <= max_tokens
        selected_seqs_idx = priority_order[selected_seqs_mask]
    else:
        selected_seqs_idx = priority_order[:max_sequences]
    return msa_tokens[selected_seqs_idx]


def greedy_sample_msa(
    tokenized_msa: core.TokenizedMSA,
    *,
    max_sequences: int | None = None,
    max_tokens: int | None = None,
    count_gap_tokens: bool = False,
) -> Integer[torch.Tensor, "s l"]:
    """Sample the MSA by descending sequence identity to the query.

    Sequences are ranked by their identity to the query (row 0) and selected
    greedily until the budget is exhausted. Used in AIDO-RAG :cite:p:`AIDO`.

    Args:
        tokenized_msa: The tokenised MSA to sample from.
        max_sequences: Maximum number of sequences to return.  Mutually
            exclusive with ``max_tokens``.
        max_tokens: Maximum total token budget.  Mutually exclusive with
            ``max_sequences``.
        count_gap_tokens: If ``True``, gap characters count towards
            ``max_tokens``.

    Returns:
        Integer tensor of shape ``(S, L)`` containing the sampled MSA tokens.
    """
    msa_tokens = tokenized_msa.msa_tokens
    seq_identity = core.get_seq_identity_to_query(msa_tokens)
    priority_order = torch.argsort(seq_identity, descending=True)
    return _ordered_sample(
        tokenized_msa,
        priority_order,
        max_sequences=max_sequences,
        max_tokens=max_tokens,
        count_gap_tokens=count_gap_tokens,
    )


def random_sample_msa(
    tokenized_msa: core.TokenizedMSA,
    *,
    max_sequences: int | None = None,
    max_tokens: int | None = None,
    count_gap_tokens: bool = False,
    generator: torch.Generator | None = None,
) -> Integer[torch.Tensor, "s l"]:
    """Sample the MSA uniformly at random.

    The query (row 0) is always placed first. The remaining sequences are
    shuffled uniformly at random before the budget is applied.

    Args:
        tokenized_msa: The tokenised MSA to sample from.
        max_sequences: Maximum number of sequences to return.  Mutually
            exclusive with ``max_tokens``.
        max_tokens: Maximum total token budget.  Mutually exclusive with
            ``max_sequences``.
        count_gap_tokens: If ``True``, gap characters count towards
            ``max_tokens``.
        generator: Generator for shuffling non-query sequences; ``None`` uses
            the default PyTorch generator.

    Returns:
        Integer tensor of shape ``(S, L)`` containing the sampled MSA tokens.
    """
    msa_tokens = tokenized_msa.msa_tokens
    priority_order = [
        0,
        *(1 + torch.randperm(msa_tokens.shape[0] - 1, generator=generator)).tolist(),
    ]
    priority_order = torch.tensor(priority_order, device=msa_tokens.device)
    return _ordered_sample(
        tokenized_msa,
        priority_order,
        max_sequences=max_sequences,
        max_tokens=max_tokens,
        count_gap_tokens=count_gap_tokens,
    )


def weighted_sample_msa(
    tokenized_msa: core.TokenizedMSA,
    *,
    max_sequences: int | None = None,
    max_tokens: int | None = None,
    count_gap_tokens: bool = False,
    threshold: float = 0.2,
    generator: torch.Generator | None = None,
) -> Integer[torch.Tensor, "s l"]:
    r"""Sample the MSA with probabilities proportional to sequence weights.

    Sequence weights are computed via the EVMutation reweighting scheme (see
    :meth:`~protuna.data.msa.core.compute_sequence_weights`).
    Sequences with fewer near-duplicates receive higher weight and are sampled
    more often. The query is always placed first.

    Used in PoET :cite:p:`PoET` and Profluent E1 :cite:p:`E1`.

    Args:
        tokenized_msa: The tokenised MSA to sample from.
        max_sequences: Maximum number of sequences to return.  Mutually
            exclusive with ``max_tokens``.
        max_tokens: Maximum total token budget.  Mutually exclusive with
            ``max_sequences``.
        count_gap_tokens: If ``True``, gap characters count towards
            ``max_tokens``.
        threshold: Hamming-distance threshold :math:`\theta` for sequence
            reweighting.  Sequences within this distance are treated as
            near-duplicates.  Defaults to ``0.2``.
        generator: Generator for drawing the weighted sequence order.
            ``None`` uses the default PyTorch generator.

    Returns:
        Integer tensor of shape ``(S, L)`` containing the sampled MSA tokens.
    """
    msa_tokens = tokenized_msa.msa_tokens
    num_seq = msa_tokens.shape[0]
    sequence_weights = tokenized_msa.get_sequence_weights(threshold=threshold)
    priority_order = torch.multinomial(
        sequence_weights[1:], num_samples=num_seq - 1, generator=generator
    )
    priority_order = [0, *(1 + priority_order).tolist()]
    priority_order = torch.tensor(priority_order, device=msa_tokens.device)
    return _ordered_sample(
        tokenized_msa,
        priority_order,
        max_sequences=max_sequences,
        max_tokens=max_tokens,
        count_gap_tokens=count_gap_tokens,
    )


def stratified_sample_msa(
    tokenized_msa: core.TokenizedMSA,
    *,
    max_sequences: int | None = None,
    max_tokens: int | None = None,
    count_gap_tokens: bool = False,
    seq_identity_buckets: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 0.98),
    shuffle_samples: bool = False,
    generator: torch.Generator | None = None,
) -> Integer[torch.Tensor, "s l"]:
    """Sample the MSA with stratified diversity over identity buckets.

    Sequences are assigned to identity buckets defined by consecutive pairs in
    ``seq_identity_buckets``, then interleaved in a round-robin fashion so
    that each stratum is represented.  Default bucket boundaries are taken from
    GEMME :cite:p:`GEMME`.

    Args:
        tokenized_msa: The tokenised MSA to sample from.
        max_sequences: Maximum number of sequences to return.  Mutually
            exclusive with ``max_tokens``.
        max_tokens: Maximum total token budget.  Mutually exclusive with
            ``max_sequences``.
        count_gap_tokens: If ``True``, gap characters count towards
            ``max_tokens``.
        seq_identity_buckets: Monotonically increasing identity thresholds
            that define bucket boundaries.  Defaults to
            ``(0.2, 0.4, 0.6, 0.8, 0.98)``.
        shuffle_samples: If ``True``, the sampled non-query sequences are
            shuffled before being returned.
        generator: Generator for within-bucket and optional output shuffling.
            ``None`` uses the default PyTorch generator.

    Returns:
        Integer tensor of shape ``(S, L)`` containing the sampled MSA tokens.
    """
    msa_tokens = tokenized_msa.msa_tokens
    seq_identities = core.get_seq_identity_to_query(msa_tokens)
    seq_identities = seq_identities[1:]
    buckets = []
    for low, high in itertools.pairwise(seq_identity_buckets):
        bucket_seqs = torch.argwhere(
            (seq_identities >= low) & (seq_identities < high)
        ).flatten()
        order = torch.randperm(bucket_seqs.numel(), generator=generator)
        buckets.append(bucket_seqs[order])
    # interleave one sequence from each bucket
    priority_order = [0]
    for idxs in itertools.zip_longest(*buckets):
        priority_order.extend([idx + 1 for idx in idxs if idx is not None])
    priority_order = torch.tensor(priority_order, device=msa_tokens.device)
    sampled_msa = _ordered_sample(
        tokenized_msa,
        priority_order,
        max_sequences=max_sequences,
        max_tokens=max_tokens,
        count_gap_tokens=count_gap_tokens,
    )
    if shuffle_samples:
        shuffled_order = [
            0,
            *(
                1 + torch.randperm(sampled_msa.shape[0] - 1, generator=generator)
            ).tolist(),
        ]
        return sampled_msa[shuffled_order]
    return sampled_msa


def sample_msa(
    tokenized_msa: core.TokenizedMSA,
    sampling_strategy: MSASamplingMethod,
    *,
    max_sequences: int | None = None,
    max_tokens: int | None = None,
    count_gap_tokens: bool = False,
    sequence_wt_threshold: float = 0.2,
    seq_identity_buckets: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 0.98),
    generator: torch.Generator | None = None,
) -> Integer[torch.Tensor, "s l"]:
    r"""Sample an MSA using the specified strategy.

    Exactly one of ``max_sequences`` and ``max_tokens`` must be provided.

    Args:
        tokenized_msa: The tokenised MSA to sample from.
        sampling_strategy: Strategy to use; one of :class:`MSASamplingMethod`.
        max_sequences: Maximum number of sequences to return.  Mutually
            exclusive with ``max_tokens``.
        max_tokens: Maximum total token budget.  Mutually exclusive with
            ``max_sequences``.
        count_gap_tokens: If ``True``, gap characters count towards
            ``max_tokens``.  Ignored when ``max_sequences`` is used.
        sequence_wt_threshold: Hamming-distance threshold :math:`\theta`
            passed to :func:`weighted_sample_msa` when
            ``sampling_strategy`` is :attr:`MSASamplingMethod.Weighted`.
        seq_identity_buckets: Bucket boundaries passed to
            :func:`stratified_sample_msa` when ``sampling_strategy``
            is :attr:`MSASamplingMethod.Stratified`.
        generator: Generator passed to the stochastic sampling strategy.
            ``None`` uses the default PyTorch generator; ignored for greedy
            sampling.

    Returns:
        Integer tensor of shape ``(S, L)`` containing the sampled MSA tokens.

    Raises:
        ValueError: If both or neither of ``max_sequences`` / ``max_tokens``
            are provided, or if ``sampling_strategy`` is unrecognised.
    """
    if (max_sequences is None and max_tokens is None) or (
        max_sequences is not None and max_tokens is not None
    ):
        raise ValueError(
            "Must provide exactly one of max_sequences and max_tokens."
            f" Got {max_sequences=}. {max_tokens=}"
        )
    if sampling_strategy == MSASamplingMethod.Greedy:
        return greedy_sample_msa(
            tokenized_msa,
            max_sequences=max_sequences,
            max_tokens=max_tokens,
            count_gap_tokens=count_gap_tokens,
        )
    if sampling_strategy == MSASamplingMethod.Random:
        return random_sample_msa(
            tokenized_msa,
            max_sequences=max_sequences,
            max_tokens=max_tokens,
            count_gap_tokens=count_gap_tokens,
            generator=generator,
        )
    if sampling_strategy == MSASamplingMethod.Weighted:
        return weighted_sample_msa(
            tokenized_msa,
            max_sequences=max_sequences,
            max_tokens=max_tokens,
            count_gap_tokens=count_gap_tokens,
            threshold=sequence_wt_threshold,
            generator=generator,
        )
    if sampling_strategy == MSASamplingMethod.Stratified:
        return stratified_sample_msa(
            tokenized_msa,
            max_sequences=max_sequences,
            max_tokens=max_tokens,
            count_gap_tokens=count_gap_tokens,
            seq_identity_buckets=seq_identity_buckets,
            generator=generator,
        )
    raise ValueError(f"Unknown Sampling Strategy: {sampling_strategy}")
