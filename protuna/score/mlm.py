"""Zero-shot fitness scoring for masked protein language models.

All functions accept a batch of tokenized mutant sequences and return a
:class:`ScoreOutput` per mutant. A higher score indicates the mutant
residues are more probable than the wild-type under the model.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Protocol

import einops
from jaxtyping import Float, Integer
import torch


class HasLogits(Protocol):
    """Expose per-token model logits for a batch of sequences.

    Attributes:
        logits: Unnormalized scores of shape ``(B, L, V)``.
    """

    logits: Float[torch.Tensor, "b l v"]


class LogitsModel(Protocol):
    """Define the callable model interface required by MLM scoring functions."""

    def __call__(self, batch: Integer[torch.Tensor, "b l"]) -> HasLogits:
        """Return vocabulary logits for each position in the token batch."""
        ...


@dataclasses.dataclass
class ScoreOutput:
    """Container for MLM fitness scores and optional logits."""

    score: Float[torch.Tensor, " b"]
    logits: Float[torch.Tensor, "b l v"] | None = None


def _log_probs_from_logits(
    logits: Float[torch.Tensor, "b l v"], temp: float = 1.0, use_log_odds: bool = False
) -> Float[torch.Tensor, "b l v"]:
    if use_log_odds:
        return torch.logit(torch.softmax(logits / temp, dim=-1))
    return torch.log_softmax(logits / temp, dim=-1)


def wildtype_marginal(
    model: LogitsModel,
    wt_tokens: Integer[torch.Tensor, "1 l"],
    mt_tokens: Integer[torch.Tensor, "b l"],
    *,
    wt_temp: float = 1.0,
    mt_temp: float = 1.0,
    use_log_odds: bool = False,
    weight: Float[torch.Tensor, " l"] | None = None,
    return_logits: bool = False,
) -> ScoreOutput:
    r"""Score variants using probabilities conditioned on the wild-type sequence.

    Runs a single forward pass on the unmasked wild-type sequence, then
    computes, at each mutated position, the difference in log-probability (or
    log-odds) between the mutant residue and the wild-type residue. The score
    for each variant is the sum of these differences over all mutated positions
    (positions where ``wt_tokens != mt_tokens``).

    .. math::
        \sum_{i \in M} w_i \cdot (\log p(x_i = x_i^{mt} \mid x_{wt})
        - \log p(x_i = x_i^{wt} \mid x_{wt}))

    Because the model is called only once regardless of batch size, this method
    is efficient for scoring large libraries of single and multi-site mutants.

    This scoring strategy was introduced alongside masked and mutant marginal
    variants in ESM-1v :cite:p:`ESM1v`.

    Independent temperature scaling of wild-type and mutant logits follows
    AIDO-RAG :cite:p:`AIDO` [#aido1]_ where ``wt_temp = 1.5`` and
    ``mt_temp = 1``.

    The ``use_log_odds`` option, which replaces log-probabilities with log-odds
    :math:`\log(p / (1 - p))`, was explored in RSALOR :cite:p:`RSALOR`.

    .. [#aido1] Not mentioned in the paper but used in the proteingym submission.

    Args:
        model: A masked protein language model whose ``forward`` pass accepts
            a token tensor of shape ``(B, L)`` and returns an object with a
            ``logits`` attribute of shape ``(B, L, V)``, where ``V`` is the
            vocabulary size.
        wt_tokens: Tokenised wild-type sequence of shape ``(1, L)``.
        mt_tokens: Tokenised mutant sequences of shape ``(B, L)``.
        wt_temp: Temperature for scaling wild-type logits. Values below
            ``1.0`` sharpen the distribution; values above flatten it.
        mt_temp: Temperature for scaling mutant logits.
        use_log_odds: If ``True``, use log-odds :math:`\log(p / (1 - p))`
            instead of log-probabilities, amplifying differences near the
            tails.
        weight: Per-position rescaling weights of shape ``(L,)``.
            Defaults to ``None`` (uniform weight of 1).
        return_logits: If ``True``, attach the wild-type logits to the
            returned :class:`ScoreOutput`.

    Returns:
        :class:`ScoreOutput` whose ``score`` tensor has shape ``(B,)``; element
        ``i`` is the wildtype-marginal score for mutant ``i``.
    """
    logits = model(wt_tokens).logits
    wt_log_p = _log_probs_from_logits(logits, wt_temp, use_log_odds)
    mt_log_p = _log_probs_from_logits(logits, mt_temp, use_log_odds)
    wt_log_p = wt_log_p.gather(dim=2, index=wt_tokens.unsqueeze(-1)).squeeze(-1)
    mt_log_p = einops.repeat(mt_log_p, "1 l v -> b l v", b=mt_tokens.shape[0])
    mt_log_p = mt_log_p.gather(dim=2, index=mt_tokens.unsqueeze(-1)).squeeze(-1)
    mutation_mask = wt_tokens != mt_tokens
    weight_ = 1 if weight is None else weight[None, :]
    score = ((mt_log_p - wt_log_p) * mutation_mask * weight_).sum(dim=1)
    return ScoreOutput(score=score, logits=logits if return_logits else None)


def wildtype_marginal_from_logits(
    logits: Float[torch.Tensor, "1 l v"],
    wt_tokens: Integer[torch.Tensor, "1 l"],
    mt_tokens: Integer[torch.Tensor, "b l"],
    *,
    wt_temp: float = 1.0,
    mt_temp: float = 1.0,
    use_log_odds: bool = False,
    weight: Float[torch.Tensor, " l"] | None = None,
    return_logits: bool = False,
) -> ScoreOutput:
    r"""Score variants from precomputed wild-type logits.

    Identical to :func:`wildtype_marginal` except that the wild-type logits are
    supplied directly instead of being obtained from a model. Use this to score
    successive batches of mutants, or several weightings of the same batch,
    against a single cached forward pass on the wild-type sequence.

    Args:
        logits: Wild-type logits of shape ``(1, L, V)``, where ``V`` is the
            vocabulary size.
        wt_tokens: Tokenised wild-type sequence of shape ``(1, L)``.
        mt_tokens: Tokenised mutant sequences of shape ``(B, L)``.
        wt_temp: Temperature for scaling wild-type logits. Values below
            ``1.0`` sharpen the distribution; values above flatten it.
        mt_temp: Temperature for scaling mutant logits.
        use_log_odds: If ``True``, use log-odds :math:`\log(p / (1 - p))`
            instead of log-probabilities.
        weight: Per-position rescaling weights of shape ``(L,)``.
            Defaults to ``None`` (uniform weight of 1).
        return_logits: If ``True``, attach ``logits`` to the returned
            :class:`ScoreOutput`.

    Returns:
        :class:`ScoreOutput` whose ``score`` tensor has shape ``(B,)``; element
        ``i`` is the wildtype-marginal score for mutant ``i``.
    """
    wt_log_p = _log_probs_from_logits(logits, wt_temp, use_log_odds)
    mt_log_p = _log_probs_from_logits(logits, mt_temp, use_log_odds)
    wt_log_p = wt_log_p.gather(dim=2, index=wt_tokens.unsqueeze(-1)).squeeze(-1)
    mt_log_p = einops.repeat(mt_log_p, "1 l v -> b l v", b=mt_tokens.shape[0])
    mt_log_p = mt_log_p.gather(dim=2, index=mt_tokens.unsqueeze(-1)).squeeze(-1)
    mutation_mask = wt_tokens != mt_tokens
    weight_ = 1 if weight is None else weight[None, :]
    score = ((mt_log_p - wt_log_p) * mutation_mask * weight_).sum(dim=1)
    return ScoreOutput(score=score, logits=logits if return_logits else None)


def mutant_marginal(
    model: LogitsModel,
    wt_tokens: Integer[torch.Tensor, "1 l"],
    mt_tokens: Integer[torch.Tensor, "b l"],
    *,
    wt_temp: float = 1.0,
    mt_temp: float = 1.0,
    use_log_odds: bool = False,
    weight: Float[torch.Tensor, " l"] | None = None,
    return_logits: bool = False,
) -> ScoreOutput:
    r"""Score variants using probabilities conditioned on each mutant sequence.

    Runs a forward pass on the unmasked mutant sequences, then computes, at
    each mutated position, the difference in log-probability (or log-odds)
    between the mutant residue and the wild-type residue under the mutant
    context.

    .. math::
        \sum_{i \in M} w_i \cdot (\log p(x_i = x_i^{mt} \mid x_{mt})
        - \log p(x_i = x_i^{wt} \mid x_{mt}))

    Unlike :func:`wildtype_marginal`, the model sees the full mutant sequence
    rather than the wild-type when producing logits, so the context used to
    evaluate each position reflects the mutant background.

    This scoring strategy was introduced alongside wildtype and masked marginal
    variants in ESM-1v :cite:p:`ESM1v` where it is evaluated as an alternative
    to masked marginals.

    Independent temperature scaling of wild-type and mutant logits follows
    AIDO-RAG :cite:p:`AIDO` [#aido2]_ where ``wt_temp = 1.5`` and
    ``mt_temp = 1``.

    The ``use_log_odds`` option, which replaces log-probabilities with log-odds
    :math:`\log(p / (1 - p))`, was explored in RSALOR :cite:p:`RSALOR`.

    .. [#aido2] Not mentioned in the paper but used in the proteingym submission.

    Args:
        model: A masked protein language model whose ``forward`` pass accepts
            a token tensor of shape ``(B, L)`` and returns an object with a
            ``logits`` attribute of shape ``(B, L, V)``.
        wt_tokens: Tokenised wild-type sequence of shape ``(1, L)``.
        mt_tokens: Tokenised mutant sequences of shape ``(B, L)``.
        wt_temp: Temperature for scaling logits when computing wild-type
            residue probabilities.
        mt_temp: Temperature for scaling logits when computing mutant residue
            probabilities.
        use_log_odds: If ``True``, use log-odds :math:`\log(p / (1 - p))`
            instead of log-probabilities, amplifying differences near the
            tails.
        weight: Per-position rescaling weights of shape ``(L,)``.
            Defaults to ``None`` (uniform weight of 1).
        return_logits: If ``True``, attach the mutant logits to the returned
            :class:`ScoreOutput`.

    Returns:
        :class:`ScoreOutput` whose ``score`` tensor has shape ``(B,)``; element
        ``i`` is the mutant-marginal score for mutant ``i``.
    """
    logits = model(mt_tokens).logits
    wt_log_p = _log_probs_from_logits(logits, wt_temp, use_log_odds)
    mt_log_p = _log_probs_from_logits(logits, mt_temp, use_log_odds)
    wt_tokens = wt_tokens.to(logits.device)
    mt_tokens = mt_tokens.to(logits.device)
    wt_tokens = einops.repeat(wt_tokens, "1 l -> b l", b=mt_tokens.shape[0])
    wt_log_p = wt_log_p.gather(dim=2, index=wt_tokens.unsqueeze(-1)).squeeze(-1)
    mt_log_p = mt_log_p.gather(dim=2, index=mt_tokens.unsqueeze(-1)).squeeze(-1)
    mutation_mask = wt_tokens != mt_tokens
    weight_ = 1 if weight is None else weight[None, :]
    score = ((mt_log_p - wt_log_p) * mutation_mask * weight_).sum(dim=1)
    return ScoreOutput(score=score, logits=logits if return_logits else None)


def masked_marginal(
    model: LogitsModel,
    wt_tokens: Integer[torch.Tensor, "1 l"],
    mt_tokens: Integer[torch.Tensor, "b l"],
    mask_token_id: int,
    *,
    wt_temp: float = 1.0,
    mt_temp: float = 1.0,
    use_log_odds: bool = False,
    weight: Float[torch.Tensor, " l"] | None = None,
    return_logits: bool = False,
) -> ScoreOutput:
    r"""Score variants using probabilities at masked mutated positions.

    For each mutant, replaces the mutated positions with the mask token before
    the forward pass, so the model predicts those positions from surrounding
    context alone. The score is the sum over mutated positions of the
    log-probability (or log-odds) difference between the mutant residue and
    the wild-type residue.

    .. math::
        \sum_{i \in M} w_i \cdot (\log p(x_i = x_i^{mt} \mid x_{-M})
        - \log p(x_i = x_i^{wt} \mid x_{-M}))

    This is the primary scoring strategy introduced by ESM-1v :cite:p:`ESM1v`
    for zero-shot fitness prediction with masked protein language models.

    Independent temperature scaling of wild-type and mutant logits follows
    AIDO-RAG :cite:p:`AIDO` [#aido3]_ where ``wt_temp = 1.5`` and
    ``mt_temp = 1``.

    The ``use_log_odds`` option, which replaces log-probabilities with log-odds
    :math:`\log(p / (1 - p))`, was explored in RSALOR :cite:p:`RSALOR`.

    .. [#aido3] Not mentioned in the paper but used in the proteingym submission.

    Args:
        model: A masked protein language model whose ``forward`` pass accepts
            a token tensor of shape ``(B, L)`` and returns an object with a
            ``logits`` attribute of shape ``(B, L, V)``.
        wt_tokens: Tokenised wild-type sequence of shape ``(1, L)``.
        mt_tokens: Tokenised mutant sequences of shape ``(B, L)``.
        mask_token_id: Integer id of the ``[MASK]`` token used to blank out
            mutated positions before the forward pass.
        wt_temp: Temperature for scaling logits when computing wild-type
            residue probabilities.
        mt_temp: Temperature for scaling logits when computing mutant residue
            probabilities.
        use_log_odds: If ``True``, use log-odds :math:`\log(p / (1 - p))`
            instead of log-probabilities, amplifying differences near the
            tails.
        weight: Per-position rescaling weights of shape ``(L,)``.
            Defaults to ``None`` (uniform weight of 1).
        return_logits: If ``True``, attach the masked logits to the returned
            :class:`ScoreOutput`.

    Returns:
        :class:`ScoreOutput` whose ``score`` tensor has shape ``(B,)``; element
        ``i`` is the masked-marginal score for mutant ``i``.
    """
    mutation_mask = wt_tokens != mt_tokens
    inp_tokens = mt_tokens.clone()
    inp_tokens[mutation_mask] = mask_token_id
    logits = model(inp_tokens).logits
    wt_log_p = _log_probs_from_logits(logits, wt_temp, use_log_odds)
    mt_log_p = _log_probs_from_logits(logits, mt_temp, use_log_odds)
    wt_tokens = einops.repeat(wt_tokens, "1 l -> b l", b=mt_tokens.shape[0])
    wt_log_p = wt_log_p.gather(dim=2, index=wt_tokens.unsqueeze(-1)).squeeze(-1)
    mt_log_p = mt_log_p.gather(dim=2, index=mt_tokens.unsqueeze(-1)).squeeze(-1)
    weight_ = 1 if weight is None else weight[None, :]
    score = ((mt_log_p - wt_log_p) * mutation_mask * weight_).sum(dim=1)
    return ScoreOutput(score=score, logits=logits if return_logits else None)


@torch.inference_mode()
def masked_marginal_inference(
    model: LogitsModel,
    wt_tokens: Integer[torch.Tensor, "1 l"],
    mt_tokens: Integer[torch.Tensor, "b l"],
    mask_token_id: int,
    *,
    wt_temp: float = 1.0,
    mt_temp: float = 1.0,
    use_log_odds: bool = False,
    weight: Float[torch.Tensor, " l"] | None = None,
    return_logits: bool = False,
    batch_size: int = 1,
) -> ScoreOutput:
    r"""Score variants with masked marginals, deduplicating masking patterns.

    Equivalent to :func:`masked_marginal` but avoids one forward pass per
    mutant. The masked input depends only on *which* positions are masked, not
    on the mutant residues there, so all mutants that share a set of mutated
    positions yield an identical masked sequence and therefore identical logits.

    Args:
        model: A masked protein language model whose ``forward`` pass accepts
            a token tensor of shape ``(N, L)`` and returns an object with a
            ``logits`` attribute of shape ``(N, L, V)``.
        wt_tokens: Tokenised wild-type sequence of shape ``(1, L)``.
        mt_tokens: Tokenised mutant sequences of shape ``(B, L)``.
        mask_token_id: Integer id of the ``[MASK]`` token used to blank out
            mutated positions before the forward pass.
        wt_temp: Temperature for scaling logits when computing wild-type
            residue probabilities.
        mt_temp: Temperature for scaling logits when computing mutant residue
            probabilities.
        use_log_odds: If ``True``, use log-odds :math:`\log(p / (1 - p))`
            instead of log-probabilities.
        weight: Per-position rescaling weights of shape ``(L,)``.
            Defaults to ``None`` (uniform weight of 1).
        return_logits: If ``True``, attach the per-mutant logits of shape
            ``(B, L, V)`` (expanded from the unique masking patterns) to the
            returned :class:`ScoreOutput`.
        batch_size: Number of unique masking patterns forwarded to the model
            per call. Defaults to ``1``.

    Returns:
        :class:`ScoreOutput` whose ``score`` tensor has shape ``(B,)``; element
        ``i`` is the masked-marginal score for mutant ``i``.
    """
    mutation_mask = wt_tokens != mt_tokens  # (B, L)
    patterns, inverse = torch.unique(mutation_mask, dim=0, return_inverse=True)
    inp_tokens = wt_tokens.expand(patterns.shape[0], -1).clone()  # (G, L)
    inp_tokens[patterns] = mask_token_id
    logits = []
    for batch in itertools.batched(inp_tokens, batch_size):
        logits.append(model(torch.stack(batch)).logits)
    logits = torch.cat(logits)  # (G, L, V)
    wt_log_p = _log_probs_from_logits(logits, wt_temp, use_log_odds)
    mt_log_p = _log_probs_from_logits(logits, mt_temp, use_log_odds)
    mt_tokens = mt_tokens.to(logits.device)
    wt_tokens = wt_tokens.to(logits.device)
    mutation_mask = mutation_mask.to(logits.device)
    inverse = inverse.to(logits.device)
    batch_idx, pos_idx = torch.nonzero(mutation_mask, as_tuple=True)  # (K,)
    group_idx = inverse[batch_idx]  # (K,)
    mt_res = mt_tokens[batch_idx, pos_idx]  # (K,)
    wt_res = wt_tokens[0, pos_idx]  # (K,)
    group_idx = group_idx.to(logits.device)
    pos_idx = pos_idx.to(logits.device)
    diff = mt_log_p[group_idx, pos_idx, mt_res] - wt_log_p[group_idx, pos_idx, wt_res]
    if weight is not None:
        weight = weight.to(diff.device)
        diff = diff * weight[pos_idx]
    score = torch.zeros(mt_tokens.shape[0], device=logits.device, dtype=diff.dtype)
    score.scatter_add_(0, batch_idx, diff)
    return ScoreOutput(score=score, logits=logits[inverse] if return_logits else None)


def pseudo_log_likelihood(
    model: LogitsModel,
    mt_tokens: Integer[torch.Tensor, "b l"],
    mask_token_id: int,
    *,
    mask_rate: int | None = 8,
) -> ScoreOutput:
    r"""Compute pseudo-log-likelihood scores via sequential strided masking.

    Approximates the pseudo-log-likelihood (PLM) of each sequence by iterating
    over ``mask_rate`` strided masking patterns. In iteration ``i``, positions
    ``{i, i + mask_rate, i + 2 * mask_rate, ...}`` are simultaneously masked,
    a forward pass is run, and the log-probabilities for those positions are
    recorded. The final score is the sum of log-probabilities over all
    positions:

    .. math::
        \sum_{i} \log p(x_i \mid x_{-i})
        \approx \sum_{i}
        \log p(x_i \mid x_{-S_{i\mod{r}}})

    where :math:`r` is ``mask_rate`` and :math:`S_k` is the set of positions
    masked in stride :math:`r` starting at position :math:`k`.

    Setting ``mask_rate=None`` reduces to true per-position masking (one
    forward pass per position, ``L`` total passes). Smaller ``mask_rate``
    values reduce the number of forward passes at the cost of masking multiple
    positions simultaneously.

    The masked modulo approximation was introduced in :cite:p:`Johnson2025` and
    further investigated in :cite:p:`Hawkins-Hooker2024`.

    For a fully vectorised variant that batches all masking patterns into a
    single model call, see :func:`batch_pseudo_log_likelihood`.

    Args:
        model: A masked protein language model whose ``forward`` pass accepts
            a token tensor of shape ``(B, L)`` and returns an object with a
            ``logits`` attribute of shape ``(B, L, V)``.
        mt_tokens: Tokenised sequences of shape ``(B, L)``.
        mask_token_id: Integer id of the ``[MASK]`` token.
        mask_rate: Number of masking strides (equal to the number of forward
            passes). If ``None``, every position is masked individually
            (``L`` passes). Defaults to ``8``.

    Returns:
        :class:`ScoreOutput` whose ``score`` tensor has shape ``(B,)``
        containing the pseudo-log-likelihood score for each sequence.
    """
    final_log_p = torch.empty(
        mt_tokens.shape, device=mt_tokens.device, dtype=torch.float32
    )
    mask_rate = mt_tokens.shape[1] if mask_rate is None else mask_rate
    for i in range(mask_rate):
        mask = torch.zeros(
            mt_tokens.shape[1], dtype=torch.bool, device=mt_tokens.device
        )
        mask[i::mask_rate] = True
        inp_tokens = mt_tokens.clone()
        inp_tokens[:, mask] = mask_token_id
        logits = model(inp_tokens).logits
        log_p = torch.log_softmax(logits, dim=-1)[:, mask]
        log_p = log_p.gather(dim=2, index=mt_tokens[:, mask, None]).squeeze(-1)
        final_log_p[:, mask] = log_p
    score = final_log_p.sum(dim=1)
    return ScoreOutput(score=score)


def batch_pseudo_log_likelihood(
    model: LogitsModel,
    mt_tokens: Integer[torch.Tensor, "b l"],
    mask_token_id: int,
    *,
    mask_rate: int | None = 8,
    return_logits: bool = False,
    batch_size: int | None = None,
) -> ScoreOutput:
    """Compute pseudo-log-likelihood scores via batched strided masking.

    Functionally equivalent to :func:`pseudo_log_likelihood` but vectorises
    all ``mask_rate`` masking patterns into a single expanded batch of shape
    ``(B * mask_rate, L)`` instead of looping over patterns sequentially.
    This reduces wall-clock time at the cost of higher peak memory.

    The expanded batch is forwarded in chunks of ``batch_size`` to bound peak
    GPU memory. When ``batch_size`` is ``None`` the entire ``(B * mask_rate)``
    tensor is processed in one model call, reproducing the original behaviour.

    The masked modulo approximation was introduced in :cite:p:`Johnson2025` and
    further investigated in :cite:p:`Hawkins-Hooker2024`.

    Args:
        model: A masked protein language model whose ``forward`` pass accepts
            a token tensor of shape ``(N, L)`` and returns an object with a
            ``logits`` attribute of shape ``(N, L, V)``.
        mt_tokens: Tokenised sequences of shape ``(B, L)``.
        mask_token_id: Integer id of the ``[MASK]`` token.
        mask_rate: Number of masking strides. The full expanded batch has
            size ``B * mask_rate`` rows. If ``None``, every position is
            masked individually, giving a batch of size ``B * L``. Defaults
            to ``8``.
        return_logits: If ``True``, attach the aggregated logits to the
            returned :class:`ScoreOutput`.
        batch_size: Maximum number of rows passed to the model in a single
            forward call. When ``None`` (default) all ``B * mask_rate`` rows
            are forwarded at once. Set to a smaller value to reduce peak
            memory at the cost of additional forward passes.

    Returns:
        :class:`ScoreOutput` whose ``score`` tensor has shape ``(B,)``
        containing the pseudo-log-likelihood score for each sequence.
    """
    mask_rate = mt_tokens.shape[1] if mask_rate is None else mask_rate
    positions = torch.arange(mt_tokens.shape[1], device=mt_tokens.device)
    mask_indices = positions % mask_rate
    freq = torch.arange(mask_rate, device=mt_tokens.device)[:, None]
    masks = mask_indices == freq
    inp_tokens = einops.repeat(mt_tokens, "b l -> b m l", m=mask_rate)
    inp_tokens = torch.where(masks[None], mask_token_id, inp_tokens)
    inp_tokens = einops.rearrange(inp_tokens, "b m l -> (b m) l")
    if batch_size is None:
        logits = model(inp_tokens).logits
    else:
        logits = []
        for batch_idx in itertools.batched(range(inp_tokens.shape[0]), batch_size):
            batch_logits = model(inp_tokens[list(batch_idx)]).logits
            logits.append(batch_logits)
        logits = torch.cat(logits)
    logits = einops.rearrange(logits, "(b m) l v -> b m l v", m=mask_rate)
    logits = logits[:, mask_indices, positions]  # b l v
    log_p = torch.log_softmax(logits, dim=-1)
    log_p = log_p.gather(dim=2, index=mt_tokens.unsqueeze(-1)).squeeze(-1)
    score = log_p.sum(dim=1)
    return ScoreOutput(score=score, logits=logits if return_logits else None)
