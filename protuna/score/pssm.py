"""MSA-based zero-shot fitness scoring via position-specific frequency matrices.

This module computes position-specific frequency matrices (PSSMs) from
:class:`~protuna.data.msa.core.TokenizedMSA` objects and uses them to score
variant sequences against a wild-type reference.

The typical workflow is:

1. Build a frequency matrix with :func:`compute_frequency_matrix`.
2. Score a batch of mutants with :func:`compute_msa_frequency_score`.
"""

import einops
from jaxtyping import Float, Integer
import torch

from protuna.data import tokenizers
from protuna.data.msa import core


def compute_frequency_matrix(
    msa: core.TokenizedMSA,
    *,
    use_sample_reweighting: bool = False,
    aa_cols_only: bool = True,
    laplace_smoothing: bool = False,
    pseudocount: float = 1e-5,
    use_frequency_regularization: bool = False,
    freq_regularization_amplitude: float = 0.01,  # see RSALOR supp material.
) -> Float[torch.Tensor, "l v"]:
    r"""Compute a position-specific frequency matrix from an MSA.

    At each alignment position :math:`l` the empirical frequency of amino acid
    :math:`a` is:

    .. math::

        f_{l,a} =
        \frac{\displaystyle\sum_i w_i \cdot \mathbf{1}[x_{i,l} = a]}
             {\displaystyle\sum_a \sum_i w_i \cdot \mathbf{1}[x_{i,l} = a]}

    where :math:`w_i = 1` by default. Setting ``use_sample_reweighting=True``
    replaces uniform weights with the EVMutation inverse-neighbours scheme
    :cite:p:`EVMutation`, which downweights sequences sharing >80 % identity
    with many other sequences.

    Two regularization strategies are available:

    * **Laplace smoothing** :cite:p:`Tranception` — adds ``pseudocount``
      :math:`\epsilon` to every count before normalising. The regularization
      effect diminishes as MSA depth grows.

      .. math::
          f_{l,a} = \frac{c_{l,a} + \epsilon}
                         {\displaystyle\sum_a (c_{l,a} + \epsilon)}

    * **Frequency regularization** :cite:p:`RSALOR` — mixes the empirical
      distribution toward the uniform distribution with a fixed amplitude
      :math:`\lambda`:

      .. math::
          f_{l,a}^{\text{reg}} =
          (1 - \lambda)\, f_{l,a} + \frac{\lambda}{V}

      Unlike pseudocount, the regularization magnitude is independent of MSA
      depth.

    .. note::
        The vocabulary dimension is ordered according to
        :attr:`~protuna.data.tokenizers.ProteinTokenizer.aa_token_ids` (20 standard
        amino acids), with a trailing
        gap column appended when ``aa_cols_only=False``.

    .. warning::
        Columns masked by ``msa.valid_cols`` (e.g. low-coverage sites) are
        set to ``nan`` in the returned matrix.  Downstream scores at those
        positions will also be ``nan``.

    Args:
        msa: A :class:`~protuna.data.msa.core.TokenizedMSA` from which
            sequences and optional column masks are read.
        use_sample_reweighting: If ``True``, weight sequences by the inverse
            of the number of neighbours within 80 % identity before computing
            frequencies.
        aa_cols_only: If ``True`` (default), the returned matrix has shape
            ``(L, 20)`` covering only standard amino acids.  If ``False``,
            an extra gap column is appended giving shape ``(L, 21)``.
        laplace_smoothing: If ``True``, add ``pseudocount`` to every count
            cell before normalising.
        pseudocount: Pseudo-count :math:`\epsilon` used for Laplace smoothing.
            Ignored when ``laplace_smoothing=False``.
        use_frequency_regularization: If ``True``, mix the frequency
            distribution with the uniform distribution.
        freq_regularization_amplitude: Mixing coefficient :math:`\lambda` for
            frequency regularization. Default follows the RSALOR supplementary
            material :cite:p:`RSALOR`.

    Returns:
        Float tensor of shape ``(L, 20)`` (or ``(L, 21)`` when
        ``aa_cols_only=False``) containing per-position amino-acid frequencies.
        Masked columns contain ``nan``.
    """
    msa_tokens = msa.msa_tokens
    tokenizer = tokenizers.tokenizer
    all_tokens = [*tokenizer.aa_token_ids, tokenizer.gap_token_id]
    all_tokens = torch.tensor(all_tokens, device=msa_tokens.device)
    one_hot_msa = (msa_tokens[:, :, None] == all_tokens[None, None, :]).long()
    if use_sample_reweighting:
        sequence_weights = msa.get_sequence_weights()[:, None, None]
        count_matrix = (sequence_weights * one_hot_msa).sum(dim=0)
    else:
        count_matrix = one_hot_msa.sum(dim=0)
    if aa_cols_only:
        count_matrix = count_matrix[:, :-1]
    if laplace_smoothing:
        count_matrix = count_matrix + pseudocount
    freq_matrix = count_matrix / count_matrix.sum(dim=-1, keepdim=True)
    if use_frequency_regularization:
        freq_matrix = (
            freq_matrix * (1 - freq_regularization_amplitude)
            + freq_regularization_amplitude / freq_matrix.shape[1]
        )
    if msa.valid_cols is not None:
        freq_matrix[~msa.valid_cols] = torch.nan
    return freq_matrix


def compute_msa_frequency_score(
    wt_tokens: Integer[torch.Tensor, "1 l"],
    mt_tokens: Integer[torch.Tensor, "b l"],
    freq_matrix: Float[torch.Tensor, "l v"],
    *,
    softmax_transform: bool = False,
    use_log_odds: bool = False,
    weight: Float[torch.Tensor, " l"] | None = None,
) -> Float[torch.Tensor, " b"]:
    r"""Score variants using a pre-computed position-specific frequency matrix.

    For each mutant, sums the log-frequency difference between the mutant and
    wild-type residue over all mutated positions :math:`M`:

    .. math::

        s = \sum_{l \in M} w_l \cdot
            \bigl(\hat{p}(x_l^{mt}) - \hat{p}(x_l^{wt})\bigr)

    where :math:`\hat{p}` is determined by the chosen transform:

    +-------------------------------+------------------------------------------+
    | Option                        | :math:`\hat{p}(f)`                       |
    +===============================+==========================================+
    | default                       | :math:`\log f`  :cite:p:`Tranception`    |
    +-------------------------------+------------------------------------------+
    | ``softmax_transform=True``    | :math:`\log \text{softmax}(f)`           |
    |                               | :cite:p:`VenusREM`                       |
    +-------------------------------+------------------------------------------+
    | ``use_log_odds=True``         | :math:`\text{logit}(f)`                  |
    |                               | :cite:p:`RSALOR`                         |
    +-------------------------------+------------------------------------------+

    .. note::
        ``use_log_odds`` takes precedence over ``softmax_transform`` when both
        are set to ``True``.

    Args:
        wt_tokens: Tokenised wild-type sequence of shape ``(1, L)``.
        mt_tokens: Tokenised mutant sequences of shape ``(B, L)``.
        freq_matrix: Position-specific frequency matrix of shape ``(L, V)``
            as returned by :func:`compute_frequency_matrix`.  ``V`` must be
            compatible with the token ids in ``wt_tokens`` and ``mt_tokens``.
        softmax_transform: If ``True``, apply :math:`\log \text{softmax}` to
            the frequency matrix before scoring.
        use_log_odds: If ``True``, apply :math:`\text{logit}` to the frequency
            matrix before scoring.
        weight: Per-position weights :math:`w_l` of shape ``(L,)``.
            Defaults to ``None`` (uniform weight of 1).

    Returns:
        Float tensor of shape ``(B,)`` where element ``i`` is the
        frequency-based score for mutant ``i``.  Higher scores indicate the
        mutant residues are more probable than the wild-type under the MSA.
    """
    if use_log_odds:
        log_p = torch.logit(freq_matrix)
    elif softmax_transform:
        log_p = torch.log_softmax(freq_matrix, dim=-1)
    else:
        log_p = torch.log(freq_matrix)
    if weight is not None:
        log_p = weight.unsqueeze(-1) * log_p
    mutation_mask = wt_tokens != mt_tokens
    wt_log_p = log_p[None].gather(dim=2, index=wt_tokens.unsqueeze(-1)).squeeze(-1)
    mt_log_p = einops.repeat(log_p, "l v -> b l v", b=mt_tokens.shape[0])
    mt_log_p = mt_log_p.gather(dim=2, index=mt_tokens.unsqueeze(-1)).squeeze(-1)
    score = ((mt_log_p - wt_log_p) * mutation_mask).sum(dim=1)
    return score
