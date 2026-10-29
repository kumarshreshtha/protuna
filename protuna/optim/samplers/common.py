"""Residue sampling utilities shared by candidate proposal strategies."""

from jaxtyping import Bool, Float
import torch

from protuna import inference
from protuna.typing import BatchTokenIDs, SequenceLike, TokenIDs


def sample_from_weights(
    weights: Float[torch.Tensor, "m v"],
    init_sequence: TokenIDs,
    input_mask: Bool[torch.Tensor, " l"],
    num_samples: int,
    *,
    dedup: bool = True,
    generator: torch.Generator | None = None,
) -> BatchTokenIDs:
    """Draw residue combinations at masked sites and keep other sites fixed.

    Args:
        weights: Nonnegative residue weights for each selected site.
        init_sequence: Tokenized sequence providing the initial residues.
        input_mask: Boolean mask over ``init_sequence``; ``True`` marks positions
            to replace. Rows of ``weights`` correspond to these positions in
            sequence order.
        num_samples: Number of sequences to draw with replacement before optional
            deduplication.
        dedup: Remove repeated combinations; fewer samples may be returned.
        generator: Optional random generator for reproducible sampling.

    Returns:
        Tokenized sequences containing the sampled residue combinations.
    """
    sampled_values = torch.multinomial(
        weights, num_samples, replacement=True, generator=generator
    )
    if sampled_values.numel() != 0 and dedup:
        sampled_values: torch.Tensor = sampled_values.unique(dim=1)
    samples = init_sequence.repeat(sampled_values.shape[1], 1)
    samples[:, input_mask] = sampled_values.T
    return samples


def top_p_mask(
    probs: Float[torch.Tensor, "m v"], top_p: float = 1.0
) -> Float[torch.Tensor, "m v"]:
    """Keep the smallest prefix of residues in descending order that reach``top_p``.

    Args:
        probs: Per-site residue probabilities.
        top_p: Cumulative probability threshold.

    Returns:
        A boolean mask in the original residue order.
    """
    prob_values, prob_indices = probs.sort(dim=-1, descending=True)
    cum_probs = prob_values.cumsum(dim=-1)
    mask = (cum_probs - prob_values) < top_p
    mask = mask.gather(-1, prob_indices.argsort(dim=-1))
    return mask


def modify_logits(
    logits: Float[torch.Tensor, "m v"],
    allowed: Bool[torch.Tensor, "m v"],
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Float[torch.Tensor, "m v"]:
    """Mask forbidden residues, scale by temperature, and truncate to top-p.

    Args:
        logits: Per-site residue logits.
        allowed: Boolean mask matching ``logits``; ``True`` allows a residue.
        temperature: Positive divisor controlling the sharpness of the distribution.
        top_p: Cumulative threshold defining the retained probability mass.

    Returns:
        Processed logits with excluded residues set to negative infinity.
    """
    logits = logits.masked_fill(~allowed, -torch.inf)
    logits = logits / temperature
    mask = top_p_mask(torch.softmax(logits, dim=-1), top_p=top_p)
    return logits.masked_fill(~mask, -torch.inf)


def sample_from_logits(
    logits: Float[torch.Tensor, "m v"],
    init_sequence: TokenIDs,
    input_mask: Bool[torch.Tensor, " l"],
    allowed: Bool[torch.Tensor, "m v"],
    num_samples: int,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> BatchTokenIDs:
    """Draw unique residue combinations from temperature-scaled allowed logits.

    See :func:`modify_logits` for temperature and top-p processing.

    Args:
        logits: Per-site residue logits for selected positions, with column indices
            used as token IDs.
        init_sequence: Tokenized sequence providing the fixed residues.
        input_mask: Boolean mask over ``init_sequence``; ``True`` marks positions
            to replace. Logit rows correspond to these positions in sequence order.
        allowed: Boolean mask matching ``logits``; ``True`` allows a residue.
        num_samples: Number of sequences to draw with replacement before removing
            duplicates. Ignored when ``temperature`` is zero.
        temperature: Positive divisor applied to logits, or zero to select the
            most likely allowed residue at each selected position.
        top_p: Cumulative probability threshold passed to :func:`modify_logits`.
            Ignored when ``temperature`` is zero.
        generator: Optional random generator for reproducible sampling. Ignored
            when ``temperature`` is zero.

    Returns:
        Unique tokenized sequences, or one sequence of the most likely allowed
        residues when ``temperature`` is zero.
    """
    if temperature == 0:
        logits = logits.masked_fill(~allowed, -torch.inf)
        sampled_values = torch.argmax(logits, dim=-1)
        samples = init_sequence.clone()
        samples[input_mask] = sampled_values
        return samples[None, :]

    logits = modify_logits(logits, allowed, temperature=temperature, top_p=top_p)
    probs = torch.softmax(logits, dim=-1)
    return sample_from_weights(
        probs, init_sequence, input_mask, num_samples, generator=generator
    )


def sample_from_mlm_response(
    response: inference.MLMOutput,
    init_sequence: TokenIDs,
    input_mask: Bool[torch.Tensor, " l"],
    allowed: Bool[torch.Tensor, "m v"],
    num_samples: int,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> BatchTokenIDs:
    """Fill masked sites from model logits, keeping other residues fixed.

    Uses :func:`sample_from_logits`.

    Args:
        response: Model output with non-``None`` logits for the selected positions,
            with column indices used as token IDs.
        init_sequence: Tokenized sequence providing the fixed residues.
        input_mask: Boolean mask over ``init_sequence``; ``True`` marks positions
            to replace. Response logit rows must follow these positions in sequence
            order.
        allowed: Boolean mask matching the response logits; ``True`` allows a
            residue.
        num_samples: Number of sequences to draw with replacement before removing
            duplicates. Ignored when ``temperature`` is zero.
        temperature: Positive divisor applied to logits, or zero to select the
            most likely allowed residue at each selected position.
        top_p: Cumulative probability threshold passed to :func:`sample_from_logits`.
            Ignored when ``temperature`` is zero.
        generator: Optional random generator for reproducible sampling. Ignored
            when ``temperature`` is zero.

    Returns:
        Tokenized sequences containing the selected residues.
    """
    assert response.logits is not None
    return sample_from_logits(
        response.logits,
        init_sequence,
        input_mask,
        allowed,
        num_samples,
        top_p=top_p,
        temperature=temperature,
        generator=generator,
    )


def execute_mlm_batch(
    batch: SequenceLike[TokenIDs],
    model: inference.MLMForLogitsLike,
    *,
    masked_positions_only: bool = True,
    aa_tokens_only: bool = True,
    device: torch.device | None = None,
) -> SequenceLike[Float[torch.Tensor, "l v"]]:
    """Return masked language model outputs for a batch of tokenized sequences.

    Args:
        batch: Tokenized sequences, including any mask tokens required
            by the model.
        model: Handler accepting masked language model requests.
        masked_positions_only: Return logits only for masked positions.
        aa_tokens_only: Return logits only for the standard amino acids.
        device: Device on which to move all results. If None, all responses are
            moved to the device of the first response.

    Returns:
        Per-sequence logits in batch order, all on the selected device.
    """
    requests = [
        inference.MLMLogitsRequest(
            token_ids,
            masked_positions_only=masked_positions_only,
            aa_tokens_only=aa_tokens_only,
        )
        for token_ids in batch
    ]
    responses = model(requests)
    if len(responses) == 0:
        return []
    device = responses[0].device if device is None else device
    return [logits.to(device) for logits in responses]
