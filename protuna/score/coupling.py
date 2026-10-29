"""Estimate residue couplings for proteins."""

from collections.abc import Iterator

from jaxtyping import Float
import torch

from protuna import inference
from protuna.data import tokenizer
from protuna.typing import TokenIDs


def categorical_jacobian(
    wt_tokens: TokenIDs, model: inference.MLMForLogitsLike
) -> Float[torch.Tensor, "l l"]:
    """Compute average-product-corrected pairwise residue coupling scores.

    Args:
        wt_tokens: Wild-type amino acid token IDs of shape ``(L,)``.
        model: Handler returning amino acid logits for each sequence position.

    Returns:
        Symmetric coupling scores of shape ``(L, L)`` with a zero diagonal.
    """
    request = inference.MLMLogitsRequest(wt_tokens, False, True)
    base_logits = model([request])[0]

    def mutant_iterator() -> Iterator[inference.MLMLogitsRequest]:
        for i in range(wt_tokens.shape[0]):
            for v in tokenizer.aa_token_ids:
                if v == int(wt_tokens[i]):
                    continue
                mutant = wt_tokens.clone()
                mutant[i] = v
                request = inference.MLMLogitsRequest(mutant, False, True)
                yield request

    jacobian = base_logits.new_zeros([*base_logits.shape, *base_logits.shape])
    for _, request, logits in model.stream(enumerate(mutant_iterator())):
        mut_pos = (request.token_ids != wt_tokens).nonzero(as_tuple=True)[0]
        aa = request.token_ids[mut_pos].long()
        jacobian[mut_pos, aa] = logits - base_logits

    symmetrized_jacobian = (jacobian + jacobian.permute(2, 3, 0, 1)) / 2
    for dim in range(4):
        symmetrized_jacobian = symmetrized_jacobian - symmetrized_jacobian.mean(
            dim=dim, keepdim=True
        )
    coupling_strengths = symmetrized_jacobian.square().sum(dim=(1, 3)).sqrt()
    coupling_strengths.fill_diagonal_(0)
    total_strength = coupling_strengths.sum()
    if total_strength == 0:
        return coupling_strengths
    background = (
        coupling_strengths.sum(dim=1, keepdim=True)
        @ coupling_strengths.sum(dim=0, keepdim=True)
    ) / total_strength
    corrected_couplings = coupling_strengths - background
    corrected_couplings.fill_diagonal_(0)
    return corrected_couplings
