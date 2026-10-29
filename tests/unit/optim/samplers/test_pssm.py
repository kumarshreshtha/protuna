from __future__ import annotations

from jaxtyping import Float
import torch

from protuna.data.msa import core
from protuna.optim.samplers import pssm
from tests.unit.optim.samplers import _helpers


def _one_hot_frequency_matrix(length: int, residue: str) -> Float[torch.Tensor, "l v"]:
    """Frequency matrix that puts all of its mass on a single residue."""
    matrix = torch.full(
        (length, len(_helpers._TOKENIZER.aa_token_ids)), _helpers._UNLIKELY
    )
    matrix[:, _helpers._TOKENIZER.encode(residue)] = 0.0
    return matrix


def test_pssm_sampler_proposes_the_highest_scoring_allowed_residue() -> None:
    """Mutable positions take the residue the frequency matrix favours."""
    space = _helpers._space("ACD", {0: "AW", 2: "DW"})
    pssm_sampler = pssm.PSSMSampler(_one_hot_frequency_matrix(3, "W"), max_samples=4)

    batch = pssm_sampler.sample([space])[0]

    assert _helpers._decode(batch) == ["WCW"]


def test_pssm_sampler_never_proposes_a_residue_outside_the_space() -> None:
    """Residues the space forbids are masked out however high they score."""
    space = _helpers._space("ACD", {0: "AW"})
    pssm_sampler = pssm.PSSMSampler(_one_hot_frequency_matrix(3, "K"), max_samples=20)

    batch = pssm_sampler.sample([space])[0]

    assert _helpers._residues_at(batch, 0) <= {"A", "W"}


def test_pssm_sampler_from_msa_scores_residues_by_alignment_frequency() -> None:
    """`from_msa` builds its logits from the alignment's frequency matrix."""
    tokens = torch.stack(_helpers._TOKENIZER.encode(["WCD", "WCE", "WCF"]), dim=0)
    msa = core.TokenizedMSA(
        msa_tokens=tokens, valid_cols=torch.ones(3, dtype=torch.bool)
    )

    pssm_sampler = pssm.PSSMSampler.from_msa(msa, max_samples=4)

    assert int(pssm_sampler.pssm_logits[0].argmax()) == _helpers._TOKENIZER.token_to_id(
        "W"
    )
