"""Tests for the search rules in protuna.optim.spaces.rules."""

from __future__ import annotations

import torch

from protuna.data import tokenizers
from protuna.data.msa import core
from protuna.optim.spaces import rules

_TOKENIZER = tokenizers.ProteinTokenizer()

# ProteinTokenizer.ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"
# A=0  C=1  D=2  E=3  F=4  G=5  H=6  I=7  K=8  L=9
# M=10 N=11 P=12 Q=13 R=14 S=15 T=16 V=17 W=18 Y=19 gap=20 unk=21


def _full_mask(length: int = 4) -> torch.Tensor:
    """An all-permissive ``(length, 20)`` search mask."""
    return torch.ones(length, 20, dtype=torch.bool)


def _make_msa(sequences: list[str]) -> core.TokenizedMSA:
    tokens = torch.stack(_TOKENIZER.encode(sequences), dim=0)
    return core.TokenizedMSA(
        msa_tokens=tokens, valid_cols=torch.ones(tokens.shape[1], dtype=torch.bool)
    )


# ---------------------------------------------------------------------------
# AllowOnly
# ---------------------------------------------------------------------------


def test_allow_only_keeps_the_listed_positions_and_closes_the_rest() -> None:
    """Listed positions keep the full alphabet; every other position is closed."""
    mask = rules.AllowOnly(positions=[0, 2])(_full_mask())
    assert mask.sum(dim=1).tolist() == [20, 0, 20, 0]


def test_allow_only_preserves_restrictions_from_earlier_rules() -> None:
    """Allowed sites retain denied residues and fully fixed sites stay closed."""
    denied = rules.Deny(positions=[0], amino_acids=("W",))(_full_mask())
    denied = rules.FixSites(positions=[2])(denied)
    mask = rules.AllowOnly(positions=[0, 2])(denied)
    assert torch.equal(mask[0], denied[0])
    assert mask.sum(dim=1).tolist() == [19, 0, 0, 0]


def test_allow_only_without_positions_closes_every_position() -> None:
    """An empty position list leaves no residue allowed anywhere."""
    mask = rules.AllowOnly(positions=[])(_full_mask())
    assert bool(mask.any()) is False


def test_allow_only_leaves_the_caller_mask_untouched() -> None:
    """AllowOnly returns a fresh mask instead of editing the one passed in."""
    original = _full_mask()
    rules.AllowOnly(positions=[0])(original)
    assert original.all()


# ---------------------------------------------------------------------------
# Allow
# ---------------------------------------------------------------------------


def test_allow_opens_the_listed_amino_acids_at_the_listed_positions() -> None:
    """Only the named residues become available at the named positions."""
    mask = rules.Allow(positions=[1], amino_acids=("A", "W"))(
        torch.zeros(4, 20, dtype=torch.bool)
    )
    assert torch.nonzero(mask[1]).flatten().tolist() == [0, 18]


def test_allow_leaves_positions_outside_the_rule_closed() -> None:
    """Positions the rule does not name are not opened."""
    mask = rules.Allow(positions=[1], amino_acids=("A", "W"))(
        torch.zeros(4, 20, dtype=torch.bool)
    )
    assert mask.sum(dim=1).tolist() == [0, 2, 0, 0]


def test_allow_defaults_to_the_whole_alphabet() -> None:
    """Omitting amino_acids opens all 20 residues at the listed positions."""
    mask = rules.Allow(positions=[2])(torch.zeros(4, 20, dtype=torch.bool))
    assert mask.sum(dim=1).tolist() == [0, 0, 20, 0]


def test_allow_without_positions_leaves_the_mask_unchanged() -> None:
    """An empty position list is a no-op."""
    mask = rules.Allow(positions=[], amino_acids=("A",))(
        torch.zeros(4, 20, dtype=torch.bool)
    )
    assert bool(mask.any()) is False


# ---------------------------------------------------------------------------
# Deny
# ---------------------------------------------------------------------------


def test_deny_closes_the_listed_amino_acids_at_the_listed_positions() -> None:
    """The named residues are removed from the named positions only."""
    mask = rules.Deny(positions=[1], amino_acids=("A", "W"))(_full_mask())
    assert mask.sum(dim=1).tolist() == [20, 18, 20, 20]


def test_deny_removes_exactly_the_named_residues() -> None:
    """Residues other than the named ones stay available."""
    mask = rules.Deny(positions=[1], amino_acids=("A", "W"))(_full_mask())
    assert torch.nonzero(~mask[1]).flatten().tolist() == [0, 18]


def test_deny_defaults_to_closing_the_whole_alphabet() -> None:
    """Omitting amino_acids denies every residue at the listed positions."""
    mask = rules.Deny(positions=[0, 3])(_full_mask())
    assert mask.sum(dim=1).tolist() == [0, 20, 20, 0]


def test_deny_without_positions_leaves_the_mask_unchanged() -> None:
    """An empty position list is a no-op."""
    mask = rules.Deny(positions=[], amino_acids=("A",))(_full_mask())
    assert mask.all()


# ---------------------------------------------------------------------------
# FixSites
# ---------------------------------------------------------------------------


def test_fix_sites_closes_every_residue_at_the_listed_positions() -> None:
    """A fixed site offers no residue at all in the raw mask."""
    mask = rules.FixSites(positions=[1, 2])(_full_mask())
    assert mask.sum(dim=1).tolist() == [20, 0, 0, 20]


def test_fix_sites_without_positions_leaves_the_mask_unchanged() -> None:
    """An empty position list is a no-op."""
    mask = rules.FixSites(positions=[])(_full_mask())
    assert mask.all()


def test_fix_sites_tolerates_a_repeated_position() -> None:
    """Listing a position twice closes it once, without error."""
    mask = rules.FixSites(positions=[1, 1])(_full_mask())
    assert mask.sum(dim=1).tolist() == [20, 0, 20, 20]


# ---------------------------------------------------------------------------
# DenyResidues
# ---------------------------------------------------------------------------


def test_deny_residues_closes_the_residue_at_every_position() -> None:
    """A denied residue disappears from the whole mask."""
    mask = rules.DenyResidues(amino_acids=["C"])(_full_mask())
    assert mask[:, 1].tolist() == [False, False, False, False]
    assert mask.sum(dim=1).tolist() == [19, 19, 19, 19]


# ---------------------------------------------------------------------------
# Preserve
# ---------------------------------------------------------------------------


def test_preserve_closes_positions_holding_a_preserved_residue() -> None:
    """Every occurrence of the residue in the target sequence becomes immutable."""
    mask = rules.Preserve(amino_acids=["C"], target_sequence="ACDC")(_full_mask())
    assert mask.sum(dim=1).tolist() == [20, 0, 20, 0]


def test_preserve_closes_positions_for_any_of_the_listed_residues() -> None:
    """Positions matching any listed residue are protected."""
    mask = rules.Preserve(amino_acids=["C", "D"], target_sequence="ACDE")(_full_mask())
    assert mask.sum(dim=1).tolist() == [20, 0, 0, 20]


def test_preserve_leaves_the_mask_unchanged_when_the_residue_is_absent() -> None:
    """A residue that never occurs in the target sequence protects nothing."""
    mask = rules.Preserve(amino_acids=["W"], target_sequence="ACDE")(_full_mask())
    assert mask.all()


# ---------------------------------------------------------------------------
# MSAConserved
# ---------------------------------------------------------------------------


def test_msa_conserved_closes_positions_above_the_threshold() -> None:
    """Columns where the query residue is conserved beyond the threshold close."""
    msa = _make_msa(["ACDE", "ACWF", "ACYG", "ACKH"])
    mask = rules.MSAConserved(tokenized_msa=msa, threshold=0.9)(_full_mask())
    assert mask.sum(dim=1).tolist() == [0, 0, 20, 20]


def test_msa_conserved_keeps_positions_below_the_threshold() -> None:
    """A threshold above the strongest conservation level protects nothing."""
    msa = _make_msa(["ACDE", "ACWF", "ACYG", "ACKH"])
    mask = rules.MSAConserved(tokenized_msa=msa, threshold=1.0)(_full_mask())
    assert mask.all()


def test_msa_conserved_with_a_zero_threshold_closes_every_position() -> None:
    """Every frequency clears a threshold of zero."""
    msa = _make_msa(["ACDE", "ACWF", "ACYG", "ACKH"])
    mask = rules.MSAConserved(tokenized_msa=msa, threshold=0.0)(_full_mask())
    assert bool(mask.any()) is False


# ---------------------------------------------------------------------------
# MSACovered
# ---------------------------------------------------------------------------


def test_msa_covered_closes_low_coverage_positions() -> None:
    """A column that is mostly gaps in the alignment becomes immutable."""
    msa = _make_msa(["ACDE", "A-DE", "A-DE", "A-DE"])
    mask = rules.MSACovered(tokenized_msa=msa, threshold=0.5)(_full_mask())
    assert mask.sum(dim=1).tolist() == [20, 0, 20, 20]


def test_msa_covered_keeps_a_position_whose_coverage_equals_the_threshold() -> None:
    """The threshold is inclusive: coverage equal to it still counts as covered."""
    msa = _make_msa(["ACDE", "A-DE", "A-DE", "A-DE"])
    mask = rules.MSACovered(tokenized_msa=msa, threshold=0.25)(_full_mask())
    assert mask.all()


def test_msa_covered_with_a_zero_threshold_keeps_every_position() -> None:
    """No coverage falls below zero, so nothing is closed."""
    msa = _make_msa(["ACDE", "----", "----", "----"])
    mask = rules.MSACovered(tokenized_msa=msa, threshold=0.0)(_full_mask())
    assert mask.all()
