from __future__ import annotations

import pytest
import torch

from protuna import constants
from protuna.data import tokenizers
from protuna.optim.spaces import rules
from protuna.optim.spaces import search

_TOKENIZER = tokenizers.ProteinTokenizer()

# ProteinTokenizer.ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"
# A=0  C=1  D=2  E=3  F=4  G=5  H=6  I=7  K=8  L=9
# M=10 N=11 P=12 Q=13 R=14 S=15 T=16 V=17 W=18 Y=19 gap=20 unk=21
_ALL_AA = "".join(constants.PROTEIN_ALPHABET)


def test_from_rules_without_rules_allows_every_amino_acid() -> None:
    """With no rules every position may take any of the 20 amino acids."""
    space = search.SearchSpace.from_rules("ACDE", [])
    assert space.allowed.shape == (4, 20)
    assert space.allowed.all()


def test_from_rules_accepts_tokenized_sequence() -> None:
    """A pre-tokenized init sequence is used as-is."""
    space = search.SearchSpace.from_rules(torch.tensor([0, 1, 2, 3]), [])
    assert space.init_sequence.tolist() == [0, 1, 2, 3]


def test_from_rules_applies_allow_only_rule() -> None:
    """AllowOnly restricts mutation to the listed positions."""
    space = search.SearchSpace.from_rules("ACDE", [rules.AllowOnly(positions=[0, 2])])
    assert space.allowed.sum(dim=1).tolist() == [20, 1, 20, 1]


def test_init_sequence_stays_allowed_after_a_denying_rule() -> None:
    """The wild-type residue survives a rule that denies it."""
    space = search.SearchSpace.from_rules(
        "ACDE", [rules.Deny(positions=[0], amino_acids=("A", "C"))]
    )
    # "A" is the wild-type residue at position 0 and is restored; "C" stays denied.
    assert space.available_aa_at_pos(0) == _ALL_AA.replace("C", "")


def test_size_is_the_product_of_per_position_choices() -> None:
    """size() multiplies the number of allowed residues at each position."""
    space = search.SearchSpace.from_rules("ACDE", [rules.FixSites(positions=[0, 1, 2])])
    assert space.size() == 20


def test_size_of_fully_fixed_space_is_one() -> None:
    """A space with no mutable position contains only the init sequence."""
    space = search.SearchSpace.from_rules(
        "ACDE", [rules.FixSites(positions=[0, 1, 2, 3])]
    )
    assert space.size() == 1


def test_mutable_positions_marks_positions_with_more_than_one_choice() -> None:
    """Positions pinned to the wild-type residue are not mutable."""
    space = search.SearchSpace.from_rules("ACDE", [rules.FixSites(positions=[1, 3])])
    assert space.mutable_positions.tolist() == [True, False, True, False]


def test_available_aa_at_fixed_position_is_the_wild_type_residue() -> None:
    """A fixed position offers only the residue from the init sequence."""
    space = search.SearchSpace.from_rules("ACDE", [rules.FixSites(positions=[1])])
    assert space.available_aa_at_pos(1) == "C"


def test_available_aa_at_free_position_is_the_full_alphabet() -> None:
    """An unconstrained position offers all 20 amino acids in alphabet order."""
    space = search.SearchSpace.from_rules("ACDE", [])
    assert space.available_aa_at_pos(0) == _ALL_AA


def test_is_valid_accepts_the_init_sequence() -> None:
    """The init sequence is always a member of its own search space."""
    space = search.SearchSpace.from_rules(
        "ACDE", [rules.FixSites(positions=[0, 1, 2, 3])]
    )
    assert space.is_valid("ACDE") is True


def test_is_valid_rejects_a_mutation_at_a_fixed_position() -> None:
    """A residue outside the allowed set makes the sequence invalid."""
    space = search.SearchSpace.from_rules("ACDE", [rules.FixSites(positions=[1])])
    assert space.is_valid("AWDE") is False


def test_is_valid_accepts_token_ids() -> None:
    """is_valid takes tokenized sequences as well as strings."""
    space = search.SearchSpace.from_rules("ACDE", [rules.FixSites(positions=[1])])
    assert space.is_valid(torch.LongTensor([18, 1, 2, 3])) is True


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_is_valid_batch_checks_each_sequence(device: str) -> None:
    """Batch validation agrees with scalar checks and preserves the device."""
    space = search.SearchSpace.from_rules(
        "ACDE", [rules.FixSites(positions=[1, 3])]
    ).to(torch.device(device))
    batch = torch.tensor(
        [[0, 1, 2, 3], [18, 1, 19, 3], [0, 18, 2, 3], [0, 1, 2, 19]],
        device=device,
    )

    result = space.is_valid_batch(batch)

    assert result.tolist() == [True, True, False, False]
    assert result.tolist() == [space.is_valid(sequence) for sequence in batch]
    assert result.shape == (batch.shape[0],)
    assert result.dtype == torch.bool
    assert result.device == batch.device


@pytest.mark.parametrize("batch_size", [0, 1, 3])
@pytest.mark.parametrize("length", [0, 2, 5])
def test_is_valid_batch_rejects_mismatched_lengths(
    batch_size: int, length: int
) -> None:
    """Wrong sequence lengths yield one false result per batch item."""
    space = search.SearchSpace.from_rules("ACDE", [])
    batch = torch.zeros((batch_size, length), dtype=torch.long)

    result = space.is_valid_batch(batch)

    assert result.shape == (batch_size,)
    assert result.dtype == torch.bool
    assert result.device == batch.device
    assert not result.any()


def test_is_valid_batch_accepts_an_empty_batch() -> None:
    """A batch with no sequences yields an empty boolean vector."""
    space = search.SearchSpace.from_rules("ACDE", [])

    result = space.is_valid_batch(torch.empty((0, 4), dtype=torch.long))

    assert result.shape == (0,)
    assert result.dtype == torch.bool


def test_is_valid_batch_accepts_a_single_sequence() -> None:
    """A singleton batch retains its batch dimension."""
    space = search.SearchSpace.from_rules("ACDE", [])

    result = space.is_valid_batch(torch.tensor([[19, 18, 17, 16]]))

    assert result.shape == (1,)
    assert result.tolist() == [True]


def test_is_valid_batch_accepts_noncontiguous_inputs() -> None:
    """Strided batches retain the same per-sequence validation results."""
    space = search.SearchSpace.from_rules("ACDE", [rules.FixSites(positions=[1])])
    batch = torch.tensor([[0, 18], [1, 1], [2, 2], [3, 3]]).T
    batch[1, 1] = 19
    assert not batch.is_contiguous()

    assert space.is_valid_batch(batch).tolist() == [True, False]


def test_equal_spaces_compare_equal() -> None:
    """Two spaces built from the same sequence and rules are equal."""
    left = search.SearchSpace.from_rules("ACDE", [rules.FixSites(positions=[1])])
    right = search.SearchSpace.from_rules("ACDE", [rules.FixSites(positions=[1])])
    assert left == right


def test_spaces_with_different_init_sequences_are_not_equal() -> None:
    """The init sequence is part of a space's identity."""
    left = search.SearchSpace.from_rules("ACDE", [])
    right = search.SearchSpace.from_rules("ACDF", [])
    assert left != right


def test_spaces_with_different_masks_are_not_equal() -> None:
    """The allowed mask is part of a space's identity."""
    left = search.SearchSpace.from_rules("ACDE", [])
    right = search.SearchSpace.from_rules("ACDE", [rules.FixSites(positions=[1])])
    assert left != right


def test_comparison_with_a_non_space_is_false() -> None:
    """Comparing against an unrelated object returns False, not an error."""
    space = search.SearchSpace.from_rules("ACDE", [])
    assert space != "ACDE"


def test_to_returns_an_equal_space_on_the_target_device() -> None:
    """Moving a space preserves its content."""
    space = search.SearchSpace.from_rules("ACDE", [])
    moved = space.to(torch.device("cpu"))
    assert moved == space
    assert moved.allowed.device.type == "cpu"
