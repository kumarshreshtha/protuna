from __future__ import annotations

import torch

from protuna import constants
from protuna.data import tokenizers
from protuna.optim.spaces import design
from protuna.optim.spaces import rules
from protuna.optim.spaces import subspaces

_TOKENIZER = tokenizers.ProteinTokenizer()

# ProteinTokenizer.ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"
# A=0  C=1  D=2  E=3  F=4  G=5  H=6  I=7  K=8  L=9
# M=10 N=11 P=12 Q=13 R=14 S=15 T=16 V=17 W=18 Y=19 gap=20 unk=21
_ALL_AA = "".join(constants.PROTEIN_ALPHABET)


def test_design_space_from_rules_builds_one_space_per_init_sequence() -> None:
    """Each init sequence gets its own SearchSpace with the shared rules applied."""
    design_space = design.DesignSpace.from_rules(
        rules=[rules.FixSites(positions=[1])],
        init_sequences=[torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 4])],
        space_sampler=subspaces.RandomMaskSampler(n_subspaces=2, max_mutations=2),
    )
    assert [s.allowed.sum(dim=1).tolist() for s in design_space.init_spaces] == [
        [20, 1, 20, 20],
        [20, 1, 20, 20],
    ]


def test_sample_search_spaces_yields_n_subspaces_per_init_space() -> None:
    """Every init space is expanded into n_subspaces subspaces."""
    design_space = design.DesignSpace.from_rules(
        rules=[],
        init_sequences=[
            torch.tensor([0, 1, 2, 3, 4, 5]),
            torch.tensor([0, 1, 2, 3, 4, 6]),
        ],
        # PartitionSampler opens a different block of sites in every subspace, so
        # none of them is dropped by the duplicate-mask check.
        space_sampler=subspaces.PartitionSampler(n_partitions=3),
    )
    assert len(list(design_space.sample_search_spaces())) == 6


def test_sample_search_spaces_respects_max_mutations() -> None:
    """Respect the sampler's mutation limit when drawing design subspaces."""
    design_space = design.DesignSpace.from_rules(
        rules=[],
        init_sequences=[torch.tensor([0, 1, 2, 3])],
        space_sampler=subspaces.RandomMaskSampler(n_subspaces=2, max_mutations=1),
    )
    sampled_spaces = list(design_space.sample_search_spaces())
    assert {int(s.mutable_positions.sum()) for s in sampled_spaces} == {1}


def test_sample_search_spaces_drops_subspaces_that_repeat_a_mask() -> None:
    """A mask already drawn for an init space is not yielded a second time."""
    design_space = design.DesignSpace.from_rules(
        # Only position 0 stays mutable, so every draw opens the same single site.
        rules=[rules.FixSites(positions=[1, 2, 3])],
        init_sequences=[torch.tensor([0, 1, 2, 3])],
        space_sampler=subspaces.RandomMaskSampler(n_subspaces=5, max_mutations=1),
    )
    assert len(list(design_space.sample_search_spaces())) == 1


def test_sample_search_spaces_deduplicates_within_one_init_space_only() -> None:
    """Two init spaces may open the same sites; the mask is only unique per parent."""
    design_space = design.DesignSpace.from_rules(
        rules=[rules.FixSites(positions=[1, 2, 3])],
        init_sequences=[torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 4])],
        space_sampler=subspaces.RandomMaskSampler(n_subspaces=3, max_mutations=1),
    )
    sampled_spaces = list(design_space.sample_search_spaces())
    assert [s.init_sequence.tolist() for s in sampled_spaces] == [
        [0, 1, 2, 3],
        [0, 1, 2, 4],
    ]
