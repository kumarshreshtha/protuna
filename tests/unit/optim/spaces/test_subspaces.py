from __future__ import annotations

import dataclasses
import operator

from jaxtyping import Float
import pytest
import torch

from protuna import constants
from protuna import inference
from protuna.data import tokenizers
from protuna.data.msa import core
from protuna.optim.spaces import rules
from protuna.optim.spaces import search
from protuna.optim.spaces import subspaces
from protuna.typing import SequenceLike, TokenIDs

_TOKENIZER = tokenizers.ProteinTokenizer()

# ProteinTokenizer.ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"
# A=0  C=1  D=2  E=3  F=4  G=5  H=6  I=7  K=8  L=9
# M=10 N=11 P=12 Q=13 R=14 S=15 T=16 V=17 W=18 Y=19 gap=20 unk=21
_ALL_AA = "".join(constants.PROTEIN_ALPHABET)


def _make_msa(sequences: list[str]) -> core.TokenizedMSA:
    tokens = torch.stack(_TOKENIZER.encode(sequences), dim=0)
    return core.TokenizedMSA(
        msa_tokens=tokens, valid_cols=torch.ones(tokens.shape[1], dtype=torch.bool)
    )


@dataclasses.dataclass
class _FixedLogitsModel(
    inference.ModelFreeLocalHandler[TokenIDs, Float[torch.Tensor, "l v"]]
):
    """Return fixed logits for each requested sequence."""

    logits: Float[torch.Tensor, "l v"]

    def __call__(
        self, requests: SequenceLike[TokenIDs]
    ) -> list[Float[torch.Tensor, "l v"]]:
        return [self.logits] * len(requests)


def test_sample_position_mask_selects_the_requested_number_of_positions() -> None:
    """Exactly n_mutation positions are selected."""
    mutable = torch.tensor([True, False, True, True])
    mask = subspaces.sample_position_mask(mutable, 2, torch.ones(4))
    assert int(mask.sum()) == 2


def test_sample_position_mask_never_selects_an_immutable_position() -> None:
    """Immutable positions are excluded from the draw."""
    mutable = torch.tensor([True, False, True, True])
    mask = subspaces.sample_position_mask(mutable, 3, torch.ones(4))
    assert mask.tolist() == [True, False, True, True]


def test_sample_position_mask_never_selects_a_zero_weight_position() -> None:
    """A position with zero weight is never drawn."""
    mutable = torch.tensor([True, True, True, True])
    weights = torch.tensor([1.0, 0.0, 1.0, 1.0])
    mask = subspaces.sample_position_mask(mutable, 3, weights)
    assert mask.tolist() == [True, False, True, True]


def test_sample_position_mask_returns_all_mutable_when_asked_for_too_many() -> None:
    """Requesting more mutations than there are mutable sites returns them all."""
    mutable = torch.tensor([True, False, True, False])
    mask = subspaces.sample_position_mask(mutable, 10, torch.ones(4))
    assert mask.tolist() == [True, False, True, False]


def test_sample_position_mask_leaves_the_caller_weights_untouched() -> None:
    """The weights passed in are not modified."""
    weights = torch.ones(4)
    subspaces.sample_position_mask(torch.tensor([True, False, True, True]), 2, weights)
    assert weights.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_sample_subspace_keeps_only_n_mutation_mutable_positions() -> None:
    """The subspace opens up exactly n_mutation sites for mutation."""
    space = search.SearchSpace.from_rules("ACDE", [])
    subspace = subspaces.sample_subspace(space, 2, torch.ones(4), generator=None)
    assert int(subspace.mutable_positions.sum()) == 2


def test_sample_subspace_preserves_the_init_sequence() -> None:
    """Sub-sampling never changes the parent sequence."""
    space = search.SearchSpace.from_rules("ACDE", [])
    subspace = subspaces.sample_subspace(space, 2, torch.ones(4), generator=None)
    assert subspace.init_sequence.tolist() == [0, 1, 2, 3]


def test_sample_subspace_keeps_the_init_sequence_valid() -> None:
    """Positions masked out of the subspace still allow the wild-type residue."""
    space = search.SearchSpace.from_rules("ACDE", [])
    subspace = subspaces.sample_subspace(space, 1, torch.ones(4), generator=None)
    assert subspace.is_valid("ACDE") is True


def test_sample_subspace_only_narrows_the_parent_space() -> None:
    """A subspace never allows anything the parent space forbids."""
    space = search.SearchSpace.from_rules(
        "ACDE", [rules.Deny(positions=[0, 1, 2, 3], amino_acids=("W",))]
    )
    subspace = subspaces.sample_subspace(space, 2, torch.ones(4), generator=None)
    assert bool((subspace.allowed & ~space.allowed).any()) is False


def test_random_mask_sampler_yields_one_subspace_per_configured_slot() -> None:
    """n_subspaces sampled_spaces are produced."""
    space = search.SearchSpace.from_rules("ACDE", [])
    sampler = subspaces.RandomMaskSampler(n_subspaces=3, max_mutations=2)
    assert len(list(sampler(space))) == 3


def test_random_mask_sampler_caps_each_subspace_at_max_mutations() -> None:
    """Every subspace exposes at most max_mutations mutable sites."""
    space = search.SearchSpace.from_rules("ACDE", [])
    sampler = subspaces.RandomMaskSampler(n_subspaces=3, max_mutations=2)
    sampled_spaces = list(sampler(space))
    assert [int(s.mutable_positions.sum()) for s in sampled_spaces] == [2, 2, 2]


def test_logits_entropy_sampler_skips_zero_entropy_positions() -> None:
    """A position the model is certain about is never opened for mutation."""
    logits = torch.zeros(4, 20)
    logits[1, 0] = 1e4  # position 1 becomes a one-hot distribution: zero entropy
    space = search.SearchSpace.from_rules("ACDE", [])
    sampler = subspaces.LogitsEntropyMaskSampler(
        _FixedLogitsModel(logits), n_subspaces=1, max_mutations=3
    )
    subspace = next(iter(sampler(space)))
    assert subspace.mutable_positions.tolist() == [True, False, True, True]


def test_logits_entropy_sampler_yields_one_subspace_per_configured_slot() -> None:
    """n_subspaces sampled_spaces are produced."""
    space = search.SearchSpace.from_rules("ACDE", [])
    sampler = subspaces.LogitsEntropyMaskSampler(
        _FixedLogitsModel(torch.zeros(4, 20)), n_subspaces=3, max_mutations=2
    )
    assert len(list(sampler(space))) == 3


def test_msa_entropy_weights_are_lowest_at_the_conserved_column() -> None:
    """A fully conserved alignment column carries the least entropy."""
    msa = _make_msa(["ACDE", "AWDF", "AYGH", "AKLM"])
    sampler = subspaces.MSAEntropyMaskSampler(msa, n_subspaces=1, max_mutations=2)
    assert int(sampler.weights.argmin()) == 0


def test_msa_entropy_sampler_caps_each_subspace_at_max_mutations() -> None:
    """Every subspace exposes at most max_mutations mutable sites."""
    msa = _make_msa(["ACDE", "AWDF", "AYGH", "AKLM"])
    space = search.SearchSpace.from_rules("ACDE", [])
    sampler = subspaces.MSAEntropyMaskSampler(msa, n_subspaces=2, max_mutations=2)
    sampled_spaces = list(sampler(space))
    assert [int(s.mutable_positions.sum()) for s in sampled_spaces] == [2, 2]


def test_linear_partition_sampler_splits_mutable_sites_into_disjoint_blocks() -> None:
    """Mutable positions are partitioned into contiguous, non-overlapping blocks."""
    space = search.SearchSpace.from_rules("ACDE", [])
    sampled_spaces = list(subspaces.PartitionSampler(n_partitions=2)(space))
    assert [s.mutable_positions.tolist() for s in sampled_spaces] == [
        [True, True, False, False],
        [False, False, True, True],
    ]


def test_coupling_mask_sampler_caps_each_subspace_at_max_mutations() -> None:
    """Every subspace exposes at most max_mutations mutable sites."""
    coupling = torch.zeros(4, 4)
    coupling[0, 1] = 1.0
    coupling[1, 0] = 1.0
    space = search.SearchSpace.from_rules("ACDE", [])
    sampler = subspaces.CouplingMaskSampler(coupling, n_subspaces=2, max_mutations=2)
    sampled_spaces = list(sampler(space))
    assert [int(s.mutable_positions.sum()) for s in sampled_spaces] == [2, 2]


@pytest.mark.parametrize("temperature", [-1.0, float("nan"), float("inf")])
def test_coupling_mask_sampler_rejects_invalid_temperatures(
    temperature: float,
) -> None:
    """Reject temperatures that cannot define positive sampling weights."""
    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        subspaces.CouplingMaskSampler(
            torch.ones(3, 3), n_subspaces=1, max_mutations=2, temperature=temperature
        )


@pytest.mark.parametrize("budget", [2, 3])
def test_coupling_mask_sampler_grows_coupled_groups_at_zero_temperature(
    budget: int,
) -> None:
    """Fill the mutation budget with coupled sites at zero temperature."""
    space = search.SearchSpace.from_rules("ACD", [])
    sampler = subspaces.CouplingMaskSampler(
        torch.ones(3, 3),
        n_subspaces=1,
        max_mutations=budget,
        temperature=0.0,
        seed=7,
    )

    result = next(sampler(space))

    assert int(result.mutable_positions.sum()) == budget


@pytest.mark.parametrize("strength", [0.25, 4.0])
@pytest.mark.parametrize("temperature", [1e-40, 1e-300, 5e-324])
def test_coupling_mask_sampler_handles_tiny_positive_temperatures(
    strength: float,
    temperature: float,
) -> None:
    """Keep equal positive gains sampleable despite extreme temperature scaling."""
    space = search.SearchSpace.from_rules("ACD", [])
    sampler = subspaces.CouplingMaskSampler(
        torch.full((3, 3), strength),
        n_subspaces=1,
        max_mutations=2,
        temperature=temperature,
        seed=7,
    )

    result = next(sampler(space))

    assert int(result.mutable_positions.sum()) == 2


@pytest.mark.parametrize("budget, expected", [(0, 0), (1, 1), (2, 2), (9, 3)])
def test_coupling_mask_sampler_caps_budgets_by_available_sites(
    budget: int, expected: int
) -> None:
    """Fill the capped budget using uniform restarts when no edges exist."""
    space = search.SearchSpace.from_rules(
        "ACDE", [rules.AllowOnly(positions=[0, 2, 3])]
    )
    sampler = subspaces.CouplingMaskSampler(
        torch.zeros(4, 4), n_subspaces=1, max_mutations=budget, seed=7
    )

    result = next(sampler(space))

    assert int(result.mutable_positions.sum()) == expected


def test_coupling_mask_sampler_accepts_fully_frozen_spaces() -> None:
    """Preserve the parent when no positions can mutate."""
    space = search.SearchSpace.from_rules("ACD", [rules.AllowOnly(positions=[])])
    sampler = subspaces.CouplingMaskSampler(
        torch.ones(3, 3), n_subspaces=1, max_mutations=2
    )

    assert next(sampler(space)) == space


@pytest.mark.parametrize(
    "temperature, expected",
    [
        (1.0, [0.1824753, 0.1824753, 0.6350494]),
        (2.0, [0.2274752, 0.2274752, 0.5450497]),
    ],
)
def test_coupling_mask_sampler_matches_temperature_weighted_probabilities(
    temperature: float, expected: list[float]
) -> None:
    """Recover pair probabilities for uniform seeds and temperature-weighted growth."""
    coupling = torch.tensor([[0.0, 4.0, 1.0], [4.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    space = search.SearchSpace.from_rules("ACD", [])
    sampler = subspaces.CouplingMaskSampler(
        coupling, n_subspaces=3000, max_mutations=2, temperature=temperature, seed=19
    )

    masks = torch.stack(
        tuple(map(operator.attrgetter("mutable_positions"), sampler(space)))
    )
    excluded_frequencies = (~masks).float().mean(dim=0)

    torch.testing.assert_close(
        excluded_frequencies, torch.tensor(expected), atol=0.035, rtol=0
    )


def test_coupling_mask_sampler_preserves_parent_constraints() -> None:
    """Keep frozen sites and amino acid restrictions in the sampled subspace."""
    space = search.SearchSpace.from_rules(
        "ACDE",
        [
            rules.AllowOnly(positions=[0, 2, 3]),
            rules.Deny(positions=[0, 1, 2, 3], amino_acids=("W",)),
        ],
    )
    sampler = subspaces.CouplingMaskSampler(
        torch.ones(4, 4), n_subspaces=1, max_mutations=2, seed=7
    )

    result = next(sampler(space))

    assert not bool(result.mutable_positions[1])
    assert not bool((result.allowed & ~space.allowed).any())


def test_coupling_mask_sampler_keeps_the_init_sequence_valid() -> None:
    """Allow the parent sequence after masking mutation sites."""
    space = search.SearchSpace.from_rules("ACDE", [])
    sampler = subspaces.CouplingMaskSampler(
        torch.ones(4, 4), n_subspaces=1, max_mutations=2, seed=7
    )

    result = next(sampler(space))

    assert result.init_sequence.tolist() == [0, 1, 2, 3]
    assert result.is_valid("ACDE")


def test_coupling_mask_sampler_leaves_the_parent_space_untouched() -> None:
    """Preserve the caller's allowed mutations during sampling."""
    space = search.SearchSpace.from_rules("ACDE", [])
    parent_allowed = space.allowed.clone()
    sampler = subspaces.CouplingMaskSampler(
        torch.ones(4, 4), n_subspaces=1, max_mutations=2, seed=7
    )

    list(sampler(space))

    assert torch.equal(space.allowed, parent_allowed)


@pytest.mark.parametrize("n_subspaces", [0, 3])
def test_coupling_mask_sampler_yields_the_requested_count(n_subspaces: int) -> None:
    """Emit the requested number of subspaces, including zero."""
    space = search.SearchSpace.from_rules("ACDE", [])
    sampler = subspaces.CouplingMaskSampler(
        torch.zeros(4, 4), n_subspaces=n_subspaces, max_mutations=2
    )

    assert len(list(sampler(space))) == n_subspaces


def test_coupling_mask_sampler_reproduces_seeded_draws() -> None:
    """Reproduce the full sampling stream across repeated calls with equal seeds."""
    space = search.SearchSpace.from_rules("ACDE", [])
    first = subspaces.CouplingMaskSampler(
        torch.ones(4, 4), n_subspaces=4, max_mutations=2, seed=9
    )
    second = subspaces.CouplingMaskSampler(
        torch.ones(4, 4), n_subspaces=4, max_mutations=2, seed=9
    )

    assert list(first(space)) == list(second(space))
    assert list(first(space)) == list(second(space))
