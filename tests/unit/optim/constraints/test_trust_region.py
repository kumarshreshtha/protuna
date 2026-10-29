import pytest
import torch

from protuna import data
from protuna.optim import constraints


@pytest.mark.parametrize(
    ("max_mutations", "expected"),
    [
        (0, [True, False, False, False]),
        (1, [True, True, False, True]),
        (2, [True, True, True, True]),
    ],
)
def test_max_mutation_constraint_accepts_the_inclusive_distance_limit(
    max_mutations: int, expected: list[bool]
) -> None:
    """Count substitutions against the parent independently for each candidate."""
    parent = data.tokenizer.encode("AC")
    candidates = data.tokenizer.encode(["AC", "CC", "CA", "AA"])
    constraint = constraints.MaxMutationConstraint(parent, max_mutations)

    verdicts = constraint(candidates)

    assert isinstance(verdicts, torch.Tensor)
    torch.testing.assert_close(verdicts, torch.tensor(expected))
    torch.testing.assert_close(parent, data.tokenizer.encode("AC"))
    torch.testing.assert_close(candidates[1], data.tokenizer.encode("CC"))


def test_max_mutation_constraint_handles_a_single_candidate() -> None:
    """Return one verdict for a single sequence."""
    constraint = constraints.MaxMutationConstraint(data.tokenizer.encode("AC"), 1)

    torch.testing.assert_close(
        constraint([data.tokenizer.encode("CC")])[0], torch.tensor(True)
    )


def test_max_mutation_constraint_accepts_empty_requests() -> None:
    """Return no verdicts when there are no candidates."""
    constraint = constraints.MaxMutationConstraint(data.tokenizer.encode("AC"), 1)

    torch.testing.assert_close(constraint([]), torch.empty(0, dtype=torch.bool))


def test_max_mutation_constraint_moves_parent_to_candidate_device() -> None:
    """Compare the reference on the device holding the candidates."""
    constraint = constraints.MaxMutationConstraint(data.tokenizer.encode("AC"), 1)

    verdict = constraint([data.tokenizer.encode("CC").to("meta")])[0]

    assert verdict.device == torch.device("meta")
    assert verdict.dtype == torch.bool
    assert verdict.shape == torch.Size([])
