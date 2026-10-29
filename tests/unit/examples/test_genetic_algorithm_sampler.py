import pathlib
import runpy

import pytest
import torch

from examples import genetic_algorithm_sampler
from protuna import data
from protuna.optim import spaces
from tests.unit.optim.samplers import _helpers


def test_genetic_algorithm_example_prints_the_optimal_toy_sequence(
    capsys: pytest.CaptureFixture[str],
    examples_dir: pathlib.Path,
) -> None:
    """Execute the documented entry point, including handler load and release."""
    runpy.run_path(
        str(examples_dir / "genetic_algorithm_sampler.py"),
        run_name="__main__",
    )

    assert capsys.readouterr().out == "AAAA\n"


def test_alanine_fitness_counts_alanines_in_request_order() -> None:
    """Return one scalar alanine count for each input sequence."""
    model = genetic_algorithm_sampler.AlanineFitness()

    scores = model(data.tokenizer.encode(["ACDE", "AAAA", "CCDD"]))

    torch.testing.assert_close(torch.stack(scores), torch.tensor([1.0, 4.0, 0.0]))


@pytest.mark.parametrize("mutation_rate", [0.0, 0.1, 1.0])
def test_genetic_algorithm_respects_constraints_during_evolution(
    mutation_rate: float,
) -> None:
    """Keep fixed sites and allowed residues through crossover and mutation."""
    space = _helpers._space("CCDE", {0: "AC", 2: "AD"})
    sampler = genetic_algorithm_sampler.GeneticAlgorithmSampler(
        genetic_algorithm_sampler.AlanineFitness(),
        population_size=16,
        generations=3,
        mutation_rate=mutation_rate,
        seed=42,
    )
    original_mask = space.allowed.clone()
    original_sequence = space.init_sequence.clone()

    batch = sampler.sample([space])[0]

    assert 0 < len(batch) <= 16
    assert _helpers._all_residues_allowed(batch, space)
    torch.testing.assert_close(space.allowed, original_mask)
    torch.testing.assert_close(space.init_sequence, original_sequence)


def test_genetic_algorithm_returns_one_batch_per_space_including_immutable() -> None:
    """Preserve request order and return a shaped empty batch for fixed spaces."""
    sampler = genetic_algorithm_sampler.GeneticAlgorithmSampler(
        genetic_algorithm_sampler.AlanineFitness(),
        population_size=16,
        generations=2,
        seed=42,
    )

    batches = sampler.sample(
        [_helpers._space("CC", {0: "AC"}), _helpers._space("CDE", {})]
    )

    assert len(batches) == 2
    assert batches[0].shape[0] > 0
    assert batches[0].shape[1] == 2
    assert batches[1].shape == (0, 3)
    assert batches[1].dtype == torch.long


def test_genetic_algorithm_reproduces_candidates_with_the_same_seed() -> None:
    """Independent samplers with the same configuration reproduce proposals."""
    space = spaces.SearchSpace.from_rules("ACDE", [])
    first = genetic_algorithm_sampler.GeneticAlgorithmSampler(
        genetic_algorithm_sampler.AlanineFitness(),
        population_size=16,
        generations=3,
        seed=42,
    )
    second = genetic_algorithm_sampler.GeneticAlgorithmSampler(
        genetic_algorithm_sampler.AlanineFitness(),
        population_size=16,
        generations=3,
        seed=42,
    )

    first_batch = first.sample([space])[0]
    second_batch = second.sample([space])[0]

    torch.testing.assert_close(first_batch, second_batch)
