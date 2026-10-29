"""Genetic algorithm sampler with a pluggable fitness handler."""

from jaxtyping import Float
import torch

from protuna import data
from protuna import inference
from protuna.optim import samplers
from protuna.optim import spaces
from protuna.typing import BatchTokenIDs, SequenceLike, TokenIDs


# init-start
class GeneticAlgorithmSampler(samplers.Sampler):
    """Evolve allowed residue combinations toward higher fitness scores."""

    def __init__(
        self,
        model: inference.Handler[TokenIDs, Float[torch.Tensor, ""]],
        population_size: int = 64,
        generations: int = 10,
        mutation_rate: float = 0.1,
        *,
        seed: int | None = None,
        batch_size: int | None = None,
    ):
        """Configure population evolution and register the fitness handler.

        Args:
            model: Handler returning one scalar fitness per sequence; higher
                scores are preferred.
            population_size: Number of individuals retained in each generation.
                Must be at least 2.
            generations: Number of selection, crossover, and mutation rounds.
            mutation_rate: Probability of redrawing the residue at each site.
            seed: Seed for each device-specific sampling generator.
            batch_size: Number of search spaces per forward pass.
        """
        super().__init__(seed=seed, batch_size=batch_size)
        self.model = model
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    # init-end
    @torch.inference_mode()
    def sample(self, spaces: SequenceLike[spaces.SearchSpace]) -> list[BatchTokenIDs]:
        """Return one batch of candidates per search space."""
        batches = []
        for space in spaces:
            device = space.allowed.device
            length = space.init_sequence.shape[0]
            # immutable space
            if not bool(space.mutable_positions.any()):
                batches.append(
                    torch.empty((0, length), dtype=torch.long, device=device)
                )
                continue
            # base-end
            rng = self.rng(device)
            weights = space.allowed.float()
            population = torch.multinomial(
                weights, self.population_size, replacement=True, generator=rng
            ).T.contiguous()
            n_elites = max(1, self.population_size // 4)
            n_children = self.population_size - n_elites

            for _ in range(self.generations):
                scores = torch.stack(list(self.model(list(population)))).to(device)
                elites = population[scores.topk(n_elites).indices]
                parent_ids = torch.randint(
                    n_elites, (2, n_children), device=device, generator=rng
                )
                crossover = (
                    torch.rand((n_children, length), device=device, generator=rng) < 0.5
                )
                children = torch.where(
                    crossover, elites[parent_ids[0]], elites[parent_ids[1]]
                )
                mutations = torch.multinomial(
                    weights, n_children, replacement=True, generator=rng
                ).T
                mutate = (
                    torch.rand(children.shape, device=device, generator=rng)
                    < self.mutation_rate
                )
                children = torch.where(mutate, mutations, children)
                population = torch.cat((elites, children))

            batches.append(population)
        return batches
        # loop-end


# fitness-toy-start
class AlanineFitness(
    inference.ModelFreeLocalHandler[TokenIDs, Float[torch.Tensor, ""]]
):
    """A toy fitness model that maximizes for the amount of alanine in the protein."""

    def __call__(
        self, requests: SequenceLike[TokenIDs]
    ) -> list[Float[torch.Tensor, ""]]:
        """Return the alanine count in input sequence."""
        alanine_id = data.tokenizer.token_to_id("A")
        return [(tokens == alanine_id).float().sum() for tokens in requests]
        # fitness-toy-end


def main() -> None:
    """Entry point for the toy example. Expect first candidate to be "AAAA"."""
    # main-start
    fitness_model = AlanineFitness()
    sampler = GeneticAlgorithmSampler(
        model=fitness_model,
        population_size=64,
        generations=10,
        seed=42,
        batch_size=1,
    )
    space = spaces.SearchSpace.from_rules("ACDE", [])
    sampler.load()
    try:
        candidates = sampler([space])[0]
        sequences = data.tokenizer.decode(list(candidates))
        print(sequences[0])  # expect "AAAA"
    finally:
        sampler.release()
    # main-end


if __name__ == "__main__":
    main()
