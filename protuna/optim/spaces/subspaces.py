"""Callables that partition and sample a search space to create smaller subspaces."""

from collections.abc import Iterator
import dataclasses
import functools
import math
from typing import Protocol

from jaxtyping import Bool, Float
import torch

from protuna import data
from protuna import inference
from protuna.optim.spaces import search
from protuna.score import pssm
from protuna.typing import TokenIDs


class SearchSpaceSampler(Protocol):
    """Structural interface shared by all callables that divide the search space."""

    def __call__(self, space: search.SearchSpace) -> Iterator[search.SearchSpace]:
        """Yield generated subspaces of ``space``.

        Args:
            space: Search space to sample subspaces from.

        Returns:
            An iterator over sampled subspaces.
        """
        ...


class PartitionSampler(SearchSpaceSampler):
    """Split mutable sites into ``n_subspaces`` disjoint contiguous groups.

    Since the subspaces do not overlap, the union of the candidates proposed in
    them covers the original space exactly once.
    """

    def __init__(self, n_partitions: int):
        """Set the number of groups into which mutable sites are split.

        Args:
            n_partitions: Requested number of disjoint site groups.
        """
        self.n_partitions = n_partitions

    def __repr__(self) -> str:
        """Return a compact summary of the partition configuration."""
        return f"{type(self).__name__}(n_partitions={self.n_partitions})"

    def __call__(self, space: search.SearchSpace) -> Iterator[search.SearchSpace]:
        """Yield one subspace per group of sites.

        Args:
            space: Search space whose mutable sites are partitioned.

        Yields:
            A subspace allowing mutations in one group of sites.
        """
        mutable = torch.nonzero(space.mutable_positions, as_tuple=True)[0]
        sites_per_partition = math.ceil(mutable.numel() / self.n_partitions)
        for sites in torch.split(mutable, sites_per_partition):
            allowed = torch.zeros_like(space.allowed)
            allowed[sites] = space.allowed[sites]
            init_sequence = space.init_sequence
            allowed[torch.arange(init_sequence.shape[0]), init_sequence] = True
            yield search.SearchSpace(init_sequence, allowed)


def sample_position_mask(
    mutable: Bool[torch.Tensor, " l"],
    n_mutation: int,
    weights: Float[torch.Tensor, " l"],
    generator: torch.Generator | None = None,
) -> Bool[torch.Tensor, " l"]:
    """Draw ``n_mutation`` mutable sites without replacement, weighted by ``weights``.

    Returns ``mutable`` unchanged when it holds no more than ``n_mutation``
    sites.

    Args:
        mutable: Mask of sites eligible for selection.
        n_mutation: Maximum number of sites to select.
        weights: Per-site sampling weights.
        generator: Random generator for sampling, or ``None`` to use the default.

    Returns:
        A mask of the selected sites.
    """
    weights = weights.clone()
    weights[~mutable] = 0
    if mutable.sum() <= n_mutation:
        return mutable
    indices = torch.multinomial(
        weights, n_mutation, replacement=False, generator=generator
    )
    pos_mask = torch.zeros(mutable.shape[0], dtype=torch.bool)
    pos_mask[indices] = True
    return pos_mask


def sample_subspace(
    space: search.SearchSpace,
    n_mutation: int | None,
    weights: Float[torch.Tensor, " l"],
    generator: torch.Generator | None,
) -> search.SearchSpace:
    """Sample a subspace that leaves at most ``n_mutation`` sites open to mutations.

    Args:
        space: Search space to sample a subspace from.
        n_mutation: Maximum open sites, or ``None`` to retain all mutable sites.
        weights: Per-site sampling weights.
        generator: Random generator for sampling, or ``None`` to use the default.

    Returns:
        A subspace with mutations restricted to the selected sites.
    """
    if n_mutation is None:
        n_mutation = int(space.mutable_positions.sum())
    position_mask = sample_position_mask(
        space.mutable_positions, n_mutation, weights, generator=generator
    )
    allowed = space.allowed.clone()
    allowed[~position_mask] = False
    return dataclasses.replace(space, allowed=allowed)


class RandomMaskSampler(SearchSpaceSampler):
    """Draw from the mutable sites of each subspace uniformly at random."""

    def __init__(
        self,
        n_subspaces: int,
        max_mutations: int,
        *,
        seed: int | None = None,
    ):
        """Configure uniformly sampled mutation masks.

        Args:
            n_subspaces: Number of subspaces to draw per input space.
            max_mutations: Maximum mutable sites per subspace.
            seed: Seed for the sampling generator.
        """
        self.n_subspaces = n_subspaces
        self.max_mutations = max_mutations
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __repr__(self) -> str:
        """Return a compact summary of the sampling configuration."""
        return (
            f"{type(self).__name__}(n_subspaces={self.n_subspaces}, "
            f"max_mutations={self.max_mutations})"
        )

    def __call__(self, space: search.SearchSpace) -> Iterator[search.SearchSpace]:
        """Yield ``n_subspaces`` random subspaces.

        Args:
            space: Search space to sample subspaces from.

        Yields:
            A subspace with uniformly sampled open sites.
        """
        for _ in range(self.n_subspaces):
            yield sample_subspace(
                space,
                self.max_mutations,
                torch.ones(space.allowed.shape[0]),
                generator=self.generator,
            )


@dataclasses.dataclass
class MSAEntropyMaskSampler(SearchSpaceSampler):
    """Sample subspaces favoring sites that vary the most across an MSA.

    Sites are drawn with probability proportional to the Shannon entropy of
    their residue distribution in the alignment, so conserved sites are rarely
    opened for mutation.
    """

    def __init__(
        self,
        tokenized_msa: data.TokenizedMSA,
        n_subspaces: int,
        max_mutations: int,
        *,
        seed: int | None = None,
    ):
        """Configure mutation masks weighted by alignment entropy.

        Args:
            tokenized_msa: Alignment used to estimate per-site residue entropy.
            n_subspaces: Number of subspaces to draw per input space.
            max_mutations: Maximum mutable sites per subspace.
            seed: Seed for the sampling generator.
        """
        self.tokenized_msa = tokenized_msa
        self.n_subspaces = n_subspaces
        self.max_mutations = max_mutations
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __repr__(self) -> str:
        """Return a compact summary of the sampling configuration."""
        return (
            f"{type(self).__name__}(n_subspaces={self.n_subspaces}, "
            f"max_mutations={self.max_mutations})"
        )

    @functools.cached_property
    def weights(self) -> Float[torch.Tensor, " l"]:
        """Per-site entropy of the alignment.

        Returns:
            The Shannon entropy at each alignment site.
        """
        p = pssm.compute_frequency_matrix(
            self.tokenized_msa, aa_cols_only=True, laplace_smoothing=True
        )
        log_p = torch.log(p)
        p = log_p.exp()
        entropy = -(p * log_p).sum(dim=1)
        return entropy

    def __call__(self, space: search.SearchSpace) -> Iterator[search.SearchSpace]:
        """Yield ``n_subspaces`` entropy-weighted subspaces.

        Args:
            space: Search space to sample subspaces from.

        Yields:
            A subspace with open sites sampled by alignment entropy.
        """
        for _ in range(self.n_subspaces):
            yield sample_subspace(
                space, self.max_mutations, self.weights, generator=self.generator
            )


type LogitsModelLike = inference.Handler[TokenIDs, Float[torch.Tensor, "l v"]]


class LogitsEntropyMaskSampler(inference.HandlerRegistryMixin, SearchSpaceSampler):
    """Sample subspaces favoring sites a model is least certain about for mutations.

    Like :class:`MSAEntropyMaskSampler`, but the per-site entropy comes from a
    model's logits on the initial sequence rather than from an alignment.
    """

    def __init__(
        self,
        model: LogitsModelLike,
        n_subspaces: int,
        max_mutations: int,
        *,
        seed: int | None = None,
    ):
        """Configure mutation masks weighted by model entropy.

        Args:
            model: Handler returning per-site residue logits for tokenized sequences.
            n_subspaces: Number of subspaces to draw per input space.
            max_mutations: Maximum open sites per subspace.
            seed: Seed for each device-specific sampling generator.
        """
        super().__init__()
        self.model = model
        self.n_subspaces = n_subspaces
        self.max_mutations = max_mutations
        self.seed = seed
        self._rngs: dict[torch.device, torch.Generator] = {}

    def __repr__(self) -> str:
        """Return a compact summary of the sampling configuration."""
        return (
            f"{type(self).__name__}(n_subspaces={self.n_subspaces}, "
            f"max_mutations={self.max_mutations})"
        )

    def rng(self, device: torch.device) -> torch.Generator:
        """Return the persistent subspace sampling generator for ``device``.

        Args:
            device: Device on which random tensors will be generated.

        Returns:
            The persistent sampling generator for the requested device.
        """
        if device not in self._rngs:
            rng = torch.Generator(device=device)
            if self.seed is not None:
                rng.manual_seed(self.seed)
            self._rngs[device] = rng
        return self._rngs[device]

    def _get_sequence_entropy(
        self,
        tokens: TokenIDs,
    ) -> Float[torch.Tensor, " l"]:
        logits = self.model([tokens])[0]
        log_p = torch.log_softmax(logits, dim=1)
        p = log_p.exp()
        entropy = -(p * log_p).sum(dim=1)
        return entropy

    def __call__(self, space: search.SearchSpace) -> Iterator[search.SearchSpace]:
        """Yield ``n_subspaces`` entropy-weighted subspaces.

        Args:
            space: Search space to sample subspaces from.

        Yields:
            A subspace on the model output device, with open sites sampled by
            model entropy.
        """
        weights = self._get_sequence_entropy(space.init_sequence).to(
            space.allowed.device
        )
        for _ in range(self.n_subspaces):
            yield sample_subspace(
                space,
                self.max_mutations,
                weights,
                generator=self.rng(weights.device),
            )


class SiteSelector(Protocol):
    """Callable protocol for selecting the next site to add to mask."""

    def __call__(
        self,
        space: search.SearchSpace,
        selected_sites: list[int],
        generator: torch.Generator,
    ) -> int:
        """Select the next site to add to the mask.

        Args:
            space: Search space containing the eligible mutation sites.
            selected_sites: Indices of sites already included in the mask.
            generator: Random generator for site selection.

        Returns:
            The index of the next site to include.
        """
        ...


def random_site_selector(
    space: search.SearchSpace, selected_sites: list[int], generator: torch.Generator
) -> int:
    """Select a site uniformly at random from available sites.

    Args:
        space: Search space containing the eligible mutation sites.
        selected_sites: Indices of sites to exclude from selection.
        generator: Random generator for site selection.

    Returns:
        The index of a mutable site not in ``selected_sites``.
    """
    weights = space.mutable_positions.clone().float()
    weights[selected_sites] = 0
    selected = torch.multinomial(weights, 1, replacement=False, generator=generator)
    return int(selected)


class CouplingMaskSampler(SearchSpaceSampler):
    r"""Grow mutation masks using pairwise coupling strengths.

    Masks are grown iteratively one site at a time, favoring eligible sites with
    larger summed coupling to sites already in the group. Only mutable sites
    with positive summed coupling to the selected sites can be added. Sites are sampled
    from these mutable sites until either the ``max_mutations`` budget is fulfilled
    or no such site remains.

    When no sites remain, a new group is started using ``site_selector`` and the same
    process is repeated until ``max_mutations`` sites are collected.

    For sparse coupling matrices, growth follows positive coupling edges and can
    exhaust a connected component before reaching the cap. For dense matrices, growth
    typically continues from the initial site until the cap is reached. Selection is
    probabilistic: strongly coupled neighbors are favored but need not all be included
    when the cap truncates growth.

    Negative strengths are treated as zero. For temperature :math:`T > 0`, a site's
    sampling weight is :math:`\exp(g/T)`, where :math:`g` is its summed coupling to
    the current group. Lower temperatures concentrate selection on larger gains;
    higher temperatures make selection more uniform among positive-gain sites.
    Temperature is measured on the coupling scale: scaling all couplings and the
    temperature by the same positive factor leaves selection probabilities unchanged.
    At zero temperature, select a site with the largest gain at each growth step.

    This sequential growth strategy is a heuristic for favoring coupled sites. It
    does not sample masks from the Gibbs distribution
    :math:`P(M) \propto \exp(F(M)/T)`, where
    :math:`F(M) = \sum_{i<j,\ i,j\in M} W_{ij}` is the total pairwise coupling
    within the selected mask :math:`M`.
    """

    def __init__(
        self,
        coupling: Float[torch.Tensor, "l l"],
        n_subspaces: int,
        max_mutations: int,
        *,
        temperature: float = 1.0,
        site_selector: SiteSelector = random_site_selector,
        seed: int | None = None,
    ):
        """Configure coupling-weighted mutation masks.

        Args:
            coupling: Finite symmetric pairwise coupling strengths between sites.
            n_subspaces: Number of subspaces to draw per input space.
            max_mutations: Maximum open sites.
            site_selector: Logic to select the next site to include in mask given
                sites selected so far and the initial search space.
            temperature: Finite nonnegative temperature for selecting coupled sites.
                Zero selects the largest coupling gain at each growth step.
            seed: Seed for the sampling generator.

        Raises:
            ValueError: If temperature is not finite or is negative.
        """
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and positive")
        self.coupling = coupling.cpu().clamp(0)
        self.max_mutations = max_mutations
        self.n_subspaces = n_subspaces
        self.temperature = temperature
        self.site_selector = site_selector
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __repr__(self) -> str:
        """Return a compact summary of the sampling configuration."""
        return (
            f"{type(self).__name__}(n_subspaces={self.n_subspaces}, "
            f"max_mutations={self.max_mutations}, temperature={self.temperature:.4g})"
        )

    def _sample_one(self, space: search.SearchSpace) -> search.SearchSpace:
        max_mutations = min(self.max_mutations, int(space.mutable_positions.sum()))
        sites = []
        while len(sites) < max_mutations:
            site = self.site_selector(space, sites, self.generator)
            selected = torch.tensor([site])
            mutations_left = max_mutations - len(sites) - 1
            for _ in range(mutations_left):
                gains = self.coupling[selected].sum(dim=0)
                gains[selected] = 0
                gains[~space.mutable_positions] = 0
                positive = gains > 0
                if not bool(positive.any()):
                    break
                if self.temperature == 0:
                    next_pos = gains.argmax(dim=0, keepdim=True)
                else:
                    log_weights = torch.full_like(
                        gains, -torch.inf, dtype=torch.float64
                    )
                    log_weights[positive] = gains[positive].double()
                    log_weights = (log_weights - log_weights.max()) / self.temperature
                    probs = log_weights.softmax(dim=0)
                    next_pos = torch.multinomial(probs, 1, generator=self.generator)
                selected = torch.cat([selected, next_pos])
            sites.extend(selected.tolist())
        allowed = torch.zeros_like(space.allowed)
        allowed[sites] = space.allowed[sites]
        return dataclasses.replace(space, allowed=allowed)

    def __call__(self, space: search.SearchSpace) -> Iterator[search.SearchSpace]:
        """Yield ``n_subspaces`` coupling-driven subspaces.

        Args:
            space: Search space to sample subspaces from.

        Yields:
            A subspace with open sites selected using pairwise couplings.
        """
        for _ in range(self.n_subspaces):
            yield self._sample_one(space)
