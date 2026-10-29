"""Select diverse candidate subsets using determinantal point processes."""

from collections.abc import Sequence
import itertools
from typing import Literal, Protocol

from jaxtyping import Float
import torch

from protuna import inference
from protuna.optim import candidates
from protuna.optim.acquisitions import base
from protuna.optim.acquisitions import scores
from protuna.typing import TokenIDs


# TODO: implement the faster version
def _kdpp_sample(
    diversity: Float[torch.Tensor, "n n"],
    log_quality: Float[torch.Tensor, " n"],
    k: int,
    n_iter: int,
    generator: torch.Generator | None = None,
) -> list[int]:
    """K-DPP sampling using MCMC, adapted from :cite:p:`MCMCSampleDPP`."""
    if k == 0:
        return []
    if k == log_quality.numel():
        return list(range(k))
    g = -torch.empty_like(log_quality).exponential_(generator=generator).log()
    _, ids = torch.topk(log_quality + g, k=k)
    mask = torch.zeros_like(log_quality, dtype=torch.bool)
    mask[ids] = True
    det = torch.logdet(diversity[mask][:, mask])
    for _ in range(n_iter):
        probs = mask.float()
        to_replace = probs.multinomial(1, generator=generator)
        replace_by = (1 - probs).multinomial(1, generator=generator)
        new_mask = mask.clone()
        new_mask[to_replace] = False
        new_mask[replace_by] = True
        new_det = torch.logdet(diversity[new_mask][:, new_mask])
        log_ratio = log_quality[replace_by] - log_quality[to_replace] + new_det - det
        eps = 1e-6
        if (torch.rand(1, generator=generator) + eps).log() <= log_ratio:
            det = new_det
            mask = new_mask

    return mask.nonzero(as_tuple=True)[0].tolist()


def _greedy_kddp_map(L: Float[torch.Tensor, "b b"], k: int) -> list[int]:
    """Greedy approximate MAP inference for a k-DPP :cite:p:`GreedyMAP`."""
    if k == 0:
        return []
    M = L.shape[0]
    # C[t, i] holds the t-th entry of c_i (row t of the Cholesky factor, column i).
    C = L.new_zeros((k, M))
    d2 = torch.diag(L).clone()  # d_i^2, initialized to L_ii

    j = int(torch.argmax(d2))
    selected = [j]

    while len(selected) < k:
        t = len(selected) - 1
        d_j = torch.sqrt(d2[j])
        # e_i = (L_ji - <c_j, c_i>) / d_j
        e = (L[j, :] - C[:t, j] @ C[:t, :]) / d_j
        C[t, :] = e
        d2 -= e**2
        d2[selected] = -torch.inf  # exclude already-selected items
        j = int(torch.argmax(d2))
        selected.append(j)

    return selected


class DiversityKernel(Protocol):
    """Compute pairwise candidate similarities for diversity-aware selection."""

    requires_std: bool
    requires_samples: bool

    def __call__(self, batch: candidates.CandidateBatch) -> Float[torch.Tensor, "b b"]:
        """Compute the similarity matrix for the given batch of candidates.

        Args:
            batch: Candidates to compare, with the predictions required by the kernel.

        Returns:
            Pairwise similarity matrix with rows and columns aligned to ``batch``.
        """
        ...


def _ensemble_covariance_kernel(
    batch: candidates.CandidateBatch,
) -> Float[torch.Tensor, "b b"]:
    assert batch.objective_samples is not None
    samples = batch.objective_samples
    samples = samples - samples.mean(dim=1, keepdim=True)
    covariance = samples @ samples.T / (samples.shape[1] - 1)
    return covariance


def _rbf_kernel(
    features: Float[torch.Tensor, "b d"],
    lengthscale: Float[torch.Tensor, "#d"],
    chunk_size: int,
) -> Float[torch.Tensor, "b b"]:
    pairwise_dist = features.new_empty(features.shape[0], features.shape[0])
    for idx in itertools.batched(range(features.shape[0]), chunk_size):
        idx = list(idx)
        # B x L
        batch_features = features[idx]
        # B x 1 x L - 1 x N x L => B x N x L => B x N
        batch_dist = (
            ((features[None, :, :] - batch_features[:, None, :]) / lengthscale)
            .square()
            .sum(dim=-1)
        )
        pairwise_dist[idx, :] = batch_dist

    return pairwise_dist.div(-2).exp()


type EmbeddingModelLike = inference.Handler[TokenIDs, Float[torch.Tensor, " d"]]


class EmbeddingRBFKernel(inference.HandlerRegistryMixin, DiversityKernel):
    r"""Measure candidate similarity by an RBF kernel over model embeddings.

    .. math::

        L_{ij} = \exp\!\left(-\frac{1}{2}
            \sum_{k=1}^{d} \frac{(z_{ik} - z_{jk})^2}{\ell_k^2}\right).

    Here, :math:`z_i` is the model embedding of candidate :math:`i`, and
    :math:`\ell_k` is the lengthscale for embedding feature :math:`k`. A scalar
    lengthscale uses the same value for every feature.
    """

    requires_std = False
    requires_samples = False

    def __init__(
        self,
        model: EmbeddingModelLike,
        lengthscale: Float[torch.Tensor, "#d"],
        *,
        chunk_size: int = 1000,
    ):
        """Set the kernel lengthscale and register the embedding model.

        Args:
            model: Model that maps candidate token ids to embedding vectors.
            lengthscale: Shared or per-feature scale for embedding distances.
            chunk_size: Number of candidate embeddings to compare against the full
                batch at a time.
        """
        super().__init__()
        self.model = model
        self.lengthscale = lengthscale
        self.chunk_size = chunk_size

    def __call__(self, batch: candidates.CandidateBatch) -> Float[torch.Tensor, "b b"]:
        """Return pairwise RBF similarities between candidate embeddings.

        Args:
            batch: Candidates whose token ids are passed to the embedding model.

        Returns:
            RBF similarity matrix with rows and columns aligned to ``batch``.
        """
        embeddings = self.model([token_ids for token_ids in batch.token_ids])
        return _rbf_kernel(
            torch.stack(list(embeddings)), self.lengthscale, chunk_size=self.chunk_size
        )


class EnsembleCovarianceKernel(DiversityKernel):
    r"""Measure candidate similarity by covariance of predicted objective samples.

    .. math::

        L_{ij} &= \frac{1}{n - 1} \sum_{s=1}^{n}
            (y_i^{(s)} - \bar{y}_i)(y_j^{(s)} - \bar{y}_j), \\
        \bar{y}_i &= \frac{1}{n} \sum_{s=1}^{n} y_i^{(s)}.

    Here, :math:`y_i^{(s)}` is the predicted objective for candidate :math:`i`
    in aligned sample :math:`s`, and :math:`n \geq 2` is the number of samples.
    The kernel estimates the unbiased covariance between candidates.

    .. note::

        With ``b`` candidates and ``n`` aligned posterior draws, the kernel's rank is
        at most ``min(b, n - 1)``. Selecting ``budget`` candidates with a k-DPP
        requires rank at least ``budget``, so ``n > budget`` is necessary. Combining
        this kernel with another valid kernel such as :class:`EmbeddingRBFKernel` can
        increase its rank.
    """

    requires_std = False
    requires_samples = True

    def __call__(self, batch: candidates.CandidateBatch) -> Float[torch.Tensor, "b b"]:
        """Return covariance after centering each candidate's posterior samples.

        Args:
            batch: Candidates with at least two aligned objective samples per
                candidate.

        Returns:
            Unbiased covariance matrix with rows and columns aligned to ``batch``.
        """
        return _ensemble_covariance_kernel(batch)


class ComposeKernel(inference.HandlerRegistryMixin, DiversityKernel):
    """Combine diversity kernels by a weighted sum and aggregate their requirements."""

    def __init__(
        self,
        kernels: Sequence[DiversityKernel],
        *,
        weights: Sequence[int] | None = None,
    ):
        """Initialize a weighted combination of diversity kernels.

        Args:
            kernels: Diversity kernels to combine.
            weights: One multiplier per kernel, or ``None`` to use unit weights.
        """
        super().__init__()
        if weights is None:
            weights = [1] * len(kernels)
        self.kernels = kernels
        self.weights = weights
        self.requires_std = any(kernel.requires_std for kernel in kernels)
        self.requires_samples = any(kernel.requires_samples for kernel in kernels)
        for i, kernel in enumerate(self.kernels):
            if isinstance(kernel, inference.HandlerRegistryMixin):
                self.register(f"kernel_{i}", kernel)

    def __call__(self, batch: candidates.CandidateBatch) -> Float[torch.Tensor, "b b"]:
        """Return the weighted sum of component similarity matrices.

        Args:
            batch: Candidates with the inputs required by all component kernels.

        Returns:
            Combined similarity matrix with rows and columns aligned to ``batch``.
        """
        final_kernel = torch.zeros(batch.seq_id.shape[0], batch.seq_id.shape[0])

        for kernel, weight in zip(self.kernels, self.weights, strict=True):
            final_kernel += kernel(batch) * weight

        return final_kernel


class DPP(base.Acquisition):
    r"""Select a diverse set of candidates using greedy k-DPP MAP or MCMC sampling.

    The :class:`~protuna.optim.acquisitions.scores.ScoreFunction` supplies a
    candidate shortlist and a log-quality score for each entry. A determinantal
    point process (DPP) then balances these scores against the diversity kernel
    to select ``budget`` candidates, favoring subsets that combine high quality
    with low redundancy.

    In ``"sample"`` mode, MCMC approximately samples from the fixed-size k-DPP
    distribution below. In ``"map"`` mode, greedy approximate maximum a
    posteriori (MAP) inference seeks a subset with high probability under the
    same distribution. The parameter ``beta`` scales the contribution of log
    quality relative to diversity:

    .. math::

        P(S) &\propto \det(L_S) \prod_{i \in S} e^{\beta r_i}
             = \det\!\left((D L D)_S\right), \\
        D &= \operatorname{diag}\!\left(e^{\beta r_1/2}, \ldots,
                                          e^{\beta r_n/2}\right).

    where:

    - :math:`S` is a subset of the :math:`n` shortlisted candidates, with
      :math:`|S| = k`, where :math:`k` is ``budget``.
    - :math:`P(S)` is the probability of selecting :math:`S` under the k-DPP.
      The proportionality constant normalizes over all size-:math:`k` subsets.
    - :math:`L` is the diversity kernel's similarity matrix, and :math:`L_S`
      is its principal submatrix containing the rows and columns indexed by
      :math:`S`.
    - :math:`r_i` is the log-quality score of candidate :math:`i`, and
      :math:`\beta` is ``beta``, which scales quality relative to diversity.
    """

    def __init__(
        self,
        quality: scores.ScoreFunction,
        diversity: DiversityKernel,
        budget: int,
        *,
        mode: Literal["map", "sample"] = "sample",
        beta: float = 1.0,
        n_iter: int | None = None,
        seed: int | None = None,
    ):
        """Initialize DPP selection balancing candidate quality and diversity.

        Args:
            quality: Scoring function that supplies log quality and a candidate
                shortlist.
            diversity: Similarity kernel for the shortlisted candidates.
            budget: Number of shortlist entries to select.
            mode: Use ``"map"`` for greedy approximate MAP inference or
                ``"sample"`` for MCMC sampling.
            beta: Multiplier applied to log quality before combining it with
                diversity.
            n_iter: Number of MCMC iterations, or ``None`` to use ten times the
                shortlist size. Only used in ``"sample"`` mode.
            seed: Seed for MCMC sampling, or ``None`` to keep the generator's
                default state.
        """
        super().__init__()
        self.budget = budget
        self.quality = quality
        self.diversity = diversity
        self.mode = mode
        self.beta = beta
        self.n_iter = n_iter
        self.requires_samples = (
            self.quality.requires_samples or self.diversity.requires_samples
        )
        self.requires_std = self.quality.requires_std or self.diversity.requires_std
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def select(self, stream: candidates.CandidateStream) -> list[int]:
        """Return candidate ids chosen by the configured DPP inference method.

        Args:
            stream: Candidate pool with the inputs required by the quality
                function and diversity kernel.

        Returns:
            Sequence ids of the selected shortlist entries.
        """
        scores, shortlist = self.quality(stream)
        diversity = self.diversity(stream.candidates_from_ids(shortlist.tolist()))
        if self.mode == "map":
            scores = scores * self.beta / 2
            quality = (scores - scores.max()).exp()
            mat = quality * diversity * quality[:, None]
            indices = _greedy_kddp_map(mat, self.budget)
        else:
            n_iter = self.n_iter if self.n_iter is not None else scores.shape[0] * 10
            scores = scores * self.beta
            indices = _kdpp_sample(
                diversity, scores, self.budget, n_iter=n_iter, generator=self.generator
            )
        return shortlist[indices].tolist()
