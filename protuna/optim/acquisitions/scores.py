"""Score and shortlist candidates for acquisition strategies."""

import math
from typing import Protocol

from jaxtyping import Float, Integer
import torch

from protuna.optim import candidates
from protuna.optim.acquisitions import base


class ScoreFunction(Protocol):
    """Build a scored candidate shortlist for an acquisition strategy.

    Calling a score function on a :class:`~protuna.optim.candidates.CandidateStream`
    evaluates candidate utility and chooses which candidates to retain, returning
    their scores and sequence ids in matching order. It controls both the utility
    measure (such as upper confidence bound or expected improvement) and the
    shortlisting rule (such as top-k selection or weighted sampling).

    Acquisition strategies such as :class:`~protuna.optim.acquisitions.kmeans.KMBBO`
    and :class:`~protuna.optim.acquisitions.dpp.DPP` use this shortlist for their
    final selection, where they can also account for candidate diversity.

    Attributes:
        requires_std: Whether scoring requires objective standard deviations.
        requires_samples: Whether scoring requires objective samples.
    """

    requires_std: bool
    requires_samples: bool

    def __call__(
        self, stream: candidates.CandidateStream
    ) -> tuple[Float[torch.Tensor, " n"], Integer[torch.Tensor, " n"]]:
        """Return utility values and ids for shortlisted candidates.

        Args:
            stream: Candidate pool to score and shortlist.

        Returns:
            A pair of aligned tensors containing utility values and sequence ids.
        """
        ...


class UCBScore(ScoreFunction):
    r"""Subsample by upper confidence bound.

    For objective mean :math:`\mu(x)` and standard deviation :math:`\sigma(x)`,
    the score is

    .. math::

        \operatorname{UCB}(x) = \mu(x) + \sqrt{\beta}\,\sigma(x).

    A zero temperature selects the highest scores; positive temperatures sample
    from softmax probabilities after filtering by joint feasibility. For the
    retained candidate set :math:`\mathcal{F}` and temperature :math:`T > 0`,
    the sampling weights are

    .. math::

        p(x) = \frac{\exp(\operatorname{UCB}(x) / T)}
                    {\sum_{x' \in \mathcal{F}}
                     \exp(\operatorname{UCB}(x') / T)}.

    """

    requires_std = True
    requires_samples = False

    def __init__(
        self,
        subsample_size: int,
        beta: float,
        *,
        temperature: float = 1.0,
        feasibility_threshold: float = 0.0,
        sample_with_replacement: bool = True,
        seed: int | None = None,
    ):
        """Initialize candidate shortlisting by upper confidence bound.

        Args:
            subsample_size: Number of candidate entries to shortlist.
            beta: Nonnegative confidence weight whose square root multiplies the
                objective standard deviation.
            temperature: Softmax temperature, or zero to select the highest scores.
            feasibility_threshold: Minimum joint feasibility, excluded at equality.
            sample_with_replacement: Whether sampling may repeat candidates.
                Ignored when ``temperature`` is zero.
            seed: Seed for shortlist sampling, or ``None`` to keep the generator's
                default state.
        """
        self.subsample_size = subsample_size
        self.sqrt_beta = math.sqrt(beta)
        self.temperature = temperature
        self.feasibility_threshold = feasibility_threshold
        self.with_replacement = sample_with_replacement
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __call__(
        self, stream: candidates.CandidateStream
    ) -> tuple[Float[torch.Tensor, " n"], Integer[torch.Tensor, " n"]]:
        """Return upper confidence bounds and their candidate ids.

        Args:
            stream: Candidate pool with objective means, standard deviations and
                optional constraint feasibility values.

        Returns:
            A pair of aligned tensors containing shortlisted upper confidence
            bounds and sequence ids.
        """
        scores = []
        seq_ids = []
        for batch in stream:
            assert batch.objective_std is not None
            assert batch.objective_mean is not None
            feasible = base.feasibility(batch) > self.feasibility_threshold
            objective_mean = batch.objective_mean[feasible]
            objective_std = batch.objective_std[feasible]
            seq_id = batch.seq_id[feasible]
            score = objective_mean + objective_std * self.sqrt_beta
            scores.append(score)
            seq_ids.append(seq_id)

        scores = torch.cat(scores)
        seq_ids = torch.cat(seq_ids)
        if self.temperature == 0:
            values, indices = scores.topk(k=self.subsample_size, dim=0)
            return values, seq_ids[indices]
        scores_ = torch.softmax(scores / self.temperature, dim=0)
        indices = torch.multinomial(
            scores_,
            num_samples=self.subsample_size,
            replacement=self.with_replacement,
            generator=self.generator,
        )
        return scores[indices], seq_ids[indices]


class EIScore(ScoreFunction):
    r"""Sample in proportion to feasibility-weighted Gaussian expected improvement.

    For objective mean :math:`\mu(x)`, standard deviation :math:`\sigma(x)`,
    and improvement threshold :math:`\tau = \text{best\_f} + \text{margin}`,
    define :math:`\Delta(x) = \mu(x) - \tau`. Gaussian expected improvement is

    .. math::

        \operatorname{EI}(x) =
        \begin{cases}
            \Delta(x)\,\Phi(z(x)) + \sigma(x)\,\phi(z(x)),
                & \sigma(x) > 0, \\[4pt]
            \max(\Delta(x), 0), & \sigma(x) = 0,
        \end{cases}
        \qquad z(x) = \frac{\Delta(x)}{\sigma(x)}.

    Here :math:`\Phi` and :math:`\phi` are the standard normal CDF and PDF,
    and :math:`z(x)` is used only when :math:`\sigma(x) > 0`. With joint
    feasibility :math:`q(x)`, the returned score and sampling weights are

    .. math::

        s(x) = q(x)\,\operatorname{EI}(x),
        \qquad p(x) = \frac{s(x)}{\sum_{x'} s(x')}.
    """

    requires_std = True
    requires_samples = False

    def __init__(
        self,
        best_f: float,
        subsample_size: int,
        *,
        margin: float = 0,
        sample_with_replacement: bool = True,
        seed: int | None = None,
    ):
        """Initialize candidate sampling by feasibility-weighted expected improvement.

        Args:
            best_f: Reference objective value to improve upon.
            subsample_size: Number of candidate entries to sample.
            margin: Additional improvement required above ``best_f``.
            sample_with_replacement: Whether sampling may repeat candidates.
            seed: Seed for shortlist sampling, or ``None`` to keep the generator's
                default state.
        """
        self.subsample_size = subsample_size
        self.with_replacement = sample_with_replacement
        self.best_f = best_f
        self.margin = margin
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __call__(
        self, stream: candidates.CandidateStream
    ) -> tuple[Float[torch.Tensor, " n"], Integer[torch.Tensor, " n"]]:
        """Return feasible expected improvements and their sampled candidate ids.

        Args:
            stream: Candidate pool with objective means, standard deviations and
                optional constraint feasibility values.

        Returns:
            A pair of aligned tensors containing feasibility-weighted expected
            improvements and sampled sequence ids.
        """
        scores = []
        seq_ids = []
        for batch in stream:
            assert batch.objective_std is not None
            assert batch.objective_mean is not None
            feasibility = base.feasibility(batch)
            mean = batch.objective_mean
            std = batch.objective_std
            improvement = mean - self.best_f
            improvement = improvement - self.margin
            positive_std = std > 0
            safe_std = torch.where(positive_std, std, torch.ones_like(std))
            z = improvement / safe_std
            cdf = 0.5 * torch.special.erfc(-z / math.sqrt(2.0))
            pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
            ei = improvement * cdf + std * pdf
            ei = torch.where(positive_std, ei.clamp_min(0), improvement.clamp_min(0))
            scores.append(ei * feasibility)
            seq_ids.append(batch.seq_id)

        scores = torch.cat(scores)
        seq_ids = torch.cat(seq_ids)
        indices = torch.multinomial(
            scores,
            self.subsample_size,
            replacement=self.with_replacement,
            generator=self.generator,
        )
        return scores[indices], seq_ids[indices]
