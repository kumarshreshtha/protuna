"""Select representatives from clusters of candidate embeddings."""

from typing import Literal

from jaxtyping import Float
import numpy as np
from sklearn import cluster
import torch

from protuna import inference
from protuna.optim import candidates
from protuna.optim.acquisitions import base
from protuna.optim.acquisitions import scores
from protuna.typing import TokenIDs

type EmbeddingModelLike = inference.Handler[TokenIDs, Float[torch.Tensor, " d"]]


class KMBBO(base.Acquisition):
    """Select candidate sequences using K-Means Batch Bayesian Optimization.

    Adapts the clustering approach of :cite:p:`KMBBO` to a discrete sequence
    space. The shortlisted candidates are clustered in embedding space into ``budget``
    groups, then candidate sequences are selected to represent either the cluster
    centers or the highest utility within each cluster.

    The :class:`~protuna.optim.acquisitions.scores.ScoreFunction` controls both
    shortlisting and utility evaluation. These choices are independent of the
    clustering strategy: a scoring function can combine, for example,
    categorical sampling, slice sampling or top-k selection with predicted mean,
    upper confidence bound (UCB) or expected improvement (EI).

    With ``mode="centroid"``, select the shortlisted candidate whose embedding
    is closest to each centroid. With ``mode="max"``, select the highest-scoring
    candidate within each cluster.

    Example:
        Given an embedding model and candidate stream, shortlist by UCB and
        select the highest-scoring member of each of eight clusters:

        >>> from protuna.optim.acquisitions import scores
        >>> acquisition = KMBBO(  # doctest: +SKIP
        ...     embedding_model,
        ...     scores.UCBScore(subsample_size=100, beta=4.0, temperature=0),
        ...     budget=8,
        ...     mode="max",
        ... )
        >>> selected_ids = acquisition.select(stream)  # doctest: +SKIP
    """

    def __init__(
        self,
        embedding_model: EmbeddingModelLike,
        score_function: scores.ScoreFunction,
        budget: int,
        *,
        mode: Literal["centroid", "max"] = "centroid",
        seed: int | None = None,
    ):
        """Initialize selection of representatives from embedding-space clusters.

        Args:
            embedding_model: Model used to embed shortlisted candidate sequences.
            score_function: Strategy that supplies shortlist scores and sequence ids.
            budget: Number of clusters and representatives to select.
            mode: Select the nearest candidate to each centroid with ``"centroid"``
                or the highest-scoring member of each cluster with ``"max"``.
            seed: Random seed for clustering, or ``None`` for unseeded clustering.
        """
        super().__init__()
        self.model = embedding_model
        self.score_function = score_function
        self.budget = budget
        self.mode = mode
        self.requires_samples = self.score_function.requires_samples
        self.requires_std = self.score_function.requires_std
        self.seed = seed

    def select(self, stream: candidates.CandidateStream) -> list[int]:
        """Select a centroid neighbor or highest-scoring member of each cluster.

        Args:
            stream: Candidate pool to shortlist and retrieve sequences from.

        Returns:
            Sequence ids of the selected representatives in cluster order.
        """
        scores, shortlist = self.score_function(stream)
        batch = stream.candidates_from_ids(shortlist.tolist())
        embeddings = torch.stack(
            list(self.model([token_ids for token_ids in batch.token_ids]))
        )
        k_means = cluster.KMeans(
            n_clusters=self.budget, random_state=self.seed, n_init="auto"
        )
        distances: np.ndarray = k_means.fit_transform(embeddings.cpu().numpy())
        if self.mode == "centroid":
            indices = np.argmin(distances, axis=0)
            selected_ids = shortlist[indices].tolist()
        else:
            labels = np.argmin(distances, axis=1)
            selected_ids = []
            for label in range(self.budget):
                mask = labels == label
                selection = scores[mask].argmax(dim=0)
                selected_ids.append(int(shortlist[mask][selection]))
        return selected_ids
