from collections.abc import Iterator, Sequence
import pathlib
from typing import Literal

from jaxtyping import Float
import pytest
import torch

from protuna import data
from protuna import inference
from protuna.inference.models import esm
from protuna.optim import candidates
from protuna.optim import storage
from protuna.optim.acquisitions import dpp
from protuna.optim.acquisitions import kmeans
from protuna.optim.acquisitions import scores
from protuna.typing import SequenceLike, TokenIDs

pytestmark = pytest.mark.integration


class _EsmMeanEmbeddings(inference.LocalHandler[TokenIDs, Float[torch.Tensor, " d"]]):
    def __init__(self):
        super().__init__()
        self.model = esm.EsmInferenceModel(esm.EsmModels.ESM2_8M, precision="fp32")

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Load the ESM checkpoint onto the requested device."""
        self.model.load(device)

    def release(self) -> None:
        """Release the ESM checkpoint."""
        self.model.release()

    def __call__(
        self, requests: SequenceLike[TokenIDs]
    ) -> list[Float[torch.Tensor, " d"]]:
        """Return one ESM mean embedding per candidate sequence."""
        outputs = self.model(
            [
                inference.MLMRequest(
                    token_ids,
                    inference.MLMOutputConfig(
                        return_logits=False, return_mean_embeddings=True
                    ),
                )
                for token_ids in requests
            ]
        )
        embeddings = []
        for output in outputs:
            assert output.mean_embeddings is not None
            embeddings.append(output.mean_embeddings)
        return embeddings


@pytest.fixture(scope="module")
def esm_embeddings() -> Iterator[_EsmMeanEmbeddings]:
    """Provide real ESM-2 8M mean embeddings on the CPU."""
    model = _EsmMeanEmbeddings()
    model.load(torch.device("cpu"))
    yield model
    model.release()


@pytest.fixture
def candidate_stream(tmp_path: pathlib.Path) -> candidates.CandidateStream:
    """Store six candidates with a four-candidate UCB shortlist."""
    token_ids = data.tokenizer.encode(
        [
            "MQIFVKTLTGKTITLEVEPSDTIENVKAKI",
            "MQVFVKTLTGKTITLEVEPSDTIENVKAKI",
            "MKIFVKTLTGKTITLEVEPSDTIENVKAKI",
            "MQIFVKTLTGKTITLEVEPSDTIENVKAKV",
            "MQIFVKTLTGKTITLEVEPSDTIENVKAKL",
            "MQIFVKTLTGKTITLEVEPSDTIENVKAKF",
        ]
    )
    store = storage.CampaignStore(protein_length=30, location=tmp_path, scan_size=2)
    store.store_samples(torch.stack(token_ids), parent_id=0, ancestor_id=0, cycle_id=1)
    # Shortlisting deliberately differs from storage order and excludes ids 1 and 4.
    store.objective_store.store(
        {
            "seq_id": [1, 2, 3, 4, 5, 6],
            "objective_mean": [-2.0, 0.3, 0.1, -1.0, 0.2, 0.4],
            "objective_std": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "objective_samples": [None, None, None, None, None, None],
        }
    )
    return store.get_candidate_stream()


def test_embedding_rbf_kernel_distinguishes_real_esm_embeddings(
    esm_embeddings: _EsmMeanEmbeddings,
    candidate_stream: candidates.CandidateStream,
) -> None:
    """Produce finite, symmetric similarities from distinct ESM sequence embeddings."""
    batch = candidate_stream.candidates_from_ids([6, 2, 5, 3])
    kernel = dpp.EmbeddingRBFKernel(
        esm_embeddings, lengthscale=torch.tensor([1.0]), chunk_size=2
    )

    similarities = kernel(batch)

    assert similarities.shape == (4, 4)
    assert torch.isfinite(similarities).all()
    torch.testing.assert_close(similarities, similarities.T)
    torch.testing.assert_close(similarities.diag(), torch.ones(4))
    assert ((similarities > 0) & (similarities <= 1)).all()
    assert (similarities[~torch.eye(4, dtype=torch.bool)] < 1).all()


@pytest.mark.parametrize("mode", ["map", "sample"])
def test_kdpp_selects_shortlisted_candidates_using_esm_embeddings(
    esm_embeddings: _EsmMeanEmbeddings,
    candidate_stream: candidates.CandidateStream,
    mode: Literal["map", "sample"],
) -> None:
    """Select a unique lab batch using real ESM embeddings and an RBF kernel."""
    acquisition = dpp.DPP(
        quality=scores.UCBScore(subsample_size=4, beta=1.0, temperature=0),
        diversity=dpp.EmbeddingRBFKernel(
            esm_embeddings, lengthscale=torch.tensor([1.0]), chunk_size=2
        ),
        budget=2,
        mode=mode,
        seed=7,
    )

    selected = acquisition.select(candidate_stream)

    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert set(selected) <= {2, 3, 5, 6}


@pytest.mark.parametrize("mode", ["centroid", "max"])
def test_kmbbo_selects_shortlisted_candidates_using_esm_embeddings(
    esm_embeddings: _EsmMeanEmbeddings,
    candidate_stream: candidates.CandidateStream,
    mode: Literal["centroid", "max"],
) -> None:
    """Select cluster representatives from real ESM mean embeddings."""
    acquisition = kmeans.KMBBO(
        embedding_model=esm_embeddings,
        score_function=scores.UCBScore(subsample_size=4, beta=1.0, temperature=0),
        budget=2,
        mode=mode,
        seed=7,
    )

    selected = acquisition.select(candidate_stream)

    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert set(selected) <= {2, 3, 5, 6}
