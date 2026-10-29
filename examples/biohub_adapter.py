"""Custom model adapter wrapping biohub's API calls for ESMC.

This example additionally requires the ``esm`` package.
"""

from collections.abc import Sequence
import os

from esm import sdk  # ty: ignore[unresolved-import]
from esm.sdk import api as esm_api  # ty: ignore[unresolved-import]
from esm.sdk import forge  # ty: ignore[unresolved-import]
from jaxtyping import Float
import torch

from protuna import data
from protuna import inference
from protuna.typing import SequenceLike, TokenIDs


# embed-start
def _embed_sequence(
    client: forge.ESMCForgeInferenceClient, sequence: str
) -> Float[torch.Tensor, " d"]:
    protein = client.encode(esm_api.ESMProtein(sequence=sequence))
    if isinstance(protein, esm_api.ESMProteinError):
        raise protein
    output = client.logits(protein, esm_api.LogitsConfig(return_mean_embedding=True))
    if isinstance(output, esm_api.ESMProteinError):
        raise output
    embedding = output.mean_embedding.squeeze()
    return embedding.float().cpu()


# embed-end


# init-start
class BiohubEmbeddingModel(
    inference.InferenceModel[TokenIDs, Float[torch.Tensor, " d"]]
):
    """Return Biohub's mean ESMC embedding for each protein sequence."""

    def __init__(
        self,
        model_name: str = "esmc-600m-2024-12",
        *,
        token_env: str = "ESM_API_KEY",
        batch_size: int = 16,
        request_timeout: int = 120,
    ):
        """Configure hosted embedding inference without connecting to Biohub.

        Args:
            model_name: Hosted ESMC model identifier available to your account.
            token_env: Environment variable containing the API token at load time.
            batch_size: Maximum sequences submitted to the parallel executor at once.
            request_timeout: Timeout in seconds for each HTTP request.
        """
        super().__init__(batch_size=batch_size)
        self.model_name = model_name
        self.token_env = token_env
        self.request_timeout = request_timeout
        self._client: forge.ESMCForgeInferenceClient | None = None

    # init-end
    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Create the API client; local device placement is unused."""
        if self._client is None:
            self._client = forge.ESMCForgeInferenceClient(
                model=self.model_name,
                url="https://biohub.ai",
                token=os.environ[self.token_env],
                request_timeout=self.request_timeout,
            )

    def release(self) -> None:
        """Close the synchronous HTTP client and allow it to be loaded again."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __getstate__(self) -> dict[str, object]:
        """Return configuration without the live client or its credentials."""
        return {**self.__dict__, "_client": None}

    # lifecycle-end
    def forward(
        self, requests: SequenceLike[TokenIDs]
    ) -> list[Float[torch.Tensor, " d"]]:
        """Return one embedding per request, preserving input order."""
        if self._client is None:
            raise RuntimeError("Call load() before requesting embeddings")
        sequences = data.tokenizer.decode(list(requests))
        with sdk.parallel_executor(show_progress=False) as executor:
            results = executor.execute_batch(
                user_func=_embed_sequence,
                client=self._client,
                sequence=sequences,
            )
        embeddings = []
        for result in results:
            if isinstance(result, esm_api.ESMProteinError):
                raise result
            embeddings.append(result)
        return embeddings
        # forward-end


def main() -> None:
    """Verify the adapter works. Requires ESM_API_KEY env variable set."""
    model = BiohubEmbeddingModel(batch_size=2, request_timeout=30)
    requests = data.tokenizer(
        [
            "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLR",
            "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE",
        ]
    )
    try:
        model.load()
        embeddings = model([requests[0], requests[1], requests[0]])
        assert len(embeddings) == 3
        for embedding in embeddings:
            assert isinstance(embedding, torch.Tensor)
            assert embedding.ndim == 1
            assert embedding.numel() == 1152  # esmc-600m hidden dim
        print("All checks passed!")
    finally:
        model.release()


if __name__ == "__main__":
    main()
