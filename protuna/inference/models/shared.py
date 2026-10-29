"""Adapters that wrap other model handler."""

from collections.abc import Sequence

from jaxtyping import Float
import torch

from protuna.inference import handlers
from protuna.inference.interfaces import mlm
from protuna.typing import SequenceLike


class SharedLogitsModel(
    handlers.InferenceModel[mlm.MLMLogitsRequest, Float[torch.Tensor, "l v"]]
):
    """Expose a logits-only interface over an existing MaskedLMLike handler.

    Requests are translated to :class:`~protuna.inference.interfaces.mlm.MLMRequest`
    and dispatched to the supplied handler, allowing an existing model to be shared
    between different interfaces. Loading and releasing this adapter also loads and
    releases that handler.
    """

    def __init__(self, model: mlm.MaskedLMLike, batch_size: int | None = None):
        """Wrap a MaskedLMLike handler that supports logits output.

        Args:
            model: Masked language model handler to reuse for inference.
            batch_size: Logits requests per forward pass; ``None`` processes
                all requests together.
        """
        super().__init__(batch_size=batch_size)
        self.model = model

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Load the wrapped handler on the requested device or devices.

        Args:
            device: Device or devices forwarded to the wrapped handler. ``None``
                leaves device selection to that handler.
        """
        self.model.load(device)

    def release(self) -> None:
        """Release resources held by the wrapped handler."""
        self.model.release()

    def forward(
        self, requests: SequenceLike[mlm.MLMLogitsRequest]
    ) -> list[Float[torch.Tensor, "l v"]]:
        """Return logits from the wrapped handler using each request's filters.

        Outputs follow request order and exclude boundary and padding tokens.

        Args:
            requests: Tokenized sequences and output filters for logits inference.

        Returns:
            A list of logits tensors, one per request.
        """
        model_requests = [
            mlm.MLMRequest(
                token_ids=request.token_ids,
                output_config=mlm.MLMOutputConfig(
                    return_logits=True,
                    masked_positions_only=request.masked_positions_only,
                    aa_tokens_only=request.aa_tokens_only,
                ),
            )
            for request in requests
        ]
        responses = []
        for response in self.model(model_requests):
            assert response.logits is not None
            responses.append(response.logits)
        return responses
