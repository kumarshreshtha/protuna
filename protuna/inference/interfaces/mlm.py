"""Request and response types for handlers of masked language models."""

from collections.abc import Sequence
import dataclasses
from typing import Self

from jaxtyping import Float
import torch

from protuna.inference import handlers
from protuna.typing import TokenIDs
from protuna.utils import devices


@dataclasses.dataclass
class MLMOutput:
    """Container for model forward-pass outputs.

    Args:
        logits: Token logits of shape ``(L, V)``.
        hidden_states: Per-layer hidden states of shape ``(L, D)``.
        embeddings: Final sequence embeddings of shape ``(L, D)``.
        mean_embeddings: Final sequence mean embeddings of shape ``(D, )``.
    """

    logits: Float[torch.Tensor, "l v"] | None = None
    hidden_states: tuple[Float[torch.Tensor, "l d"], ...] | None = None
    embeddings: Float[torch.Tensor, "l d"] | None = None
    mean_embeddings: Float[torch.Tensor, " d"] | None = None

    def to(self, device: torch.device) -> Self:
        """Return these outputs with every populated tensor on *device*."""
        return dataclasses.replace(
            self,
            **{
                field.name: tuple(state.to(device=device) for state in v)
                if field.name == "hidden_states"
                else v.to(device=device)
                for field in dataclasses.fields(self)
                if (v := getattr(self, field.name)) is not None
            },
        )

    @classmethod
    def to_device_batch(
        cls, responses: Sequence[Self], device: torch.device
    ) -> list[Self]:
        """Move model outputs together while preserving their optional fields.

        Args:
            responses: Outputs to transfer, in response order.
            device: Destination device.

        Returns:
            New outputs with all populated tensors on the destination device.
            Compatible tensors share a transfer, including across hidden layers.
        """
        tensors = []
        for response in responses:
            for value in (
                response.logits,
                response.hidden_states,
                response.embeddings,
                response.mean_embeddings,
            ):
                if isinstance(value, tuple):
                    tensors.extend(value)
                elif value is not None:
                    tensors.append(value)
        moved = iter(devices.to_device_batch(tensors, device))
        return [
            dataclasses.replace(
                response,
                logits=next(moved) if response.logits is not None else None,
                hidden_states=tuple(next(moved) for _ in response.hidden_states)
                if response.hidden_states is not None
                else None,
                embeddings=next(moved) if response.embeddings is not None else None,
                mean_embeddings=next(moved)
                if response.mean_embeddings is not None
                else None,
            )
            for response in responses
        ]


@dataclasses.dataclass
class MLMOutputConfig:
    """Config stating what values a :py:type:`MaskedLMLike` handler should return.

    Args:
        return_logits: Return token logits.
        masked_positions_only: Restrict logits to masked positions rather than
            the whole sequence.
        aa_tokens_only: Restrict logits to the 20 amino acid tokens, dropping
            special tokens.
        return_mean_embeddings: Return the sequence embedding averaged over
            positions.
        return_embeddings: Return per-position final-layer embeddings.
        return_hidden_states: Return the hidden states of every layer.
    """

    return_logits: bool = True
    masked_positions_only: bool = False
    aa_tokens_only: bool = False
    return_mean_embeddings: bool = False
    return_embeddings: bool = False
    return_hidden_states: bool = False


@dataclasses.dataclass
class MLMRequest:
    """A single input request to run through a :py:type:`MaskedLMLike` handler.

    Args:
        token_ids: Sequence in protuna's token space.
        output_config: Which outputs to compute. Defaults to logits only.
    """

    token_ids: TokenIDs
    output_config: MLMOutputConfig = dataclasses.field(default_factory=MLMOutputConfig)


@dataclasses.dataclass
class MLMLogitsRequest:
    """Request for logits of a single sequence.

    Args:
        token_ids: Input token IDs of shape ``(L,)``, including any mask tokens.
        masked_positions_only: Return rows only for input mask positions.
            Otherwise, return all sequence positions without boundary or padding
            tokens.
        aa_tokens_only: Return only the 20 amino acid columns rather than the
            full protuna vocabulary.
    """

    token_ids: TokenIDs
    masked_positions_only: bool = False
    aa_tokens_only: bool = False


type MaskedLMLike = handlers.Handler[MLMRequest, MLMOutput]

type MLMForLogitsLike = handlers.Handler[MLMLogitsRequest, Float[torch.Tensor, "l v"]]
