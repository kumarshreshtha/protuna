"""Inference adapters for ESMC :cite:p:`ESMC` through FastPLMs :cite:p:`FastPLMs`."""

from collections.abc import Sequence
import enum
import os
from typing import Literal

from jaxtyping import Float
import torch

from protuna.inference.interfaces import mlm
from protuna.inference.models import hf_mlm
from protuna.typing import SequenceLike


class EsmcModels(enum.StrEnum):
    """ESMC checkpoint names."""

    ESMC_300M = "Synthyra/ESMplusplus_small"
    ESMC_600M = "Synthyra/ESMplusplus_large"
    ESMC_6B = "Synthyra/ESMplusplus_6B"


class _EsmcBase[RequestT, ResponseT](hf_mlm.HuggingFaceMLMBase[RequestT, ResponseT]):
    def __init__(
        self,
        model_name: EsmcModels,
        *,
        pretrained_path: str | os.PathLike | None = None,
        precision: Literal["bf16", "fp32"] | None = None,
        batch_size: int | None = None,
        trust_remote_code: bool | None = None,
    ):
        """Configure the ESMC checkpoint.

        Args:
            model_name: Checkpoint architecture and tokenizer to use.
            pretrained_path: Optional replacement weights for this checkpoint.
            precision: Inference precision; ``None`` uses bf16.
            batch_size: Requests per forward pass, or ``None`` for all requests.
            trust_remote_code: Whether to allow Synthyra's custom adaptation
                code. ``None`` (default) lets Transformers prompt when required.
                ``True`` allows remote code without prompting; ``False`` refuses
                it. Prompting requires interactive input and may time out.
        """
        model_name = EsmcModels(model_name)
        precision = "bf16" if precision is None else precision
        if precision not in ("bf16", "fp32"):
            raise ValueError(f"Unsupported precision: {precision}")
        super().__init__(
            model_name,
            pretrained_path=pretrained_path,
            dtype=torch.bfloat16 if precision == "bf16" else torch.float32,
            batch_size=batch_size,
            trust_remote_code=trust_remote_code,
        )

    def _forward_kwargs(
        self, configs: Sequence[mlm.MLMOutputConfig]
    ) -> dict[str, bool]:
        return {
            **super()._forward_kwargs(configs),
            "compute_logits": any(cfg.return_logits for cfg in configs),
        }


class EsmcInferenceModel(_EsmcBase[mlm.MLMRequest, mlm.MLMOutput]):
    """:py:type:`~protuna.inference.interfaces.mlm.MaskedLMLike` adapter for ESMC.

    Uses the FastPLMs implementation :cite:p:`FastPLMs` from Synthyra.
    """

    def forward(self, requests: SequenceLike[mlm.MLMRequest]) -> list[mlm.MLMOutput]:
        """Return the outputs selected by each request's configuration.

        Args:
            requests: Tokenized sequences and output configurations for inference.

        Returns:
            A list of model outputs in request order.
        """
        return self._forward_mlm(requests)


class EsmcLogitsModel(_EsmcBase[mlm.MLMLogitsRequest, Float[torch.Tensor, "l v"]]):
    """:py:type:`~protuna.inference.interfaces.mlm.MLMForLogitsLike` adapter for ESMC.

    Uses the FastPLMs implementation :cite:p:`FastPLMs` from Synthyra.
    """

    def forward(
        self, requests: SequenceLike[mlm.MLMLogitsRequest]
    ) -> list[Float[torch.Tensor, "l v"]]:
        """Return one logits tensor per request, excluding padding and boundaries.

        Args:
            requests: Tokenized sequences and position and vocabulary filters.

        Returns:
            A list of filtered logits tensors in request order.
        """
        return self._forward_logits(requests)
