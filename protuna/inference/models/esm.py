"""Inference adapters for ESM-1 :cite:p:`ESM1b, ESM1v` and ESM-2 :cite:p:`ESM2`."""

from collections.abc import Sequence
import enum
import os
from typing import Literal

from jaxtyping import Float
import torch

from protuna.inference.interfaces import mlm
from protuna.inference.models import hf_mlm
from protuna.typing import SequenceLike


class EsmModels(enum.StrEnum):
    """ESM1b, ESM1v and ESM2 checkpoint names."""

    ESM2_15B = "facebook/esm2_t48_15B_UR50D"
    ESM2_3B = "facebook/esm2_t36_3B_UR50D"
    ESM2_650M = "facebook/esm2_t33_650M_UR50D"
    ESM2_150M = "facebook/esm2_t30_150M_UR50D"
    ESM2_35M = "facebook/esm2_t12_35M_UR50D"
    ESM2_8M = "facebook/esm2_t6_8M_UR50D"
    ESM_1B = "facebook/esm1b_t33_650M_UR50S"
    ESM_1V_1 = "facebook/esm1v_t33_650M_UR90S_1"
    ESM_1V_2 = "facebook/esm1v_t33_650M_UR90S_2"
    ESM_1V_3 = "facebook/esm1v_t33_650M_UR90S_3"
    ESM_1V_4 = "facebook/esm1v_t33_650M_UR90S_4"
    ESM_1V_5 = "facebook/esm1v_t33_650M_UR90S_5"


class _EsmBase[RequestT, ResponseT](hf_mlm.HuggingFaceMLMBase[RequestT, ResponseT]):
    def __init__(
        self,
        model_name: EsmModels,
        *,
        pretrained_path: str | os.PathLike | None = None,
        precision: Literal["fp16", "fp32"] | None = None,
        batch_size: int | None = None,
    ):
        """Configure the ESM checkpoint.

        Args:
            model_name: Checkpoint architecture and tokenizer to use.
            pretrained_path: Optional replacement weights for this checkpoint.
            precision: Inference precision; ``None`` uses fp16.
            batch_size: Requests per forward pass, or ``None`` for all requests.
        """
        model_name = EsmModels(model_name)
        precision = "fp16" if precision is None else precision
        if precision not in ("fp16", "fp32"):
            raise ValueError(f"Unsupported precision: {precision}")
        super().__init__(
            model_name,
            pretrained_path=pretrained_path,
            dtype=torch.float16 if precision == "fp16" else torch.float32,
            batch_size=batch_size,
            trust_remote_code=True,
        )

    def _forward_kwargs(
        self, configs: Sequence[mlm.MLMOutputConfig]
    ) -> dict[str, bool]:
        return {
            "output_hidden_states": any(
                cfg.return_hidden_states
                or cfg.return_embeddings
                or cfg.return_mean_embeddings
                for cfg in configs
            ),
        }


class EsmInferenceModel(_EsmBase[mlm.MLMRequest, mlm.MLMOutput]):
    """:py:type:`~protuna.inference.interfaces.mlm.MaskedLMLike` adapter for ESM.

    Supports ESM-1b :cite:p:`ESM1b`, ESM-1v :cite:p:`ESM1v` and ESM-2 :cite:p:`ESM2`.
    """

    def forward(self, requests: SequenceLike[mlm.MLMRequest]) -> list[mlm.MLMOutput]:
        """Return the outputs selected by each request's configuration.

        Args:
            requests: Tokenized sequences and output configurations for inference.

        Returns:
            A list of model outputs in request order.
        """
        return self._forward_mlm(requests)


class EsmLogitsModel(_EsmBase[mlm.MLMLogitsRequest, Float[torch.Tensor, "l v"]]):
    """:py:type:`~protuna.inference.interfaces.mlm.MLMForLogitsLike` adapter for ESM.

    Supports ESM-1b :cite:p:`ESM1b`, ESM-1v :cite:p:`ESM1v` and ESM-2 :cite:p:`ESM2`.
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
