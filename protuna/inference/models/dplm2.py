"""Inference adapters for DPLM2 :cite:p:`DPLM2` through FastPLMs :cite:p:`FastPLMs`."""

from collections.abc import Sequence
import enum
import os
from typing import Literal

from jaxtyping import Float
import torch
import transformers

from protuna.inference.interfaces import mlm
from protuna.inference.models import hf_mlm
from protuna.typing import SequenceLike, TokenIDs


class Dplm2Models(enum.StrEnum):
    """DPLM2 checkpoint names."""

    DPLM2_150M = "Synthyra/DPLM2-150M"
    DPLM2_650M = "Synthyra/DPLM2-650M"
    DPLM2_3B = "Synthyra/DPLM2-3B"


class _Dplm2Base[RequestT, ResponseT](hf_mlm.HuggingFaceMLMBase[RequestT, ResponseT]):
    _special_tokens = {"*": "<pad>", "^": "<cls_aa>", "_": "<mask_aa>", "?": "<unk_aa>"}

    def __init__(
        self,
        model_name: Dplm2Models,
        pretrained_path: str | os.PathLike | None = None,
        precision: Literal["bf16", "fp32"] | None = None,
        batch_size: int | None = None,
        *,
        trust_remote_code: bool | None = None,
    ):
        """Configure the DPLM2 checkpoint.

        Args:
            model_name: Checkpoint architecture and tokenizer to use.
            pretrained_path: Optional replacement weights for this checkpoint.
            precision: Inference precision; ``None`` uses fp32.
                Bf16 uses autocast while keeping weights in fp32.
            batch_size: Requests per forward pass, or ``None`` for all requests.
            trust_remote_code: Whether to allow Synthyra's custom adaptation
                code. ``None`` (default) lets Transformers prompt when required.
                ``True`` allows remote code without prompting; ``False`` refuses
                it. Prompting requires interactive input and may time out.
        """
        precision = "fp32" if precision is None else precision
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
        return {**super()._forward_kwargs(configs), "return_dict": True}

    def _load_model(self) -> torch.nn.Module:
        # DPLM2 does not support Hugging Face device maps.
        return transformers.AutoModelForMaskedLM.from_pretrained(
            self._pretrained_path,
            dtype=torch.float32,
            trust_remote_code=self._trust_remote_code,
        ).to(self._device)

    def prepare_inputs(self, inputs: list[TokenIDs]) -> transformers.BatchEncoding:
        """Translate amino acids and add boundary and padding tokens."""
        vocab = self.tokenizer.get_vocab()
        width = max(len(tokens) for tokens in inputs) + 2
        input_ids = torch.full(
            (len(inputs), width), vocab["<pad>"], dtype=torch.long, device=self._device
        )
        for i, tokens in enumerate(inputs):
            input_ids[i, 0] = vocab["<cls_aa>"]
            input_ids[i, 1 : len(tokens) + 1] = self.translation[
                tokens.to(self._device)
            ]
            input_ids[i, len(tokens) + 1] = vocab["<eos_aa>"]
        return transformers.BatchEncoding(
            {
                "input_ids": input_ids,
                "attention_mask": input_ids.ne(vocab["<pad>"]).long(),
            }
        )


class Dplm2InferenceModel(_Dplm2Base[mlm.MLMRequest, mlm.MLMOutput]):
    """:py:type:`~protuna.inference.interfaces.mlm.MaskedLMLike` adapter for DPLM2.

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


class Dplm2LogitsModel(_Dplm2Base[mlm.MLMLogitsRequest, Float[torch.Tensor, "l v"]]):
    """:py:type:`~protuna.inference.interfaces.mlm.MLMForLogitsLike` adapter for DPLM2.

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
