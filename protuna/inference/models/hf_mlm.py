"""Shared operations for protein masked language models hosted on huggingface hub.

:class:`HuggingFaceMLMBase` handles checkpoint resources, token translation and output
selection. Family specific adapters supply precision rules and forward options.
"""

from collections.abc import Sequence
import logging
import os
from typing import Any, NotRequired, TypedDict

from jaxtyping import Float
import torch
import transformers

from protuna import data
from protuna.inference import handlers
from protuna.inference.interfaces import mlm
from protuna.typing import SequenceLike, TokenIDs
from protuna.utils import devices

logger = logging.getLogger(__name__)


class _ModelOutput(TypedDict):
    """Collect the checkpoint outputs consumed by the adapters."""

    logits: NotRequired[torch.Tensor]
    hidden_states: NotRequired[tuple[torch.Tensor, ...]]
    last_hidden_state: NotRequired[torch.Tensor]


class HuggingFaceMLMBase[RequestT, ResponseT](
    handlers.InferenceModel[RequestT, ResponseT]
):
    """Shared loading and inference operations across MaskedLM hosted on HF Hub."""

    _special_tokens = {"*": "<pad>", "^": "<cls>", "_": "<mask>", "?": "<unk>"}
    _fp32_weights = False

    def __init__(
        self,
        model_name: str,
        *,
        pretrained_path: str | os.PathLike | None,
        dtype: torch.dtype,
        batch_size: int | None,
        trust_remote_code: bool | None,
    ):
        """Configure a checkpoint without loading its weights.

        Args:
            model_name: Checkpoint architecture and tokenizer to use.
            pretrained_path: Optional replacement weights.
            dtype: Precision for forward computation.
            batch_size: Requests per forward pass, or ``None`` for all requests.
            trust_remote_code: Allow checkpoint code, refuse it, or let
                Transformers prompt when required with ``None``.
        """
        super().__init__(batch_size=batch_size)
        self._trust_remote_code = trust_remote_code
        transformers.AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self._model_name = model_name
        self._pretrained_path = (
            model_name if pretrained_path is None else pretrained_path
        )
        self.precision = dtype
        self._reset()

    def _load_model(self) -> torch.nn.Module:
        return transformers.AutoModelForMaskedLM.from_pretrained(
            self._pretrained_path,
            dtype=torch.float32 if self._fp32_weights else self.precision,
            trust_remote_code=self._trust_remote_code,
            device_map=self._device,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Return the picklable state, dropping anything :meth:`load` rebuilds."""
        state = self.__dict__.copy()
        state["_model"] = None
        state["_tokenizer"] = None
        state["_translation"] = None
        state["_device"] = None
        return state

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Load the checkpoint onto device.

        Args:
            device: Device or devices on which to load the checkpoint. ``None``
                selects the first visible CUDA device, or CPU if CUDA is unavailable.

        .. warning:: Sharding across devices is not supported; given several devices,
           only the first is used.
        """
        if (
            device is not None
            and not isinstance(device, torch.device)
            and len(device) > 1
        ):
            logger.warning(
                "%s does not support multi-device placement. Found "
                "%s devices. Only first device will be used.",
                type(self).__name__,
                len(device),
            )
        device = devices.resolve_devices() if device is None else device
        if not isinstance(device, torch.device):
            device = device[0]
        # no-op if nothing changed
        if self._model is not None and self._device == device:
            return
        _ = self.tokenizer
        self._device = device
        self._translation = None
        self._model = self._load_model().eval()

    def _reset(self) -> None:
        self._tokenizer = None
        self._model = None
        self._translation = None
        self._device = torch.device("cpu")

    def release(self) -> None:
        """Release GPU resources held by the handler."""
        self._reset()

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        """Return the checkpoint's own tokenizer.

        Returns:
            The tokenizer configured for the checkpoint.
        """
        if self._tokenizer is None:
            self._tokenizer: transformers.PreTrainedTokenizerBase = (
                transformers.AutoTokenizer.from_pretrained(
                    self._model_name, trust_remote_code=self._trust_remote_code
                )  # ty: ignore[invalid-assignment]
            )
        return self._tokenizer

    @property
    def translation(self) -> torch.Tensor:
        """Return the mapping from protuna token IDs to checkpoint token IDs.

        Returns:
            A tensor indexed by protuna token ID containing checkpoint token IDs.
        """
        if self._translation is None:
            self._translation = torch.tensor(
                [
                    self.tokenizer.convert_tokens_to_ids(
                        self._special_tokens.get(token, token)
                    )
                    for token in data.tokenizer.ordered_tokens
                ],
                dtype=torch.long,
                device=self._device,
            )
        return self._translation

    def prepare_inputs(self, inputs: list[TokenIDs]) -> transformers.BatchEncoding:
        """Translate protuna tokens and pad sequences for the checkpoint.

        Args:
            inputs: Sequences encoded with the protuna tokenizer.

        Returns:
            Padded token IDs and attention masks on the model's device.
        """
        sequences = data.tokenizer.decode(inputs)
        sequences = [
            seq.translate(str.maketrans(self._special_tokens)) for seq in sequences
        ]
        return self.tokenizer(
            sequences, padding=True, return_tensors="pt", return_attention_mask=True
        ).to(self._device)

    def _forward_kwargs(
        self, configs: Sequence[mlm.MLMOutputConfig]
    ) -> dict[str, bool]:
        return {
            "output_hidden_states": any(cfg.return_hidden_states for cfg in configs),
        }

    def _run_model(
        self, token_ids: list[TokenIDs], configs: Sequence[mlm.MLMOutputConfig]
    ) -> tuple[torch.Tensor, _ModelOutput]:
        if self._model is None:
            raise RuntimeError("forward called before loading the model")
        inputs = self.prepare_inputs(token_ids)
        with (
            torch.inference_mode(),
            torch.autocast(
                enabled=self.precision != torch.float32,
                device_type=self._device.type,
                dtype=self.precision,
            ),
        ):
            outputs = self._model(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **self._forward_kwargs(configs),
            )
        return inputs["input_ids"], outputs

    def _select_logits(
        self,
        logits: torch.Tensor,
        inputs: torch.Tensor,
        config: mlm.MLMOutputConfig | mlm.MLMLogitsRequest,
    ) -> Float[torch.Tensor, "l v"]:
        if config.masked_positions_only:
            mask_id = self.tokenizer.convert_tokens_to_ids(self._special_tokens["_"])
            logits = logits[inputs == mask_id]
        else:
            logits = logits[inputs != self.tokenizer.pad_token_id][1:-1]
        return logits[:, :20] if config.aa_tokens_only else logits

    def _forward_mlm(
        self, requests: SequenceLike[mlm.MLMRequest]
    ) -> list[mlm.MLMOutput]:
        configs = [req.output_config for req in requests]
        inputs, model_outputs = self._run_model(
            [req.token_ids for req in requests], configs
        )
        logits = (
            model_outputs["logits"][:, :, self.translation].float()
            if any(cfg.return_logits for cfg in configs)
            else None
        )
        outputs = []
        for i, cfg in enumerate(configs):
            output = mlm.MLMOutput()
            valid = inputs[i] != self.tokenizer.pad_token_id
            if cfg.return_logits:
                assert logits is not None
                output.logits = self._select_logits(logits[i], inputs[i], cfg)
            if cfg.return_embeddings or cfg.return_mean_embeddings:
                embeddings = (
                    model_outputs["last_hidden_state"]
                    if "last_hidden_state" in model_outputs
                    else model_outputs["hidden_states"][-1]
                )[i, valid][1:-1].float()
                if cfg.return_embeddings:
                    output.embeddings = embeddings
                if cfg.return_mean_embeddings:
                    output.mean_embeddings = embeddings.mean(dim=0)
            if cfg.return_hidden_states:
                output.hidden_states = tuple(
                    state[i, valid][1:-1].float()
                    for state in model_outputs["hidden_states"]
                )
            outputs.append(output)
        return outputs

    def _forward_logits(
        self, requests: SequenceLike[mlm.MLMLogitsRequest]
    ) -> list[Float[torch.Tensor, "l v"]]:
        inputs, model_outputs = self._run_model(
            [req.token_ids for req in requests], [mlm.MLMOutputConfig()]
        )
        logits = model_outputs["logits"][:, :, self.translation].float()
        return [
            self._select_logits(logits[i], inputs[i], req)
            for i, req in enumerate(requests)
        ]
