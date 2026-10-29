"""Fixtures for checkpoint adapter tests."""

import dataclasses
import pathlib
from typing import Any
from unittest import mock

import pytest
import torch
import transformers


class _DplmModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_states_requested = False
        self.autocast_dtype: torch.dtype | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> dict[str, Any]:
        self.hidden_states_requested = output_hidden_states
        self.autocast_dtype = (
            torch.get_autocast_dtype(input_ids.device.type)
            if torch.is_autocast_enabled(input_ids.device.type)
            else None
        )
        embeddings = input_ids.float().unsqueeze(-1).expand(-1, -1, 2)
        logits = torch.arange(33, dtype=torch.float32).expand(*input_ids.shape, 33)
        return {
            "logits": logits,
            "last_hidden_state": embeddings,
            "hidden_states": (embeddings * 0, embeddings)
            if output_hidden_states
            else None,
        }


@dataclasses.dataclass
class DplmCheckpoint:
    """Hold a strict forward double and mocked external loading entry points."""

    model: _DplmModel
    config_loader: mock.MagicMock
    tokenizer_loader: mock.MagicMock
    model_loader: mock.MagicMock


@pytest.fixture
def dplm_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> DplmCheckpoint:
    """Provide the real ESM tokenizer vocabulary without checkpoint downloads."""
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "<cls>\n<pad>\n<eos>\n<unk>\nL\nA\nG\nV\nS\nE\nR\nT\nI\nD\n"
        "P\nK\nQ\nN\nF\nY\nM\nH\nW\nC\nX\nB\nU\nZ\nO\n.\n-\n<null_1>\n<mask>\n"
    )
    tokenizer = transformers.EsmTokenizer(vocab_file=str(vocab))
    model = _DplmModel()
    config_loader = mock.create_autospec(
        transformers.AutoConfig.from_pretrained, return_value=object()
    )
    tokenizer_loader = mock.create_autospec(
        transformers.AutoTokenizer.from_pretrained, return_value=tokenizer
    )
    model_loader = mock.create_autospec(
        transformers.AutoModelForMaskedLM.from_pretrained, return_value=model
    )
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", config_loader)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(
        transformers.AutoModelForMaskedLM, "from_pretrained", model_loader
    )
    return DplmCheckpoint(model, config_loader, tokenizer_loader, model_loader)
