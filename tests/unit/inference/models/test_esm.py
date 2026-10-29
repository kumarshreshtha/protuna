"""Tests for the ESM inference wrapper in protuna.inference.models.esm."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import re
from typing import Any

import pytest
import torch
import transformers

from protuna import inference
from protuna.data import tokenizer as protuna_tokenizer
from protuna.inference.models import esm
from protuna.inference.models import esmc
from protuna.inference.models import hf_mlm

# The real ESM vocabulary, verified against
# `AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D").get_vocab()`.
_ESM_VOCAB = {
    "<cls>": 0, "<pad>": 1, "<eos>": 2, "<unk>": 3,
    "L": 4, "A": 5, "G": 6, "V": 7, "S": 8, "E": 9, "R": 10, "T": 11,
    "I": 12, "D": 13, "P": 14, "K": 15, "Q": 16, "N": 17, "F": 18, "Y": 19,
    "M": 20, "H": 21, "W": 22, "C": 23, "X": 24, "B": 25, "U": 26, "Z": 27,
    "O": 28, ".": 29, "-": 30, "<null_1>": 31, "<mask>": 32,
}  # fmt: skip

_ESM_VOCAB_SIZE = 33
_HIDDEN_SIZE = 4

# protuna's `ordered_tokens` (A C D E F G H I K L M N P Q R S T V W Y - _ ^ * ?)
# expressed as ESM token IDs. This is what `EsmInferenceModel.translation` must
# produce so that logits can be reindexed into protuna's vocabulary order.
_TRANSLATION = [
    5, 23, 13, 9, 18, 6, 21, 12, 15, 4, 20, 17, 14, 16, 10, 8, 11, 7, 22, 19,
    30, 32, 0, 1, 3,
]  # fmt: skip


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _FakeEsmTokenizer:
    """Stand-in for the ESM tokenizer that mirrors its real vocabulary.

    Encodes like the HuggingFace ESM tokenizer: a ``<cls>`` prefix, an ``<eos>``
    suffix, right padding to the longest sequence in the batch, characters
    outside the alphabet mapped to ``<unk>``, and the literal string
    ``"<mask>"`` mapped to the mask token.
    """

    mask_token = "<mask>"
    mask_token_id = _ESM_VOCAB["<mask>"]
    pad_token_id = _ESM_VOCAB["<pad>"]

    def convert_tokens_to_ids(self, token: str) -> int:
        """Returns the ESM ID for a token symbol."""
        return _ESM_VOCAB.get(token, _ESM_VOCAB["<unk>"])

    def __call__(
        self,
        sequences: Sequence[str],
        padding: bool = True,
        return_tensors: str = "pt",
        return_attention_mask: bool = True,
    ) -> transformers.BatchEncoding:
        """Encodes a batch of sequences into padded ID and mask tensors."""
        encoded = [self._encode(sequence) for sequence in sequences]
        width = max(len(ids) for ids in encoded)
        input_ids = torch.full((len(encoded), width), self.pad_token_id)
        attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids)
            attention_mask[row, : len(ids)] = 1
        return transformers.BatchEncoding(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

    def _encode(self, sequence: str) -> list[int]:
        ids = [_ESM_VOCAB["<cls>"]]
        for chunk in re.split(r"(<mask>)", sequence):
            if chunk == self.mask_token:
                ids.append(self.mask_token_id)
            else:
                ids.extend(_ESM_VOCAB.get(char, _ESM_VOCAB["<unk>"]) for char in chunk)
        ids.append(_ESM_VOCAB["<eos>"])
        return ids


class _FakeEsmModel:
    """Masked LM stand-in returning position- and vocabulary-indexed tensors.

    Logits at ``(b, l, v)`` equal ``v``, so reindexing them through the
    translation table yields the translation table itself. Hidden states at
    ``(b, l, d)`` equal ``l``, so slicing and masking are visible in the values.
    Every call is recorded on :attr:`calls`; the placement and mode requested by
    :meth:`to` and :meth:`eval` are recorded on :attr:`device` and
    :attr:`training`.
    """

    def __init__(self, *, emit_last_hidden_state: bool = False):
        """Configures whether the model reports a ``last_hidden_state``.

        Args:
            emit_last_hidden_state: If ``True``, the output carries a
                ``last_hidden_state`` entry, as the ESMC models do.
        """
        self._emit_last_hidden_state = emit_last_hidden_state
        self.calls: list[dict[str, Any]] = []
        self.device: torch.device | None = None
        self.training = True

    def to(self, device: torch.device) -> _FakeEsmModel:
        """Records the requested placement, as `torch.nn.Module.to` would."""
        self.device = device
        return self

    def eval(self) -> _FakeEsmModel:
        """Records the switch to evaluation mode."""
        self.training = False
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        compute_logits: bool = True,
    ) -> dict[str, Any]:
        """Records the call and returns the configured outputs."""
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask,
                "output_hidden_states": output_hidden_states,
                "compute_logits": compute_logits,
            }
        )
        batch, length = input_ids.shape
        logits = torch.arange(_ESM_VOCAB_SIZE, dtype=torch.float).expand(
            batch, length, _ESM_VOCAB_SIZE
        )
        positions = (
            torch.arange(length, dtype=torch.float)[None, :, None]
            .expand(batch, length, _HIDDEN_SIZE)
            .contiguous()
        )
        outputs: dict[str, Any] = {}
        if compute_logits:
            outputs["logits"] = logits
        if self._emit_last_hidden_state:
            outputs["last_hidden_state"] = positions
        if output_hidden_states:
            outputs["hidden_states"] = (torch.zeros_like(positions), positions)
        return outputs


class _Stubs:
    """Replacement `from_pretrained` entry points, recording what was asked for."""

    def __init__(self):
        self.model = _FakeEsmModel()
        self.config_names: list[str] = []
        self.tokenizer_names: list[str] = []
        self.model_calls: list[dict[str, Any]] = []

    def config(self, name: str, **kwargs: object) -> object:
        """Stands in for `AutoConfig.from_pretrained`."""
        self.config_names.append(str(name))
        return object()

    def tokenizer(self, name: str, **kwargs: object) -> _FakeEsmTokenizer:
        """Stands in for `AutoTokenizer.from_pretrained`."""
        self.tokenizer_names.append(str(name))
        return _FakeEsmTokenizer()

    def masked_lm(self, name: str, **kwargs: object) -> _FakeEsmModel:
        """Stands in for `AutoModelForMaskedLM.from_pretrained`."""
        self.model_calls.append({"name": str(name), **kwargs})
        return self.model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> _Stubs:
    """Patches the transformers auto-classes with in-test doubles."""
    stubs = _Stubs()
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", stubs.config)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", stubs.tokenizer)
    monkeypatch.setattr(
        transformers.AutoModelForMaskedLM, "from_pretrained", stubs.masked_lm
    )
    return stubs


def _loaded(
    stubs: _Stubs,
    model_name: esm.EsmModels = esm.EsmModels.ESM2_8M,
    device: torch.device | Sequence[torch.device] | None = None,
) -> esm.EsmInferenceModel:
    """Builds an `EsmInferenceModel` already loaded onto *device*."""
    model = esm.EsmInferenceModel(model_name)
    model.load(torch.device("cpu") if device is None else device)
    return model


def _present[T](value: T | None) -> T:
    """Asserts an optional output field was populated and returns it."""
    assert value is not None
    return value


def _request(sequence: str, **config: bool) -> inference.MLMRequest:
    """Builds an `MLMRequest` for *sequence* with the given output config."""
    return inference.MLMRequest(
        token_ids=protuna_tokenizer.encode(sequence),
        output_config=inference.MLMOutputConfig(**config),
    )


# ---------------------------------------------------------------------------
# Construction and loading
# ---------------------------------------------------------------------------


def test_init_checks_the_config_without_loading_weights(stubs: _Stubs) -> None:
    """Construction validates the model name but downloads nothing."""
    esm.EsmInferenceModel(esm.EsmModels.ESM2_8M)

    assert stubs.config_names == ["facebook/esm2_t6_8M_UR50D"]
    assert stubs.model_calls == []
    assert stubs.tokenizer_names == []


@pytest.mark.parametrize("device", ["cpu", "cuda:1", "mps"])
def test_load_places_the_model_on_the_given_device(stubs: _Stubs, device: str) -> None:
    """Load moves the weights onto the device it was given."""
    _loaded(stubs, device=torch.device(device))

    assert stubs.model_calls[0]["device_map"] == torch.device(device)


def test_load_puts_the_model_in_evaluation_mode(stubs: _Stubs) -> None:
    """Loaded weights are not left in training mode."""
    _loaded(stubs, device=torch.device("cpu"))

    assert stubs.model.training is False


def test_load_requests_the_default_precision_for_the_model_family(
    stubs: _Stubs,
) -> None:
    """Load asks from_pretrained for the checkpoint's default precision."""
    _loaded(stubs, device=torch.device("cpu"))

    assert stubs.model_calls == [
        {
            "name": "facebook/esm2_t6_8M_UR50D",
            "dtype": torch.float16,
            "trust_remote_code": True,
            "device_map": torch.device("cpu"),
        }
    ]


def test_load_requests_the_precision_it_was_configured_with(stubs: _Stubs) -> None:
    """An explicit precision overrides the family default."""
    model = esm.EsmInferenceModel(esm.EsmModels.ESM2_8M, precision="fp32")
    model.load(torch.device("cpu"))

    assert stubs.model_calls[0]["dtype"] == torch.float32


def test_load_uses_only_the_first_of_several_devices(stubs: _Stubs) -> None:
    """A multi-device placement collapses to the first device."""
    _loaded(
        stubs,
        device=[torch.device("cpu"), torch.device("cuda:1"), torch.device("cuda:2")],
    )

    assert stubs.model_calls[0]["device_map"] == torch.device("cpu")


def test_load_warns_when_given_several_devices(
    stubs: _Stubs, caplog: pytest.LogCaptureFixture
) -> None:
    """A multi-device placement is reported as unsupported."""
    _loaded(
        stubs,
        device=[torch.device("cpu"), torch.device("cuda:1"), torch.device("cuda:2")],
    )

    assert caplog.messages == [
        "EsmInferenceModel does not support multi-device placement. Found 3 devices."
        " Only first device will be used."
    ]


def test_load_is_quiet_for_a_single_device_sequence(
    stubs: _Stubs, caplog: pytest.LogCaptureFixture
) -> None:
    """A one-element device sequence is not a multi-device placement."""
    _loaded(stubs, device=[torch.device("cpu")])

    assert caplog.messages == []


def test_load_reads_weights_from_the_pretrained_path(stubs: _Stubs) -> None:
    """A pretrained path overrides the hub name for the weights only."""
    model = esm.EsmInferenceModel(
        esm.EsmModels.ESM2_8M, pretrained_path="/checkpoints/esm"
    )
    model.load(torch.device("cpu"))

    assert stubs.model_calls[0]["name"] == "/checkpoints/esm"
    assert stubs.tokenizer_names == ["facebook/esm2_t6_8M_UR50D"]


def test_forward_before_load_raises(stubs: _Stubs) -> None:
    """Calling forward without loading is an error, not a crash inside torch."""
    model = esm.EsmInferenceModel(esm.EsmModels.ESM2_8M)

    with pytest.raises(RuntimeError, match="forward called before loading the model"):
        model.forward([_request("ACDEF")])


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_getstate_drops_the_loaded_model_and_tokenizer(stubs: _Stubs) -> None:
    """Pickled state carries no live model, so it can cross a Ray boundary."""
    model = _loaded(stubs)

    state = model.__getstate__()

    assert state["_model"] is None
    assert state["_tokenizer"] is None
    assert state["_translation"] is None
    assert state["_device"] is None


def test_getstate_keeps_the_configuration_needed_to_reload(stubs: _Stubs) -> None:
    """Pickled state keeps the identifiers a fresh worker needs to reload."""
    model = _loaded(stubs)

    state = model.__getstate__()

    assert state["_model_name"] == esm.EsmModels.ESM2_8M
    assert state["_pretrained_path"] == esm.EsmModels.ESM2_8M


def test_getstate_leaves_the_original_instance_loaded(stubs: _Stubs) -> None:
    """Serializing a model does not unload the model being serialized."""
    model = _loaded(stubs)

    model.__getstate__()

    assert model.forward([_request("ACDEF")])[0].logits is not None


# ---------------------------------------------------------------------------
# Vocabulary translation
# ---------------------------------------------------------------------------


def test_translation_maps_protuna_tokens_to_esm_ids(stubs: _Stubs) -> None:
    """Every protuna token, specials included, maps to its ESM counterpart."""
    model = _loaded(stubs)

    assert model.translation.tolist() == _TRANSLATION


def test_translation_is_available_before_the_model_is_loaded(stubs: _Stubs) -> None:
    """The translation table only needs the tokenizer, not the weights."""
    model = esm.EsmInferenceModel(esm.EsmModels.ESM2_8M)

    assert model.translation.tolist() == _TRANSLATION
    assert stubs.model_calls == []


# ---------------------------------------------------------------------------
# forward: logits
# ---------------------------------------------------------------------------


def test_forward_returns_one_output_per_request(stubs: _Stubs) -> None:
    """Outputs line up one-to-one with requests."""
    model = _loaded(stubs)

    outputs = model.forward([_request("ACDEF"), _request("AC"), _request("WY")])

    assert len(outputs) == 3


def test_forward_reindexes_logits_into_protuna_vocab_order(stubs: _Stubs) -> None:
    """Logit columns are gathered in protuna's token order."""
    model = _loaded(stubs)

    outputs = model.forward([_request("ACDEF")])

    assert _present(outputs[0].logits)[0].tolist() == [float(i) for i in _TRANSLATION]


def test_forward_strips_the_special_tokens_from_logits(stubs: _Stubs) -> None:
    """The cls and eos rows are dropped, leaving one row per residue."""
    model = _loaded(stubs)

    outputs = model.forward([_request("ACDEF")])

    assert _present(outputs[0].logits).shape == (5, 25)


def test_forward_strips_padding_from_shorter_sequences_in_a_batch(
    stubs: _Stubs,
) -> None:
    """Each output is trimmed to its own sequence length, not the batch width."""
    model = _loaded(stubs)

    outputs = model.forward([_request("ACDEF"), _request("AC")])

    assert _present(outputs[0].logits).shape == (5, 25)
    assert _present(outputs[1].logits).shape == (2, 25)


def test_forward_aa_tokens_only_keeps_the_twenty_amino_acid_columns(
    stubs: _Stubs,
) -> None:
    """aa_tokens_only drops the gap and special-token columns."""
    model = _loaded(stubs)

    outputs = model.forward([_request("ACDEF", aa_tokens_only=True)])

    assert _present(outputs[0].logits).shape == (5, 20)
    assert _present(outputs[0].logits)[0].tolist() == [
        float(i) for i in _TRANSLATION[:20]
    ]


def test_forward_omits_logits_when_they_are_not_requested(stubs: _Stubs) -> None:
    """return_logits=False leaves the logits field unset."""
    model = _loaded(stubs)

    outputs = model.forward([_request("ACDEF", return_logits=False)])

    assert outputs[0].logits is None


def test_forward_masked_positions_only_returns_just_the_masked_rows(
    stubs: _Stubs,
) -> None:
    """masked_positions_only keeps one row per masked residue."""
    model = _loaded(stubs)

    outputs = model.forward([_request("AC_EF", masked_positions_only=True)])

    assert _present(outputs[0].logits).shape == (1, 25)


# ---------------------------------------------------------------------------
# forward: embeddings and hidden states
# ---------------------------------------------------------------------------


def test_forward_returns_embeddings_for_every_non_pad_position(stubs: _Stubs) -> None:
    """Embeddings cover the cls and eos rows but not the padding."""
    model = _loaded(stubs)

    outputs = model.forward(
        [
            _request("ACDEF", return_embeddings=True),
            _request("AC", return_embeddings=True),
        ]
    )

    assert _present(outputs[0].embeddings).shape == (5, _HIDDEN_SIZE)
    assert _present(outputs[1].embeddings).shape == (2, _HIDDEN_SIZE)


def test_forward_averages_embeddings_over_non_pad_positions(stubs: _Stubs) -> None:
    """Mean embeddings average only the positions the sequence occupies."""
    model = _loaded(stubs)

    outputs = model.forward(
        [
            _request("ACDEF", return_mean_embeddings=True),
            _request("AC", return_mean_embeddings=True),
        ]
    )

    assert _present(outputs[0].mean_embeddings).tolist() == [3.0] * _HIDDEN_SIZE
    assert _present(outputs[1].mean_embeddings).tolist() == [1.5] * _HIDDEN_SIZE


def test_forward_returns_every_layer_of_hidden_states(stubs: _Stubs) -> None:
    """return_hidden_states hands back the full per-layer tuple."""
    model = _loaded(stubs)

    outputs = model.forward([_request("ACDEF", return_hidden_states=True)])

    assert len(_present(outputs[0].hidden_states)) == 2
    assert _present(outputs[0].hidden_states)[-1].shape == (5, _HIDDEN_SIZE)


def test_forward_applies_each_requests_own_output_config(stubs: _Stubs) -> None:
    """A mixed batch gives each request only what it asked for."""
    model = _loaded(stubs)

    outputs = model.forward(
        [
            _request("ACDEF", return_logits=False, return_mean_embeddings=True),
            _request("AC"),
        ]
    )

    assert outputs[0].logits is None
    assert outputs[0].mean_embeddings is not None
    assert outputs[1].logits is not None
    assert outputs[1].mean_embeddings is None


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, False),
        ({"return_hidden_states": True}, True),
        ({"return_embeddings": True}, True),
        ({"return_mean_embeddings": True}, True),
    ],
)
def test_forward_asks_for_hidden_states_only_when_needed(
    stubs: _Stubs, config: dict[str, bool], expected: bool
) -> None:
    """The hidden-state flag is set only when some request needs them."""
    model = _loaded(stubs)

    model.forward([_request("ACDEF", **config)])

    assert stubs.model.calls[0]["output_hidden_states"] is expected


def test_forward_asks_for_hidden_states_when_any_request_needs_them(
    stubs: _Stubs,
) -> None:
    """One embedding request in the batch turns hidden states on for the batch."""
    model = _loaded(stubs)

    model.forward([_request("ACDEF"), _request("AC", return_embeddings=True)])

    assert stubs.model.calls[0]["output_hidden_states"] is True


def test_forward_pads_the_batch_to_the_longest_sequence(stubs: _Stubs) -> None:
    """Ragged batches are padded before they reach the model."""
    model = _loaded(stubs)

    model.forward([_request("ACDEF"), _request("AC")])

    assert stubs.model.calls[0]["input_ids"].shape == (2, 7)
    assert stubs.model.calls[0]["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0],
    ]


# ---------------------------------------------------------------------------
# ESMC models
# ---------------------------------------------------------------------------


def test_esmc_forward_does_not_ask_for_hidden_states_to_get_embeddings(
    stubs: _Stubs,
) -> None:
    """ESMC always reports a last hidden state, so the flag stays off."""
    stubs.model = _FakeEsmModel(emit_last_hidden_state=True)
    model = esmc.EsmcInferenceModel(esmc.EsmcModels.ESMC_300M)
    model.load(torch.device("cpu"))

    model.forward([_request("ACDEF", return_mean_embeddings=True)])

    assert stubs.model.calls[0]["output_hidden_states"] is False


def test_esmc_forward_takes_embeddings_from_the_last_hidden_state(
    stubs: _Stubs,
) -> None:
    """Embeddings come from last_hidden_state when the model provides one."""
    stubs.model = _FakeEsmModel(emit_last_hidden_state=True)
    model = esmc.EsmcInferenceModel(esmc.EsmcModels.ESMC_300M)
    model.load(torch.device("cpu"))

    outputs = model.forward([_request("ACDEF", return_embeddings=True)])

    assert _present(outputs[0].embeddings).shape == (5, _HIDDEN_SIZE)


def test_esmc_forward_skips_logits_when_they_are_not_requested(
    stubs: _Stubs,
) -> None:
    """ESMC computes its logit head only when some request wants logits."""
    stubs.model = _FakeEsmModel(emit_last_hidden_state=True)
    model = esmc.EsmcInferenceModel(esmc.EsmcModels.ESMC_300M)
    model.load(torch.device("cpu"))

    model.forward([_request("ACDEF", return_logits=False, return_mean_embeddings=True)])

    assert stubs.model.calls[0]["compute_logits"] is False


def test_esmc_forward_asks_for_logits_when_they_are_requested(stubs: _Stubs) -> None:
    """A logit request turns the ESMC logit head back on."""
    stubs.model = _FakeEsmModel(emit_last_hidden_state=True)
    model = esmc.EsmcInferenceModel(esmc.EsmcModels.ESMC_300M)
    model.load(torch.device("cpu"))

    model.forward([_request("ACDEF", return_logits=True)])

    assert stubs.model.calls[0]["compute_logits"] is True


def test_esmc_loads_at_bf16_by_default(stubs: _Stubs) -> None:
    """ESMC checkpoints default to bf16 rather than the ESM-2 default."""
    model = esmc.EsmcInferenceModel(esmc.EsmcModels.ESMC_300M)
    model.load(torch.device("cpu"))

    assert stubs.model_calls[0]["dtype"] == torch.bfloat16


def test_esmc_forward_still_returns_hidden_states_when_asked(stubs: _Stubs) -> None:
    """An explicit hidden-state request overrides the ESMC shortcut."""
    stubs.model = _FakeEsmModel(emit_last_hidden_state=True)
    model = esmc.EsmcInferenceModel(esmc.EsmcModels.ESMC_300M)
    model.load(torch.device("cpu"))

    outputs = model.forward([_request("ACDEF", return_hidden_states=True)])

    assert stubs.model.calls[0]["output_hidden_states"] is True
    assert len(_present(outputs[0].hidden_states)) == 2


@pytest.mark.parametrize(
    ("adapter_type", "name", "precision", "dtype"),
    [
        (esm.EsmInferenceModel, esm.EsmModels.ESM2_8M, "fp16", torch.float16),
        (esm.EsmLogitsModel, esm.EsmModels.ESM_1B, "fp32", torch.float32),
        (esmc.EsmcInferenceModel, esmc.EsmcModels.ESMC_300M, "bf16", torch.bfloat16),
        (esmc.EsmcLogitsModel, esmc.EsmcModels.ESMC_600M, "fp32", torch.float32),
    ],
)
def test_family_precision(
    stubs: _Stubs,
    adapter_type: Callable[..., hf_mlm.HuggingFaceMLMBase],
    name: str,
    precision: str,
    dtype: torch.dtype,
) -> None:
    """Load weights using the requested family precision."""
    adapter = adapter_type(name, precision=precision)
    adapter.load(torch.device("cpu"))

    assert stubs.model_calls[0]["dtype"] == dtype


@pytest.mark.parametrize(
    ("adapter_type", "name", "precision"),
    [
        (esm.EsmInferenceModel, esm.EsmModels.ESM2_8M, "bf16"),
        (esm.EsmLogitsModel, esm.EsmModels.ESM2_8M, "bf16"),
        (esmc.EsmcInferenceModel, esmc.EsmcModels.ESMC_300M, "fp16"),
        (esmc.EsmcLogitsModel, esmc.EsmcModels.ESMC_300M, "fp16"),
        (esm.EsmInferenceModel, esm.EsmModels.ESM2_8M, "invalid"),
        (esmc.EsmcInferenceModel, esmc.EsmcModels.ESMC_300M, "invalid"),
    ],
)
def test_reject_unsupported_precision(
    stubs: _Stubs,
    adapter_type: Callable[..., hf_mlm.HuggingFaceMLMBase],
    name: str,
    precision: str,
) -> None:
    """Reject unsupported precision before resolving the checkpoint."""
    with pytest.raises(ValueError, match="Unsupported precision"):
        adapter_type(name, precision=precision)

    assert stubs.config_names == []


@pytest.mark.parametrize(
    ("adapter_type", "name"),
    [
        (esm.EsmInferenceModel, esmc.EsmcModels.ESMC_300M),
        (esmc.EsmcInferenceModel, esm.EsmModels.ESM2_8M),
    ],
)
def test_reject_checkpoint_from_other_family(
    stubs: _Stubs,
    adapter_type: Callable[..., hf_mlm.HuggingFaceMLMBase],
    name: str,
) -> None:
    """Reject a checkpoint from the other ESM family."""
    with pytest.raises(ValueError, match="is not a valid"):
        adapter_type(name)

    assert stubs.config_names == []


@pytest.mark.parametrize(
    ("adapter_type", "name"),
    [
        (esm.EsmLogitsModel, esm.EsmModels.ESM2_8M),
        (esmc.EsmcLogitsModel, esmc.EsmcModels.ESMC_300M),
    ],
)
def test_direct_logits_filter_ragged_requests(
    stubs: _Stubs,
    adapter_type: Callable[..., hf_mlm.HuggingFaceMLMBase],
    name: str,
) -> None:
    """Filter masked, full and empty sequences through each logits adapter."""
    adapter = adapter_type(name, precision="fp32", batch_size=2)
    adapter.load(torch.device("cpu"))

    masked, full, empty = adapter(
        [
            inference.MLMLogitsRequest(protuna_tokenizer.encode("A_"), True, True),
            inference.MLMLogitsRequest(protuna_tokenizer.encode("C")),
            inference.MLMLogitsRequest(protuna_tokenizer.encode(""), True, True),
        ]
    )

    assert masked.tolist() == [
        [5, 23, 13, 9, 18, 6, 21, 12, 15, 4, 20, 17, 14, 16, 10, 8, 11, 7, 22, 19]
    ]
    assert full.shape == (1, 25)
    assert empty.shape == (0, 20)
    assert len(stubs.model_calls) == 1


def test_hidden_states_exclude_other_sequences_and_padding(stubs: _Stubs) -> None:
    """Return each sequence's residue hidden states independently."""
    model = _loaded(stubs)

    first, second = model(
        [
            _request("ACD", return_hidden_states=True),
            _request("A", return_hidden_states=True),
        ]
    )

    assert _present(first.hidden_states)[-1].shape == (3, _HIDDEN_SIZE)
    assert _present(second.hidden_states)[-1].shape == (1, _HIDDEN_SIZE)
    assert _present(second.hidden_states)[-1][0].tolist() == [1.0] * _HIDDEN_SIZE
