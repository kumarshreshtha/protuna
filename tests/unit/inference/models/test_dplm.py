import pickle
from typing import Literal

import pytest
import torch

from protuna import data
from protuna import inference
from protuna.inference.models import dplm
from protuna.inference.models import shared

from . import conftest


def test_construction_defers_weights(dplm_checkpoint: conftest.DplmCheckpoint) -> None:
    """Validate the checkpoint without loading weights or a tokenizer."""
    dplm.DplmInferenceModel(dplm.DplmModels.DPLM_150M)

    dplm_checkpoint.config_loader.assert_called_once_with(
        dplm.DplmModels.DPLM_150M, trust_remote_code=None
    )
    dplm_checkpoint.model_loader.assert_not_called()
    dplm_checkpoint.tokenizer_loader.assert_not_called()


@pytest.mark.parametrize("name", list(dplm.DplmModels))
def test_load_checkpoint(
    name: dplm.DplmModels, dplm_checkpoint: conftest.DplmCheckpoint
) -> None:
    """Load every published size in evaluation mode at the default precision."""
    adapter = dplm.DplmInferenceModel(name)
    adapter.load(torch.device("cpu"))

    dplm_checkpoint.model_loader.assert_called_once_with(
        name,
        dtype=torch.float32,
        trust_remote_code=None,
        device_map=torch.device("cpu"),
    )
    assert not dplm_checkpoint.model.training


@pytest.mark.parametrize(
    ("precision", "dtype"),
    [("fp32", torch.float32), ("bf16", torch.float32)],
)
def test_custom_weights_and_precision(
    precision: Literal["fp32", "bf16"],
    dtype: torch.dtype,
    dplm_checkpoint: conftest.DplmCheckpoint,
) -> None:
    """Use replacement weights with the original checkpoint tokenizer."""
    adapter = dplm.DplmInferenceModel(
        dplm.DplmModels.DPLM_650M, pretrained_path="local-weights", precision=precision
    )
    adapter.load(torch.device("cpu"))

    dplm_checkpoint.model_loader.assert_called_once_with(
        "local-weights",
        dtype=dtype,
        trust_remote_code=None,
        device_map=torch.device("cpu"),
    )
    dplm_checkpoint.tokenizer_loader.assert_called_once_with(
        dplm.DplmModels.DPLM_650M, trust_remote_code=None
    )


def test_translate_and_pad_inputs(dplm_checkpoint: conftest.DplmCheckpoint) -> None:
    """Translate masks, unknowns and gaps while adding boundary and padding tokens."""
    adapter = dplm.DplmInferenceModel(dplm.DplmModels.DPLM_150M)

    batch = adapter.prepare_inputs(
        [data.tokenizer.encode("A_?-"), data.tokenizer.encode("C")]
    )

    assert batch["input_ids"].tolist() == [[0, 5, 32, 3, 30, 2], [0, 23, 2, 1, 1, 1]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]


def test_mixed_requests_filter_logits(dplm_checkpoint: conftest.DplmCheckpoint) -> None:
    """Respect each request's vocabulary and position filters in a ragged batch."""
    adapter = dplm.DplmInferenceModel(dplm.DplmModels.DPLM_150M)
    adapter.load(torch.device("cpu"))
    requests = [
        inference.MLMRequest(
            data.tokenizer.encode("A_"),
            inference.MLMOutputConfig(masked_positions_only=True, aa_tokens_only=True),
        ),
        inference.MLMRequest(data.tokenizer.encode("C")),
    ]

    masked, full = adapter(requests)

    assert masked.logits is not None
    assert full.logits is not None
    assert masked.logits.tolist() == [
        [5, 23, 13, 9, 18, 6, 21, 12, 15, 4, 20, 17, 14, 16, 10, 8, 11, 7, 22, 19]
    ]
    assert full.logits.tolist() == [
        [
            5,
            23,
            13,
            9,
            18,
            6,
            21,
            12,
            15,
            4,
            20,
            17,
            14,
            16,
            10,
            8,
            11,
            7,
            22,
            19,
            30,
            32,
            0,
            1,
            3,
        ]
    ]


def test_embeddings_use_last_hidden_state(
    dplm_checkpoint: conftest.DplmCheckpoint,
) -> None:
    """Return residue embeddings and their mean without requesting all layers."""
    adapter = dplm.DplmInferenceModel(dplm.DplmModels.DPLM_150M)
    adapter.load(torch.device("cpu"))

    result = adapter(
        [
            inference.MLMRequest(
                data.tokenizer.encode("AC"),
                inference.MLMOutputConfig(
                    return_logits=False,
                    return_embeddings=True,
                    return_mean_embeddings=True,
                ),
            )
        ]
    )[0]

    assert result.logits is None
    assert result.embeddings is not None
    assert result.mean_embeddings is not None
    assert result.embeddings.tolist() == [[5, 5], [23, 23]]
    assert result.mean_embeddings.tolist() == [14, 14]
    assert not dplm_checkpoint.model.hidden_states_requested


def test_hidden_states_belong_to_each_sequence(
    dplm_checkpoint: conftest.DplmCheckpoint,
) -> None:
    """Remove other sequences, padding and boundary tokens from layer outputs."""
    adapter = dplm.DplmInferenceModel(dplm.DplmModels.DPLM_150M)
    adapter.load(torch.device("cpu"))
    config = inference.MLMOutputConfig(return_hidden_states=True)

    first, second = adapter(
        [
            inference.MLMRequest(data.tokenizer.encode("AC"), config),
            inference.MLMRequest(data.tokenizer.encode("D"), config),
        ]
    )

    assert first.hidden_states is not None
    assert second.hidden_states is not None
    assert len(first.hidden_states) == 2
    assert first.hidden_states[-1].tolist() == [[5, 5], [23, 23]]
    assert second.hidden_states[-1].tolist() == [[13, 13]]


@pytest.mark.parametrize("sequence", ["AC", ""])
def test_no_masks_returns_no_logits_rows(
    sequence: str, dplm_checkpoint: conftest.DplmCheckpoint
) -> None:
    """Return a correctly shaped empty tensor when no positions are masked."""
    adapter = dplm.DplmLogitsModel(dplm.DplmModels.DPLM_150M)
    adapter.load(torch.device("cpu"))

    result = adapter(
        [inference.MLMLogitsRequest(data.tokenizer.encode(sequence), True, True)]
    )[0]

    assert result.shape == (0, 20)


def test_shared_adapter_reuses_weights(
    dplm_checkpoint: conftest.DplmCheckpoint,
) -> None:
    """Share one checkpoint across full and logits-only requests."""
    adapter = dplm.DplmInferenceModel(dplm.DplmModels.DPLM_150M)
    logits = shared.SharedLogitsModel(adapter)
    logits.load(torch.device("cpu"))
    adapter.load(torch.device("cpu"))

    result = logits(
        [inference.MLMLogitsRequest(data.tokenizer.encode("_"), True, True)]
    )[0]

    assert result.shape == (1, 20)
    dplm_checkpoint.model_loader.assert_called_once()


def test_shared_adapter_batches_requests(
    dplm_checkpoint: conftest.DplmCheckpoint,
) -> None:
    """Batch logits requests through the common shared adapter."""
    adapter = dplm.DplmInferenceModel(dplm.DplmModels.DPLM_150M)
    logits = shared.SharedLogitsModel(adapter, batch_size=1)
    logits.load(torch.device("cpu"))

    first, second = logits(
        [
            inference.MLMLogitsRequest(data.tokenizer.encode("AC")),
            inference.MLMLogitsRequest(data.tokenizer.encode("D")),
        ]
    )

    assert first.shape == (2, 25)
    assert second.shape == (1, 25)


def test_pickle_drops_loaded_resources(
    dplm_checkpoint: conftest.DplmCheckpoint,
) -> None:
    """Deserialize a weightless adapter that can reload in a worker."""
    adapter = dplm.DplmInferenceModel(dplm.DplmModels.DPLM_150M)
    adapter.load(torch.device("cpu"))
    _ = adapter.translation

    restored = pickle.loads(pickle.dumps(adapter))

    with pytest.raises(RuntimeError, match="forward called before loading"):
        restored.forward([inference.MLMRequest(data.tokenizer.encode("A"))])
    restored.load(torch.device("cpu"))
    assert (
        restored([inference.MLMRequest(data.tokenizer.encode("A"))])[0].logits
        is not None
    )


@pytest.mark.parametrize(
    "adapter_type", [dplm.DplmInferenceModel, dplm.DplmLogitsModel]
)
def test_release_requires_reload(
    adapter_type: type[dplm.DplmInferenceModel] | type[dplm.DplmLogitsModel],
    dplm_checkpoint: conftest.DplmCheckpoint,
) -> None:
    """Reject forward calls after releasing checkpoint resources."""
    adapter = adapter_type(dplm.DplmModels.DPLM_150M)
    adapter.load(torch.device("cpu"))
    adapter.release()

    with pytest.raises(RuntimeError, match="forward called before loading"):
        adapter.forward([])


def test_bf16_uses_autocast_with_fp32_weights(
    dplm_checkpoint: conftest.DplmCheckpoint,
) -> None:
    """Keep weights in fp32 while enabling bf16 computation for forward calls."""
    adapter = dplm.DplmInferenceModel(dplm.DplmModels.DPLM_150M, precision="bf16")
    adapter.load(torch.device("cpu"))

    adapter([inference.MLMRequest(data.tokenizer.encode("AC"))])

    dplm_checkpoint.model_loader.assert_called_once_with(
        dplm.DplmModels.DPLM_150M,
        dtype=torch.float32,
        trust_remote_code=None,
        device_map=torch.device("cpu"),
    )
    assert dplm_checkpoint.model.autocast_dtype == torch.bfloat16
