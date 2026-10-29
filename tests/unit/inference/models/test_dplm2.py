import pathlib
import pickle

import pytest
import torch
import transformers

from protuna import data
from protuna import inference
from protuna.inference.models import dplm2
from protuna.inference.models import shared

from . import conftest


@pytest.fixture
def checkpoint(
    dplm_checkpoint: conftest.DplmCheckpoint, tmp_path: pathlib.Path
) -> conftest.DplmCheckpoint:
    """Use DPLM2's explicit modality boundaries with the existing model double."""
    vocab = tmp_path / "dplm2_vocab.txt"
    vocab.write_text(
        "<cls_aa>\n<pad>\n<eos_aa>\n<unk_aa>\nL\nA\nG\nV\nS\nE\nR\nT\nI\nD\n"
        "P\nK\nQ\nN\nF\nY\nM\nH\nW\nC\nX\nB\nU\nZ\nO\n.\n-\n<null_1>\n"
        "<mask_aa>\n"
    )
    dplm_checkpoint.tokenizer_loader.return_value = transformers.EsmTokenizer(
        vocab_file=str(vocab)
    )
    return dplm_checkpoint


def test_sequence_only_ragged_outputs(checkpoint: conftest.DplmCheckpoint) -> None:
    """Return amino-acid outputs without requiring structure dependencies."""
    adapter = dplm2.Dplm2InferenceModel(dplm2.Dplm2Models.DPLM2_150M)
    adapter.load(torch.device("cpu"))
    cfg = inference.MLMOutputConfig(
        return_embeddings=True, return_mean_embeddings=True, return_hidden_states=True
    )

    first, second = adapter(
        [
            inference.MLMRequest(data.tokenizer.encode("AC"), cfg),
            inference.MLMRequest(data.tokenizer.encode("_"), cfg),
        ]
    )

    assert first.embeddings is not None
    assert first.mean_embeddings is not None
    assert second.hidden_states is not None
    assert first.embeddings.tolist() == [[5, 5], [23, 23]]
    assert first.mean_embeddings.tolist() == [14, 14]
    assert second.hidden_states[-1].tolist() == [[32, 32]]
    assert checkpoint.model.hidden_states_requested


def test_translate_and_pad_inputs(checkpoint: conftest.DplmCheckpoint) -> None:
    """Build only an amino-acid track with explicit modality boundaries."""
    adapter = dplm2.Dplm2InferenceModel(dplm2.Dplm2Models.DPLM2_150M)

    batch = adapter.prepare_inputs(
        [data.tokenizer.encode("A_?-"), data.tokenizer.encode("C")]
    )

    assert batch["input_ids"].tolist() == [[0, 5, 32, 3, 30, 2], [0, 23, 2, 1, 1, 1]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]


def test_mixed_requests_filter_logits(checkpoint: conftest.DplmCheckpoint) -> None:
    """Respect per-request position and vocabulary filters in sequence inference."""
    adapter = dplm2.Dplm2InferenceModel(dplm2.Dplm2Models.DPLM2_150M)
    adapter.load(torch.device("cpu"))

    masked, full = adapter(
        [
            inference.MLMRequest(
                data.tokenizer.encode("A_"),
                inference.MLMOutputConfig(
                    masked_positions_only=True, aa_tokens_only=True
                ),
            ),
            inference.MLMRequest(data.tokenizer.encode("C")),
        ]
    )

    assert masked.logits is not None
    assert full.logits is not None
    assert masked.logits.tolist() == [
        [5, 23, 13, 9, 18, 6, 21, 12, 15, 4, 20, 17, 14, 16, 10, 8, 11, 7, 22, 19]
    ]
    assert full.logits.shape == (1, 25)
    assert full.logits[0, 20:].tolist() == [30, 32, 0, 1, 3]


def test_logits_only_filters_sequence_outputs(
    checkpoint: conftest.DplmCheckpoint,
) -> None:
    """Apply position and vocabulary filters through the direct logits adapter."""
    adapter = dplm2.Dplm2LogitsModel(dplm2.Dplm2Models.DPLM2_150M)
    adapter.load(torch.device("cpu"))

    masked, full = adapter(
        [
            inference.MLMLogitsRequest(data.tokenizer.encode("A_"), True, True),
            inference.MLMLogitsRequest(data.tokenizer.encode("C")),
        ]
    )

    assert masked.tolist() == [
        [5, 23, 13, 9, 18, 6, 21, 12, 15, 4, 20, 17, 14, 16, 10, 8, 11, 7, 22, 19]
    ]
    assert full.shape == (1, 25)
    assert full[0, 20:].tolist() == [30, 32, 0, 1, 3]


@pytest.mark.parametrize("sequence", ["AC", ""])
def test_no_masks_returns_no_logits_rows(
    checkpoint: conftest.DplmCheckpoint, sequence: str
) -> None:
    """Return an empty tensor when no positions are masked."""
    adapter = dplm2.Dplm2LogitsModel(dplm2.Dplm2Models.DPLM2_150M)
    adapter.load(torch.device("cpu"))

    output = adapter(
        [inference.MLMLogitsRequest(data.tokenizer.encode(sequence), True, True)]
    )[0]

    assert output.shape == (0, 20)


def test_pickle_drops_loaded_resources(checkpoint: conftest.DplmCheckpoint) -> None:
    """Deserialize an adapter that can reload its sequence model in a worker."""
    adapter = dplm2.Dplm2LogitsModel(dplm2.Dplm2Models.DPLM2_150M)
    adapter.load(torch.device("cpu"))
    request = inference.MLMLogitsRequest(data.tokenizer.encode("A_"), True, True)
    expected = adapter([request])[0]

    restored = pickle.loads(pickle.dumps(adapter))

    with pytest.raises(RuntimeError, match="forward called before loading"):
        restored.forward([request])
    restored.load(torch.device("cpu"))
    torch.testing.assert_close(restored([request])[0], expected)


def test_release_requires_reload(checkpoint: conftest.DplmCheckpoint) -> None:
    """Reload sequence inference after releasing model resources."""
    adapter = dplm2.Dplm2LogitsModel(dplm2.Dplm2Models.DPLM2_150M)
    adapter.load(torch.device("cpu"))
    adapter.release()

    with pytest.raises(RuntimeError, match="forward called before loading"):
        adapter.forward([])
    adapter.load(torch.device("cpu"))
    output = adapter(
        [inference.MLMLogitsRequest(data.tokenizer.encode("A_"), True, True)]
    )[0]
    assert output.shape == (1, 20)


def test_shared_adapter_batches_sequence_requests(
    checkpoint: conftest.DplmCheckpoint,
) -> None:
    """Reuse one sequence model across batches from the shared logits adapter."""
    adapter = dplm2.Dplm2InferenceModel(dplm2.Dplm2Models.DPLM2_150M)
    logits = shared.SharedLogitsModel(adapter, batch_size=1)
    logits.load(torch.device("cpu"))

    first, second = logits(
        [
            inference.MLMLogitsRequest(data.tokenizer.encode("A_"), True, True),
            inference.MLMLogitsRequest(data.tokenizer.encode("AC"), True, True),
        ]
    )

    assert first.shape == (1, 20)
    assert second.shape == (0, 20)
    checkpoint.model_loader.assert_called_once()
