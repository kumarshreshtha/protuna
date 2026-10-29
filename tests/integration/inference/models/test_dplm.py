"""Exercise Synthyra's real DPLM-150M checkpoint on CUDA."""

import pytest
import torch

from protuna import data
from protuna import inference
from protuna.inference.models import dplm


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="DPLM requires CUDA")
def test_real_checkpoint_outputs_and_padding() -> None:
    """Preserve residue outputs when batching sequences of different lengths."""
    adapter = dplm.DplmInferenceModel(dplm.DplmModels.DPLM_150M, trust_remote_code=True)
    adapter.load(torch.device("cuda"))
    config = inference.MLMOutputConfig(
        aa_tokens_only=True,
        return_embeddings=True,
        return_mean_embeddings=True,
        return_hidden_states=True,
    )
    short_request = inference.MLMRequest(data.tokenizer.encode("AC_"), config)
    long_request = inference.MLMRequest(data.tokenizer.encode("ACDEFGHIKLMN"))

    try:
        short, long = adapter([short_request, long_request])
        alone = adapter([short_request])[0]

        assert short.logits is not None
        assert long.logits is not None
        assert alone.logits is not None
        assert short.embeddings is not None
        assert short.mean_embeddings is not None
        assert short.hidden_states is not None
        assert short.logits.shape == (3, 20)
        assert long.logits.shape == (12, 25)
        assert short.embeddings.shape == (3, 640)
        assert short.mean_embeddings.shape == (640,)
        assert len(short.hidden_states) == 31
        assert short.hidden_states[-1].shape == (3, 640)
        torch.testing.assert_close(short.logits, alone.logits, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(short.mean_embeddings, short.embeddings.mean(dim=0))
    finally:
        adapter.release()
