from collections.abc import Callable

import pytest
import torch

from protuna.inference.models import dplm
from protuna.inference.models import dplm2
from protuna.inference.models import esmc
from protuna.inference.models import hf_mlm

from . import conftest


@pytest.mark.parametrize(
    ("factory", "name"),
    [
        (esmc.EsmcInferenceModel, esmc.EsmcModels.ESMC_300M),
        (esmc.EsmcLogitsModel, esmc.EsmcModels.ESMC_300M),
        (dplm.DplmInferenceModel, dplm.DplmModels.DPLM_150M),
        (dplm.DplmLogitsModel, dplm.DplmModels.DPLM_150M),
        (dplm2.Dplm2InferenceModel, dplm2.Dplm2Models.DPLM2_150M),
        (dplm2.Dplm2LogitsModel, dplm2.Dplm2Models.DPLM2_150M),
    ],
)
def test_synthyra_remote_code_defers_to_transformers_by_default(
    factory: Callable[..., hf_mlm.HuggingFaceMLMBase],
    name: str,
    dplm_checkpoint: conftest.DplmCheckpoint,
) -> None:
    """Defer to Transformers for configuration, tokenizer and weight loading."""
    adapter = factory(name)

    adapter.load(torch.device("cpu"))

    assert dplm_checkpoint.config_loader.call_args.kwargs["trust_remote_code"] is None
    assert (
        dplm_checkpoint.tokenizer_loader.call_args.kwargs["trust_remote_code"] is None
    )
    assert dplm_checkpoint.model_loader.call_args.kwargs["trust_remote_code"] is None


@pytest.mark.parametrize("trust_remote_code", [None, False, True])
@pytest.mark.parametrize(
    ("factory", "name"),
    [
        (esmc.EsmcInferenceModel, esmc.EsmcModels.ESMC_300M),
        (esmc.EsmcLogitsModel, esmc.EsmcModels.ESMC_300M),
        (dplm.DplmInferenceModel, dplm.DplmModels.DPLM_150M),
        (dplm.DplmLogitsModel, dplm.DplmModels.DPLM_150M),
        (dplm2.Dplm2InferenceModel, dplm2.Dplm2Models.DPLM2_150M),
        (dplm2.Dplm2LogitsModel, dplm2.Dplm2Models.DPLM2_150M),
    ],
)
def test_synthyra_remote_code_setting_reaches_all_loaders(
    factory: Callable[..., hf_mlm.HuggingFaceMLMBase],
    name: str,
    trust_remote_code: bool | None,
    dplm_checkpoint: conftest.DplmCheckpoint,
) -> None:
    """Forward the explicit setting, including when using replacement weights."""
    adapter = factory(
        name, pretrained_path="local-weights", trust_remote_code=trust_remote_code
    )

    adapter.load(torch.device("cpu"))

    assert (
        dplm_checkpoint.config_loader.call_args.kwargs["trust_remote_code"]
        is trust_remote_code
    )
    assert (
        dplm_checkpoint.tokenizer_loader.call_args.kwargs["trust_remote_code"]
        is trust_remote_code
    )
    assert (
        dplm_checkpoint.model_loader.call_args.kwargs["trust_remote_code"]
        is trust_remote_code
    )
