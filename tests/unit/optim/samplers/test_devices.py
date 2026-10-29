from collections.abc import Iterator
import itertools

from jaxtyping import Float
import pytest
import torch

from protuna import inference
from protuna.optim.samplers import base
from protuna.optim.samplers import beam
from protuna.optim.samplers import common
from protuna.optim.samplers import maskgit
from protuna.optim.samplers import mlm
from protuna.optim.samplers import pssm
from protuna.optim.samplers import remdm
from protuna.optim.samplers import rlxf
from protuna.typing import SequenceLike
from tests.unit.optim.samplers import _helpers


class _MixedDeviceMLM(_helpers._FakeMLM):
    def __init__(self, device: torch.device, alternate_device: torch.device):
        super().__init__({"W": 0.5, "Y": 0.5})
        self.devices: Iterator[torch.device] = itertools.cycle(
            (device, alternate_device)
        )

    def __call__(
        self, requests: SequenceLike[inference.MLMLogitsRequest]
    ) -> list[Float[torch.Tensor, "m v"]]:
        return [logits.to(next(self.devices)) for logits in super().__call__(requests)]


class _MixedDeviceGuidance(
    inference.ModelFreeLocalHandler[tuple[torch.Tensor, float], torch.Tensor]
):
    def __init__(self, device: torch.device):
        super().__init__()
        self.devices: Iterator[torch.device] = itertools.cycle(
            (device, torch.device("cpu"))
        )

    def __call__(
        self, requests: SequenceLike[tuple[torch.Tensor, float]]
    ) -> list[torch.Tensor]:
        return [torch.tensor(0.0, device=next(self.devices)) for _ in requests]


@pytest.fixture(
    params=[
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ]
)
def accelerator(request: pytest.FixtureRequest) -> torch.device:
    """Return an available accelerator with its concrete device index."""
    return torch.empty(0, device=request.param).device


@pytest.fixture(params=[("cpu", "accelerator"), ("accelerator", "cpu")])
def device_pair(
    request: pytest.FixtureRequest, accelerator: torch.device
) -> tuple[torch.device, torch.device]:
    """Pair opposite input and output devices."""
    devices = {"cpu": torch.device("cpu"), "accelerator": accelerator}
    input_device, output_device = request.param
    return devices[input_device], devices[output_device]


@pytest.fixture(
    params=[
        "mlm",
        "maskgit",
        "pssm",
        "beam",
        "beam_random",
        "rlxf",
        "remdm",
        "remdm_confidence",
        "remdm_guidance",
    ]
)
def device_sampler(
    request: pytest.FixtureRequest, device_pair: tuple[torch.device, torch.device]
) -> base.Sampler:
    """Build a sampler whose model alternates response devices."""
    input_device, output_device = device_pair
    if request.param.startswith("remdm") and output_device.type == "mps":
        pytest.skip("ReMDM transition kernel requires float64, unsupported on MPS")
    model = _MixedDeviceMLM(output_device, input_device)
    logits = torch.full((3, 20), -50.0, device=output_device)
    logits[:, _helpers._TOKENIZER.token_to_id("W")] = 0
    factories = {
        "mlm": lambda: mlm.MLMSampler(model, max_samples=1, seed=7),
        "maskgit": lambda: maskgit.MaskGITSampler(model, max_samples=1, seed=7),
        "pssm": lambda: pssm.PSSMSampler(logits, max_samples=1, seed=7),
        "beam": lambda: beam.MLMBeamSampler(model, max_samples=1, seed=7),
        "beam_random": lambda: beam.MLMBeamSampler(
            model, max_samples=1, sampling_order="random", seed=7
        ),
        "rlxf": lambda: rlxf.RLXFSampler(model, max_samples=1, seed=7),
        "remdm": lambda: _remdm_sampler(model),
        "remdm_confidence": lambda: _remdm_sampler(model, confidence=True),
        "remdm_guidance": lambda: _remdm_sampler(
            model, confidence=True, guidance=_MixedDeviceGuidance(input_device)
        ),
    }
    return factories[request.param]()


def _remdm_sampler(
    model: _MixedDeviceMLM,
    confidence: bool = False,
    guidance: _MixedDeviceGuidance | None = None,
) -> remdm.ReMDMSampler:
    return remdm.ReMDMSampler(
        model,
        scheduler=remdm.ReMDMSwitchScheduler(
            remdm.MaxCappedRemaskScheduler(0.1), t_on=1.0
        ),
        n_steps=4,
        max_samples=1,
        confidence_remasking=confidence,
        guidance_model=guidance,
        seed=7,
    )


def test_remdm_denoising_aligns_masks_with_mixed_device_responses(
    device_pair: tuple[torch.device, torch.device],
) -> None:
    """Apply allowed-residue masks on the denoising response device."""
    input_device, output_device = device_pair
    model = _MixedDeviceMLM(output_device, input_device)
    space = _helpers._space("ACD", {0: "AW", 1: "CW"}).to(input_device)
    tokens = torch.tensor([_helpers._MASK, _helpers._MASK, 2], device=input_device)

    logits = _remdm_sampler(model).denoising_logits([tokens, tokens], space)

    assert logits[0].device == output_device
    assert logits[1].device == output_device
    assert _helpers._decode(logits[0].argmax(-1).cpu()[None, :]) == ["WW"]


def test_samplers_keep_proposals_on_the_first_response_device(
    device_sampler: base.Sampler, device_pair: tuple[torch.device, torch.device]
) -> None:
    """Keep proposals and sampling state aligned despite changing response devices."""
    input_device, output_device = device_pair
    first = _helpers._space("ACD", {0: "AW", 1: "CW"}).to(input_device)
    second = _helpers._space("ACE", {0: "AW", 1: "CW"}).to(input_device)

    proposals = device_sampler.sample([first, second])

    assert proposals[0].device == output_device
    assert proposals[1].device == output_device
    assert _helpers._decode(proposals[0].cpu()) == ["WWD"]
    assert _helpers._decode(proposals[1].cpu()) == ["WWE"]
    assert first.init_sequence.device == input_device


def test_mlm_batch_combines_responses_from_different_devices(
    accelerator: torch.device,
) -> None:
    """Normalize mixed responses to the first response's device."""
    model = _MixedDeviceMLM(accelerator, torch.device("cpu"))
    batch = torch.tensor([[_helpers._MASK], [_helpers._MASK]])

    responses = common.execute_mlm_batch(batch, model)

    assert responses[0].device == accelerator
    assert responses[1].device == accelerator


def test_mlm_batch_moves_responses_to_the_requested_device(
    accelerator: torch.device,
) -> None:
    """Use an explicit device even when the first response is elsewhere."""
    model = _MixedDeviceMLM(accelerator, torch.device("cpu"))
    batch = torch.tensor([[_helpers._MASK], [_helpers._MASK]])

    responses = common.execute_mlm_batch(batch, model, device=torch.device("cpu"))

    assert responses[0].device == torch.device("cpu")
    assert responses[1].device == torch.device("cpu")


def test_mlm_batch_accepts_an_empty_response_batch() -> None:
    """Return no logits when the model receives no sequences."""
    model = _helpers._FakeMLM({"W": 1.0})

    responses = common.execute_mlm_batch([], model)

    assert responses == []
