from collections.abc import Sequence

from jaxtyping import Float
import pytest
import torch

from protuna import data
from protuna import inference
from protuna.optim import predictors
from protuna.optim.predictors import zeroshot
from protuna.typing import SequenceLike


class _FixedLogitsMLM(
    inference.InferenceModel[inference.MLMLogitsRequest, Float[torch.Tensor, "l v"]]
):
    def __init__(self):
        super().__init__()
        self.requests: list[inference.MLMLogitsRequest] = []
        self.loaded_devices: list[torch.device | Sequence[torch.device] | None] = []
        self.released = False
        self.device = torch.device("cpu")
        self.mutant_logit = 2.0

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Record the requested device and place subsequent logits on it."""
        self.loaded_devices.append(device)
        if device is not None:
            self.device = device if isinstance(device, torch.device) else device[0]

    def release(self) -> None:
        self.released = True

    def forward(
        self, requests: SequenceLike[inference.MLMLogitsRequest]
    ) -> list[Float[torch.Tensor, "l v"]]:
        """Assign zero A logits and configurable C logits at each position."""
        self.requests.extend(requests)
        outputs = []
        for request in requests:
            logits = torch.full(
                (len(request.token_ids), len(data.tokenizer.all_token_ids)),
                -torch.inf,
            )
            logits[:, data.tokenizer.encode("A")] = 0.0
            logits[:, data.tokenizer.encode("C")] = self.mutant_logit
            if request.masked_positions_only:
                logits = logits[request.token_ids.cpu() == data.tokenizer.mask_token_id]
            if request.aa_tokens_only:
                logits = logits[:, :20]
            outputs.append(logits.to(self.device))
        return outputs


@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_masked_marginal_scores_mutations_in_candidate_order(
    batch_size: int | None,
) -> None:
    """Sum log-likelihood ratios only at mutated sites, including duplicates."""
    model = _FixedLogitsMLM()
    predictor = predictors.MaskedMarginalPredictor(
        data.tokenizer.encode("AA"), model, batch_size=batch_size
    )

    predictions = predictor(
        [
            predictors.PredictionRequest(data.tokenizer.encode("CC")),
            predictors.PredictionRequest(data.tokenizer.encode("AA")),
            predictors.PredictionRequest(data.tokenizer.encode("AC")),
            predictors.PredictionRequest(data.tokenizer.encode("AC")),
        ]
    )

    assert predictions[0].mean == pytest.approx(4.0)
    assert predictions[1].mean == pytest.approx(0.0)
    assert predictions[2].mean == pytest.approx(2.0)
    assert predictions[3].mean == pytest.approx(2.0)
    torch.testing.assert_close(predictions[0].mean, torch.tensor(4.0))
    assert predictions[0].std is None
    assert predictions[0].samples is None


@pytest.mark.parametrize(
    ("wt_temp", "mt_temp", "expected"),
    [(1.0, 1.0, 2.0), (2.0, 2.0, 1.0), (1.0, 2.0, 1.813666)],
)
def test_masked_marginal_applies_both_temperatures(
    wt_temp: float, mt_temp: float, expected: float
) -> None:
    """Scale wild-type and mutant probabilities by their own temperatures."""
    predictor = predictors.MaskedMarginalPredictor(
        data.tokenizer.encode("AA"),
        _FixedLogitsMLM(),
        wt_temp=wt_temp,
        mt_temp=mt_temp,
    )

    prediction = predictor([predictors.PredictionRequest(data.tokenizer.encode("AC"))])[
        0
    ]

    assert float(prediction.mean) == pytest.approx(expected)


def test_masked_marginal_only_masks_mutated_positions_without_changing_inputs() -> None:
    """Request full logits for the masked context and preserve input sequences."""
    model = _FixedLogitsMLM()
    wild_type = data.tokenizer.encode("AA")
    mutant = data.tokenizer.encode("AC")
    predictor = predictors.MaskedMarginalPredictor(wild_type, model)

    predictor([predictors.PredictionRequest(mutant)])

    assert len(model.requests) == 1
    torch.testing.assert_close(model.requests[0].token_ids, data.tokenizer.encode("A_"))
    assert model.requests[0].masked_positions_only is False
    assert model.requests[0].aa_tokens_only is False
    torch.testing.assert_close(wild_type, data.tokenizer.encode("AA"))
    torch.testing.assert_close(mutant, data.tokenizer.encode("AC"))


def test_masked_marginal_delegates_loading() -> None:
    """Pass device selection through to the language model."""
    model = _FixedLogitsMLM()
    predictor = predictors.MaskedMarginalPredictor(data.tokenizer.encode("AA"), model)

    predictor.load(torch.device("cpu"))

    assert model.loaded_devices == [torch.device("cpu")]


def test_masked_marginal_delegates_release() -> None:
    """Release the language model's resources."""
    model = _FixedLogitsMLM()
    predictor = predictors.MaskedMarginalPredictor(data.tokenizer.encode("AA"), model)

    predictor.release()

    assert model.released


def test_wild_type_marginal_reuses_logits_between_batches() -> None:
    """Score successive candidate batches with one wild-type model request."""
    model = _FixedLogitsMLM()
    predictor = zeroshot.WildTypeMarginalPredictor(data.tokenizer.encode("AA"), model)

    predictor([predictors.PredictionRequest(data.tokenizer.encode("AC"))])
    prediction = predictor([predictors.PredictionRequest(data.tokenizer.encode("CC"))])

    assert prediction[0].mean == pytest.approx(4.0)
    assert len(model.requests) == 1
    assert model.requests[0].aa_tokens_only is True


def test_wild_type_marginal_loading_refreshes_cached_logits() -> None:
    """Recompute wild-type logits after the model is reloaded."""
    model = _FixedLogitsMLM()
    predictor = zeroshot.WildTypeMarginalPredictor(data.tokenizer.encode("AA"), model)
    requests = [predictors.PredictionRequest(data.tokenizer.encode("AC"))]
    predictor(requests)
    model.mutant_logit = 4.0

    predictor.load(torch.device("cpu"))
    prediction = predictor(requests)

    assert prediction[0].mean == pytest.approx(4.0)
    assert len(model.requests) == 2
    assert model.loaded_devices == [torch.device("cpu")]


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda:0"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
        pytest.param(
            torch.device("mps"),
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_wild_type_marginal_release_clears_resources(device: torch.device) -> None:
    """Drop cached logits and return the wild type to CPU when releasing."""
    model = _FixedLogitsMLM()
    predictor = zeroshot.WildTypeMarginalPredictor(data.tokenizer.encode("AA"), model)
    predictor.load(device)
    predictor([predictors.PredictionRequest(data.tokenizer.encode("AC"))])

    predictor.release()

    assert predictor._wt_logits is None
    assert predictor.wt_tokens.device == torch.device("cpu")
    assert model.released
