from unittest import mock

import pytest
import torch

from protuna import data
from protuna.optim import predictors
from tests.unit.optim import _helpers


@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_ensemble_preserves_candidate_and_member_order(batch_size: int | None) -> None:
    """Average each candidate across members and retain ordered samples."""
    first = _helpers._LookupPredictor({"A": 1.0, "C": 10.0})
    second = _helpers._LookupPredictor({"A": 3.0, "C": 14.0})
    ensemble = predictors.EnsemblePredictor([first, second], batch_size=batch_size)

    predictions = ensemble(
        [
            predictors.PredictionRequest(data.tokenizer.encode("C"), True, True),
            predictors.PredictionRequest(data.tokenizer.encode("A"), True, True),
        ]
    )

    assert predictions[0].mean == 12.0
    assert predictions[0].std == pytest.approx(2.828427)
    torch.testing.assert_close(predictions[0].samples, torch.tensor([10.0, 14.0]))
    assert predictions[1].mean == 2.0
    assert predictions[1].std == pytest.approx(1.414214)
    torch.testing.assert_close(predictions[1].samples, torch.tensor([1.0, 3.0]))


@pytest.mark.parametrize(
    ("return_std", "return_samples"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_ensemble_only_reports_requested_uncertainty(
    return_std: bool, return_samples: bool
) -> None:
    """Use member means without asking members for their own uncertainty."""
    first = _helpers._LookupPredictor({"A": 1.0})
    second = _helpers._LookupPredictor({"A": 3.0})
    ensemble = predictors.EnsemblePredictor([first, second])
    request = predictors.PredictionRequest(
        data.tokenizer.encode("A"), return_std, return_samples
    )

    prediction = ensemble([request])[0]

    assert (prediction.std is not None) == return_std
    assert (prediction.samples is not None) == return_samples
    assert first.batches[0][0].return_std is False
    assert first.batches[0][0].return_samples is False
    assert second.batches[0][0].return_std is False
    assert second.batches[0][0].return_samples is False
    assert request.return_std == return_std
    assert request.return_samples == return_samples


def test_single_member_ensemble_returns_its_mean_and_sample() -> None:
    """Keep a single member's score when standard deviation is not requested."""
    ensemble = predictors.EnsemblePredictor([_helpers._LookupPredictor({"A": 3.0})])

    prediction = ensemble(
        [predictors.PredictionRequest(data.tokenizer.encode("A"), return_samples=True)]
    )[0]

    assert prediction.mean == 3.0
    assert prediction.std is None
    torch.testing.assert_close(prediction.samples, torch.tensor([3.0]))


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
def test_ensemble_loads_all_members_on_a_single_device(device: torch.device) -> None:
    """Pass a single device through to every ensemble member."""
    first = _helpers._LookupPredictor({})
    second = _helpers._LookupPredictor({})
    ensemble = predictors.EnsemblePredictor([first, second])

    ensemble.load(device)

    assert first.loaded_devices == [device]
    assert second.loaded_devices == [device]


def test_ensemble_cycles_devices_when_there_are_more_members() -> None:
    """Assign members to devices in round-robin order."""
    first = _helpers._LookupPredictor({})
    second = _helpers._LookupPredictor({})
    third = _helpers._LookupPredictor({})
    ensemble = predictors.EnsemblePredictor([first, second, third])

    ensemble.load([torch.device("cpu"), torch.device("cuda:0")])

    assert first.loaded_devices == [torch.device("cpu")]
    assert second.loaded_devices == [torch.device("cuda:0")]
    assert third.loaded_devices == [torch.device("cpu")]


def test_ensemble_defaults_to_cpu_when_cuda_is_unavailable() -> None:
    """Resolve the default device and load each member onto CPU."""
    first = _helpers._LookupPredictor({})
    second = _helpers._LookupPredictor({})
    ensemble = predictors.EnsemblePredictor([first, second])

    with mock.patch.object(
        torch.cuda, "is_available", autospec=True, return_value=False
    ):
        ensemble.load()

    assert first.loaded_devices == [torch.device("cpu")]
    assert second.loaded_devices == [torch.device("cpu")]


def test_ensemble_releases_all_members() -> None:
    """Release every member's resources."""
    first = _helpers._LookupPredictor({})
    second = _helpers._LookupPredictor({})
    ensemble = predictors.EnsemblePredictor([first, second])

    ensemble.release()

    assert first.released
    assert second.released


def test_ensemble_preserves_tensor_dtype_and_gradients() -> None:
    """Combine tensor scores without copying them into default CPU scalars."""
    score = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    ensemble = predictors.EnsemblePredictor(
        [
            _helpers._LookupPredictor({"A": score}),
            _helpers._LookupPredictor({"A": torch.tensor(3.0, dtype=torch.float64)}),
        ]
    )

    prediction = ensemble(
        [predictors.PredictionRequest(data.tokenizer.encode("A"), True, True)]
    )[0]
    prediction.mean.backward()

    torch.testing.assert_close(prediction.mean, torch.tensor(2.0, dtype=torch.float64))
    torch.testing.assert_close(
        prediction.std, torch.tensor(1.4142135623730951, dtype=torch.float64)
    )
    torch.testing.assert_close(
        prediction.samples, torch.tensor([1.0, 3.0], dtype=torch.float64)
    )
    torch.testing.assert_close(score.grad, torch.tensor(0.5, dtype=torch.float64))


def test_single_member_ensemble_has_zero_standard_deviation() -> None:
    """Report zero disagreement for a lone member instead of NaN."""
    ensemble = predictors.EnsemblePredictor([_helpers._LookupPredictor({"A": 3.0})])

    prediction = ensemble(
        [predictors.PredictionRequest(data.tokenizer.encode("A"), return_std=True)]
    )[0]

    torch.testing.assert_close(prediction.std, torch.tensor(0.0))


def test_empty_ensemble_rejected_at_initialization() -> None:
    """Require a member when initializing an ensemble."""
    with pytest.raises(ValueError, match="at least one member"):
        predictors.EnsemblePredictor([])
