import pytest
import torch

from protuna import data
from protuna.optim import predictors
from tests.unit.optim import _helpers


@pytest.mark.parametrize("batch_size", [None, 1, 3])
def test_calibration_interpolates_and_clips_to_measured_endpoints(
    batch_size: int | None,
) -> None:
    """Map raw scores onto assay values and clamp scores outside the anchors."""
    model = _helpers._LookupPredictor(
        {"A": 0.0, "C": 2.0, "D": 1.0, "E": -1.0, "F": 3.0}
    )
    calibrated = predictors.CalibratedPredictor(
        model,
        {
            "token_ids": torch.stack(data.tokenizer.encode(["A", "C"])),
            "measurement": torch.tensor([10.0, 30.0]),
        },
        batch_size=batch_size,
    )

    predictions = calibrated(
        [
            predictors.PredictionRequest(data.tokenizer.encode("D")),
            predictors.PredictionRequest(data.tokenizer.encode("E")),
            predictors.PredictionRequest(data.tokenizer.encode("F")),
            predictors.PredictionRequest(data.tokenizer.encode("A")),
            predictors.PredictionRequest(data.tokenizer.encode("C")),
        ]
    )

    assert predictions == [
        predictors.Prediction(torch.tensor(20.0)),
        predictors.Prediction(torch.tensor(10.0)),
        predictors.Prediction(torch.tensor(30.0)),
        predictors.Prediction(torch.tensor(10.0)),
        predictors.Prediction(torch.tensor(30.0)),
    ]


def test_calibration_scores_history_once_across_repeated_calls() -> None:
    """Reuse measured-sequence scores when scoring subsequent candidates."""
    model = _helpers._LookupPredictor({"A": 0.0, "C": 2.0, "D": 1.0})
    calibrated = predictors.CalibratedPredictor(
        model,
        {
            "token_ids": torch.stack(data.tokenizer.encode(["A", "C"])),
            "measurement": torch.tensor([10.0, 30.0]),
        },
    )
    request = predictors.PredictionRequest(data.tokenizer.encode("D"))

    first = calibrated([request])
    second = calibrated([request])

    assert first == second == [predictors.Prediction(torch.tensor(20.0))]
    assert len(model.batches) == 3
    assert len(model.batches[0]) == 1
    assert len(model.batches[1]) == 2
    assert len(model.batches[2]) == 1
    torch.testing.assert_close(
        model.batches[1][0].token_ids, data.tokenizer.encode("A")
    )
    torch.testing.assert_close(
        model.batches[1][1].token_ids, data.tokenizer.encode("C")
    )


def test_calibration_with_one_anchor_returns_the_measurement() -> None:
    """Map all raw scores to the lone assay measurement."""
    model = _helpers._LookupPredictor({"A": 0.0, "C": 2.0})
    calibrated = predictors.CalibratedPredictor(
        model,
        {
            "token_ids": data.tokenizer.encode("A")[None, :],
            "measurement": torch.tensor([7.0]),
        },
    )

    assert calibrated([predictors.PredictionRequest(data.tokenizer.encode("C"))]) == [
        predictors.Prediction(torch.tensor(7.0))
    ]


def test_calibration_delegates_loading() -> None:
    """Pass device selection through to the wrapped predictor."""
    model = _helpers._LookupPredictor({})
    calibrated = predictors.CalibratedPredictor(
        model,
        {
            "token_ids": data.tokenizer.encode("A")[None],
            "measurement": torch.tensor([7.0]),
        },
    )

    calibrated.load([torch.device("cpu"), torch.device("cuda:0")])

    assert model.loaded_devices == [[torch.device("cpu"), torch.device("cuda:0")]]


def test_calibration_delegates_release() -> None:
    """Release the wrapped predictor's resources."""
    model = _helpers._LookupPredictor({})
    calibrated = predictors.CalibratedPredictor(
        model,
        {
            "token_ids": data.tokenizer.encode("A")[None],
            "measurement": torch.tensor([7.0]),
        },
    )

    calibrated.release()

    assert model.released


@pytest.mark.parametrize(
    ("sequences", "measurements", "expected"),
    [
        (["C", "A"], [30.0, 10.0], [10.0, 20.0, 30.0]),
        (["A", "A", "C"], [5.0, 10.0, 30.0], [10.0, 20.0, 30.0]),
        (["A", "C", "C"], [10.0, 25.0, 30.0], [10.0, 17.5, 30.0]),
        (["A", "A"], [5.0, 10.0], [10.0, 10.0, 10.0]),
    ],
)
def test_calibration_sorts_anchors_and_handles_duplicate_scores(
    sequences: list[str], measurements: list[float], expected: list[float]
) -> None:
    """Keep measurements paired with sorted scores and use the last exact match."""
    model = _helpers._LookupPredictor({"A": 0.0, "C": 2.0, "D": 1.0})
    calibrated = predictors.CalibratedPredictor(
        model,
        {
            "token_ids": torch.stack(data.tokenizer.encode(sequences)),
            "measurement": torch.tensor(measurements),
        },
    )

    predictions = calibrated(
        [
            predictors.PredictionRequest(data.tokenizer.encode("A")),
            predictors.PredictionRequest(data.tokenizer.encode("D")),
            predictors.PredictionRequest(data.tokenizer.encode("C")),
        ]
    )

    torch.testing.assert_close(
        torch.stack([prediction.mean for prediction in predictions]),
        torch.tensor(expected),
    )


@pytest.mark.parametrize("measurements", [[], [10.0, 30.0]])
def test_calibration_rejects_missing_or_mismatched_measurements(
    measurements: list[float],
) -> None:
    """Reject unequal sequence and measurement counts at construction."""
    model = _helpers._LookupPredictor({"A": 0.0})
    with pytest.raises(ValueError, match="equal lengths"):
        predictors.CalibratedPredictor(
            model,
            {
                "token_ids": data.tokenizer.encode("A")[None],
                "measurement": torch.tensor(measurements),
            },
        )


def test_calibration_rejects_empty_history_at_construction() -> None:
    """Require a measured anchor when constructing the predictor."""
    with pytest.raises(ValueError, match="at least one measured sequence"):
        predictors.CalibratedPredictor(
            _helpers._LookupPredictor({}),
            {
                "token_ids": torch.empty((0, 1), dtype=torch.long),
                "measurement": torch.empty(0),
            },
        )


def test_calibration_forward_accepts_empty_requests() -> None:
    """Return no predictions without trying to stack an empty batch."""
    calibrated = predictors.CalibratedPredictor(
        _helpers._LookupPredictor({}),
        {
            "token_ids": data.tokenizer.encode("A")[None],
            "measurement": torch.tensor([7.0]),
        },
    )

    assert calibrated.forward([]) == []


def test_calibration_preserves_tensor_precision_and_gradients() -> None:
    """Interpolate tensor scores without a NumPy conversion or detachment."""
    score = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    calibrated = predictors.CalibratedPredictor(
        _helpers._LookupPredictor(
            {
                "A": torch.tensor(0.0, dtype=torch.float64),
                "C": torch.tensor(2.0, dtype=torch.float64),
                "D": score,
            }
        ),
        {
            "token_ids": torch.stack(data.tokenizer.encode(["A", "C"])),
            "measurement": torch.tensor([10.0, 30.0], dtype=torch.float64),
        },
    )

    prediction = calibrated([predictors.PredictionRequest(data.tokenizer.encode("D"))])[
        0
    ]
    prediction.mean.backward()

    torch.testing.assert_close(prediction.mean, torch.tensor(20.0, dtype=torch.float64))
    torch.testing.assert_close(score.grad, torch.tensor(10.0, dtype=torch.float64))
