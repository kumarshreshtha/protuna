import torch

from protuna.optim import predictors


def test_prediction_to_moves_all_tensors_without_changing_the_original() -> None:
    """Move statistics and samples while preserving the original tensors."""
    prediction = predictors.Prediction(
        torch.tensor(2.0), torch.tensor(0.5), torch.tensor([1.0, 3.0])
    )

    moved = prediction.to(torch.device("meta"))

    assert moved is not prediction
    assert moved.mean.device == torch.device("meta")
    assert moved.std is not None
    assert moved.std.device == torch.device("meta")
    assert moved.samples is not None
    assert moved.samples.device == torch.device("meta")
    assert prediction.samples is not None
    torch.testing.assert_close(prediction.mean, torch.tensor(2.0))
    torch.testing.assert_close(prediction.std, torch.tensor(0.5))
    torch.testing.assert_close(prediction.samples, torch.tensor([1.0, 3.0]))


def test_prediction_to_moves_mean_without_uncertainty() -> None:
    """Move the mean even when optional uncertainty tensors are absent."""
    prediction = predictors.Prediction(torch.tensor(2.0))

    moved = prediction.to(torch.device("meta"))

    assert moved.mean.device == torch.device("meta")
    assert moved.std is None
    assert moved.samples is None
    torch.testing.assert_close(prediction.mean, torch.tensor(2.0))


def test_prediction_batch_preserves_mixed_optional_fields() -> None:
    """Keep optional statistics and differently sized samples aligned."""
    predictions = [
        predictors.Prediction(torch.tensor(2.0), samples=torch.tensor([1.0, 3.0])),
        predictors.Prediction(torch.tensor(4.0), torch.tensor(0.5)),
        predictors.Prediction(torch.tensor(6.0), samples=torch.tensor([6.0])),
    ]

    moved = predictors.Prediction.to_device_batch(predictions, torch.device("cpu"))

    torch.testing.assert_close(moved[0].mean, torch.tensor(2.0))
    torch.testing.assert_close(moved[0].samples, torch.tensor([1.0, 3.0]))
    assert moved[0].std is None
    torch.testing.assert_close(moved[1].mean, torch.tensor(4.0))
    torch.testing.assert_close(moved[1].std, torch.tensor(0.5))
    assert moved[1].samples is None
    torch.testing.assert_close(moved[2].samples, torch.tensor([6.0]))
    assert moved[0] is not predictions[0]


def test_prediction_batch_accepts_empty_input() -> None:
    """Allow a batch without predictions."""
    assert predictors.Prediction.to_device_batch([], torch.device("cpu")) == []
