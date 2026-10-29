import pytest
import torch

from protuna import data
from protuna.optim import constraints
from protuna.optim import predictors
from tests.unit.optim import _helpers


@pytest.mark.parametrize(
    ("ge", "le", "expected"),
    [
        (1.0, None, [False, True, True, True, True]),
        (None, 3.0, [True, True, True, True, False]),
        (1.0, 3.0, [False, True, True, True, False]),
        (2.0, 2.0, [False, False, True, False, False]),
        (0.0, 0.0, [True, False, False, False, False]),
    ],
)
@pytest.mark.parametrize("batch_size", [None, 2])
def test_prediction_range_accepts_inclusive_bounds(
    ge: float | None, le: float | None, expected: list[bool], batch_size: int | None
) -> None:
    """Accept exactly the predictions within every configured bound."""
    model = _helpers._LookupPredictor(
        {"A": 0.0, "C": 1.0, "D": 2.0, "E": 3.0, "F": 4.0}
    )
    constraint = constraints.HardPredictionRangeConstraint(
        model, ge=ge, le=le, batch_size=batch_size
    )

    verdicts = constraint(data.tokenizer.encode(["A", "C", "D", "E", "F"]))

    torch.testing.assert_close(torch.stack(list(verdicts)), torch.tensor(expected))


def test_prediction_range_requires_at_least_one_bound() -> None:
    """Reject an unconstrained prediction range."""
    with pytest.raises(ValueError, match="Must provide at least one"):
        constraints.HardPredictionRangeConstraint(_helpers._LookupPredictor({}))


@pytest.mark.parametrize(
    ("beta", "expected"),
    [(0, [False, False, False, True]), (4, [False, False, True, True])],
)
def test_ucb_constraint_uses_uncertainty_and_a_strict_threshold(
    beta: int, expected: list[bool]
) -> None:
    """Accept an upper confidence bound strictly above the threshold."""
    model = _helpers._LookupPredictor({"A": 0.0, "C": 1.0, "D": 2.0, "E": 3.0}, std=0.5)
    constraint = constraints.HardUCBInequalityConstraint(
        model, threshold=2.0, beta=beta, batch_size=2
    )

    verdicts = constraint(data.tokenizer.encode(["A", "C", "D", "E"]))

    torch.testing.assert_close(torch.stack(list(verdicts)), torch.tensor(expected))
    assert model.batches[0][0].return_std is True
    assert model.batches[0][0].return_samples is False
    assert model.batches[1][0].return_std is True


def test_ucb_constraint_rejects_negative_beta() -> None:
    """Reject a confidence multiplier whose square root is undefined."""
    with pytest.raises(ValueError, match="math domain error"):
        constraints.HardUCBInequalityConstraint(
            _helpers._LookupPredictor({}), threshold=2.0, beta=-1
        )


@pytest.mark.parametrize(
    ("temperature", "expected"),
    [
        (1.0, [0.119202922, 0.5, 0.880797078]),
        (2.0, [0.268941421, 0.5, 0.731058579]),
    ],
)
def test_soft_constraint_returns_temperature_scaled_feasibility(
    temperature: float, expected: list[float]
) -> None:
    """Return probabilities below, at and above the threshold in request order."""
    model = _helpers._LookupPredictor({"A": 0.0, "C": 2.0, "D": 4.0})
    constraint = constraints.SoftInequalityConstraint(
        model, threshold=2.0, temperature=temperature, batch_size=2
    )

    probabilities = constraint(data.tokenizer.encode(["A", "C", "D"]))

    torch.testing.assert_close(torch.stack(list(probabilities)), torch.tensor(expected))
    assert model.batches[0][0].return_std is False
    assert model.batches[0][0].return_samples is False


def test_model_constraint_delegates_loading() -> None:
    """Load the predictor registered by a model-based constraint."""
    model = _helpers._LookupPredictor({})
    constraint = constraints.HardPredictionRangeConstraint(model, ge=0.0)

    constraint.load(torch.device("cpu"))

    assert model.loaded_devices == [torch.device("cpu")]


def test_model_constraint_delegates_release() -> None:
    """Release the predictor registered by a model-based constraint."""
    model = _helpers._LookupPredictor({})
    constraint = constraints.HardPredictionRangeConstraint(model, ge=0.0)

    constraint.release()

    assert model.released


def test_prediction_range_preserves_the_prediction_device() -> None:
    """Evaluate both bounds without converting a tensor verdict to a Python bool."""
    model = _helpers._LookupPredictor({"A": torch.tensor(2.0, device="meta")})
    constraint = constraints.HardPredictionRangeConstraint(model, ge=1.0, le=3.0)

    verdict = constraint([data.tokenizer.encode("A")])[0]

    assert verdict.device == torch.device("meta")
    assert verdict.dtype == torch.bool
    assert verdict.shape == torch.Size([])


def test_soft_constraint_preserves_prediction_dtype_and_gradients() -> None:
    """Keep tensor precision and gradients while computing feasibility."""
    score = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    constraint = constraints.SoftInequalityConstraint(
        _helpers._LookupPredictor({"A": score}), threshold=2.0, temperature=1.0
    )

    probability = constraint([data.tokenizer.encode("A")])[0]
    probability.backward()

    torch.testing.assert_close(probability, torch.tensor(0.5, dtype=torch.float64))
    torch.testing.assert_close(score.grad, torch.tensor(0.25, dtype=torch.float64))


def test_ucb_constraint_preserves_the_prediction_device() -> None:
    """Keep confidence bounds on the predictor's device."""
    model = _helpers._LookupPredictor(
        {"A": torch.tensor(2.0, device="meta")}, std=torch.tensor(0.5, device="meta")
    )
    constraint = constraints.HardUCBInequalityConstraint(model, threshold=2.0, beta=4)

    verdict = constraint([data.tokenizer.encode("A")])[0]

    assert verdict.device == torch.device("meta")
    assert verdict.dtype == torch.bool
    assert verdict.shape == torch.Size([])


def test_ucb_constraint_requires_standard_deviations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a predictor that omits the requested uncertainty."""
    model = _helpers._LookupPredictor({"A": 0.0})
    monkeypatch.setattr(
        model,
        "forward",
        lambda requests: [
            predictors.Prediction(mean=torch.tensor(0.0)) for _ in requests
        ],
    )
    constraint = constraints.HardUCBInequalityConstraint(model, threshold=2.0, beta=4)

    with pytest.raises(ValueError, match="require prediction standard deviations"):
        constraint([data.tokenizer.encode("A")])


@pytest.mark.parametrize("kind", ["range", "ucb", "soft"])
def test_prediction_constraints_accept_empty_requests(kind: str) -> None:
    """Return an empty tensor when no candidates are submitted."""
    model = _helpers._LookupPredictor({})
    constraint = {
        "range": constraints.HardPredictionRangeConstraint(model, ge=0.0),
        "ucb": constraints.HardUCBInequalityConstraint(model, threshold=0.0, beta=1),
        "soft": constraints.SoftInequalityConstraint(
            model, threshold=0.0, temperature=1.0
        ),
    }[kind]

    responses = constraint.forward([])

    assert isinstance(responses, torch.Tensor)
    assert responses.shape == torch.Size([0])
