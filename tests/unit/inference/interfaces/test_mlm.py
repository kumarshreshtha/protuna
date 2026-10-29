import torch

from protuna.inference.interfaces import mlm


def test_output_batch_preserves_hidden_layers_and_optional_fields() -> None:
    """Keep ragged outputs and hidden-layer tuples aligned."""
    outputs = [
        mlm.MLMOutput(
            logits=torch.tensor([[1.0, 2.0]]),
            hidden_states=(torch.tensor([[3.0]]), torch.tensor([[4.0]])),
            embeddings=torch.tensor([[5.0]]),
            mean_embeddings=torch.tensor([6.0]),
        ),
        mlm.MLMOutput(logits=torch.tensor([[7.0, 8.0], [9.0, 10.0]])),
        mlm.MLMOutput(hidden_states=()),
    ]

    moved = mlm.MLMOutput.to_device_batch(outputs, torch.device("cpu"))

    torch.testing.assert_close(moved[0].logits, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(
        moved[0].hidden_states, (torch.tensor([[3.0]]), torch.tensor([[4.0]]))
    )
    torch.testing.assert_close(moved[0].embeddings, torch.tensor([[5.0]]))
    torch.testing.assert_close(moved[0].mean_embeddings, torch.tensor([6.0]))
    torch.testing.assert_close(moved[1].logits, torch.tensor([[7.0, 8.0], [9.0, 10.0]]))
    assert moved[1].hidden_states is None
    assert moved[1].embeddings is None
    assert moved[1].mean_embeddings is None
    assert moved[2].hidden_states == ()
    assert moved[0] is not outputs[0]


def test_output_batch_moves_all_populated_fields() -> None:
    """Transfer tensors while leaving the original output unchanged."""
    output = mlm.MLMOutput(
        logits=torch.ones(2, 3),
        hidden_states=(torch.ones(2, 4),),
        embeddings=torch.ones(2, 4),
        mean_embeddings=torch.ones(4),
    )

    moved = mlm.MLMOutput.to_device_batch([output], torch.device("meta"))[0]

    assert moved.logits is not None
    assert moved.logits.device.type == "meta"
    assert moved.hidden_states is not None
    assert moved.hidden_states[0].device.type == "meta"
    assert moved.embeddings is not None
    assert moved.embeddings.device.type == "meta"
    assert moved.mean_embeddings is not None
    assert moved.mean_embeddings.device.type == "meta"
    assert output.logits is not None
    assert output.logits.device.type == "cpu"


def test_output_batch_accepts_empty_input() -> None:
    """Allow a batch without outputs."""
    assert mlm.MLMOutput.to_device_batch([], torch.device("cpu")) == []
