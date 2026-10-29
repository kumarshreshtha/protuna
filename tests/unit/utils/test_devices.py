import pytest
import torch

from protuna.utils import devices


def test_resolve_devices_returns_cpu_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A machine with no CUDA device serves on the CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert devices.resolve_devices() == torch.device("cpu")


def test_resolve_devices_unwraps_a_single_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """One visible GPU is returned as a bare device rather than a list."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    assert devices.resolve_devices() == torch.device("cuda:0")


def test_resolve_devices_lists_every_visible_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Several visible GPUs are returned in device order."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

    assert devices.resolve_devices() == [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
        torch.device("cuda:2"),
    ]
