"""Utilities for managing device discovery and movement management."""

from collections.abc import Sequence

import torch


def resolve_devices() -> torch.device | Sequence[torch.device]:
    """Return the devices visible to this process.

    Returns:
        ``torch.device("cpu")`` when CUDA is unavailable, the single CUDA device
        when only one is visible, and the list of CUDA devices otherwise.
    """
    device = (
        torch.device("cpu")
        if not torch.cuda.is_available()
        else [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    )
    if isinstance(device, list) and len(device) == 1:
        return device[0]
    return device


def to_device_batch(
    tensors: Sequence[torch.Tensor], device: torch.device
) -> list[torch.Tensor]:
    """Move compatible tensors together, preserving their order and shapes.

    Args:
        tensors: Tensors to transfer; shapes, dtypes and source devices may differ.
        device: Destination device.

    Returns:
        Tensors on the destination device in input order. Compatible dense
        tensors share a stacked transfer; other layouts transfer individually.
    """
    moved = list(tensors)
    groups: dict[tuple[torch.Size, torch.dtype, torch.device], list[int]] = {}
    for i, tensor in enumerate(tensors):
        if tensor.layout != torch.strided or tensor.is_nested:
            moved[i] = tensor.to(device)
        else:
            groups.setdefault((tensor.shape, tensor.dtype, tensor.device), []).append(i)
    for indices in groups.values():
        batch = torch.stack([tensors[i] for i in indices]).to(device)
        for i, tensor in zip(indices, batch.unbind(0), strict=True):
            # without clone ray serializes the entire underlying storage for each
            # view
            moved[i] = tensor.clone()
    return moved
