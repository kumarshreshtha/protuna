from collections.abc import Iterator, Sequence

import pytest
import torch

from protuna.inference import handlers
from protuna.typing import SequenceLike
from tests.unit.inference import _helpers


def test_local_handler_streams_every_request_in_one_call_when_unbatched() -> None:
    """Without a batch size the whole stream is handed to the handler at once."""
    handler = _helpers._RecordingHandler()

    streamed = list(handler.stream(iter([(0, 1), (1, 2), (2, 3)])))

    assert streamed == [(0, 1, 2), (1, 2, 4), (2, 3, 6)]
    assert handler.batches == [[1, 2, 3]]


def test_local_handler_splits_the_stream_into_batches() -> None:
    """A batch size caps how many requests reach the handler per call."""
    handler = _helpers._RecordingHandler(batch_size=2)

    list(handler.stream(iter([(0, 1), (1, 2), (2, 3), (3, 4)])))

    assert handler.batches == [[1, 2], [3, 4]]


def test_local_handler_emits_a_final_short_batch() -> None:
    """A stream that does not divide evenly still yields its trailing requests."""
    handler = _helpers._RecordingHandler(batch_size=3)

    streamed = list(handler.stream(iter([(0, 1), (1, 2), (2, 3), (3, 4)])))

    assert streamed == [(0, 1, 2), (1, 2, 4), (2, 3, 6), (3, 4, 8)]
    assert handler.batches == [[1, 2, 3], [4]]


@pytest.mark.parametrize("batch_size", [None, 2])
def test_local_handler_streaming_an_empty_stream_yields_nothing(
    batch_size: int | None,
) -> None:
    """No requests means no handler calls and no results, batched or not."""
    handler = _helpers._RecordingHandler(batch_size=batch_size)

    streamed = list(handler.stream(iter([])))

    assert streamed == []
    assert handler.batches == []


def test_local_handler_shutdown_is_a_no_op() -> None:
    """Local handlers own no remote resources, so shutdown does nothing."""
    assert _helpers._RecordingHandler().shutdown() is None


@pytest.mark.parametrize(
    "device",
    [None, torch.device("cpu"), [torch.device("cuda:0"), torch.device("cuda:1")]],
)
def test_model_free_handler_loading_is_a_no_op(
    device: torch.device | Sequence[torch.device] | None,
) -> None:
    """Allow model-free handlers to participate in explicit loading."""
    handler = _helpers._RecordingHandler()

    assert handler.load(device) is None
    assert handler([1, 2]) == [2, 4]


def test_model_free_handler_release_is_a_no_op() -> None:
    """Keep model-free handlers usable after releasing resources."""
    handler = _helpers._RecordingHandler()

    assert handler.release() is None
    assert handler([1, 2]) == [2, 4]


@pytest.mark.parametrize("batch_size", [None, 2])
def test_local_handler_streams_tensor_batch_responses(batch_size: int | None) -> None:
    """Yield one tensor response per request in input order."""
    handler = _helpers._TensorBatchHandler(batch_size=batch_size)

    streamed = list(handler.stream(iter([(7, 1), (8, 2), (9, 3)])))

    assert [(rid, request) for rid, request, _ in streamed] == [(7, 1), (8, 2), (9, 3)]
    torch.testing.assert_close(
        torch.stack([response for _, _, response in streamed]), torch.tensor([2, 4, 6])
    )


class _RecordingHandler(handlers.LocalHandler[int, int]):
    def __init__(self):
        super().__init__()
        self.loaded_devices: list[torch.device | Sequence[torch.device] | None] = []
        self.release_count = 0

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        self.loaded_devices.append(device)

    def release(self) -> None:
        self.release_count += 1

    def __call__(self, requests: SequenceLike[int]) -> list[int]:
        return list(requests)


class _StructuralHandler:
    def __init__(self):
        self.loaded_devices: list[torch.device | Sequence[torch.device] | None] = []
        self.release_count = 0

    def __call__(self, requests: SequenceLike[int]) -> list[int]:
        return list(requests)

    def stream(
        self, requests: Iterator[tuple[int, int]]
    ) -> Iterator[tuple[int, int, int]]:
        return ((rid, request, request) for rid, request in requests)

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        self.loaded_devices.append(device)

    def release(self) -> None:
        self.release_count += 1

    def shutdown(self) -> None:
        pass


class _Registry(handlers.HandlerRegistryMixin):
    component: handlers.Handler[int, int] | handlers.HandlerRegistryMixin | str | None
    label: str


@pytest.mark.parametrize("handler_type", [_RecordingHandler, _StructuralHandler])
def test_handler_protocol_accepts_handlers_at_runtime(
    handler_type: type[_RecordingHandler | _StructuralHandler],
) -> None:
    """Recognize handlers with and without explicit protocol inheritance."""
    assert isinstance(handler_type(), handlers.Handler)


def test_handler_protocol_rejects_a_registry_without_request_operations() -> None:
    """Exclude lifecycle-only registries from the handler protocol."""
    assert not isinstance(_Registry(), handlers.Handler)


@pytest.mark.parametrize("handler_type", [_RecordingHandler, _StructuralHandler])
def test_register_forwards_handler_lifecycle_calls(
    handler_type: type[_RecordingHandler | _StructuralHandler],
) -> None:
    """Manage a handler registered explicitly by name."""
    registry = _Registry()
    registered = handler_type()

    registry.register("encoder", registered)
    registry.load(torch.device("cpu"))
    registry.release()

    assert registered.loaded_devices == [torch.device("cpu")]
    assert registered.release_count == 1


def test_register_forwards_nested_registry_lifecycle_calls() -> None:
    """Allow an initialized registry to manage a registered child registry."""
    parent = _Registry()
    child = _Registry()
    registered = _RecordingHandler()
    child.register("encoder", registered)

    parent.register("child", child)
    parent.load(torch.device("cpu"))
    parent.release()

    assert registered.loaded_devices == [torch.device("cpu")]
    assert registered.release_count == 1


@pytest.mark.parametrize("handler_type", [_RecordingHandler, _StructuralHandler])
def test_assigning_a_handler_registers_it(
    handler_type: type[_RecordingHandler | _StructuralHandler],
) -> None:
    """Manage a handler assigned as an ordinary attribute."""
    registry = _Registry()
    registered = handler_type()

    registry.component = registered
    registry.load()
    registry.release()

    assert registry.component is registered
    assert registered.loaded_devices == [None]
    assert registered.release_count == 1


def test_assigning_a_subregistry_registers_it() -> None:
    """Manage a child registry assigned as an ordinary attribute."""
    parent = _Registry()
    child = _Registry()
    registered = _RecordingHandler()
    child.component = registered

    parent.component = child
    parent.load()
    parent.release()

    assert parent.component is child
    assert registered.loaded_devices == [None]
    assert registered.release_count == 1


@pytest.mark.parametrize(
    "device",
    [None, torch.device("cpu"), [torch.device("cuda:0"), torch.device("cuda:1")]],
)
def test_load_forwards_device_to_direct_and_nested_handlers(
    device: torch.device | Sequence[torch.device] | None,
) -> None:
    """Forward the same device through every level of a registry tree."""
    parent = _Registry()
    child = _Registry()
    grandchild = _Registry()
    first = _RecordingHandler()
    second = _RecordingHandler()
    nested = _RecordingHandler()
    parent.register("first", first)
    parent.register("second", second)
    grandchild.component = nested
    child.component = grandchild
    parent.component = child

    parent.load(device)

    assert first.loaded_devices == [device]
    assert second.loaded_devices == [device]
    assert nested.loaded_devices == [device]


def test_release_reaches_direct_and_nested_handlers() -> None:
    """Release every registered handler, including deeply nested handlers."""
    parent = _Registry()
    child = _Registry()
    grandchild = _Registry()
    first = _RecordingHandler()
    second = _RecordingHandler()
    nested = _RecordingHandler()
    parent.register("first", first)
    parent.register("second", second)
    grandchild.component = nested
    child.component = grandchild
    parent.component = child
    parent.load()

    parent.release()

    assert first.release_count == 1
    assert second.release_count == 1
    assert nested.release_count == 1


def test_empty_registry_accepts_lifecycle_calls() -> None:
    """Allow loading and releasing a registry with no components."""
    registry = _Registry()

    assert registry.load() is None
    assert registry.release() is None


def test_registry_can_reload_after_release() -> None:
    """Keep registrations available across repeated lifecycle calls."""
    registry = _Registry()
    registered = _RecordingHandler()
    registry.component = registered
    registry.load(torch.device("cpu"))
    registry.release()

    registry.load(torch.device("cuda:1"))
    registry.release()

    assert registered.loaded_devices == [torch.device("cpu"), torch.device("cuda:1")]
    assert registered.release_count == 2


def test_registrations_are_isolated_between_instances() -> None:
    """Keep handlers and subregistries local to their owning registry."""
    first = _Registry()
    second = _Registry()
    child = _Registry()
    direct = _RecordingHandler()
    nested = _RecordingHandler()
    first.register("encoder", direct)
    child.component = nested
    first.component = child

    second.load()
    second.release()

    assert direct.loaded_devices == []
    assert direct.release_count == 0
    assert nested.loaded_devices == []
    assert nested.release_count == 0


def test_plain_attributes_do_not_participate_in_lifecycle_calls() -> None:
    """Preserve normal attribute assignment and deletion for metadata."""
    registry = _Registry()
    registry.label = "campaign"

    registry.load()
    registry.release()

    assert registry.label == "campaign"
    del registry.label
    assert not hasattr(registry, "label")


def test_deleting_a_handler_unregisters_it() -> None:
    """Exclude a deleted handler from subsequent lifecycle calls."""
    registry = _Registry()
    removed = _RecordingHandler()
    retained = _RecordingHandler()
    registry.component = removed
    registry.register("retained", retained)

    del registry.component
    registry.load()
    registry.release()

    assert not hasattr(registry, "component")
    assert removed.loaded_devices == []
    assert removed.release_count == 0
    assert retained.loaded_devices == [None]
    assert retained.release_count == 1


def test_deleting_a_subregistry_unregisters_its_handlers() -> None:
    """Exclude a deleted child registry from subsequent lifecycle calls."""
    parent = _Registry()
    child = _Registry()
    removed = _RecordingHandler()
    retained = _RecordingHandler()
    child.component = removed
    parent.component = child
    parent.register("retained", retained)

    del parent.component
    parent.load()
    parent.release()

    assert not hasattr(parent, "component")
    assert removed.loaded_devices == []
    assert removed.release_count == 0
    assert retained.loaded_devices == [None]
    assert retained.release_count == 1


def test_replacing_a_handler_manages_only_the_replacement() -> None:
    """Replace an existing handler registration when its attribute changes."""
    registry = _Registry()
    old = _RecordingHandler()
    replacement = _RecordingHandler()
    registry.component = old

    registry.component = replacement
    registry.load()
    registry.release()

    assert registry.component is replacement
    assert old.loaded_devices == []
    assert old.release_count == 0
    assert replacement.loaded_devices == [None]
    assert replacement.release_count == 1


def test_replacing_a_subregistry_manages_only_the_replacement() -> None:
    """Replace an existing child registry when its attribute changes."""
    parent = _Registry()
    old_child = _Registry()
    new_child = _Registry()
    old = _RecordingHandler()
    replacement = _RecordingHandler()
    old_child.component = old
    new_child.component = replacement
    parent.component = old_child

    parent.component = new_child
    parent.load()
    parent.release()

    assert parent.component is new_child
    assert old.loaded_devices == []
    assert old.release_count == 0
    assert replacement.loaded_devices == [None]
    assert replacement.release_count == 1


@pytest.mark.parametrize("replacement", [None, "metadata"])
def test_replacing_a_handler_with_plain_data_unregisters_it(
    replacement: str | None,
) -> None:
    """Stop managing a handler when its attribute becomes ordinary data."""
    registry = _Registry()
    old = _RecordingHandler()
    registry.component = old

    registry.component = replacement
    registry.load()
    registry.release()

    assert registry.component is replacement
    assert old.loaded_devices == []
    assert old.release_count == 0


def test_replacing_a_handler_with_a_subregistry_unregisters_the_handler() -> None:
    """Remove the old registration when an attribute changes component type."""
    parent = _Registry()
    old = _RecordingHandler()
    child = _Registry()
    replacement = _RecordingHandler()
    child.component = replacement
    parent.component = old

    parent.component = child
    parent.load()
    parent.release()

    assert old.loaded_devices == []
    assert old.release_count == 0
    assert replacement.loaded_devices == [None]
    assert replacement.release_count == 1


def test_register_replaces_a_handler_with_a_registry() -> None:
    """Replace a named handler when registering a child under the same name."""
    parent = _Registry()
    old = _RecordingHandler()
    child = _Registry()
    replacement = _RecordingHandler()
    child.component = replacement
    parent.register("encoder", old)

    parent.register("encoder", child)
    parent.load()
    parent.release()

    assert old.loaded_devices == []
    assert old.release_count == 0
    assert replacement.loaded_devices == [None]
    assert replacement.release_count == 1


def test_register_replaces_a_registry_with_a_handler() -> None:
    """Replace a named child when registering a handler under the same name."""
    parent = _Registry()
    child = _Registry()
    old = _RecordingHandler()
    replacement = _RecordingHandler()
    child.component = old
    parent.register("encoder", child)

    parent.register("encoder", replacement)
    parent.load()
    parent.release()

    assert old.loaded_devices == []
    assert old.release_count == 0
    assert replacement.loaded_devices == [None]
    assert replacement.release_count == 1


def test_register_handler_before_initialization_raises() -> None:
    """Reject explicit handler registration before the mixin is initialized."""
    registry = _Registry.__new__(_Registry)
    registered = _RecordingHandler()

    with pytest.raises(AttributeError, match="cannot register component before super"):
        registry.register("encoder", registered)


def test_register_subregistry_before_initialization_raises() -> None:
    """Reject explicit child registration before the mixin is initialized."""
    registry = _Registry.__new__(_Registry)
    child = _Registry()

    with pytest.raises(AttributeError, match="cannot register component before super"):
        registry.register("child", child)


def test_assigning_a_handler_before_initialization_raises() -> None:
    """Reject handler assignment before the mixin is initialized."""
    registry = _Registry.__new__(_Registry)
    registered = _RecordingHandler()

    with pytest.raises(AttributeError, match="cannot register component before super"):
        registry.component = registered


def test_assigning_a_subregistry_before_initialization_raises() -> None:
    """Reject child registry assignment before the mixin is initialized."""
    registry = _Registry.__new__(_Registry)
    child = _Registry()

    with pytest.raises(AttributeError, match="cannot register component before super"):
        registry.component = child


def test_deleting_an_unknown_attribute_raises() -> None:
    """Preserve the normal error for deleting an attribute that does not exist."""
    registry = _Registry()

    with pytest.raises(AttributeError, match="component"):
        del registry.component
