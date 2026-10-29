from __future__ import annotations

import pathlib

import pytest
import torch

from protuna.optim import predictors
from protuna.optim import storage


@pytest.fixture
def store(tmp_path: pathlib.Path) -> storage.CampaignStore:
    """A campaign store for length-3 proteins writing under a temp directory."""
    return storage.CampaignStore(protein_length=3, location=tmp_path)


@pytest.fixture
def soft_store(tmp_path: pathlib.Path) -> storage.CampaignStore:
    """Register stability and solubility as soft constraints."""
    return storage.CampaignStore(
        protein_length=3,
        location=tmp_path,
        soft_constraint_names=("stability", "solubility"),
    )


def _sorted_samples(
    store: storage.CampaignStore, **kwargs
) -> list[tuple[int, list[int]]]:
    """Collect iter_samples into plain values, ordered by seq_id."""
    samples = [
        (seq_id, token_ids.tolist())
        for seq_id, token_ids in store.iter_samples(**kwargs)
    ]
    return sorted(samples)


def _token_lists(rows: list[storage.SampleRow]) -> list[list[int]]:
    """Collect the token ids of sample rows into plain lists."""
    return [row.token_ids.tolist() for row in rows]  # ty: ignore[unresolved-attribute]


def test_stored_samples_round_trip_after_finalize(store: storage.CampaignStore) -> None:
    """Samples written to parquet come back with their ids and token ids."""
    store.store_samples(
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        parent_id=0,
        ancestor_id=0,
        cycle_id=1,
    )
    store.finalize()

    assert _sorted_samples(store) == [(1, [1, 2, 3]), (2, [4, 5, 6])]


def test_buffered_samples_are_visible_before_flush(
    store: storage.CampaignStore,
) -> None:
    """Rows still in the in-memory chunk are readable without finalizing."""
    store.store_samples(
        torch.tensor([[7, 8, 9]]), parent_id=0, ancestor_id=0, cycle_id=1
    )

    assert _sorted_samples(store) == [(1, [7, 8, 9])]


def test_seq_ids_start_at_one(store: storage.CampaignStore) -> None:
    """Ids are one-based, leaving 0 free to mean "no parent"."""
    store.store_samples(
        torch.tensor([[1, 1, 1]]), parent_id=0, ancestor_id=0, cycle_id=1
    )
    store.finalize()

    assert [seq_id for seq_id, _ in _sorted_samples(store)] == [1]


def test_seq_ids_continue_across_store_calls(store: storage.CampaignStore) -> None:
    """Ids are assigned from a single counter shared by every batch."""
    store.store_samples(
        torch.tensor([[1, 1, 1], [2, 2, 2]]),
        parent_id=0,
        ancestor_id=0,
        cycle_id=1,
    )
    store.store_samples(
        torch.tensor([[3, 3, 3]]), parent_id=1, ancestor_id=1, cycle_id=1
    )
    store.finalize()

    assert [seq_id for seq_id, _ in _sorted_samples(store)] == [1, 2, 3]


def test_iter_samples_filters_by_cycle_id(store: storage.CampaignStore) -> None:
    """Only samples proposed in the requested cycle are yielded."""
    store.store_base_sequences(torch.tensor([[1, 1, 1]]))
    store.store_samples(
        torch.tensor([[2, 2, 2]]), parent_id=1, ancestor_id=1, cycle_id=1
    )
    store.finalize()

    assert _sorted_samples(store, cycle_id=1) == [(2, [2, 2, 2])]


def test_iter_samples_drops_sequences_that_failed_a_constraint(
    store: storage.CampaignStore,
) -> None:
    """A failing constraint hides the sequence by default."""
    store.store_samples(
        torch.tensor([[1, 1, 1], [2, 2, 2]]),
        parent_id=0,
        ancestor_id=0,
        cycle_id=1,
    )
    store.store_constraint("stability", 1, torch.tensor(False))
    store.finalize()

    assert _sorted_samples(store) == [(2, [2, 2, 2])]


def test_iter_samples_keeps_sequences_that_passed_a_constraint(
    store: storage.CampaignStore,
) -> None:
    """A passing constraint leaves the sequence in place."""
    store.store_samples(
        torch.tensor([[1, 1, 1]]), parent_id=0, ancestor_id=0, cycle_id=1
    )
    store.store_constraint("stability", 1, torch.tensor(True))
    store.finalize()

    assert _sorted_samples(store) == [(1, [1, 1, 1])]


def test_iter_samples_without_exclude_failed_keeps_everything(
    store: storage.CampaignStore,
) -> None:
    """Setting exclude_failed to False ignores the constraint table."""
    store.store_samples(
        torch.tensor([[1, 1, 1], [2, 2, 2]]),
        parent_id=0,
        ancestor_id=0,
        cycle_id=1,
    )
    store.store_constraint("stability", 1, torch.tensor(False))
    store.finalize()

    assert _sorted_samples(store, exclude_failed=False) == [
        (1, [1, 1, 1]),
        (2, [2, 2, 2]),
    ]


def test_get_samples_unordered_returns_only_the_requested_ids(
    store: storage.CampaignStore,
) -> None:
    """Look-up by id selects the matching rows and nothing else."""
    store.store_base_sequences(torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]]))
    store.finalize()

    rows = store.get_samples_unordered([1, 3])

    assert sorted(_token_lists(rows)) == [[1, 1, 1], [3, 3, 3]]


def test_get_samples_unordered_with_no_ids_returns_nothing(
    store: storage.CampaignStore,
) -> None:
    """An empty id list yields no rows."""
    store.store_base_sequences(torch.tensor([[1, 1, 1]]))
    store.finalize()

    assert store.get_samples_unordered([]) == []


def test_get_samples_unordered_reads_the_requested_columns(
    store: storage.CampaignStore,
) -> None:
    """The lineage columns come back on the row when they are asked for."""
    store.store_samples(
        torch.tensor([[1, 1, 1]]), parent_id=7, ancestor_id=3, cycle_id=2
    )
    store.finalize()

    row = store.get_samples_unordered(
        [1], columns=["seq_id", "parent_id", "ancestor_id", "cycle_id", "token_ids"]
    )[0]

    assert (row.seq_id, row.parent_id, row.ancestor_id, row.cycle_id) == (1, 7, 3, 2)


def test_get_samples_unordered_leaves_unread_columns_unset(
    store: storage.CampaignStore,
) -> None:
    """Columns outside the requested set stay None on the returned row."""
    store.store_samples(
        torch.tensor([[1, 1, 1]]), parent_id=7, ancestor_id=3, cycle_id=2
    )
    store.finalize()

    row = store.get_samples_unordered([1])[0]

    assert (row.seq_id, row.parent_id, row.ancestor_id, row.cycle_id) == (
        None,
        None,
        None,
        None,
    )


def test_base_sequences_record_a_zero_lineage(
    store: storage.CampaignStore,
) -> None:
    """A starting sequence has no parent, which is stored as id 0."""
    store.store_base_sequences(torch.tensor([[1, 1, 1]]))
    store.finalize()

    row = store.get_samples_unordered(
        [1], columns=["parent_id", "ancestor_id", "token_ids"]
    )[0]

    assert (row.parent_id, row.ancestor_id) == (0, 0)


def test_every_sample_in_a_batch_shares_the_batch_lineage(
    store: storage.CampaignStore,
) -> None:
    """One store call records the same parent and ancestor for all its samples."""
    store.store_samples(
        torch.tensor([[1, 1, 1], [2, 2, 2]]), parent_id=5, ancestor_id=2, cycle_id=1
    )
    store.finalize()

    rows = store.get_samples_unordered(
        [1, 2], columns=["parent_id", "ancestor_id", "token_ids"]
    )

    assert [(row.parent_id, row.ancestor_id) for row in rows] == [(5, 2), (5, 2)]


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        (predictors.Prediction(mean=torch.tensor(1.5)), (1, 1.5, None, None)),
        (
            predictors.Prediction(mean=torch.tensor(1.5), std=torch.tensor(0.25)),
            (1, 1.5, 0.25, None),
        ),
        (
            predictors.Prediction(
                mean=torch.tensor(1.5), samples=torch.tensor([1.0, 2.0])
            ),
            (1, 1.5, None, [1.0, 2.0]),
        ),
    ],
    ids=["mean", "standard-deviation", "posterior-samples"],
)
def test_objective_predictions_are_recorded(
    store: storage.CampaignStore,
    prediction: predictors.Prediction,
    expected: tuple[int, float, float | None, list[float] | None],
) -> None:
    """Persist the objective mean and any requested uncertainty."""
    store.store_objective(1, prediction)
    store.finalize()

    assert store.objective_store.get_lf().collect().rows() == [expected]


@pytest.mark.parametrize("passed", [False, True])
def test_hard_constraint_verdicts_are_recorded(
    store: storage.CampaignStore, passed: bool
) -> None:
    """Persist boolean verdicts in the hard constraint table."""
    store.store_constraint("stability", 7, torch.tensor(passed))
    store.finalize()

    assert store.hard_constraint_store.get_lf().collect().rows() == [
        (7, "stability", passed)
    ]
    assert store.soft_constraint_store.get_lf().collect().is_empty()


def test_predictions_stay_on_their_device_while_buffering(
    store: storage.CampaignStore,
) -> None:
    """Collect predictions without transferring or extracting tensor values."""
    prediction = predictors.Prediction(
        mean=torch.empty((), device="meta"),
        std=torch.empty((), device="meta"),
        samples=torch.empty(2, device="meta"),
    )

    store.store_objective(1, prediction)

    assert store.objective_store._chunk["objective_mean"][0].device.type == "meta"
    assert store.objective_store._chunk["objective_std"][0].device.type == "meta"
    assert store.objective_store._chunk["objective_samples"][0].device.type == "meta"


@pytest.mark.parametrize("constraint_id", ["stability", "activity"])
def test_constraints_stay_on_their_device_while_buffering(
    soft_store: storage.CampaignStore, constraint_id: str
) -> None:
    """Collect hard and soft constraints without extracting tensor values."""
    value = torch.empty((), device="meta")

    soft_store.store_constraint(constraint_id, 1, value)

    buffered = (
        soft_store.soft_constraint_store._chunk["feasibility"]
        + soft_store.hard_constraint_store._chunk["passed"]
    )
    assert len(buffered) == 1
    assert buffered[0].device.type == "meta"


@pytest.mark.parametrize("feasibility", [0.0, 0.5, 1.0])
def test_soft_constraint_probabilities_are_recorded(
    soft_store: storage.CampaignStore, feasibility: float
) -> None:
    """Persist float probabilities in the soft constraint table."""
    soft_store.store_constraint("stability", 7, torch.tensor(feasibility))
    soft_store.finalize()

    assert soft_store.soft_constraint_store.get_lf().collect().rows() == [
        (7, "stability", feasibility)
    ]
    assert soft_store.hard_constraint_store.get_lf().collect().is_empty()


@pytest.mark.parametrize(("value", "expected"), [(False, 0.0), (True, 1.0)])
def test_registered_soft_constraints_route_boolean_values_to_feasibility(
    soft_store: storage.CampaignStore, value: bool, expected: float
) -> None:
    """Route verdicts by the registered name even for boolean values."""
    soft_store.store_constraint("stability", 7, torch.tensor(value))
    soft_store.finalize()

    assert soft_store.soft_constraint_store.get_lf().collect().rows() == [
        (7, "stability", expected)
    ]
    assert soft_store.hard_constraint_store.get_lf().collect().is_empty()


def test_unregistered_constraints_remain_hard_with_soft_names_configured(
    soft_store: storage.CampaignStore,
) -> None:
    """Keep unregistered names in the hard constraint table."""
    soft_store.store_constraint("activity", 7, torch.tensor(False))
    soft_store.finalize()

    assert soft_store.hard_constraint_store.get_lf().collect().rows() == [
        (7, "activity", False)
    ]
    assert soft_store.soft_constraint_store.get_lf().collect().is_empty()


def test_supplied_task_id_determines_artifact_directory(tmp_path: pathlib.Path) -> None:
    """Write campaign artifacts under the supplied task identifier."""
    store = storage.CampaignStore(
        protein_length=3, location=tmp_path, task_id="campaign-42"
    )
    store.store_samples(
        torch.tensor([[1, 2, 3]]), parent_id=0, ancestor_id=0, cycle_id=1
    )
    store.finalize()

    assert (tmp_path / "pipelines/campaign-42/samples_part00001.parquet").is_file()
    assert _sorted_samples(store) == [(1, [1, 2, 3])]


def test_samples_spill_into_multiple_files_past_max_size(
    tmp_path: pathlib.Path,
) -> None:
    """Flush each batch when it reaches the configured memory limit."""
    store = storage.CampaignStore(
        protein_length=3, location=tmp_path, max_size_mb=0.001
    )

    store.store_samples(
        torch.tensor([[1, 1, 1], [2, 2, 2]]),
        parent_id=0,
        ancestor_id=0,
        cycle_id=1,
    )
    store.store_samples(
        torch.tensor([[3, 3, 3], [4, 4, 4]]),
        parent_id=0,
        ancestor_id=0,
        cycle_id=1,
    )

    assert len(list(tmp_path.glob("pipelines/*/samples_part*.parquet"))) == 2


@pytest.mark.parametrize("max_size_mb", [0, -1, float("inf"), float("nan")])
def test_invalid_memory_limit_is_rejected(max_size_mb: float) -> None:
    """Require a positive finite memory budget."""
    with pytest.raises(ValueError, match="max_size_mb must be positive and finite"):
        storage.CampaignStore(protein_length=3, max_size_mb=max_size_mb)


@pytest.mark.parametrize(("sample_count", "expected_files"), [(1, 0), (4096, 1)])
def test_objective_flush_accounts_for_actual_sample_size(
    tmp_path: pathlib.Path, sample_count: int, expected_files: int
) -> None:
    """Flush large posterior samples while keeping small predictions buffered."""
    store = storage.CampaignStore(
        protein_length=3, location=tmp_path, max_size_mb=0.004
    )

    store.store_objective(
        1,
        predictors.Prediction(mean=torch.tensor(1.0), samples=torch.ones(sample_count)),
    )

    assert (
        len(list(tmp_path.glob("pipelines/*/objective_part*.parquet")))
        == expected_files
    )
    assert store.objective_store.get_lf().collect().height == 1


@pytest.mark.parametrize("soft", [False, True], ids=["hard", "soft"])
@pytest.mark.parametrize(("name_length", "expected_files"), [(8, 0), (4096, 1)])
def test_constraint_flush_accounts_for_actual_name_size(
    tmp_path: pathlib.Path, soft: bool, name_length: int, expected_files: int
) -> None:
    """Flush long constraint names while keeping short verdicts buffered."""
    name = "x" * name_length
    store = storage.CampaignStore(
        protein_length=3,
        location=tmp_path,
        max_size_mb=0.004,
        soft_constraint_names=[name] * soft,
    )

    store.store_constraint(name, 1, torch.tensor(True))

    assert (
        len(list(tmp_path.glob("pipelines/*/*constraints_part*.parquet")))
        == expected_files
    )


def test_memory_budget_resets_after_flush(tmp_path: pathlib.Path) -> None:
    """Keep a small prediction buffered after a large one was flushed."""
    store = storage.CampaignStore(
        protein_length=3, location=tmp_path, max_size_mb=0.004
    )

    store.store_objective(
        1, predictors.Prediction(mean=torch.tensor(1.0), samples=torch.ones(4096))
    )
    store.store_objective(2, predictors.Prediction(mean=torch.tensor(2.0)))

    assert len(list(tmp_path.glob("pipelines/*/objective_part*.parquet"))) == 1
    assert store.objective_store.get_lf().collect().height == 2


def test_samples_split_across_files_are_read_back_together(
    tmp_path: pathlib.Path,
) -> None:
    """Reads span every part file plus the unflushed chunk."""
    store = storage.CampaignStore(
        protein_length=3, location=tmp_path, max_size_mb=0.001
    )

    store.store_samples(
        torch.tensor([[1, 1, 1], [2, 2, 2]]),
        parent_id=0,
        ancestor_id=0,
        cycle_id=1,
    )
    store.store_samples(
        torch.tensor([[3, 3, 3]]), parent_id=0, ancestor_id=0, cycle_id=1
    )

    assert _sorted_samples(store) == [(1, [1, 1, 1]), (2, [2, 2, 2]), (3, [3, 3, 3])]


def test_each_store_writes_to_its_own_task_directory(tmp_path: pathlib.Path) -> None:
    """Two stores at the same location never share a run directory."""
    first = storage.CampaignStore(protein_length=3, location=tmp_path)
    second = storage.CampaignStore(protein_length=3, location=tmp_path)

    first.store_samples(
        torch.tensor([[1, 1, 1]]), parent_id=0, ancestor_id=0, cycle_id=1
    )
    first.finalize()
    second.store_samples(
        torch.tensor([[2, 2, 2]]), parent_id=0, ancestor_id=0, cycle_id=1
    )
    second.finalize()

    assert _sorted_samples(second) == [(1, [2, 2, 2])]


def test_store_samples_can_flush_before_the_memory_limit(
    store: storage.CampaignStore, tmp_path: pathlib.Path
) -> None:
    """Write a parquet part immediately when flushing is requested."""
    store.store_samples(
        torch.tensor([[1, 2, 3]]),
        parent_id=0,
        ancestor_id=0,
        cycle_id=1,
        flush=True,
    )

    assert len(list(tmp_path.glob("pipelines/*/samples_part*.parquet"))) == 1
    assert _sorted_samples(store) == [(1, [1, 2, 3])]


def test_iter_samples_keeps_sequences_with_zero_soft_feasibility(
    soft_store: storage.CampaignStore,
) -> None:
    """Keep soft constraint probabilities out of hard constraint filtering."""
    soft_store.store_samples(
        torch.tensor([[1, 2, 3]]), parent_id=0, ancestor_id=0, cycle_id=1
    )
    soft_store.store_constraint("stability", 1, torch.tensor(0.0))

    assert _sorted_samples(soft_store) == [(1, [1, 2, 3])]


def test_candidate_stream_returns_buffered_samples_and_objectives(
    store: storage.CampaignStore,
) -> None:
    """Read aligned candidate tensors before finalizing the store."""
    store.store_samples(
        torch.tensor([[1, 2, 3]]), parent_id=7, ancestor_id=3, cycle_id=2
    )
    store.store_objective(1, predictors.Prediction(mean=torch.tensor(1.5)))

    stream = store.get_candidate_stream()
    assert isinstance(stream, storage.CandidateStreamFromStorage)
    (batch,) = list(stream)

    assert batch.seq_id.tolist() == [1]
    assert batch.token_ids.tolist() == [[1, 2, 3]]
    assert batch.token_ids.dtype == torch.long
    assert batch.parent_id.tolist() == [7]
    assert batch.ancestor_id.tolist() == [3]
    assert batch.cycle_id.tolist() == [2]
    assert batch.objective_mean is not None
    torch.testing.assert_close(batch.objective_mean, torch.tensor([1.5]))
    assert batch.objective_std is None
    assert batch.objective_samples is None
    assert batch.feasibility is None


def test_candidate_stream_round_trips_uncertainty_and_soft_constraints(
    soft_store: storage.CampaignStore,
) -> None:
    """Read persisted uncertainty and feasibility alongside the candidate."""
    soft_store.store_samples(
        torch.tensor([[1, 2, 3]]), parent_id=0, ancestor_id=0, cycle_id=1
    )
    soft_store.store_objective(
        1,
        predictors.Prediction(
            mean=torch.tensor(1.5),
            std=torch.tensor(0.25),
            samples=torch.tensor([1.0, 2.0]),
        ),
    )
    soft_store.store_constraint("stability", 1, torch.tensor(0.75))
    soft_store.store_constraint("solubility", 1, torch.tensor(0.5))
    soft_store.finalize()

    (batch,) = list(soft_store.get_candidate_stream())

    assert batch.objective_mean is not None
    assert batch.objective_std is not None
    assert batch.objective_samples is not None
    assert batch.feasibility is not None
    torch.testing.assert_close(batch.objective_mean, torch.tensor([1.5]))
    torch.testing.assert_close(batch.objective_std, torch.tensor([0.25]))
    torch.testing.assert_close(batch.objective_samples, torch.tensor([[1.0, 2.0]]))
    assert set(batch.feasibility) == {"stability", "solubility"}
    torch.testing.assert_close(batch.feasibility["stability"], torch.tensor([0.75]))
    torch.testing.assert_close(batch.feasibility["solubility"], torch.tensor([0.5]))


def test_candidate_stream_filters_by_cycle_id(store: storage.CampaignStore) -> None:
    """Yield only scored candidates from the requested cycle."""
    store.store_base_sequences(torch.tensor([[1, 1, 1]]))
    store.store_samples(
        torch.tensor([[2, 2, 2]]), parent_id=1, ancestor_id=1, cycle_id=1
    )
    store.store_objective(1, predictors.Prediction(mean=torch.tensor(1.0)))
    store.store_objective(2, predictors.Prediction(mean=torch.tensor(2.0)))

    (batch,) = list(store.get_candidate_stream(cycle_id=1))

    assert batch.seq_id.tolist() == [2]
    assert batch.cycle_id.tolist() == [1]


def test_candidate_stream_excludes_any_failed_hard_constraint(
    store: storage.CampaignStore,
) -> None:
    """Exclude a candidate even when another hard constraint passed."""
    store.store_samples(
        torch.tensor([[1, 1, 1], [2, 2, 2]]),
        parent_id=0,
        ancestor_id=0,
        cycle_id=1,
    )
    store.store_objective(1, predictors.Prediction(mean=torch.tensor(1.0)))
    store.store_objective(2, predictors.Prediction(mean=torch.tensor(2.0)))
    store.store_constraint("stability", 1, torch.tensor(False))
    store.store_constraint("solubility", 1, torch.tensor(True))
    store.store_constraint("stability", 2, torch.tensor(True))
    store.finalize()

    (batch,) = list(store.get_candidate_stream())

    assert batch.seq_id.tolist() == [2]


def test_candidate_stream_keeps_samples_without_objectives(
    store: storage.CampaignStore,
) -> None:
    """Keep unscored candidates aligned with available objective values."""
    store.store_samples(
        torch.tensor([[1, 1, 1], [2, 2, 2]]),
        parent_id=0,
        ancestor_id=0,
        cycle_id=1,
    )
    store.store_objective(2, predictors.Prediction(mean=torch.tensor(2.0)))

    (streamed,) = list(store.get_candidate_stream())
    assert sorted(streamed.seq_id.tolist()) == [1, 2]

    batch = store.get_candidate_stream().candidates_from_ids([1, 2])
    assert batch.seq_id.tolist() == [1, 2]
    assert batch.objective_mean is not None
    torch.testing.assert_close(
        batch.objective_mean, torch.tensor([float("nan"), 2.0]), equal_nan=True
    )


def test_candidate_stream_keeps_samples_without_soft_constraint_evaluations(
    soft_store: storage.CampaignStore,
) -> None:
    """Keep missing soft evaluations aligned with zero feasibility results."""
    soft_store.store_samples(
        torch.tensor([[1, 1, 1], [2, 2, 2]]),
        parent_id=0,
        ancestor_id=0,
        cycle_id=1,
    )
    soft_store.store_objective(1, predictors.Prediction(mean=torch.tensor(1.0)))
    soft_store.store_objective(2, predictors.Prediction(mean=torch.tensor(2.0)))
    soft_store.store_constraint("stability", 2, torch.tensor(0.0))

    (streamed,) = list(soft_store.get_candidate_stream())
    assert sorted(streamed.seq_id.tolist()) == [1, 2]
    batch = soft_store.get_candidate_stream().candidates_from_ids([1, 2])

    assert batch.seq_id.tolist() == [1, 2]
    assert batch.feasibility is not None
    torch.testing.assert_close(
        batch.feasibility["stability"],
        torch.tensor([float("nan"), 0.0]),
        equal_nan=True,
    )
    assert "solubility" not in batch.feasibility


def test_candidate_stream_without_samples_is_empty(
    store: storage.CampaignStore,
) -> None:
    """Yield no candidate batches from an unused store."""
    assert list(store.get_candidate_stream()) == []


def test_candidate_stream_without_objectives_keeps_candidates(
    store: storage.CampaignStore,
) -> None:
    """An objective is optional for generated candidates."""
    store.store_samples(
        torch.tensor([[1, 2, 3]]), parent_id=0, ancestor_id=0, cycle_id=1
    )

    (batch,) = list(store.get_candidate_stream())

    assert batch.seq_id.tolist() == [1]
    assert batch.objective_mean is None
    assert batch.objective_std is None
    assert batch.objective_samples is None


def test_candidate_stream_respects_scan_size_across_disk_and_buffer(
    tmp_path: pathlib.Path,
) -> None:
    """Read all candidates in batches bounded by the configured scan size."""
    store = storage.CampaignStore(
        protein_length=3,
        location=tmp_path,
        max_size_mb=0.001,
        scan_size=1,
    )
    store.store_samples(
        torch.tensor([[1, 1, 1], [2, 2, 2]]),
        parent_id=0,
        ancestor_id=0,
        cycle_id=1,
    )
    store.store_objective(1, predictors.Prediction(mean=torch.tensor(1.0)))
    store.store_objective(2, predictors.Prediction(mean=torch.tensor(2.0)))
    store.finalize()
    store.store_samples(
        torch.tensor([[3, 3, 3]]), parent_id=1, ancestor_id=1, cycle_id=1
    )
    store.store_objective(3, predictors.Prediction(mean=torch.tensor(3.0)))

    first, second, third = list(store.get_candidate_stream())

    assert (
        first.token_ids.shape,
        second.token_ids.shape,
        third.token_ids.shape,
    ) == ((1, 3), (1, 3), (1, 3))
    assert sorted([first.seq_id.item(), second.seq_id.item(), third.seq_id.item()]) == [
        1,
        2,
        3,
    ]


def test_candidates_from_ids_preserves_requested_order_and_alignment(
    soft_store: storage.CampaignStore,
) -> None:
    """Return requested candidates with their matching scores and feasibility."""
    soft_store.store_samples(
        torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
        parent_id=5,
        ancestor_id=2,
        cycle_id=1,
    )
    soft_store.store_objective(
        1,
        predictors.Prediction(
            mean=torch.tensor(1.0),
            std=torch.tensor(0.25),
            samples=torch.tensor([0.5, 1.5]),
        ),
    )
    soft_store.store_objective(
        2, predictors.Prediction(mean=torch.tensor(2.0), std=torch.tensor(0.5))
    )
    soft_store.store_objective(
        3,
        predictors.Prediction(
            mean=torch.tensor(3.0),
            std=torch.tensor(0.75),
            samples=torch.tensor([2.5, 3.5]),
        ),
    )
    soft_store.store_constraint("stability", 3, torch.tensor(0.75))
    soft_store.store_constraint("stability", 1, torch.tensor(0.25))
    soft_store.store_constraint("stability", 2, torch.tensor(0.5))
    soft_store.finalize()

    batch = soft_store.get_candidate_stream().candidates_from_ids([3, 1])

    assert batch.seq_id.tolist() == [3, 1]
    assert batch.token_ids.tolist() == [[3, 3, 3], [1, 1, 1]]
    assert batch.parent_id.tolist() == [5, 5]
    assert batch.ancestor_id.tolist() == [2, 2]
    assert batch.cycle_id.tolist() == [1, 1]
    assert batch.objective_mean is not None
    assert batch.objective_std is not None
    assert batch.objective_samples is not None
    assert batch.feasibility is not None
    torch.testing.assert_close(batch.objective_mean, torch.tensor([3.0, 1.0]))
    torch.testing.assert_close(batch.objective_std, torch.tensor([0.75, 0.25]))
    torch.testing.assert_close(
        batch.objective_samples, torch.tensor([[2.5, 3.5], [0.5, 1.5]])
    )
    torch.testing.assert_close(
        batch.feasibility["stability"], torch.tensor([0.75, 0.25])
    )


def test_candidates_from_ids_can_read_unscored_samples(
    store: storage.CampaignStore,
) -> None:
    """Return absent objective data as None when looking up unscored samples."""
    store.store_samples(
        torch.tensor([[1, 2, 3]]), parent_id=0, ancestor_id=0, cycle_id=1
    )

    batch = store.get_candidate_stream().candidates_from_ids([1])

    assert batch.seq_id.tolist() == [1]
    assert batch.token_ids.tolist() == [[1, 2, 3]]
    assert batch.objective_mean is None
    assert batch.objective_std is None
    assert batch.objective_samples is None
    assert batch.feasibility is None


@pytest.mark.parametrize("seq_ids", [[], [99]], ids=["empty", "unknown"])
def test_candidates_from_ids_without_matches_returns_an_empty_batch(
    store: storage.CampaignStore, seq_ids: list[int]
) -> None:
    """Return an empty candidate batch when no requested ids exist."""
    store.store_samples(
        torch.tensor([[1, 2, 3]]), parent_id=0, ancestor_id=0, cycle_id=1
    )

    batch = store.get_candidate_stream().candidates_from_ids(seq_ids)

    assert batch.seq_id.tolist() == []
    assert batch.token_ids.shape == (0, 3)


@pytest.mark.parametrize("flush", [False, True])
def test_base_sequences_are_separate_from_candidates(
    soft_store: storage.CampaignStore, flush: bool
) -> None:
    """Base reads preserve registration order and bypass candidate filtering."""
    soft_store.store_base_sequences(torch.tensor([[1, 1, 1], [2, 2, 2]]), flush=flush)
    soft_store.store_samples(
        torch.tensor([[3, 3, 3]]), parent_id=1, ancestor_id=1, cycle_id=1
    )
    soft_store.store_constraint("hard", 1, torch.tensor(False))
    soft_store.store_objective(2, predictors.Prediction(mean=torch.tensor(2.0)))

    base = list(soft_store.iter_base_sequences())
    (batch,) = list(soft_store.get_candidate_stream())

    assert [(seq_id, tokens.tolist()) for seq_id, tokens in base] == [
        (1, [1, 1, 1]),
        (2, [2, 2, 2]),
    ]
    assert _sorted_samples(soft_store, exclude_failed=False) == [(3, [3, 3, 3])]
    assert batch.seq_id.tolist() == [3]
    assert batch.objective_mean is None
    assert batch.feasibility is None


def test_iter_samples_rejects_cycle_zero(store: storage.CampaignStore) -> None:
    """Direct base reads must use the dedicated iterator."""
    with pytest.raises(ValueError, match="use iter_base_sequences"):
        list(store.iter_samples(cycle_id=0))


def test_candidate_stream_rejects_cycle_zero(store: storage.CampaignStore) -> None:
    """Cycle zero cannot be requested as a candidate cycle."""
    with pytest.raises(ValueError, match="Only candidate cycles are supported"):
        store.get_candidate_stream(cycle_id=0)


def test_candidates_from_ids_omits_base_sequences(
    store: storage.CampaignStore,
) -> None:
    """Candidate lookup cannot emit a base sequence alongside candidates."""
    store.store_base_sequences(torch.tensor([[1, 1, 1]]))
    store.store_samples(
        torch.tensor([[2, 2, 2]]), parent_id=1, ancestor_id=1, cycle_id=1
    )

    batch = store.get_candidate_stream().candidates_from_ids([2, 1])

    assert batch.seq_id.tolist() == [2]
    assert batch.token_ids.tolist() == [[2, 2, 2]]


def test_iter_base_sequences_without_samples_is_empty(
    store: storage.CampaignStore,
) -> None:
    """An unused store has no base sequences."""
    assert list(store.iter_base_sequences()) == []


@pytest.mark.parametrize("max_size_mb", [500, 0.00001])
def test_candidate_stream_preserves_partially_missing_prediction_samples(
    tmp_path: pathlib.Path, max_size_mb: float
) -> None:
    """Missing predictions remain aligned with available posterior samples."""
    store = storage.CampaignStore(
        protein_length=3, location=tmp_path, max_size_mb=max_size_mb
    )
    store.store_samples(
        torch.tensor([[1, 1, 1], [2, 2, 2]]),
        parent_id=0,
        ancestor_id=0,
        cycle_id=1,
    )
    store.store_objective(
        2,
        predictors.Prediction(
            mean=torch.tensor(2.0),
            std=torch.tensor(0.5),
            samples=torch.tensor([1.0, 3.0]),
        ),
    )

    (batch,) = list(store.get_candidate_stream())
    order = batch.seq_id.argsort()

    assert batch.objective_samples is not None
    assert batch.objective_std is not None
    torch.testing.assert_close(
        batch.objective_samples[order],
        torch.tensor([[float("nan"), float("nan")], [1.0, 3.0]]),
        equal_nan=True,
    )
    torch.testing.assert_close(
        batch.objective_std[order], torch.tensor([float("nan"), 0.5]), equal_nan=True
    )


def test_base_and_candidate_sequences_share_the_sequence_id_counter(
    store: storage.CampaignStore,
) -> None:
    """Registering base sequences and candidates assigns disjoint sequence ids."""
    store.store_base_sequences(torch.tensor([[1, 1, 1]]))
    cycle_id = store.get_next_cycle_id()
    store.store_samples(
        torch.tensor([[2, 2, 2]]), parent_id=1, ancestor_id=1, cycle_id=cycle_id
    )
    store.store_base_sequences(torch.tensor([[3, 3, 3]]))

    assert [seq_id for seq_id, _ in store.iter_base_sequences()] == [1, 3]
    assert _sorted_samples(store) == [(2, [2, 2, 2])]


def test_cycle_ids_start_at_one_and_advance_independently_of_base_sequences(
    store: storage.CampaignStore,
) -> None:
    """Registering base sequences does not consume candidate cycle ids."""
    first = store.get_next_cycle_id()
    store.store_base_sequences(torch.tensor([[1, 1, 1]]))
    second = store.get_next_cycle_id()

    assert (first, second) == (1, 2)


@pytest.mark.parametrize("seq_ids", [[], [1], [1, 99]])
def test_candidates_from_ids_with_only_base_or_unknown_ids_returns_empty_batch(
    store: storage.CampaignStore, seq_ids: list[int]
) -> None:
    """Base sequence ids are omitted just like unknown candidate ids."""
    store.store_base_sequences(torch.tensor([[1, 1, 1]]))

    batch = store.get_candidate_stream().candidates_from_ids(seq_ids)

    assert batch.seq_id.tolist() == []
    assert batch.token_ids.shape == (0, 3)
