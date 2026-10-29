"""Tests for the campaign runner in protuna.optim.pipeline."""

from __future__ import annotations

from collections.abc import Sequence
import pathlib

import lmdb
import pytest
import torch

from protuna import data
from protuna import inference
from protuna.optim import acquisitions
from protuna.optim import candidates
from protuna.optim import constraints
from protuna.optim import pipeline
from protuna.optim import predictors
from protuna.optim import spaces
from protuna.optim import storage
from protuna.optim.samplers import base
from protuna.optim.samplers import baselines
from protuna.typing import BatchTokenIDs, SequenceLike

# Wild type of the campaign, mutable to "A" at site 1 and to "C" at site 2, so
# the single mutants of the first cycle are exactly "MAT" and "MKC".
_WILD_TYPE = "MKT"
_ALLOWED_AT = {1: "KA", 2: "TC"}


def _space(init_sequence: str, allowed_at: dict[int, str]) -> spaces.SearchSpace:
    """Build a space allowing exactly ``allowed_at[pos]`` at each named position.

    Positions missing from ``allowed_at`` keep only their wild-type residue, which
    :class:`~protuna.optim.spaces.search.SearchSpace` restores on construction.
    """
    allowed = torch.zeros(
        len(init_sequence), len(data.tokenizer.aa_token_ids), dtype=torch.bool
    )
    for position, residues in allowed_at.items():
        allowed[position, data.tokenizer.encode(residues)] = True
    return spaces.SearchSpace(data.tokenizer.encode(init_sequence), allowed)


def _design_space(
    init_spaces: list[spaces.SearchSpace], n_subspaces: int = 1
) -> spaces.DesignSpace:
    """Bundle spaces into a design space cut up by a partition sampler."""
    return spaces.DesignSpace(
        init_spaces=init_spaces,
        space_sampler=spaces.PartitionSampler(n_partitions=n_subspaces),
    )


def _sample_rows(
    store: storage.CampaignStore, cycle_id: int | None = None
) -> list[tuple[int | None, str, int | None, int | None]]:
    """Collect a cycle's records as ``(seq_id, sequence, parent_id, ancestor_id)``."""
    seq_ids = [
        seq_id
        for seq_id, _ in (
            store.iter_base_sequences()
            if cycle_id == 0
            else store.iter_samples(cycle_id=cycle_id, exclude_failed=False)
        )
    ]
    rows = store.get_samples_unordered(
        seq_ids, columns=["seq_id", "parent_id", "ancestor_id", "token_ids"]
    )
    return sorted(
        (
            row.seq_id,
            data.tokenizer.decode(row.token_ids),  # ty: ignore[no-matching-overload]
            row.parent_id,
            row.ancestor_id,
        )
        for row in rows
    )


def _recorded_cycles(store: storage.CampaignStore) -> list[int]:
    """List the distinct cycle ids the store holds, in ascending order."""
    seq_ids = [
        seq_id
        for seq_id, _ in [
            *store.iter_base_sequences(),
            *store.iter_samples(exclude_failed=False),
        ]
    ]
    rows = store.get_samples_unordered(seq_ids, columns=["cycle_id"])
    return sorted({int(row.cycle_id) for row in rows})  # ty: ignore[invalid-argument-type]


class _RecordingSampler(baselines.SiteSaturationSampler):
    """Site saturation sampler that records the spaces and devices it was given.

    Attributes:
        spaces: Every search space the pipeline asked it to propose in, in order.
        devices: Every device it was asked to load onto.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spaces: list[spaces.SearchSpace] = []
        self.devices: list[torch.device | Sequence[torch.device] | None] = []

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        self.devices.append(device)
        super().load(device)

    def sample(self, spaces: SequenceLike[spaces.SearchSpace]) -> list[BatchTokenIDs]:
        self.spaces.extend(spaces)
        return super().sample(spaces)


class _LookupPredictor(
    inference.InferenceModel[predictors.PredictionRequest, predictors.Prediction]
):
    """Predictor returning a score looked up per sequence, defaulting to zero.

    Attributes:
        requests: Every request it answered, in order.
        devices: Every device it was asked to load onto.
    """

    def __init__(self, scores: dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self.scores = scores
        self.requests: list[predictors.PredictionRequest] = []
        self.devices: list[torch.device | Sequence[torch.device] | None] = []

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        self.devices.append(device)

    def release(self) -> None:
        pass

    def forward(
        self, requests: SequenceLike[predictors.PredictionRequest]
    ) -> list[predictors.Prediction]:
        self.requests.extend(requests)
        return [
            predictors.Prediction(
                mean=torch.tensor(
                    self.scores.get(data.tokenizer.decode(request.token_ids), 0.0)
                ),
                std=torch.tensor(0.1) if request.return_std else None,
                samples=torch.tensor([0.0, 1.0]) if request.return_samples else None,
            )
            for request in requests
        ]


class _RecordingRecycler(acquisitions.Greedy):
    """Record the cycles offered for selecting the next parent.

    Attributes:
        cycles: Cycle ids seen in each selection call.
    """

    def __init__(self, **kwargs):
        super().__init__(budget=1, **kwargs)
        self.cycles: list[list[int]] = []

    def select(self, stream: candidates.CandidateStream) -> list[int]:
        self.cycles.append(
            sorted(
                {cycle_id for batch in stream for cycle_id in batch.cycle_id.tolist()}
            )
        )
        return super().select(stream)


class _UncertainGreedy(acquisitions.Greedy):
    """Greedy selection that also asks for the uncertainty of every prediction."""

    requires_std = True
    requires_samples = True


@pytest.fixture
def store(tmp_path: pathlib.Path) -> storage.CampaignStore:
    """A campaign store for the length-3 proteins of these tests."""
    return storage.CampaignStore(protein_length=3, location=tmp_path)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROTUNA_HOME", str(tmp_path))


@pytest.fixture
def design_space() -> spaces.DesignSpace:
    """A design space over the single wild type, open at two sites."""
    return _design_space([_space(_WILD_TYPE, _ALLOWED_AT)])


def test_initial_sequences_are_recorded_as_cycle_zero(
    design_space: spaces.DesignSpace, store: storage.CampaignStore
) -> None:
    """The starting sequences are stored before any proposal, without a parent."""
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=_RecordingSampler(),
        objective=_LookupPredictor({}),
        acquisition=acquisitions.Greedy(budget=1),
    )

    assert _sample_rows(store, cycle_id=0) == []
    campaign.run(store=store)
    assert _sample_rows(store, cycle_id=0) == [(1, "MKT", 0, 0)]


def test_models_are_not_loaded_when_the_pipeline_is_built(
    design_space: spaces.DesignSpace, store: storage.CampaignStore
) -> None:
    """Defer loading the sampler, objective and constraints until the run."""
    candidate_sampler = _RecordingSampler()
    objective = _LookupPredictor({})
    constraint_model = _LookupPredictor({})

    pipeline.Pipeline(
        design_space=design_space,
        sampler=candidate_sampler,
        objective=objective,
        acquisition=acquisitions.Greedy(budget=1),
        hard_constraints={
            "stability": constraints.HardPredictionRangeConstraint(
                constraint_model, ge=0.5
            )
        },
    )

    assert (
        len(candidate_sampler.devices),
        len(objective.devices),
        len(constraint_model.devices),
    ) == (0, 0, 0)


def test_every_model_is_loaded_when_the_pipeline_runs(
    design_space: spaces.DesignSpace, store: storage.CampaignStore
) -> None:
    """Load the sampler, objective and constraint model once for a single cycle."""
    candidate_sampler = _RecordingSampler()
    objective = _LookupPredictor({})
    constraint_model = _LookupPredictor({"MAT": 1.0, "MKC": 1.0})
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=candidate_sampler,
        objective=objective,
        acquisition=acquisitions.Greedy(budget=1),
        hard_constraints={
            "stability": constraints.HardPredictionRangeConstraint(
                constraint_model, ge=0.5
            )
        },
    )

    campaign.run(store=store)

    assert (
        len(candidate_sampler.devices),
        len(objective.devices),
        len(constraint_model.devices),
    ) == (1, 1, 1)


def test_run_returns_the_sequences_the_acquisition_selected(
    design_space: spaces.DesignSpace, store: storage.CampaignStore
) -> None:
    """The campaign answers with the decoded top-scoring candidates."""
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=_RecordingSampler(),
        objective=_LookupPredictor({"MAT": 1.0, "MKC": 0.5}),
        acquisition=acquisitions.Greedy(budget=1),
    )

    assert data.tokenizer.decode(
        list(campaign.run(store=store).selection.token_ids.unbind())
    ) == ["MAT"]


def test_proposals_are_recorded_against_their_parent_and_ancestor(
    design_space: spaces.DesignSpace, store: storage.CampaignStore
) -> None:
    """Every candidate of the first cycle points back at the sequence it came from."""
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=_RecordingSampler(),
        objective=_LookupPredictor({}),
        acquisition=acquisitions.Greedy(budget=1),
    )

    campaign.run(store=store)

    assert _sample_rows(store, cycle_id=1) == [(2, "MAT", 1, 1), (3, "MKC", 1, 1)]


def test_each_starting_sequence_keeps_its_own_ancestor(
    store: storage.CampaignStore,
) -> None:
    """Candidates are traced back to the starting sequence they were mutated from."""
    design_space = _design_space([_space("MKT", {1: "KA"}), _space("WYT", {2: "TC"})])
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=_RecordingSampler(),
        objective=_LookupPredictor({}),
        acquisition=acquisitions.Greedy(budget=1),
    )

    campaign.run(store=store)

    assert _sample_rows(store, cycle_id=1) == [(3, "MAT", 1, 1), (4, "WYC", 2, 2)]


def test_candidates_are_proposed_in_one_subspace_per_partition(
    store: storage.CampaignStore,
) -> None:
    """The design space is cut up before proposing, one open site per subspace here."""
    candidate_sampler = _RecordingSampler()
    campaign = pipeline.Pipeline(
        design_space=_design_space([_space(_WILD_TYPE, _ALLOWED_AT)], n_subspaces=2),
        sampler=candidate_sampler,
        objective=_LookupPredictor({}),
        acquisition=acquisitions.Greedy(budget=1),
    )

    campaign.run(store=store)

    assert [space.mutable_positions.tolist() for space in candidate_sampler.spaces] == [
        [False, True, False],
        [False, False, True],
    ]


def test_multiple_cycles_require_a_recycler(
    design_space: spaces.DesignSpace, store: storage.CampaignStore
) -> None:
    """Reject multiple proposal cycles without a recycling acquisitions."""
    with pytest.raises(
        ValueError, match="Must provide a recycling acquisition for num_cycles > 1"
    ):
        pipeline.Pipeline(
            design_space=design_space,
            sampler=_RecordingSampler(),
            objective=_LookupPredictor({}),
            acquisition=acquisitions.Greedy(budget=1),
            num_cycles=3,
        )


def test_run_writes_the_campaign_out_to_parquet(
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
    tmp_path: pathlib.Path,
) -> None:
    """Everything still buffered is flushed to disk before the run returns."""
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=_RecordingSampler(),
        objective=_LookupPredictor({}),
        acquisition=acquisitions.Greedy(budget=1),
    )

    campaign.run(store=store)

    assert [path.name for path in tmp_path.glob("pipelines/*/samples_*.parquet")] == [
        "samples_part00001.parquet"
    ]


def test_candidates_failing_a_constraint_are_never_scored(
    design_space: spaces.DesignSpace, store: storage.CampaignStore
) -> None:
    """The objective only sees the candidates every constraint accepted."""
    objective = _LookupPredictor({"MAT": 0.5, "MKC": 9.0})
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=_RecordingSampler(),
        objective=objective,
        acquisition=acquisitions.Greedy(budget=1),
        hard_constraints={
            "stability": constraints.HardPredictionRangeConstraint(
                _LookupPredictor({"MAT": 1.0, "MKC": 0.0}), ge=0.5
            )
        },
    )

    campaign.run(store=store)

    assert [
        data.tokenizer.decode(request.token_ids) for request in objective.requests
    ] == ["MAT"]


def test_candidates_failing_a_constraint_cannot_be_selected(
    design_space: spaces.DesignSpace, store: storage.CampaignStore
) -> None:
    """A rejected candidate is dropped even when it would score best."""
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=_RecordingSampler(),
        objective=_LookupPredictor({"MAT": 0.5, "MKC": 9.0}),
        acquisition=acquisitions.Greedy(budget=1),
        hard_constraints={
            "stability": constraints.HardPredictionRangeConstraint(
                _LookupPredictor({"MAT": 1.0, "MKC": 0.0}), ge=0.5
            )
        },
    )

    assert data.tokenizer.decode(
        list(campaign.run(store=store).selection.token_ids.unbind())
    ) == ["MAT"]


def test_default_store_registers_soft_constraints_before_objective_scoring(
    design_space: spaces.DesignSpace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Score zero-feasibility candidates before acquisition filters them."""
    monkeypatch.setenv("PROTUNA_HOME", str(tmp_path))
    objective = _LookupPredictor({"MAT": 1.0, "MKC": 0.5})
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=_RecordingSampler(),
        objective=objective,
        acquisition=acquisitions.Greedy(budget=2),
        soft_constraints={
            "stability": constraints.SoftInequalityConstraint(
                _LookupPredictor({"MAT": -1000.0, "MKC": 0.0}),
                threshold=0.0,
                temperature=1.0,
            )
        },
    )

    assert data.tokenizer.decode(list(campaign.run().selection.token_ids.unbind())) == [
        "MKC"
    ]
    assert sorted(
        data.tokenizer.decode(request.token_ids) for request in objective.requests
    ) == ["MAT", "MKC"]


def test_soft_constraints_only_evaluate_candidates_passing_hard_constraints(
    design_space: spaces.DesignSpace, tmp_path: pathlib.Path
) -> None:
    """Run hard filters before recording soft feasibility in a supplied store."""
    store = storage.CampaignStore(
        protein_length=3, location=tmp_path, soft_constraint_names=("solubility",)
    )
    soft_model = _LookupPredictor({})
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=_RecordingSampler(),
        objective=_LookupPredictor({}),
        acquisition=acquisitions.Greedy(budget=1),
        hard_constraints={
            "stability": constraints.HardPredictionRangeConstraint(
                _LookupPredictor({"MAT": 1.0, "MKC": 0.0}), ge=0.5
            )
        },
        soft_constraints={
            "solubility": constraints.SoftInequalityConstraint(
                soft_model, threshold=0.0, temperature=1.0
            )
        },
    )

    campaign.run(store=store)

    assert [
        data.tokenizer.decode(request.token_ids) for request in soft_model.requests
    ] == ["MAT"]
    assert store.soft_constraint_store.get_lf().collect().rows() == [
        (2, "solubility", 0.5)
    ]


@pytest.mark.parametrize(
    ("acquisition_cls", "expected"),
    [
        (acquisitions.Greedy, {(False, False)}),
        (_UncertainGreedy, {(True, True)}),
    ],
)
def test_prediction_requests_ask_for_what_the_acquisition_needs(
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
    acquisition_cls: type[acquisitions.Greedy],
    expected: set[tuple[bool, bool]],
) -> None:
    """The uncertainty asked of the objective follows the acquisition's requirements."""
    objective = _LookupPredictor({})
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=_RecordingSampler(),
        objective=objective,
        acquisition=acquisition_cls(budget=1),
    )

    campaign.run(store=store)

    assert {
        (request.return_std, request.return_samples) for request in objective.requests
    } == expected


def test_recycled_parent_seeds_the_proposals_of_the_next_cycle(
    design_space: spaces.DesignSpace, store: storage.CampaignStore
) -> None:
    """The second cycle mutates the candidate the recycler picked, not the wild type."""
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=_RecordingSampler(),
        objective=_LookupPredictor({"MAT": 1.0, "MKC": 0.5, "MAC": 2.0, "MKT": 0.1}),
        acquisition=acquisitions.Greedy(budget=1),
        recycler=acquisitions.Greedy(budget=1),
        num_cycles=2,
    )

    campaign.run(store=store)

    assert _sample_rows(store, cycle_id=2) == [(4, "MAC", 2, 1)]


def test_recycling_selects_across_every_cycle(
    design_space: spaces.DesignSpace, store: storage.CampaignStore
) -> None:
    """The acquisition ranks the candidates of all cycles together."""
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=_RecordingSampler(),
        objective=_LookupPredictor({"MAT": 1.0, "MKC": 0.5, "MAC": 2.0, "MKT": 0.1}),
        acquisition=acquisitions.Greedy(budget=1),
        recycler=acquisitions.Greedy(budget=1),
        num_cycles=2,
    )

    assert data.tokenizer.decode(
        list(campaign.run(store=store).selection.token_ids.unbind())
    ) == ["MAC"]


def test_locked_mutations_are_not_reopened_on_recycle(
    design_space: spaces.DesignSpace, store: storage.CampaignStore
) -> None:
    """A site already mutated stays fixed, so the wild-type residue never comes back."""
    sampler = _RecordingSampler()
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=sampler,
        objective=_LookupPredictor({"MAT": 1.0, "MKC": 0.5, "MAC": 2.0, "MKT": 0.1}),
        acquisition=acquisitions.Greedy(budget=1),
        recycler=acquisitions.Greedy(budget=1),
        num_cycles=2,
        lock_mutations_on_recycle=True,
    )

    campaign.run(store=store)

    assert _sample_rows(store, cycle_id=2) == [(4, "MAC", 2, 1)]
    assert sampler.spaces[1].mutable_positions.tolist() == [False, False, True]


@pytest.mark.parametrize(
    ("num_cycles", "expected"), [(1, [0, 1]), (2, [0, 1, 2]), (3, [0, 1, 2])]
)
def test_number_of_proposal_cycles_follows_num_cycles(
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
    num_cycles: int,
    expected: list[int],
) -> None:
    """Record proposal cycles until the cycle budget or unique designs run out."""
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=_RecordingSampler(),
        objective=_LookupPredictor({"MAT": 1.0, "MAC": 2.0}),
        acquisition=acquisitions.Greedy(budget=1),
        recycler=acquisitions.Greedy(budget=1),
        num_cycles=num_cycles,
    )

    campaign.run(store=store)

    assert _recorded_cycles(store) == expected


@pytest.mark.parametrize(
    ("num_cycles", "expected"), [(1, []), (2, [[1]]), (3, [[1], [2]])]
)
def test_recycler_selects_from_each_completed_cycle_except_the_last(
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
    num_cycles: int,
    expected: list[list[int]],
) -> None:
    """Offer only the current cycle to the recycler before starting the next one."""
    campaign_recycler = _RecordingRecycler()
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=_RecordingSampler(),
        objective=_LookupPredictor({"MAT": 1.0, "MAC": 2.0}),
        acquisition=acquisitions.Greedy(budget=1),
        recycler=campaign_recycler,
        num_cycles=num_cycles,
    )

    campaign.run(store=store)

    assert campaign_recycler.cycles == expected


def test_nothing_is_proposed_when_num_cycles_is_zero(
    design_space: spaces.DesignSpace, store: storage.CampaignStore
) -> None:
    """A campaign with no cycles selects nothing and records only its wild types."""
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=_RecordingSampler(),
        objective=_LookupPredictor({}),
        acquisition=acquisitions.Greedy(budget=1),
        recycler=acquisitions.Greedy(budget=1),
        num_cycles=0,
    )

    assert (
        data.tokenizer.decode(
            list(campaign.run(store=store).selection.token_ids.unbind())
        )
        == []
    )


def test_store_defaults_to_the_protuna_cache_directory(
    design_space: spaces.DesignSpace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a store the campaign records itself under the protuna cache."""
    monkeypatch.setenv("PROTUNA_HOME", str(tmp_path))
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=_RecordingSampler(),
        objective=_LookupPredictor({"MAT": 1.0}),
        acquisition=acquisitions.Greedy(budget=1),
    )

    campaign.run()

    assert [
        path.name for path in tmp_path.glob("protuna/pipelines/*/samples_*.parquet")
    ] == ["samples_part00001.parquet"]


def test_pipeline_without_objective_returns_generated_candidates(
    design_space: spaces.DesignSpace, store: storage.CampaignStore
) -> None:
    """Default acquisition returns unscored candidates and excludes base sequences."""
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=_RecordingSampler(),
    )

    result = campaign.run(store=store)

    assert sorted(data.tokenizer.decode(list(result.selection.token_ids.unbind()))) == [
        "MAT",
        "MKC",
    ]


@pytest.mark.parametrize(("num_cycles", "expected"), [(1, []), (2, [[2]])])
def test_recycling_uses_round_position_when_cycle_ids_start_above_one(
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
    num_cycles: int,
    expected: list[list[int]],
) -> None:
    """An advanced cycle counter must not change which rounds need recycling."""
    store.get_next_cycle_id()
    campaign_recycler = _RecordingRecycler()
    campaign = pipeline.Pipeline(
        design_space=design_space,
        sampler=_RecordingSampler(),
        objective=_LookupPredictor({"MAT": 1.0, "MAC": 2.0}),
        acquisition=acquisitions.Greedy(budget=1),
        recycler=campaign_recycler,
        num_cycles=num_cycles,
    )

    campaign.run(store=store)

    assert campaign_recycler.cycles == expected


@pytest.mark.parametrize(
    ("evaluated", "expected"),
    [([], ["MAT", "MKC"]), (["MAT"], ["MKC"]), (["MAT", "MKC"], [])],
)
def test_evaluated_mutants_are_excluded_before_scoring(
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
    evaluated: list[str],
    expected: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Filter evaluated mutants and report the number removed from each batch."""
    objective = _LookupPredictor({})
    campaign = pipeline.Pipeline(
        design_space,
        _RecordingSampler(),
        objective=objective,
        evaluated_mutants=data.tokenizer.encode(evaluated),
    )

    with caplog.at_level("INFO", logger=pipeline.__name__):
        assert (
            sorted(
                data.tokenizer.decode(
                    list(campaign.run(store=store).selection.token_ids.unbind())
                )
            )
            == expected
        )

    assert (
        sorted(
            data.tokenizer.decode(request.token_ids) for request in objective.requests
        )
        == expected
    )
    assert (
        f"stored {len(expected)} candidates, excluded {len(evaluated)}" in caplog.text
    )
    assert [row[1] for row in _sample_rows(store, cycle_id=1)] == expected


def test_proposals_shared_by_starting_spaces_are_stored_once(
    store: storage.CampaignStore,
) -> None:
    """Keep one record and one objective evaluation for overlapping proposals."""
    objective = _LookupPredictor({})
    campaign = pipeline.Pipeline(
        _design_space([_space("MAT", _ALLOWED_AT), _space("MKC", _ALLOWED_AT)]),
        _RecordingSampler(batch_size=1),
        objective=objective,
    )

    assert sorted(
        data.tokenizer.decode(
            list(campaign.run(store=store).selection.token_ids.unbind())
        )
    ) == ["MAC", "MKT"]
    assert _sample_rows(store, cycle_id=1) == [(3, "MKT", 1, 1), (4, "MAC", 1, 1)]
    assert len(objective.requests) == 2


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_raw_sampler_proposals_are_deduplicated_before_evaluation(
    store: storage.CampaignStore, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """Exclude starting sequences and repeated proposals across batches and cycles."""
    sampler = _RecordingSampler(batch_size=1)
    proposals = torch.stack(data.tokenizer.encode(["MKT", "MAT", "MKC", "MKC", "MAC"]))

    sampled_spaces: list[spaces.SearchSpace] = []

    def sample(requests: SequenceLike[spaces.SearchSpace]) -> list[BatchTokenIDs]:
        sampled_spaces.extend(requests)
        return [proposals.to(device).clone() for _ in requests]

    monkeypatch.setattr(sampler, "sample", sample)
    objective = _LookupPredictor({})
    campaign = pipeline.Pipeline(
        _design_space([_space("MKT", _ALLOWED_AT), _space("MAT", _ALLOWED_AT)]),
        sampler,
        objective=objective,
        recycler=acquisitions.IdentityAcquisition(),
        num_cycles=2,
    )

    result = campaign.run(store=store)

    assert sorted(data.tokenizer.decode(list(result.selection.token_ids.unbind()))) == [
        "MAC",
        "MKC",
    ]
    assert _sample_rows(store, cycle_id=1) == [(3, "MKC", 1, 1), (4, "MAC", 1, 1)]
    assert _sample_rows(store, cycle_id=2) == []
    assert len(sampled_spaces) == 4
    assert len(objective.requests) == 2


@pytest.mark.parametrize("sequences", [["MKT", "WAT"], ["MKT", "MWT"], ["MK"]])
def test_invalid_sampler_proposals_are_rejected_before_storage_and_evaluation(
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
    monkeypatch: pytest.MonkeyPatch,
    sequences: list[str],
) -> None:
    """Reject forbidden fixed residues, forbidden mutations, and wrong lengths."""
    sampler = _RecordingSampler()

    def sample(requests: SequenceLike[spaces.SearchSpace]) -> list[BatchTokenIDs]:
        return [torch.stack(data.tokenizer.encode(sequences)) for _ in requests]

    monkeypatch.setattr(sampler, "sample", sample)
    objective = _LookupPredictor({})
    campaign = pipeline.Pipeline(design_space, sampler, objective=objective)

    with pytest.raises(ValueError, match="Sampler produced invalid candidates"):
        campaign.run(store=store)

    assert _sample_rows(store, cycle_id=1) == []
    assert objective.requests == []
    assert objective.devices == []


@pytest.mark.parametrize(
    "sampler", [baselines.SiteSaturationSampler(), baselines.BruteForceSampler()]
)
def test_empty_sampler_batches_pass_validation(
    store: storage.CampaignStore, sampler: base.Sampler
) -> None:
    """Accept empty proposal batches from immutable search spaces."""
    campaign = pipeline.Pipeline(_design_space([_space("MKT", {})]), sampler)

    result = campaign.run(store=store)

    assert result.selection.token_ids.shape == (0, 3)
    assert _sample_rows(store, cycle_id=1) == []


def test_empty_cycle_stops_without_loading_downstream_handlers(
    design_space: spaces.DesignSpace, store: storage.CampaignStore
) -> None:
    """Stop on exhaustion and acquire from the earlier cycles' unique candidates."""
    sampler = _RecordingSampler()
    objective = _LookupPredictor({"MAT": 1.0, "MAC": 2.0})
    constraint_model = _LookupPredictor({"MAT": 1.0, "MKC": 1.0, "MAC": 1.0})
    recycler = _RecordingRecycler()
    campaign = pipeline.Pipeline(
        design_space,
        sampler,
        objective=objective,
        hard_constraints={
            "stability": constraints.HardPredictionRangeConstraint(
                constraint_model, ge=0.5
            )
        },
        recycler=recycler,
        num_cycles=5,
    )

    assert sorted(
        data.tokenizer.decode(
            list(campaign.run(store=store).selection.token_ids.unbind())
        )
    ) == ["MAC", "MAT", "MKC"]
    assert len(sampler.devices) == 3
    assert len(objective.devices) == len(constraint_model.devices) == 2
    assert len(objective.requests) == len(constraint_model.requests) == 3
    assert recycler.cycles == [[1], [2]]
    assert _recorded_cycles(store) == [0, 1, 2]


def test_duplicate_starting_sequences_are_rejected_before_sampling(
    store: storage.CampaignStore,
) -> None:
    """Reject ambiguous ancestry even when duplicate parents have different masks."""
    sampler = _RecordingSampler()
    campaign = pipeline.Pipeline(
        _design_space([_space("MKT", {1: "KA"}), _space("MKT", {2: "TC"})]),
        sampler,
    )

    with pytest.raises(ValueError, match="Found 1/2 duplicates"):
        campaign.run(store=store)

    assert sampler.devices == []
    assert _sample_rows(store, cycle_id=0) == []


def test_each_cycle_loads_and_releases_its_current_design_space(
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply resource lifecycle calls to the recycled design space in later cycles."""
    loaded: list[spaces.DesignSpace] = []
    released: list[spaces.DesignSpace] = []

    def record_load(self: spaces.DesignSpace) -> None:
        loaded.append(self)

    def record_release(self: spaces.DesignSpace) -> None:
        released.append(self)

    monkeypatch.setattr(spaces.DesignSpace, "load", record_load)
    monkeypatch.setattr(spaces.DesignSpace, "release", record_release)
    campaign = pipeline.Pipeline(
        design_space,
        _RecordingSampler(),
        recycler=acquisitions.IdentityAcquisition(),
        num_cycles=2,
        release_handlers_after_use=True,
    )

    campaign.run(store=store)

    assert len(loaded) == len(released) == 2
    assert loaded[0] is released[0] is design_space
    assert loaded[1] is released[1]
    assert loaded[1] is not design_space


@pytest.mark.parametrize("fail", [False, True])
def test_fingerprint_index_is_removed_after_run(
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
) -> None:
    """Remove the temporary index after successful runs and model failures."""
    objective = _LookupPredictor({})

    def fail_forward(
        requests: SequenceLike[predictors.PredictionRequest],
    ) -> list[predictors.Prediction]:
        raise RuntimeError("objective failed")

    if fail:
        monkeypatch.setattr(objective, "forward", fail_forward)
    campaign = pipeline.Pipeline(design_space, _RecordingSampler(), objective=objective)

    if fail:
        with pytest.raises(RuntimeError, match="objective failed"):
            campaign.run(store=store)
    else:
        campaign.run(store=store)

    assert list((tmp_path / "protuna").iterdir()) == []


def test_fingerprint_index_retries_full_transactions_after_resizing() -> None:
    """Preserve duplicate detection when a full LMDB map aborts a batch."""
    index = pipeline._FingerprintIndex()
    try:
        index.env.set_mapsize(32768)
        initial_size = index.env.info()["map_size"]
        fingerprints = [number.to_bytes(16, "big") for number in range(4000)]
        assert index.add(fingerprints[:1]).tolist() == [True]

        accepted = index.add([*fingerprints, fingerprints[1]])

        assert accepted.tolist() == [False, *([True] * 3999), False]
        assert index.env.info()["map_size"] > initial_size
        assert not index.add(fingerprints).any()
        assert index.add([]).shape == (0,)
    finally:
        index.close()

    assert not pathlib.Path(index.index_path).exists()
    with pytest.raises(lmdb.Error):
        index.env.begin()


def test_output_preserves_ranking_evaluations_and_lineage(
    design_space: spaces.DesignSpace, store: storage.CampaignStore
) -> None:
    """Return the selected batch in score order with its evaluations and history."""
    campaign = pipeline.Pipeline(
        design_space,
        _RecordingSampler(),
        objective=_LookupPredictor({"MAT": 2.0, "MKC": 1.0}),
        acquisition=_UncertainGreedy(budget=2),
    )

    output = campaign.run(store=store)

    assert output.store is store
    batch = output.selection
    assert data.tokenizer.decode(list(batch.token_ids.unbind())) == ["MKC", "MAT"]
    assert batch.seq_id.tolist() == [3, 2]
    assert batch.parent_id.tolist() == batch.ancestor_id.tolist() == [1, 1]
    assert batch.cycle_id.tolist() == [1, 1]
    torch.testing.assert_close(batch.objective_mean, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(batch.objective_std, torch.tensor([0.1, 0.1]))
    torch.testing.assert_close(
        batch.objective_samples, torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    )


def test_repeated_runs_have_independent_history_and_recycling_state(
    design_space: spaces.DesignSpace,
) -> None:
    """Start each run from the initial design space with a fresh history."""
    campaign = pipeline.Pipeline(
        design_space,
        _RecordingSampler(),
        recycler=acquisitions.IdentityAcquisition(),
        num_cycles=2,
    )

    first = campaign.run()
    second = campaign.run()

    assert first.store is not second.store
    assert _sample_rows(first.store) == _sample_rows(second.store)
    assert _recorded_cycles(first.store) == _recorded_cycles(second.store) == [0, 1, 2]
    assert sorted(
        data.tokenizer.decode(list(first.selection.token_ids.unbind()))
    ) == sorted(data.tokenizer.decode(list(second.selection.token_ids.unbind())))


def test_initialization_failure_closes_fingerprint_index(
    store: storage.CampaignStore,
    tmp_path: pathlib.Path,
) -> None:
    """Preserve initialization errors and remove the temporary fingerprint index."""
    campaign = pipeline.Pipeline(
        _design_space([_space("MKT", {1: "KA"}), _space("MKT", {2: "TC"})]),
        _RecordingSampler(),
    )

    with pytest.raises(ValueError, match="Found 1/2 duplicates"):
        campaign.run(store=store)

    assert list((tmp_path / "protuna").iterdir()) == []
