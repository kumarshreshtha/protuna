from collections.abc import Callable, Iterator, Sequence
import os
import pathlib
import sys

from jaxtyping import Float
import pytest
import ray.cloudpickle
import ray.exceptions
import torch

from protuna import data
from protuna import inference
from protuna.inference.serve import api
from protuna.inference.serve import pool as serving_pool
from protuna.optim import acquisitions
from protuna.optim import constraints
from protuna.optim import pipeline
from protuna.optim import predictors
from protuna.optim import spaces
from protuna.optim import storage
from protuna.optim.samplers import baselines
from protuna.optim.samplers import mlm
from protuna.typing import BatchTokenIDs, SequenceLike

pytestmark = pytest.mark.integration


class _Lifecycle:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loaded = False
        self.driver_pid = os.getpid()

    def load(self, device: torch.device | Sequence[torch.device] | None = None) -> None:
        """Make the test handler ready for inference."""
        self.loaded = True

    def release(self) -> None:
        """Make inference fail until the handler is loaded again."""
        self.loaded = False

    def _check_ready(self) -> None:
        if not self.loaded:
            raise RuntimeError("test handler is not loaded")
        assert os.getpid() != self.driver_pid, "inference must run in a Ray worker"


class _Sampler(_Lifecycle, baselines.SiteSaturationSampler):
    def sample(self, spaces: SequenceLike[spaces.SearchSpace]) -> list[BatchTokenIDs]:
        """Enumerate single mutants in a loaded Ray worker."""
        self._check_ready()
        return super().sample(spaces)


class _Predictor(
    _Lifecycle,
    inference.InferenceModel[predictors.PredictionRequest, predictors.Prediction],
):
    def __init__(
        self,
        scores: dict[str, float],
        expected_flags: tuple[bool, bool] = (False, False),
    ):
        super().__init__()
        self.scores = scores
        self.expected_flags = expected_flags

    def forward(
        self, requests: SequenceLike[predictors.PredictionRequest]
    ) -> list[predictors.Prediction]:
        """Return deterministic predictions and validate the request flags."""
        self._check_ready()
        outputs = []
        for request in requests:
            assert (request.return_std, request.return_samples) == self.expected_flags
            score = self.scores[data.tokenizer.decode(request.token_ids)]
            outputs.append(
                predictors.Prediction(
                    torch.tensor(score),
                    std=torch.tensor(0.25) if request.return_std else None,
                    samples=torch.tensor([score - 0.25, score + 0.25])
                    if request.return_samples
                    else None,
                )
            )
        return outputs


class _LanguageModel(
    _Lifecycle,
    inference.InferenceModel[inference.MLMLogitsRequest, Float[torch.Tensor, "l v"]],
):
    def forward(
        self, requests: SequenceLike[inference.MLMLogitsRequest]
    ) -> list[Float[torch.Tensor, "l v"]]:
        """Assign fixed logits to alanine and cysteine at every masked site."""
        self._check_ready()
        outputs = []
        for request in requests:
            assert not request.masked_positions_only
            assert not request.aa_tokens_only
            assert (request.token_ids == data.tokenizer.mask_token_id).any()
            logits = torch.zeros(
                len(request.token_ids), len(data.tokenizer.ordered_tokens)
            )
            logits[:, data.tokenizer.encode("A")] = 3.0
            logits[:, data.tokenizer.encode("C")] = 5.0
            outputs.append(logits)
        return outputs


class _UncertainGreedy(acquisitions.Greedy):
    requires_std = True
    requires_samples = True


class _LogitsModel(
    _Lifecycle,
    inference.InferenceModel[inference.MLMLogitsRequest, Float[torch.Tensor, "l v"]],
):
    def forward(
        self, requests: SequenceLike[inference.MLMLogitsRequest]
    ) -> list[Float[torch.Tensor, "l v"]]:
        """Return logits making the allowed alanine and cysteine mutations certain."""
        self._check_ready()
        outputs = []
        for request in requests:
            assert request.masked_positions_only
            assert request.aa_tokens_only
            assert (request.token_ids == data.tokenizer.mask_token_id).any()
            logits = torch.zeros(len(request.token_ids), 20)
            logits[:, data.tokenizer.encode("A")] = 3.0
            logits[:, data.tokenizer.encode("C")] = 5.0
            outputs.append(logits[request.token_ids == data.tokenizer.mask_token_id])
        return outputs


class _FailingPredictor(_Predictor):
    def forward(
        self, requests: SequenceLike[predictors.PredictionRequest]
    ) -> list[predictors.Prediction]:
        """Raise a recognizable error inside a loaded Ray worker."""
        self._check_ready()
        raise RuntimeError("distributed objective failed")


@pytest.fixture(scope="module", autouse=True)
def _serialize_test_handlers(_ray_cluster: None) -> Iterator[None]:
    # Test modules are not installed in workers; serialize their classes by value.
    module = sys.modules[__name__]
    ray.cloudpickle.register_pickle_by_value(module)
    yield
    ray.cloudpickle.unregister_pickle_by_value(module)


@pytest.fixture(scope="module", params=["queue", "routed"])
def pool_kind(request: pytest.FixtureRequest) -> str:
    """Group pipeline tests by serving mode to reuse their sampler workers."""
    return request.param


@pytest.fixture(scope="module")
def _sampler_pool(
    _serialize_test_handlers: None, pool_kind: str
) -> Iterator[serving_pool.ModelPool]:
    factory = api.serve_queue_pool if pool_kind == "queue" else api.serve_routed_pool
    pool = factory(
        _Sampler(),
        num_replicas=2,
        max_batch_size=2,
        max_wait_ms=1,
        actor_options={"num_cpus": 0.1},
    )
    try:
        yield pool
    finally:
        pool.shutdown()


@pytest.fixture
def distributed_sampler(
    _sampler_pool: serving_pool.ModelPool,
) -> serving_pool.ModelPool:
    """Reuse sampler workers while requiring each pipeline to load them afresh."""
    # A previous pipeline may leave the handler loaded. Reset its lifecycle so
    # sharing workers cannot hide a missing load() in the next test.
    _sampler_pool.release()
    return _sampler_pool


@pytest.fixture
def distributed(
    serve_pool: Callable[..., serving_pool.ModelPool], pool_kind: str
) -> Callable[..., serving_pool.ModelPool]:
    """Serve lightweight handlers through either public pool API."""

    def _serve(handler: inference.Handler) -> serving_pool.ModelPool:
        return serve_pool(
            handler,
            pool_kind=pool_kind,
            num_replicas=2,
            max_batch_size=2,
            max_wait_ms=1,
            actor_options={"num_cpus": 0.1},
        )

    return _serve


@pytest.fixture
def design_space() -> spaces.DesignSpace:
    """Allow exactly MAT and MKC as single mutants of MKT."""
    allowed = torch.zeros(3, 20, dtype=torch.bool)
    allowed[1, data.tokenizer.encode("A")] = True
    allowed[2, data.tokenizer.encode("C")] = True
    return spaces.DesignSpace(
        [spaces.SearchSpace(data.tokenizer.encode("MKT"), allowed)]
    )


@pytest.fixture
def store(tmp_path: pathlib.Path) -> storage.CampaignStore:
    """Keep campaign outputs isolated."""
    return storage.CampaignStore(protein_length=3, location=tmp_path)


def _proposals(
    store: storage.CampaignStore, cycle_id: int
) -> list[tuple[str, int, int]]:
    ids = [
        seq_id
        for seq_id, _ in store.iter_samples(cycle_id=cycle_id, exclude_failed=False)
    ]
    rows = []
    for row in store.get_samples_unordered(
        ids, columns=["token_ids", "parent_id", "ancestor_id"]
    ):
        assert row.token_ids is not None
        assert row.parent_id is not None
        assert row.ancestor_id is not None
        rows.append(
            (data.tokenizer.decode(row.token_ids), row.parent_id, row.ancestor_id)
        )
    return sorted(rows)


def test_distributed_constraints_filter_candidates_before_the_objective(
    distributed: Callable[..., serving_pool.ModelPool],
    distributed_sampler: serving_pool.ModelPool,
    design_space: spaces.DesignSpace,
    tmp_path: pathlib.Path,
) -> None:
    """Filter failed candidates before sending requests to the remote objective."""
    store = storage.CampaignStore(
        protein_length=3, location=tmp_path, soft_constraint_names=["solubility"]
    )
    hard = constraints.HardPredictionRangeConstraint(
        _Predictor({"MAT": 1.0, "MKC": 0.0}), ge=0.5
    )
    soft = constraints.SoftInequalityConstraint(
        _Predictor({"MAT": 0.0}), threshold=0.0, temperature=1.0
    )
    campaign = pipeline.Pipeline(
        design_space,
        distributed_sampler,
        objective=distributed(_Predictor({"MAT": 4.0})),
        acquisition=acquisitions.Greedy(budget=1),
        hard_constraints={"stability": distributed(hard)},
        soft_constraints={"solubility": distributed(soft)},
    )

    selected = campaign.run(store=store)

    assert data.tokenizer.decode(list(selected.selection.token_ids.unbind())) == ["MAT"]
    assert _proposals(store, 1) == [("MAT", 1, 1), ("MKC", 1, 1)]
    batch = next(iter(store.get_candidate_stream()))
    assert data.tokenizer.decode(list(batch.token_ids.unbind())) == ["MAT"]
    torch.testing.assert_close(batch.objective_mean, torch.tensor([4.0]))
    assert batch.feasibility is not None
    torch.testing.assert_close(batch.feasibility["solubility"], torch.tensor([0.5]))


def test_distributed_objective_preserves_uncertainty_and_posterior_samples(
    distributed: Callable[..., serving_pool.ModelPool],
    distributed_sampler: serving_pool.ModelPool,
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
) -> None:
    """Carry requested uncertainty through Ray and campaign storage."""
    campaign = pipeline.Pipeline(
        design_space,
        distributed_sampler,
        objective=distributed(_Predictor({"MAT": 4.0, "MKC": 1.0}, (True, True))),
        acquisition=_UncertainGreedy(budget=1),
    )

    selected = campaign.run(store=store)

    assert data.tokenizer.decode(list(selected.selection.token_ids.unbind())) == ["MAT"]
    batch = store.get_candidate_stream().candidates_from_ids([2, 3])
    assert data.tokenizer.decode(list(batch.token_ids.unbind())) == ["MAT", "MKC"]
    torch.testing.assert_close(batch.objective_mean, torch.tensor([4.0, 1.0]))
    torch.testing.assert_close(batch.objective_std, torch.tensor([0.25, 0.25]))
    torch.testing.assert_close(
        batch.objective_samples, torch.tensor([[3.75, 4.25], [0.75, 1.25]])
    )


@pytest.mark.parametrize("release_handlers", [False, True])
def test_distributed_handlers_recycle_candidates_and_reload_between_cycles(
    distributed: Callable[..., serving_pool.ModelPool],
    distributed_sampler: serving_pool.ModelPool,
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
    release_handlers: bool,
) -> None:
    """Use recycled parents after optionally releasing and reloading replicas."""
    campaign = pipeline.Pipeline(
        design_space,
        distributed_sampler,
        objective=distributed(_Predictor({"MAT": 4.0, "MKC": 1.0, "MAC": 6.0})),
        acquisition=acquisitions.Greedy(budget=1),
        recycler=acquisitions.Greedy(budget=1),
        num_cycles=2,
        lock_mutations_on_recycle=True,
        release_handlers_after_use=release_handlers,
    )

    selected = campaign.run(store=store)

    assert data.tokenizer.decode(list(selected.selection.token_ids.unbind())) == ["MAC"]
    assert _proposals(store, 2) == [("MAC", 2, 1)]


def test_pipeline_releases_distributed_handlers_after_use(
    distributed: Callable[..., serving_pool.ModelPool],
    distributed_sampler: serving_pool.ModelPool,
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
) -> None:
    """Release remote sampler and objective resources when requested."""
    sampler = distributed_sampler
    objective = distributed(_Predictor({"MAT": 4.0, "MKC": 1.0}))
    campaign = pipeline.Pipeline(
        design_space,
        sampler,
        objective=objective,
        release_handlers_after_use=True,
    )

    campaign.run(store=store)

    with pytest.raises(ray.exceptions.RayTaskError, match="test handler is not loaded"):
        sampler([design_space.init_spaces[0]])
    with pytest.raises(ray.exceptions.RayTaskError, match="test handler is not loaded"):
        objective([predictors.PredictionRequest(data.tokenizer.encode("MAT"))])


def test_local_predictor_scores_candidates_with_a_distributed_language_model(
    distributed: Callable[..., serving_pool.ModelPool],
    distributed_sampler: serving_pool.ModelPool,
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
) -> None:
    """Compute masked-marginal scores through a remote language model."""
    objective = predictors.MaskedMarginalPredictor(
        data.tokenizer.encode("MKT"), distributed(_LanguageModel()), batch_size=1
    )
    campaign = pipeline.Pipeline(
        design_space,
        distributed_sampler,
        objective=objective,
        acquisition=acquisitions.Greedy(budget=1),
    )

    selected = campaign.run(store=store)

    assert data.tokenizer.decode(list(selected.selection.token_ids.unbind())) == ["MKC"]
    batch = store.get_candidate_stream().candidates_from_ids([2, 3])
    torch.testing.assert_close(batch.objective_mean, torch.tensor([3.0, 5.0]))


def test_local_constraint_loads_its_distributed_predictor(
    distributed: Callable[..., serving_pool.ModelPool],
    distributed_sampler: serving_pool.ModelPool,
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
) -> None:
    """Load a served predictor through the local constraint wrapper."""
    constraint = constraints.HardPredictionRangeConstraint(
        distributed(_Predictor({"MAT": 1.0, "MKC": 0.0})), ge=0.5
    )
    campaign = pipeline.Pipeline(
        design_space,
        distributed_sampler,
        objective=distributed(_Predictor({"MAT": 4.0})),
        hard_constraints={"stability": constraint},
    )

    selected = campaign.run(store=store)

    assert data.tokenizer.decode(list(selected.selection.token_ids.unbind())) == ["MAT"]


def test_pipeline_propagates_distributed_objective_errors(
    distributed: Callable[..., serving_pool.ModelPool],
    distributed_sampler: serving_pool.ModelPool,
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
) -> None:
    """Surface worker failures to the pipeline caller."""
    campaign = pipeline.Pipeline(
        design_space,
        distributed_sampler,
        objective=distributed(_FailingPredictor({})),
    )

    with pytest.raises(
        ray.exceptions.RayTaskError, match="distributed objective failed"
    ):
        campaign.run(store=store)


def test_distributed_sampler_preserves_ancestry_across_starting_sequences(
    distributed: Callable[..., serving_pool.ModelPool],
    distributed_sampler: serving_pool.ModelPool,
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
) -> None:
    """Associate concurrently generated proposals with their original parents."""
    first = design_space.init_spaces[0]
    allowed = first.allowed.clone()
    allowed[0] = False
    design = spaces.DesignSpace(
        [first, spaces.SearchSpace(data.tokenizer.encode("LKT"), allowed)]
    )
    campaign = pipeline.Pipeline(
        design,
        distributed_sampler,
        objective=distributed(
            _Predictor({"MAT": 4.0, "MKC": 1.0, "LAT": 8.0, "LKC": 2.0})
        ),
        acquisition=acquisitions.Greedy(budget=2),
    )

    selected = campaign.run(store=store)

    assert sorted(
        data.tokenizer.decode(list(selected.selection.token_ids.unbind()))
    ) == ["LAT", "MAT"]
    assert _proposals(store, 1) == [
        ("LAT", 2, 2),
        ("LKC", 2, 2),
        ("MAT", 1, 1),
        ("MKC", 1, 1),
    ]


def test_distributed_constraints_can_reject_every_proposal(
    distributed: Callable[..., serving_pool.ModelPool],
    distributed_sampler: serving_pool.ModelPool,
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
) -> None:
    """Handle empty downstream streams after all remote constraint checks fail."""
    campaign = pipeline.Pipeline(
        design_space,
        distributed_sampler,
        objective=distributed(_FailingPredictor({})),
        hard_constraints={
            "stability": distributed(
                constraints.HardPredictionRangeConstraint(
                    _Predictor({"MAT": 0.0, "MKC": 0.0}), ge=0.5
                )
            )
        },
    )

    selected = campaign.run(store=store)

    assert data.tokenizer.decode(list(selected.selection.token_ids.unbind())) == []
    assert list(store.iter_samples(cycle_id=1)) == []
    assert _proposals(store, 1) == [("MAT", 1, 1), ("MKC", 1, 1)]


def test_local_sampler_loads_and_releases_its_distributed_model(
    distributed: Callable[..., serving_pool.ModelPool],
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
) -> None:
    """Manage a remote model through a local model-guided sampler."""
    model = distributed(_LogitsModel())
    sampler = mlm.MLMSampler(model, max_samples=1, temperature=0.0, seed=7)
    campaign = pipeline.Pipeline(
        design_space,
        sampler,
        objective=distributed(_Predictor({"MAC": 6.0})),
        release_handlers_after_use=True,
    )

    selected = campaign.run(store=store)

    assert data.tokenizer.decode(list(selected.selection.token_ids.unbind())) == ["MAC"]
    with pytest.raises(ray.exceptions.RayTaskError, match="test handler is not loaded"):
        model([inference.MLMLogitsRequest(data.tokenizer.encode("M__"))])


def test_local_ensemble_combines_distributed_predictors(
    distributed: Callable[..., serving_pool.ModelPool],
    distributed_sampler: serving_pool.ModelPool,
    design_space: spaces.DesignSpace,
    store: storage.CampaignStore,
) -> None:
    """Combine concurrently served member predictions into stored posterior samples."""
    objective = predictors.EnsemblePredictor(
        [
            distributed(_Predictor({"MAT": 2.0, "MKC": 1.0})),
            distributed(_Predictor({"MAT": 4.0, "MKC": 1.0})),
        ]
    )
    campaign = pipeline.Pipeline(
        design_space,
        distributed_sampler,
        objective=objective,
        acquisition=_UncertainGreedy(budget=1),
    )

    selected = campaign.run(store=store)

    assert data.tokenizer.decode(list(selected.selection.token_ids.unbind())) == ["MAT"]
    batch = store.get_candidate_stream().candidates_from_ids([2, 3])
    torch.testing.assert_close(batch.objective_mean, torch.tensor([3.0, 1.0]))
    torch.testing.assert_close(batch.objective_std, torch.tensor([1.4142135624, 0.0]))
    torch.testing.assert_close(
        batch.objective_samples, torch.tensor([[2.0, 4.0], [1.0, 1.0]])
    )
