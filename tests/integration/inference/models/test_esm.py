"""Integration tests for the ESM wrapper served through the Ray serving layer.

Every test here loads the real `facebook/esm2_t6_8M_UR50D` weights — the smallest
published ESM-2 checkpoint — into Ray actors and drives them through
`protuna.inference.serve`. What is under test is the seam between the two: that
an `EsmInferenceModel` survives being pickled into a worker, reloads its own
weights there, and hands numerically equivalent outputs back across the actor
boundary however the pool routes, batches and streams the requests.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import sys
from typing import Literal

import pytest
import ray
import ray.cloudpickle
import torch

from protuna import data
from protuna import inference
from protuna.inference import serve
from protuna.inference.models import esm
from protuna.inference.serve import pool as serving_pool

# The smallest ESM-2 checkpoint: 6 layers, 320 hidden dims. Small enough that a
# handful of actors can each hold their own copy on the CPU.
# Use fp32 to keep comparisons across worker processes and padded batches tight.
_MODEL = esm.EsmModels.ESM2_8M
_HIDDEN_SIZE = 320
# One entry per layer plus the input embeddings.
_NUM_HIDDEN_STATES = 7

# protuna's vocabulary: 20 amino acids followed by the gap and special tokens.
_VOCAB_SIZE = 25
_AA_VOCAB_SIZE = 20

# The first 30 residues of human ubiquitin, and a single-substitution mutant of
# it (I3V). Short sequences keep the forward passes cheap.
_WILD_TYPE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKI"
_MUTANT = "MQVFVKTLTGKTITLEVEPSDTIENVKAKI"
_SHORT = "MQIFVKTLTG"
# Two masked positions, so masked-only outputs have a length of their own.
_MASKED = "MQ_FVKTLTGKTITLEVEPSDTIENVKAK_"

# Long enough that a batch never fires on the timer while the test is still
# submitting, so a batch size in the assertions is deterministic.
_MAX_WAIT_MS = 500.0

# Ray schedules on these numbers alone, and the actors are tiny, so keep the
# reservation small enough that concurrently live pools never queue for a slot.
_TINY_ACTOR: serve.RayActorOptions = {"num_cpus": 0.25}

# Worker processes and padding can perturb float32 forward passes slightly.
_LOGITS_TOLERANCE = 1e-4


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _RecordingRouter:
    """Round-robin router that records the replica each request was sent to."""

    def __init__(self):
        self.assigned: list[int] = []

    def __call__(self, request: inference.MLMRequest, num_replicas: int) -> int:
        replica_id = len(self.assigned) % num_replicas
        self.assigned.append(replica_id)
        return replica_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _ray_cluster() -> Iterator[None]:
    """Start one local Ray cluster shared by every pool in the session."""
    ray.init(
        address="local",
        num_cpus=4,
        include_dashboard=False,
        log_to_driver=False,
        ignore_reinit_error=True,
    )
    # Worker processes cannot import this file by name, so the router defined
    # here has to travel by value rather than by reference.
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    yield
    ray.cloudpickle.unregister_pickle_by_value(sys.modules[__name__])
    ray.shutdown()


@pytest.fixture
def serve_pool(_ray_cluster: None) -> Iterator[Callable[..., serving_pool.ModelPool]]:
    """Create and load pools through the public APIs, then shut them down."""
    pools: list[serving_pool.ModelPool] = []

    def _serve(
        *args, pool_kind: Literal["queue", "routed"] = "queue", **kwargs
    ) -> serving_pool.ModelPool:
        factory = (
            serve.serve_routed_pool if pool_kind == "routed" else serve.serve_queue_pool
        )
        pool = factory(*args, **kwargs)
        pools.append(pool)
        pool.load()
        return pool

    try:
        yield _serve
    finally:
        for pool in pools:
            pool.shutdown()


@pytest.fixture(scope="session")
def local_model() -> esm.EsmInferenceModel:
    """An ESM-2 8M loaded in the test process, to compare served outputs against.

    Held for the session because it is read-only: a forward pass leaves no state
    behind, so tests can share one copy of the weights.
    """
    model = esm.EsmInferenceModel(_MODEL, precision="fp32")
    model.load(torch.device("cpu"))
    return model


def _request(sequence: str, **config: bool) -> inference.MLMRequest:
    """Builds an `MLMRequest` for *sequence* with the given output config."""
    return inference.MLMRequest(
        token_ids=data.tokenizer.encode(sequence),
        output_config=inference.MLMOutputConfig(**config),
    )


def _present[T](value: T | None) -> T:
    """Asserts an optional output field was populated and returns it."""
    assert value is not None
    return value


@pytest.mark.integration
class TestQueuePool:
    """Inference outputs and streaming through one shared queue pool."""

    @pytest.fixture(scope="class")
    @classmethod
    def queue_pool(cls, _ray_cluster: None) -> Iterator[serving_pool.ModelPool]:
        """A two-replica pull-based pool of ESM-2 8M, released after the class.

        Every replica serves one request per forward pass, so a served output is
        padded exactly as an unbatched local forward is and the two are comparable
        within float32 tolerance.
        """
        pool = serve.serve_queue_pool(
            serve.ReplicaGroup(
                esm.EsmInferenceModel(_MODEL, precision="fp32"),
                num_replicas=2,
                actor_options=_TINY_ACTOR,
            )
        )
        try:
            pool.load()
            yield pool
        finally:
            pool.shutdown()

    def test_queue_pool_logits_match_a_local_forward(
        self, queue_pool: serving_pool.ModelPool, local_model: esm.EsmInferenceModel
    ) -> None:
        """A pull-based pool matches the model outputs computed in-process."""
        request = _request(_WILD_TYPE)

        served = queue_pool([request])

        torch.testing.assert_close(
            _present(served[0].logits),
            _present(local_model([request])[0].logits),
            atol=_LOGITS_TOLERANCE,
            rtol=_LOGITS_TOLERANCE,
        )

    def test_pool_returns_one_output_per_request_in_order(
        self,
        queue_pool: serving_pool.ModelPool,
    ) -> None:
        """Responses come back aligned with the requests that produced them.

        The three sequences have distinct lengths, so the shapes alone identify which
        request each output belongs to.
        """
        served = queue_pool([_request(_WILD_TYPE), _request(_SHORT), _request(_MUTANT)])

        assert [_present(output.logits).shape for output in served] == [
            (30, _VOCAB_SIZE),
            (10, _VOCAB_SIZE),
            (30, _VOCAB_SIZE),
        ]

    def test_pool_distinguishes_a_mutant_from_its_wild_type(
        self,
        queue_pool: serving_pool.ModelPool,
    ) -> None:
        """Two sequences differing at one site get different logits at that site.

        Requests of equal length are the ones a mis-wired pool could silently swap or
        deduplicate, so the substitution has to show up in the response.
        """
        served = queue_pool([_request(_WILD_TYPE), _request(_MUTANT)])

        assert not torch.equal(_present(served[0].logits), _present(served[1].logits))

    def test_pool_returns_responses_on_the_host(
        self, queue_pool: serving_pool.ModelPool
    ) -> None:
        """Output tensors arrive on the CPU, whatever device the worker served from."""
        served = queue_pool([_request(_WILD_TYPE, return_embeddings=True)])

        assert _present(served[0].logits).device.type == "cpu"
        assert _present(served[0].embeddings).device.type == "cpu"

    def test_queue_pool_serves_more_requests_than_it_has_replicas(
        self,
        queue_pool: serving_pool.ModelPool,
    ) -> None:
        """Requests beyond the replica count are queued rather than dropped."""
        served = queue_pool([_request(_SHORT) for _ in range(6)])

        assert [_present(output.logits).shape for output in served] == [
            (10, _VOCAB_SIZE)
        ] * 6

    def test_pool_returns_embeddings_for_every_residue(
        self,
        queue_pool: serving_pool.ModelPool,
    ) -> None:
        """Return one embedding per residue after removing special-token rows."""
        served = queue_pool([_request(_WILD_TYPE, return_embeddings=True)])

        assert _present(served[0].embeddings).shape == (30, _HIDDEN_SIZE)

    def test_pool_returns_mean_embeddings(
        self, queue_pool: serving_pool.ModelPool
    ) -> None:
        """A mean embedding arrives as one vector per sequence."""
        served = queue_pool([_request(_WILD_TYPE, return_mean_embeddings=True)])

        assert _present(served[0].mean_embeddings).shape == (_HIDDEN_SIZE,)

    def test_pool_returns_every_layer_of_hidden_states(
        self,
        queue_pool: serving_pool.ModelPool,
    ) -> None:
        """The whole per-layer tuple is moved to the host, not just the last layer."""
        served = queue_pool([_request(_SHORT, return_hidden_states=True)])

        hidden_states = _present(served[0].hidden_states)
        assert len(hidden_states) == _NUM_HIDDEN_STATES
        for state in hidden_states:
            assert state.shape == (len(_SHORT), _HIDDEN_SIZE)
            assert state.device.type == "cpu"

    def test_pool_returns_only_the_masked_positions_when_asked(
        self,
        queue_pool: serving_pool.ModelPool,
    ) -> None:
        """A masked-marginals request comes back with one row per masked residue."""
        served = queue_pool(
            [_request(_MASKED, masked_positions_only=True, aa_tokens_only=True)]
        )

        assert _present(served[0].logits).shape == (2, _AA_VOCAB_SIZE)

    def test_pool_applies_each_requests_own_output_config(
        self,
        queue_pool: serving_pool.ModelPool,
    ) -> None:
        """A mixed batch gives every request only the fields it asked for."""
        served = queue_pool(
            [
                _request(_WILD_TYPE, return_logits=False, return_mean_embeddings=True),
                _request(_SHORT),
            ]
        )

        assert served[0].logits is None
        assert served[0].mean_embeddings is not None
        assert served[1].logits is not None
        assert served[1].mean_embeddings is None

    def test_pool_streams_each_response_with_the_request_that_produced_it(
        self,
        queue_pool: serving_pool.ModelPool,
    ) -> None:
        """Streamed results carry their request id, however the pool interleaves them.

        The sequences have distinct lengths, so pairing a request id with the wrong
        response would show up as the wrong number of rows.
        """
        requests = [(0, _request(_WILD_TYPE)), (1, _request(_SHORT))]

        streamed = queue_pool.stream(iter(requests))

        assert sorted(
            (request_id, _present(output.logits).shape[0])
            for request_id, _, output in streamed
        ) == [(0, 30), (1, 10)]


@pytest.mark.integration
class TestRoutedPool:
    """Local and replica agreement through one shared routed pool."""

    @pytest.fixture(scope="class")
    @classmethod
    def routed_pool(cls, _ray_cluster: None) -> Iterator[serving_pool.ModelPool]:
        """A two-replica push-based pool of ESM-2 8M, released after the class."""
        pool = serve.serve_routed_pool(
            serve.ReplicaGroup(
                esm.EsmInferenceModel(_MODEL, precision="fp32"),
                num_replicas=2,
                actor_options=_TINY_ACTOR,
            ),
            router=serve.RoundRobinRouter(),
        )
        try:
            pool.load()
            yield pool
        finally:
            pool.shutdown()

    def test_routed_pool_logits_match_a_local_forward(
        self, routed_pool: serving_pool.ModelPool, local_model: esm.EsmInferenceModel
    ) -> None:
        """A push-based pool matches the model outputs computed in-process."""
        request = _request(_WILD_TYPE)

        served = routed_pool([request])

        torch.testing.assert_close(
            _present(served[0].logits),
            _present(local_model([request])[0].logits),
            atol=_LOGITS_TOLERANCE,
            rtol=_LOGITS_TOLERANCE,
        )

    def test_every_replica_serves_the_same_logits(
        self,
        routed_pool: serving_pool.ModelPool,
    ) -> None:
        """Replicas load weights independently and agree residue by residue."""
        served = routed_pool([_request(_WILD_TYPE), _request(_WILD_TYPE)])

        torch.testing.assert_close(
            _present(served[0].logits),
            _present(served[1].logits),
            atol=_LOGITS_TOLERANCE,
            rtol=_LOGITS_TOLERANCE,
        )


# Tests with custom pool configurations own and release their pools per test.


@pytest.mark.integration
def test_pool_batches_a_ragged_batch_without_changing_its_logits(
    serve_pool: Callable[..., serving_pool.ModelPool],
    local_model: esm.EsmInferenceModel,
) -> None:
    """Coalescing sequences of different lengths leaves each output unchanged.

    Batching pads every sequence to the longest in the batch, so this is the test
    that the padding is masked out rather than read as residues.
    """
    pool = serve_pool(
        serve.ReplicaGroup(
            esm.EsmInferenceModel(_MODEL, precision="fp32"),
            max_batch_size=4,
            max_wait_ms=_MAX_WAIT_MS,
            actor_options=_TINY_ACTOR,
        )
    )
    requests = [_request(_WILD_TYPE), _request(_SHORT), _request(_MUTANT)]

    served = pool(requests)

    torch.testing.assert_close(
        _present(served[1].logits),
        _present(local_model([requests[1]])[0].logits),
        atol=_LOGITS_TOLERANCE,
        rtol=_LOGITS_TOLERANCE,
    )


@pytest.mark.integration
def test_pool_serves_a_model_that_was_never_loaded_locally(
    serve_pool: Callable[..., serving_pool.ModelPool],
) -> None:
    """Weights are fetched by the worker, so the driver never needs them."""
    pool = serve_pool(
        serve.ReplicaGroup(
            esm.EsmInferenceModel(_MODEL, precision="fp32"), actor_options=_TINY_ACTOR
        )
    )

    served = pool([_request(_WILD_TYPE)])

    assert _present(served[0].logits).shape == (30, _VOCAB_SIZE)


@pytest.mark.integration
def test_serving_a_model_leaves_the_local_instance_unloaded(
    serve_pool: Callable[..., serving_pool.ModelPool],
) -> None:
    """The driver's copy stays weightless; only the pickled copies are loaded."""
    model = esm.EsmInferenceModel(_MODEL, precision="fp32")
    serve_pool(serve.ReplicaGroup(model, actor_options=_TINY_ACTOR))

    with pytest.raises(RuntimeError, match="forward called before loading the model"):
        model.forward([_request(_WILD_TYPE)])


@pytest.mark.integration
def test_routed_pool_dispatches_across_every_replica(
    serve_pool: Callable[..., serving_pool.ModelPool],
) -> None:
    """A pool of several replicas hands work to each of them."""
    router = _RecordingRouter()
    pool = serve_pool(
        serve.ReplicaGroup(
            esm.EsmInferenceModel(_MODEL, precision="fp32"),
            num_replicas=3,
            actor_options=_TINY_ACTOR,
        ),
        pool_kind="routed",
        router=router,
    )

    pool([_request(_SHORT) for _ in range(6)])

    assert router.assigned == [0, 1, 2, 0, 1, 2]


@pytest.mark.integration
def test_pool_serves_requests_from_several_replica_groups(
    serve_pool: Callable[..., serving_pool.ModelPool],
) -> None:
    """One pool can front groups configured differently and still answer them all.

    The groups here differ in how they batch, so a request is served correctly
    whichever group's worker happens to pull it off the queue.
    """
    pool = serve_pool(
        [
            serve.ReplicaGroup(
                esm.EsmInferenceModel(_MODEL, precision="fp32"),
                actor_options=_TINY_ACTOR,
            ),
            serve.ReplicaGroup(
                esm.EsmInferenceModel(_MODEL, precision="fp32"),
                max_batch_size=4,
                max_wait_ms=_MAX_WAIT_MS,
                actor_options=_TINY_ACTOR,
            ),
        ]
    )

    served = pool([_request(_SHORT) for _ in range(8)])

    assert [_present(output.logits).shape for output in served] == [
        (10, _VOCAB_SIZE)
    ] * 8


@pytest.mark.integration
def test_mask_affinity_router_gives_every_request_its_own_logits(
    serve_pool: Callable[..., serving_pool.ModelPool],
    local_model: esm.EsmInferenceModel,
) -> None:
    """Routing by mutation pattern still answers each request from its own sequence.

    `MaskAffinityRouter` keys on which sites a sequence mutates and pins each
    pattern to one replica, so the two wild-type requests here are guaranteed to
    share a replica while the mutant draws its own assignment. That sharing is
    what affinity routing adds over round-robin, and it is where a pool could
    hand one request another's response, so each output is checked against the
    forward pass for its own sequence.
    """
    pool = serve_pool(
        serve.ReplicaGroup(
            esm.EsmInferenceModel(_MODEL, precision="fp32"),
            num_replicas=2,
            actor_options=_TINY_ACTOR,
        ),
        pool_kind="routed",
        router=inference.MaskAffinityRouter(data.tokenizer.encode(_WILD_TYPE)),
    )
    requests = [_request(_WILD_TYPE), _request(_MUTANT), _request(_WILD_TYPE)]

    served = pool(requests)

    wild_type_logits = _present(local_model([requests[0]])[0].logits)
    mutant_logits = _present(local_model([requests[1]])[0].logits)
    torch.testing.assert_close(
        _present(served[0].logits),
        wild_type_logits,
        atol=_LOGITS_TOLERANCE,
        rtol=_LOGITS_TOLERANCE,
    )
    torch.testing.assert_close(
        _present(served[1].logits),
        mutant_logits,
        atol=_LOGITS_TOLERANCE,
        rtol=_LOGITS_TOLERANCE,
    )
    torch.testing.assert_close(
        _present(served[2].logits),
        wild_type_logits,
        atol=_LOGITS_TOLERANCE,
        rtol=_LOGITS_TOLERANCE,
    )


@pytest.mark.integration
def test_execute_parallel_fans_the_same_requests_across_two_pools(
    serve_pool: Callable[..., serving_pool.ModelPool],
) -> None:
    """A pull-based and a push-based pool of the same model give the same answers."""
    pools = [
        serve_pool(
            serve.ReplicaGroup(
                esm.EsmInferenceModel(_MODEL, precision="fp32"),
                actor_options=_TINY_ACTOR,
            )
        ),
        serve_pool(
            serve.ReplicaGroup(
                esm.EsmInferenceModel(_MODEL, precision="fp32"),
                actor_options=_TINY_ACTOR,
            ),
            pool_kind="routed",
        ),
    ]

    from_queue, from_routed = serve.execute_parallel(pools, [_request(_WILD_TYPE)])

    torch.testing.assert_close(
        _present(from_queue[0].logits),
        _present(from_routed[0].logits),
        atol=_LOGITS_TOLERANCE,
        rtol=_LOGITS_TOLERANCE,
    )
