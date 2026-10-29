"""Fixtures shared across the test suite."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import os
import pathlib
from typing import Any

# Local test workers share the installed environment. Ray's uv integration would
# copy the project and rebuild its dependencies in /tmp for every test session.
# Set this before importing Ray, which reads the flag at import time.
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import pytest
import ray
import ray.cloudpickle
import torch

from protuna import data
from protuna.inference.serve import api
from protuna.inference.serve import pool as serving_pool
from tests.unit.inference import _helpers


@pytest.fixture
def examples_dir() -> pathlib.Path:
    """Return the directory containing the runnable documentation examples."""
    return pathlib.Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def check_getting_started_result() -> Callable[[dict[str, Any]], None]:
    """Provide checks for the quickstart's budget and mutation constraints."""

    def _check(namespace: dict[str, Any]) -> None:
        selected = namespace["candidates"]
        assert isinstance(selected, list)
        assert 0 < len(selected) <= 20
        assert len(set(selected)) == len(selected)
        assert isinstance(selected[0], str)
        tokens = torch.stack(data.tokenizer.encode(selected))
        initial = namespace["tokens"]
        allowed = namespace["design_space"].init_spaces[0].allowed
        assert tokens.shape[1] == len(initial)
        assert allowed[torch.arange(len(initial)), tokens].all()
        mutations = (tokens != initial).sum(dim=1)
        assert (mutations >= 1).all()
        assert (mutations <= 3).all()

    return _check


@pytest.fixture(scope="session")
def a3m_file() -> pathlib.Path:
    """Path to the example alignment used by the MSA and scoring tests."""
    return pathlib.Path(__file__).parent / "fixtures" / "alignment.a3m"


@pytest.fixture(scope="session")
def _ray_cluster() -> Iterator[None]:
    """Start one local Ray cluster shared by every pool test in the session."""
    ray.init(
        address="local",
        num_cpus=4,
        include_dashboard=False,
        log_to_driver=False,
        ignore_reinit_error=True,
    )
    # Ray workers receive the shared handler doubles by value because the test
    # helper module is not installed in their environment.
    ray.cloudpickle.register_pickle_by_value(_helpers)
    yield
    ray.cloudpickle.unregister_pickle_by_value(_helpers)
    ray.shutdown()


@pytest.fixture(params=["queue", "routed"])
def pool_kind(request: pytest.FixtureRequest) -> str:
    """Select queue or routed serving for shared pool behavior tests."""
    return request.param


@pytest.fixture
def serve_pool(_ray_cluster: None) -> Iterator[Callable[..., serving_pool.ModelPool]]:
    """Create pools through the public APIs and shut them down after each test."""
    pools: list[serving_pool.ModelPool] = []

    def _serve(*args, pool_kind: str = "queue", **kwargs) -> serving_pool.ModelPool:
        factory = (
            api.serve_routed_pool if pool_kind == "routed" else api.serve_queue_pool
        )
        pool = factory(*args, **kwargs)
        pools.append(pool)
        return pool

    yield _serve
    for pool in pools:
        pool.shutdown()
