"""Local dataset caches for loader tests."""

import pathlib

import pytest


@pytest.fixture
def dataset_cache(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> pathlib.Path:
    """Use a temporary cache so loader tests never read the user's datasets."""
    monkeypatch.setenv("PROTUNA_HOME", str(tmp_path))
    cache = tmp_path / "protuna"
    cache.mkdir()
    return cache
