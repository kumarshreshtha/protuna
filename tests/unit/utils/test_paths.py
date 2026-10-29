import pathlib
from typing import Literal

import pytest

from protuna.utils import paths


@pytest.mark.parametrize("cache_source", ["default", "xdg", "protuna"])
def test_get_cache_dir_creates_missing_parents(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_source: Literal["default", "xdg", "protuna"],
) -> None:
    """Create nested cache directories and reuse them on subsequent calls."""
    home = tmp_path / "home"
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    monkeypatch.delenv("PROTUNA_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    base = home / ".cache"
    if cache_source in {"xdg", "protuna"}:
        base = tmp_path / "custom" / "xdg"
        monkeypatch.setenv("XDG_CACHE_HOME", str(base))
    if cache_source == "protuna":
        base = tmp_path / "custom" / "protuna-home"
        monkeypatch.setenv("PROTUNA_HOME", str(base))

    assert not base.exists()

    cache_dir = paths.get_cache_dir()

    assert cache_dir == base / "protuna"
    assert cache_dir.is_dir()
    assert paths.get_cache_dir() == cache_dir
