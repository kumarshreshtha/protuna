"""Utilities for path resolution."""

import os
import pathlib


def get_cache_dir() -> pathlib.Path:
    """Get the default cache directory for storing datasets, models and embeddings.

    Returns:
        Path to the cache directory. Creates the directory if it doesn't exist.
    """
    base = pathlib.Path(
        os.environ.get(
            "PROTUNA_HOME",
            default=os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache"),
        )
    )

    cache_dir = base / "protuna"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
