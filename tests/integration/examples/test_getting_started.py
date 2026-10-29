from collections.abc import Callable
import pathlib
import runpy
from typing import Any

import pytest

pytestmark = pytest.mark.integration


def test_getting_started_runs_with_the_published_dataset(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    examples_dir: pathlib.Path,
    check_getting_started_result: Callable[[dict[str, Any]], None],
) -> None:
    """Run the exact quickstart with its real MSA download and pipeline."""
    monkeypatch.setenv("PROTUNA_HOME", str(tmp_path))

    namespace = runpy.run_path(str(examples_dir / "getting_started.py"))

    check_getting_started_result(namespace)
