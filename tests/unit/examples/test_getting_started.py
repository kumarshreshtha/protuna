from collections.abc import Callable
import pathlib
import runpy
from typing import Any
from unittest import mock

import pytest

from protuna import data
from protuna.data.datasets import flip


def test_getting_started_runs_with_a_local_alignment(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    examples_dir: pathlib.Path,
    check_getting_started_result: Callable[[dict[str, Any]], None],
) -> None:
    """Run the published quickstart with only its remote download substituted."""
    monkeypatch.setenv("PROTUNA_HOME", str(tmp_path))
    sequence = data.get_dataset("gb1", source=data.DatasetSource.FLIP).target_sequence
    alignment = tmp_path / "gb1.a3m"
    alignment.write_text(
        f">query\n{sequence}\n"
        f">homolog_a\n{sequence[:100]}{'A' * 25}{sequence[125:]}\n"
        f">homolog_g\n{sequence[:100]}{'G' * 25}{sequence[125:]}\n"
    )

    with mock.patch.object(
        flip, "_download_from_hub", autospec=True, return_value=str(alignment)
    ) as download:
        namespace = runpy.run_path(str(examples_dir / "getting_started.py"))

    download.assert_called_once()
    check_getting_started_result(namespace)
