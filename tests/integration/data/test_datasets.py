from __future__ import annotations

import pathlib

import polars as pl
import pytest

from protuna.data import datasets
from protuna.data.datasets import align_bio
from protuna.data.datasets import base
from protuna.data.datasets import flip
from protuna.data.datasets import proteinguide
from protuna.data.datasets import sgpo

# ProteinGym ships 217 assays behind a single multi-gigabyte archive, and listing
# them downloads the metadata file at collection time. Pin a small subset instead:
# the smallest assay of each `coarse_selection_type`, plus one assay that includes
# multiple mutants per sequence (PIN1).
_PROTEINGYM_DATASETS = [
    "NPC1_HUMAN_Erwood_2022_RPE1",
    "OPSD_HUMAN_Wan_2019",
    "B2L11_HUMAN_Dutta_2010_binding-Mcl-1",
    "SCN5A_HUMAN_Glazer_2019",
    "PIN1_HUMAN_Tsuboyama_2023_1I6C",
]

_FITNESS_COLUMNS = ["mutant", "sequence", "prop", "fitness", "split"]


def _assert_valid_fitness_df(
    fitness_df: pl.DataFrame, dataset: base.MutantFitnessDataset
) -> None:
    """Assert the loaded frame carries the standardized fitness columns.

    ``get_fitness_df`` validates against
    :class:`~protuna.data.datasets.base.FitnessDFSchema` itself, so reaching this
    helper already means the schema passed; the assertions here pin the parts of
    the contract the schema does not cover.
    """
    assert isinstance(fitness_df, pl.DataFrame)
    assert not fitness_df.is_empty()
    assert fitness_df.columns == _FITNESS_COLUMNS
    assert set(fitness_df["prop"].unique()) == set(dataset.properties)


# ---------------------------------------------------------------------------
# Every dataset of every source loads and validates
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("name", _PROTEINGYM_DATASETS)
def test_proteingym_fitness_df_loads(name: str) -> None:
    """Each ProteinGym assay loads into a schema-valid fitness dataframe."""
    dataset = datasets.get_dataset(name, datasets.DatasetSource.PROTEINGYM)

    fitness_df = dataset.get_fitness_df()

    _assert_valid_fitness_df(fitness_df, dataset)


@pytest.mark.integration
@pytest.mark.parametrize("name", align_bio.list_datasets())
def test_align_fitness_df_loads(name: str) -> None:
    """Each Align tournament assay loads into a schema-valid fitness dataframe."""
    dataset = datasets.get_dataset(name, datasets.DatasetSource.ALIGN)

    fitness_df = dataset.get_fitness_df()

    _assert_valid_fitness_df(fitness_df, dataset)


@pytest.mark.integration
@pytest.mark.parametrize("name", flip.list_datasets())
def test_flip_fitness_df_loads(name: str) -> None:
    """Each FLIP dataset loads into a schema-valid fitness dataframe."""
    dataset = datasets.get_dataset(name, datasets.DatasetSource.FLIP)

    fitness_df = dataset.get_fitness_df()

    _assert_valid_fitness_df(fitness_df, dataset)


@pytest.mark.integration
@pytest.mark.parametrize("name", sgpo.list_datasets())
def test_sgpo_fitness_df_loads(name: str) -> None:
    """Each SGPO dataset loads into a schema-valid fitness dataframe."""
    dataset = datasets.get_dataset(name, datasets.DatasetSource.SGPO)

    fitness_df = dataset.get_fitness_df()

    _assert_valid_fitness_df(fitness_df, dataset)


@pytest.mark.integration
@pytest.mark.parametrize("name", proteinguide.list_datasets())
def test_proteinguide_fitness_df_loads(name: str) -> None:
    """Each ProteinGuide dataset loads into a schema-valid fitness dataframe."""
    dataset = datasets.get_dataset(name, datasets.DatasetSource.PROTEINGUIDE)

    fitness_df = dataset.get_fitness_df()

    _assert_valid_fitness_df(fitness_df, dataset)


# ---------------------------------------------------------------------------
# The `prop` and `drop_wt` arguments of `get_fitness_df`
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    ("name", "source", "prop"),
    [
        ("NPC1_HUMAN_Erwood_2022_RPE1", datasets.DatasetSource.PROTEINGYM, "Activity"),
        ("beta_glucosidase", datasets.DatasetSource.ALIGN, "activity"),
        ("gb1", datasets.DatasetSource.FLIP, "binding"),
        ("creiLOV", datasets.DatasetSource.SGPO, "fluorescence"),
        ("PbrR", datasets.DatasetSource.PROTEINGUIDE, "pb_affinity"),
    ],
)
def test_fitness_df_for_a_single_prop_loads(
    name: str, source: datasets.DatasetSource, prop: str
) -> None:
    """Restricting to one property loads a schema-valid frame for that property."""
    dataset = datasets.get_dataset(name, source)

    fitness_df = dataset.get_fitness_df(prop=prop)

    assert fitness_df.columns == _FITNESS_COLUMNS
    assert fitness_df["prop"].unique().to_list() == [prop]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("name", "source"),
    [
        ("NPC1_HUMAN_Erwood_2022_RPE1", datasets.DatasetSource.PROTEINGYM),
        ("beta_glucosidase", datasets.DatasetSource.ALIGN),
        ("gb1", datasets.DatasetSource.FLIP),
        ("creiLOV", datasets.DatasetSource.SGPO),
        ("PbrR", datasets.DatasetSource.PROTEINGUIDE),
    ],
)
def test_fitness_df_without_wild_type_loads(
    name: str, source: datasets.DatasetSource
) -> None:
    """Dropping the wild type loads a schema-valid frame with no wild-type row."""
    dataset = datasets.get_dataset(name, source)

    fitness_df = dataset.get_fitness_df(drop_wt=True)

    _assert_valid_fitness_df(fitness_df, dataset)
    assert dataset.target_sequence not in fitness_df["sequence"]


# ---------------------------------------------------------------------------
# Wild-type structures and MSAs
# ---------------------------------------------------------------------------


def _assert_valid_structure_file(path: pathlib.Path) -> None:
    """Assert the path points at a downloaded PDB file with coordinate records."""
    assert path.exists()
    assert path.suffix == ".pdb"
    assert "ATOM  " in path.read_text()


def _assert_valid_msa_file(
    path: pathlib.Path, dataset: base.MutantFitnessDataset
) -> None:
    """Assert the path points at a ColabFold a3m aligned against the wild type.

    ColabFold prefixes the FASTA records with a ``#<query length>`` header line, so
    the first three lines are that header, the query description and the query row.
    The query row is the sequence the alignment is indexed against, so it has to be
    the dataset's own wild type for per-position statistics to line up.
    """
    assert path.exists()
    assert path.suffix == ".a3m"
    header, description, query = path.read_text().split("\n", 3)[:3]
    assert header.startswith("#")
    assert description.startswith(">")
    assert query == dataset.target_sequence


@pytest.mark.integration
@pytest.mark.parametrize("name", _PROTEINGYM_DATASETS)
def test_proteingym_structure_path_downloads(name: str) -> None:
    """Each ProteinGym assay resolves to its AlphaFold2 wild-type structure."""
    dataset = datasets.get_dataset(name, datasets.DatasetSource.PROTEINGYM)

    structure_path = dataset.get_structure_path(base.StructureSource.AlphaFold2)

    _assert_valid_structure_file(structure_path)


@pytest.mark.integration
@pytest.mark.parametrize("name", align_bio.list_datasets())
def test_align_structure_path_downloads(name: str) -> None:
    """Each Align tournament assay resolves to its AlphaFold2 wild-type structure."""
    dataset = datasets.get_dataset(name, datasets.DatasetSource.ALIGN)

    structure_path = dataset.get_structure_path(base.StructureSource.AlphaFold2)

    _assert_valid_structure_file(structure_path)


@pytest.mark.integration
@pytest.mark.parametrize("name", flip.list_datasets())
def test_flip_structure_path_downloads(name: str) -> None:
    """Each FLIP dataset resolves to its AlphaFold2 wild-type structure."""
    dataset = datasets.get_dataset(name, datasets.DatasetSource.FLIP)

    structure_path = dataset.get_structure_path(base.StructureSource.AlphaFold2)

    _assert_valid_structure_file(structure_path)


@pytest.mark.integration
@pytest.mark.parametrize("name", sgpo.list_datasets())
def test_sgpo_structure_path_downloads(name: str) -> None:
    """Each SGPO dataset resolves to its AlphaFold2 wild-type structure."""
    dataset = datasets.get_dataset(name, datasets.DatasetSource.SGPO)

    structure_path = dataset.get_structure_path(base.StructureSource.AlphaFold2)

    _assert_valid_structure_file(structure_path)


@pytest.mark.integration
@pytest.mark.parametrize("name", proteinguide.list_datasets())
def test_proteinguide_structure_path_downloads(name: str) -> None:
    """Each ProteinGuide dataset resolves to its AlphaFold2 wild-type structure."""
    dataset = datasets.get_dataset(name, datasets.DatasetSource.PROTEINGUIDE)

    structure_path = dataset.get_structure_path(base.StructureSource.AlphaFold2)

    _assert_valid_structure_file(structure_path)


@pytest.mark.integration
@pytest.mark.parametrize("name", align_bio.list_datasets())
def test_align_msa_path_downloads(name: str) -> None:
    """Each Align tournament assay resolves to its ColabFold MSA."""
    dataset = datasets.get_dataset(name, datasets.DatasetSource.ALIGN)

    msa_path = dataset.get_msa_path(base.MSAProtocol.ColabFold)

    _assert_valid_msa_file(msa_path, dataset)


@pytest.mark.integration
@pytest.mark.parametrize("name", flip.list_datasets())
def test_flip_msa_path_downloads(name: str) -> None:
    """Each FLIP dataset resolves to its ColabFold MSA."""
    dataset = datasets.get_dataset(name, datasets.DatasetSource.FLIP)

    msa_path = dataset.get_msa_path(base.MSAProtocol.ColabFold)

    _assert_valid_msa_file(msa_path, dataset)


@pytest.mark.integration
@pytest.mark.parametrize("name", sgpo.list_datasets())
def test_sgpo_msa_path_downloads(name: str) -> None:
    """Each SGPO dataset resolves to its ColabFold MSA."""
    dataset = datasets.get_dataset(name, datasets.DatasetSource.SGPO)

    msa_path = dataset.get_msa_path(base.MSAProtocol.ColabFold)

    _assert_valid_msa_file(msa_path, dataset)


@pytest.mark.integration
@pytest.mark.parametrize("name", proteinguide.list_datasets())
def test_proteinguide_msa_path_downloads(name: str) -> None:
    """Each ProteinGuide dataset resolves to its ColabFold MSA."""
    dataset = datasets.get_dataset(name, datasets.DatasetSource.PROTEINGUIDE)

    msa_path = dataset.get_msa_path(base.MSAProtocol.ColabFold)

    _assert_valid_msa_file(msa_path, dataset)


# ---------------------------------------------------------------------------
# Structure sources and MSA protocols that are not available yet
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    ("name", "source"),
    [
        ("beta_glucosidase", datasets.DatasetSource.ALIGN),
        ("gb1", datasets.DatasetSource.FLIP),
        ("creiLOV", datasets.DatasetSource.SGPO),
        ("PbrR", datasets.DatasetSource.PROTEINGUIDE),
    ],
)
def test_structure_path_rejects_unavailable_source(
    name: str, source: datasets.DatasetSource
) -> None:
    """Asking for a non-AlphaFold2 structure reports that it is unavailable."""
    dataset = datasets.get_dataset(name, source)

    with pytest.raises(NotImplementedError, match="Only alphafold2 structures"):
        dataset.get_structure_path(base.StructureSource.Pdb)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("name", "source"),
    [
        ("beta_glucosidase", datasets.DatasetSource.ALIGN),
        ("gb1", datasets.DatasetSource.FLIP),
        ("creiLOV", datasets.DatasetSource.SGPO),
        ("PbrR", datasets.DatasetSource.PROTEINGUIDE),
    ],
)
def test_msa_path_rejects_unavailable_protocol(
    name: str, source: datasets.DatasetSource
) -> None:
    """Asking for a non-ColabFold MSA reports that the protocol is unavailable."""
    dataset = datasets.get_dataset(name, source)

    with pytest.raises(NotImplementedError, match="Only colabfold msa protocol"):
        dataset.get_msa_path(base.MSAProtocol.EVCoupling)


@pytest.mark.integration
def test_proteingym_structure_path_rejects_unavailable_source() -> None:
    """ProteinGym only ships AlphaFold2 structures."""
    dataset = datasets.get_dataset(
        "NPC1_HUMAN_Erwood_2022_RPE1", datasets.DatasetSource.PROTEINGYM
    )

    with pytest.raises(NotImplementedError, match="only alphafold2 structures"):
        dataset.get_structure_path(base.StructureSource.ESMFold2)
