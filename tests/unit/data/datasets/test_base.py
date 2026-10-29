from __future__ import annotations

import dataclasses
import pathlib

import polars as pl
import pytest

from protuna.data.datasets import base

# The stub dataset below is a four-row fitness frame over the wild type "ACDE".
# "A1G" appears under both properties so `get_all_sequences` has duplicates to
# collapse.
_TARGET_SEQUENCE = "ACDE"

_ROWS = [
    ("WT", "ACDE", "binding", 0.0, "train"),
    ("A1G", "GCDE", "binding", 1.5, "train"),
    ("C2W", "AWDE", "binding", -0.5, "test"),
    ("A1G", "GCDE", "stability", 2.0, "train"),
]


@dataclasses.dataclass(frozen=True, repr=False)
class _StubDataset(base.MutantFitnessDataset):
    """Minimal concrete dataset backed by an in-memory frame."""

    def get_structure_path(self, source: base.StructureSource) -> pathlib.Path:
        """Not exercised by these tests."""
        raise NotImplementedError

    def get_msa_path(self, protocol: base.MSAProtocol) -> pathlib.Path:
        """Not exercised by these tests."""
        raise NotImplementedError

    def get_fitness_df(
        self, prop: str | None = None, drop_wt: bool = False
    ) -> pl.DataFrame:
        """Return the stub rows, optionally filtered like a real dataset would."""
        fitness_df = pl.DataFrame(
            _ROWS, schema=base.FitnessDFSchema.schema, orient="row"
        )
        if prop is not None:
            fitness_df = fitness_df.filter(pl.col("prop") == prop)
        if drop_wt:
            fitness_df = fitness_df.filter(pl.col("sequence") != self.target_sequence)
        return base.FitnessDFSchema.validate(fitness_df)


@pytest.fixture
def dataset() -> _StubDataset:
    """A stub dataset over the wild type "ACDE" with two properties."""
    return _StubDataset(
        dataset_name="STUB_DATASET",
        target_sequence=_TARGET_SEQUENCE,
        properties=["binding", "stability"],
    )


# ---------------------------------------------------------------------------
# create_mutant_uuids
# ---------------------------------------------------------------------------


def test_create_mutant_uuids_names_single_substitution() -> None:
    """A single substitution is named "<wt aa><1-based position><mt aa>"."""
    assert base.create_mutant_uuids(["GCDE"], "ACDE") == {"A1G": "GCDE"}


def test_create_mutant_uuids_joins_multiple_substitutions_with_colons() -> None:
    """Multiple substitutions are joined in position order by ":"."""
    assert base.create_mutant_uuids(["GCWE"], "ACDE") == {"A1G:D3W": "GCWE"}


def test_create_mutant_uuids_names_unmutated_sequence_wt() -> None:
    """A sequence identical to the wild type is keyed as "WT"."""
    assert base.create_mutant_uuids(["ACDE"], "ACDE") == {"WT": "ACDE"}


def test_create_mutant_uuids_maps_every_input_sequence() -> None:
    """Each input sequence gets its own entry in the returned map."""
    uuids = base.create_mutant_uuids(["GCDE", "AWDE", "ACDE"], "ACDE")

    assert uuids == {"A1G": "GCDE", "C2W": "AWDE", "WT": "ACDE"}


def test_create_mutant_uuids_rejects_length_mismatch() -> None:
    """A mutant of a different length than the wild type is an error."""
    with pytest.raises(ValueError, match="shorter"):
        base.create_mutant_uuids(["GCD"], "ACDE")


# ---------------------------------------------------------------------------
# wt_from_mutants
# ---------------------------------------------------------------------------


def test_wt_from_mutants_takes_the_majority_amino_acid_per_position() -> None:
    """The wild type is the most frequent amino acid at each position."""
    mutants = ["ACDE", "GCDE", "ACDE", "ACWE", "ACDE"]

    assert base.wt_from_mutants(mutants) == "ACDE"


def test_wt_from_mutants_recovers_wild_type_from_a_single_mutant() -> None:
    """With one sequence every position is unanimous, so it is the wild type."""
    assert base.wt_from_mutants(["ACDE"]) == "ACDE"


def test_wt_from_mutants_marks_unreliable_positions_unknown() -> None:
    """A position whose top amino acid falls below `min_freq` becomes "?"."""
    assert base.wt_from_mutants(["ACD", "GCD"], min_freq=0.9) == "?CD"


# ---------------------------------------------------------------------------
# MutantFitnessDataset.get_all_sequences
# ---------------------------------------------------------------------------


def test_get_all_sequences_maps_mutant_uuid_to_sequence(
    dataset: _StubDataset,
) -> None:
    """Every mutant in the fitness frame is returned, keyed by its uuid."""
    assert dataset.get_all_sequences() == {
        "WT": "ACDE",
        "A1G": "GCDE",
        "C2W": "AWDE",
    }


def test_get_all_sequences_includes_wild_type_even_when_absent_from_frame() -> None:
    """The wild type is added from `target_sequence`, not read off the frame."""
    dataset = _StubDataset(
        dataset_name="STUB_DATASET",
        target_sequence="ACDE",
        properties=["stability"],
    )

    # The "stability" rows carry no WT measurement, yet "WT" is still reported.
    assert dataset.get_all_sequences()["WT"] == "ACDE"


def test_get_all_sequences_excludes_wild_type_when_dropped(
    dataset: _StubDataset,
) -> None:
    """`drop_wt` removes the wild type from the returned map."""
    assert dataset.get_all_sequences(drop_wt=True) == {
        "A1G": "GCDE",
        "C2W": "AWDE",
    }


# ---------------------------------------------------------------------------
# MutantFitnessDataset.__repr__
# ---------------------------------------------------------------------------


def test_repr_reports_dataset_name_length_and_properties(dataset: _StubDataset) -> None:
    """The representation names the class, dataset, target length, and properties."""
    assert repr(dataset) == (
        "_StubDataset(dataset_name='STUB_DATASET', length=4, "
        "properties=['binding', 'stability'])"
    )


# ---------------------------------------------------------------------------
# FitnessDFSchema
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mutant", ["WT", "A1G", "A1G:D3W", "A12G"])
def test_fitness_schema_accepts_valid_mutant_names(mutant: str) -> None:
    """Both the WT marker and colon-joined substitution strings validate."""
    fitness_df = pl.DataFrame(
        [(mutant, "ACDE", "binding", 0.0, "train")],
        schema=base.FitnessDFSchema.schema,
        orient="row",
    )

    assert base.FitnessDFSchema.validate(fitness_df)["mutant"].to_list() == [mutant]


@pytest.mark.parametrize(
    "mutant",
    [
        "A0G",  # positions are 1-based
        "a1g",  # amino acids are uppercase
        "B1G",  # B is not a standard amino acid
        "A1G,D3W",  # substitutions are colon-joined
        "1G",  # missing wild-type amino acid
    ],
)
def test_fitness_schema_rejects_malformed_mutant_names(mutant: str) -> None:
    """Mutant names outside the WT/substitution grammar fail validation."""
    fitness_df = pl.DataFrame(
        [(mutant, "ACDE", "binding", 0.0, "train")],
        schema=base.FitnessDFSchema.schema,
        orient="row",
    )

    with pytest.raises(ValueError, match="str_matches"):
        base.FitnessDFSchema.validate(fitness_df)


@pytest.mark.parametrize("column", ["mutant", "sequence", "prop", "fitness", "split"])
def test_fitness_schema_rejects_missing_columns(column: str) -> None:
    """Reject tables missing any required column."""
    fitness_df = pl.DataFrame(
        _ROWS, schema=base.FitnessDFSchema.schema, orient="row"
    ).drop(column)

    with pytest.raises(ValueError, match="Missing required column"):
        base.FitnessDFSchema.validate(fitness_df)


@pytest.mark.parametrize("column", ["mutant", "sequence", "prop", "fitness", "split"])
def test_fitness_schema_rejects_null_values(column: str) -> None:
    """Reject null values in every required column."""
    fitness_df = pl.DataFrame(
        _ROWS, schema=base.FitnessDFSchema.schema, orient="row"
    ).with_columns(
        pl.lit(None, dtype=base.FitnessDFSchema.schema[column]).alias(column)
    )

    with pytest.raises(ValueError, match="contains null"):
        base.FitnessDFSchema.validate(fitness_df)


def test_fitness_schema_rejects_nan_measurements() -> None:
    """Reject NaN measurements as well as nulls."""
    fitness_df = pl.DataFrame(
        _ROWS, schema=base.FitnessDFSchema.schema, orient="row"
    ).with_columns(pl.lit(float("nan")).alias("fitness"))

    with pytest.raises(ValueError, match="contains NaN"):
        base.FitnessDFSchema.validate(fitness_df)


def test_fitness_schema_rejects_unknown_split() -> None:
    """Reject split labels outside the train/test convention."""
    fitness_df = pl.DataFrame(
        _ROWS, schema=base.FitnessDFSchema.schema, orient="row"
    ).with_columns(pl.lit("validation").alias("split"))

    with pytest.raises(ValueError, match="only train or test"):
        base.FitnessDFSchema.validate(fitness_df)


def test_fitness_schema_rejects_integer_measurements() -> None:
    """Require floating measurements in the continuous fitness schema."""
    fitness_df = pl.DataFrame(
        _ROWS, schema=base.FitnessDFSchema.schema, orient="row"
    ).with_columns(pl.col("fitness").cast(pl.Int64))

    with pytest.raises(ValueError, match="dtype Float64"):
        base.FitnessDFSchema.validate(fitness_df)


def test_binned_fitness_schema_preserves_integer_measurements() -> None:
    """Validate discrete measurements without coercing their dtype."""
    fitness_df = pl.DataFrame(
        [("WT", "ACDE", "binding", 2, "train")],
        schema=base.BinnedFitnessDFSchema.schema,
        orient="row",
    )

    assert base.BinnedFitnessDFSchema.validate(fitness_df) is fitness_df
    assert fitness_df["fitness"].dtype == pl.Int64


def test_fitness_schema_accepts_empty_typed_frame() -> None:
    """Accept an empty frame with the expected column types."""
    fitness_df = pl.DataFrame(schema=base.FitnessDFSchema.schema)

    assert base.FitnessDFSchema.validate(fitness_df) is fitness_df
