"""Defines base class for all protein fitness datasets."""

import abc
from collections.abc import Iterable, Sequence
import dataclasses
import enum
import logging
import pathlib

import polars as pl
import torch

from protuna import constants
from protuna.data.tokenizers import tokenizer

logger = logging.getLogger(__name__)


def _mutation_pos_from_seq(wt_seq: str, mt_seq: str) -> list[int]:
    """Extract mutation positions by comparing wild-type and mutant sequences."""
    return [
        pos
        for pos, (wt_aa, mt_aa) in enumerate(zip(wt_seq, mt_seq, strict=True))
        if wt_aa != mt_aa
    ]


def create_mutant_uuids(mt_seqs: Iterable[str], wt_seq: str) -> dict[str, str]:
    """Creates a map between sequences and there mutation uuids."""
    uuids_map = {}

    def create_uuid(seq: str) -> str:
        uuid = ":".join(
            [
                f"{wt_seq[pos]}{pos + 1}{seq[pos]}"
                for pos in _mutation_pos_from_seq(wt_seq, seq)
            ]
        )
        return uuid if uuid != "" else "WT"

    uuids_map.update({create_uuid(mt_seq): mt_seq for mt_seq in mt_seqs})
    return uuids_map


def wt_from_mutants(mt_seqs: Sequence[str], *, min_freq: float = 0.2) -> str:
    """Infer the wild-type sequence from the mutation dataset.

    Assumes that at a given position majority of mutants will have the wild-type amino
    acid. To assign an amino acid to the wild type sequence it must be the most frequent
    amino acid at that position and it's frequency must be greater that `min_freq`.

    If the `min_freq` criteria is not satisfied for a position "?" is returned at
    that position.
    """
    vocab = tokenizer.encode(constants.AA_UPPERCASE)
    tokens = torch.stack(tokenizer.encode(mt_seqs))
    mt_freq = torch.stack([(tokens == v).float().mean(dim=0) for v in vocab], dim=1)
    max_freq, wt_tokens = mt_freq.max(dim=1)
    if max_freq.min() < min_freq:
        positions = torch.argwhere(max_freq < min_freq).flatten().tolist()
        freqs = max_freq[positions].tolist()
        pair_str = ", ".join(
            [f"({pos}, {freq:.2f})" for pos, freq in zip(positions, freqs, strict=True)]
        )
        logger.warning(
            "Amino acids for one or more locations cannot be inferred reliably."
            "The following positions have the most frequent amino acid with frequency"
            " < %s. %s. Wild-type sequence will default to the unknown"
            " token ('?') at those locations.",
            min_freq,
            pair_str,
        )
        wt_tokens[max_freq < min_freq] = tokenizer.unk_token_id

    return tokenizer.decode(wt_tokens)


class MSAProtocol(enum.StrEnum):
    """Protocol used to construct the MSA."""

    ColabFold = "colabfold"
    EVCoupling = "ev_coupling"


class StructureSource(enum.StrEnum):
    """Origin of the wild-type structure, either predicted or experimental."""

    AlphaFold2 = "alphafold2"
    ESMFold2 = "esmfold2"
    Pdb = "pdb"


AA_REGEX = f"[{constants.AA_UPPERCASE}]"
MUT_STR = rf"{AA_REGEX}[1-9]\d*{AA_REGEX}"


class FitnessDFSchema:
    """Schema of a mutant fitness table.

    Attributes:
        mutant: Mutation identifier relative to the target sequence, colon separated
            for multi-site mutants (e.g. ``"A1C:D2E"``), or ``"WT"``.
        sequence: Full amino acid sequence of the mutant.
        prop: Name of the measured property.
        fitness: Measured fitness for ``prop``.
        split: Whether the row belongs to the ``"train"`` or ``"test"`` split.
    """

    schema = pl.Schema(
        {
            "mutant": pl.String,
            "sequence": pl.String,
            "prop": pl.String,
            "fitness": pl.Float64,
            "split": pl.String,
        }
    )

    @classmethod
    def validate(cls, df: pl.DataFrame) -> pl.DataFrame:
        """Validate column types and values without changing the measurements.

        Returns:
            The input dataframe, preserving row and column order.

        Raises:
            ValueError: If required columns are missing, have incorrect types or
                contain nulls, or mutant identifiers or split labels are invalid.
        """
        for column, dtype in cls.schema.items():
            if column not in df.columns:
                raise ValueError(f"Missing required column: {column}")
            if df.schema[column] != dtype:
                raise ValueError(
                    f"Expected {column} to have dtype {dtype}, got {df.schema[column]}"
                )
            if df[column].null_count():
                raise ValueError(f"Column {column} contains null values")
        if df["fitness"].is_nan().any():
            raise ValueError("Column fitness contains NaN values")
        if not df["mutant"].str.contains(rf"\A(?:WT|{MUT_STR}(?::{MUT_STR})*)\z").all():
            raise ValueError("Column mutant fails str_matches validation")
        if not df["split"].is_in(["train", "test"]).all():
            raise ValueError("Column split must contain only train or test")
        return df


class BinnedFitnessDFSchema(FitnessDFSchema):
    """Schema of a mutant fitness table whose fitness is discretized into bins.

    .. seealso:: :class:`FitnessDFSchema` for the remaining columns.

    Attributes:
        fitness: Index of the bin the measured fitness falls into.
    """

    schema = pl.Schema({**FitnessDFSchema.schema, "fitness": pl.Int64})


@dataclasses.dataclass(frozen=True)
class MutantFitnessDataset(abc.ABC):
    """Abstract base class for mutant fitness datasets.

    Args:
        dataset_name: Name identifying the dataset within its source collection.
        target_sequence: Wild-type amino acid sequence used to identify mutations.
        properties: Names of the measured properties.
    """

    dataset_name: str
    target_sequence: str
    properties: list[str]

    @abc.abstractmethod
    def get_structure_path(self, source: StructureSource) -> pathlib.Path:
        """Return the path to the wild-type structure.

        Args:
            source: Source of the structure.

        Returns:
            Path to the wild-type ``.pdb`` file.
        """
        ...

    @abc.abstractmethod
    def get_msa_path(self, protocol: MSAProtocol) -> pathlib.Path:
        """Return the path to the wild-type multiple sequence alignment.

        Args:
            protocol: Protocol used to construct the MSA.

        Returns:
            Path to the MSA file.
        """
        ...

    @abc.abstractmethod
    def get_fitness_df(
        self, *, prop: str | None = None, drop_wt: bool = False
    ) -> pl.DataFrame:
        """Return experimental fitness as a Polars dataframe.

        Args:
            prop: Property to include, or ``None`` to include all properties.
            drop_wt: Whether to exclude rows matching the wild-type sequence.
                Defaults to ``False``.

        Returns:
            A dataframe with columns ``mutant``, ``sequence``, ``prop``, ``fitness``
            and ``split``, following :class:`FitnessDFSchema`. Datasets with binned
            measurements follow :class:`BinnedFitnessDFSchema` instead.
        """
        ...

    def get_all_sequences(self, *, drop_wt: bool = False) -> dict[str, str]:
        """Return sequences keyed by their mutation identifiers.

        Args:
            drop_wt: Whether to exclude the wild-type sequence. Defaults to
                ``False``, which includes ``target_sequence`` under ``"WT"`` even
                when it has no fitness measurement.

        Returns:
            A mapping from mutation identifiers to amino acid sequences.
        """
        df = self.get_fitness_df(drop_wt=drop_wt)
        df = df.select("mutant", "sequence").unique(maintain_order=True)
        all_sequences = dict(df.iter_rows())
        if not drop_wt:
            all_sequences["WT"] = self.target_sequence
        return all_sequences

    def __repr__(self) -> str:
        """Return a compact summary of the mutant fitness dataset."""
        return (
            f"{type(self).__name__}(dataset_name={self.dataset_name!r}, "
            f"length={len(self.target_sequence)}, properties={self.properties!r})"
        )
